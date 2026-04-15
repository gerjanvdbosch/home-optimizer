"""Weather-augmented sensor backend.

Wraps any :class:`SensorBackend` and injects current Open-Meteo GTI (for windows
and PV panels), a seasonal T_mains estimate, and the forecast T_out for the current
hour into every :class:`LiveReadings` snapshot.

Architecture
------------
* The *inner* backend (e.g. :class:`HomeAssistantBackend`) handles all HA sensor
  readings, but produces placeholder values (``0.0``) for the weather fields.
* :class:`WeatherAugmentedBackend` queries Open-Meteo **at most once per UTC hour**,
  caches the full :class:`~home_optimizer.sensors.WeatherForecast` (N-step horizon)
  thread-safely, and uses ``dataclasses.replace()`` to overwrite the weather fields
  in every snapshot before returning it.
* The full cached forecast is exposed via :attr:`latest_forecast` so that
  :class:`~home_optimizer.telemetry.ForecastPersister` can persist all N steps to the
  ``forecast_snapshots`` table without making a second API call.
* :class:`SeasonalMainsModel` computes T_mains purely from the calendar date
  (no network call) and is evaluated on every read.

Typical production setup::

    from home_optimizer.sensors import (
        HomeAssistantBackend, HAEntityConfig,
        OpenMeteoClient, SeasonalMainsModel, WeatherAugmentedBackend,
    )

    ha_backend = HomeAssistantBackend(...)
    weather_client = OpenMeteoClient(
        latitude=52.37, longitude=4.90,
        tilt=90, azimuth=0,       # south-facing windows (§4, Q_solar)
        pv_tilt=35, pv_azimuth=0, # south-facing PV panels
    )
    mains_model = SeasonalMainsModel.for_netherlands()

    backend = WeatherAugmentedBackend(
        ha_backend, weather_client, mains_model,
        forecast_horizon_hours=48,   # full horizon cached for ForecastPersister
    )
    readings = backend.read_all()
    # readings.gti_w_per_m2        ← current-hour GTI for windows [W/m²]
    # readings.gti_pv_w_per_m2     ← current-hour GTI for PV panels [W/m²]
    # readings.t_mains_estimated_c ← seasonal cold mains estimate [°C]
    # readings.t_out_forecast_c    ← Open-Meteo forecast T_out for current hour [°C]

Units
-----
GTI     : W/m²
T_out   : °C
T_mains : °C
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

from .base import LiveReadings, SensorBackend
from .open_meteo import OpenMeteoClient, SeasonalMainsModel, WeatherForecast

#: Default forecast horizon fetched from Open-Meteo [h].
#: 48 h gives two days of lookahead — sufficient for MPC and forecast-error training.
#: Must be ≥ 2 to avoid hour-boundary edge cases at step 0.
DEFAULT_FORECAST_HORIZON_HOURS: int = 48


@dataclass(frozen=True, slots=True)
class WeatherCurrentValues:
    """Cached current-hour weather values extracted from the Open-Meteo forecast.

    Attributes
    ----------
    gti_w_per_m2:
        Global Tilted Irradiance for south-facing windows [W/m²].  Always ≥ 0.
        Corresponds to ``WeatherForecast.gti_w_per_m2[0]`` (step 0 = current hour).
    gti_pv_w_per_m2:
        Global Tilted Irradiance for PV panels [W/m²].  0.0 when ``pv_tilt``
        is not set on the :class:`OpenMeteoClient`.  Always ≥ 0.
    t_out_forecast_c:
        Open-Meteo forecast outdoor temperature for the current UTC hour [°C].
        Compare with ``LiveReadings.outdoor_temperature_c`` (measured) to compute
        forecast error (§16, training requirement 7).
    hour_utc:
        UTC hour (0–23) for which these values are valid.  The cache is
        invalidated when the UTC clock advances to a new hour.
    """

    gti_w_per_m2: float
    gti_pv_w_per_m2: float
    t_out_forecast_c: float
    hour_utc: int


class WeatherAugmentedBackend(SensorBackend):
    """SensorBackend wrapper that injects Open-Meteo weather data into readings.

    The wrapper calls the *inner* backend first, then replaces the four weather
    fields (``gti_w_per_m2``, ``gti_pv_w_per_m2``, ``t_mains_estimated_c``,
    ``t_out_forecast_c``) using the current Open-Meteo forecast and the seasonal
    mains model.

    Open-Meteo is queried **at most once per UTC hour**; the full N-step forecast
    is cached so that :class:`~home_optimizer.telemetry.ForecastPersister` can
    read :attr:`latest_forecast` without triggering another API call.

    Parameters
    ----------
    inner:
        Any :class:`SensorBackend` that populates all non-weather
        :class:`LiveReadings` fields.  Its weather-field values are discarded
        and replaced by this wrapper.
    weather_client:
        Configured :class:`OpenMeteoClient` for the site location and
        window / PV orientation.  Set ``pv_tilt`` to enable
        ``gti_pv_w_per_m2``; leave it ``None`` for ``gti_pv = 0.0``.
    mains_model:
        :class:`SeasonalMainsModel` instance.  Use
        ``SeasonalMainsModel.for_netherlands()`` for Dutch sites.
    forecast_horizon_hours:
        How many hours ahead to fetch from Open-Meteo each refresh.
        Default ``48`` (two days).  Must be ≥ 2.
        The full forecast is available via :attr:`latest_forecast` for
        :class:`~home_optimizer.telemetry.ForecastPersister` to persist.
    """

    def __init__(
        self,
        inner: SensorBackend,
        weather_client: OpenMeteoClient,
        mains_model: SeasonalMainsModel,
        forecast_horizon_hours: int = DEFAULT_FORECAST_HORIZON_HOURS,
    ) -> None:
        if not isinstance(inner, SensorBackend):
            raise TypeError(
                f"inner must be a SensorBackend instance, got {type(inner).__name__}."
            )
        if forecast_horizon_hours < 2:
            raise ValueError("forecast_horizon_hours must be ≥ 2.")
        self._inner = inner
        self._weather = weather_client
        self._mains = mains_model
        self._forecast_horizon_hours = forecast_horizon_hours
        self._cached_forecast: WeatherForecast | None = None
        self._cached_current: WeatherCurrentValues | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Weather cache management
    # ------------------------------------------------------------------

    def _refresh_cache(self, hour_now: int) -> None:
        """Fetch a new Open-Meteo forecast and update both caches.

        Must be called while ``self._lock`` is held.

        Fetches ``forecast_horizon_hours`` steps at 1-hour resolution.
        Extracts step-0 values for ``WeatherCurrentValues``.
        """
        forecast = self._weather.get_forecast(
            horizon_hours=self._forecast_horizon_hours,
            dt_hours=1.0,
        )
        # Step 0 = current UTC hour.  Clamp irradiance to ≥ 0.
        gti = max(float(forecast.gti_w_per_m2[0]), 0.0)
        gti_pv = (
            max(float(forecast.gti_pv_w_per_m2[0]), 0.0)
            if forecast.gti_pv_w_per_m2 is not None
            else 0.0
        )
        t_out_fc = float(forecast.outdoor_temperature_c[0])

        self._cached_forecast = forecast
        self._cached_current = WeatherCurrentValues(
            gti_w_per_m2=gti,
            gti_pv_w_per_m2=gti_pv,
            t_out_forecast_c=t_out_fc,
            hour_utc=hour_now,
        )

    def _get_current_weather(self) -> WeatherCurrentValues:
        """Return cached step-0 weather values, refreshing if the UTC hour changed.

        Thread-safe: protected by ``self._lock``.  All threads within the same
        UTC hour share a single API call.
        """
        hour_now = datetime.now(tz=timezone.utc).hour
        with self._lock:
            if self._cached_current is None or self._cached_current.hour_utc != hour_now:
                self._refresh_cache(hour_now)
            return self._cached_current  # type: ignore[return-value]

    @property
    def latest_forecast(self) -> WeatherForecast | None:
        """The most recently cached full Open-Meteo forecast, or ``None`` if not yet fetched.

        Exposed for :class:`~home_optimizer.telemetry.ForecastPersister` so the
        persister can access the N-step arrays without making a second API call.

        Returns
        -------
        WeatherForecast | None
            The cached forecast (all N steps, 1-hour resolution).
            ``None`` until the first ``read_all()`` or explicit warm-up call.
        """
        with self._lock:
            return self._cached_forecast

    def warm_up(self) -> WeatherForecast:
        """Force an immediate forecast fetch and cache population.

        Call this at startup before the first ``read_all()`` to ensure the
        :class:`~home_optimizer.telemetry.ForecastPersister` has data available
        from the very first scheduler tick.

        Returns
        -------
        WeatherForecast
            The freshly fetched forecast.
        """
        hour_now = datetime.now(tz=timezone.utc).hour
        with self._lock:
            self._refresh_cache(hour_now)
            return self._cached_forecast  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def read_all(self) -> LiveReadings:
        """Read sensor snapshot from the inner backend and inject weather fields.

        Sequence:
        1. Fetch (or return cached) full forecast; extract step-0 current values.
        2. Compute T_mains from the seasonal model (pure date function, no I/O).
        3. Call ``inner.read_all()`` for all HA/local sensor values.
        4. Replace the four weather fields using ``dataclasses.replace()``.

        Returns
        -------
        LiveReadings
            Fully populated snapshot with live weather data injected.
        """
        weather = self._get_current_weather()
        t_mains = self._mains.estimate(datetime.now(tz=timezone.utc))
        reading = self._inner.read_all()
        # dataclasses.replace() works on frozen dataclasses with __slots__:
        # it creates a new instance with the specified fields overwritten.
        return dataclasses.replace(
            reading,
            gti_w_per_m2=weather.gti_w_per_m2,
            gti_pv_w_per_m2=weather.gti_pv_w_per_m2,
            t_mains_estimated_c=t_mains,
            t_out_forecast_c=weather.t_out_forecast_c,
        )

    def close(self) -> None:
        """Close the inner backend's resources.

        The forecast cache is in-process; only the inner backend may hold connections.
        """
        self._inner.close()
