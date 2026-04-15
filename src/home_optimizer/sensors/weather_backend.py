"""Weather-augmented sensor backend.

Wraps any :class:`SensorBackend` and injects current Open-Meteo GTI (for windows
and PV panels) plus a seasonal T_mains estimate into every :class:`LiveReadings`
snapshot.

Architecture
------------
* The *inner* backend (e.g. :class:`HomeAssistantBackend`) handles all HA sensor
  readings, but produces placeholder values (``0.0``) for the three weather fields.
* :class:`WeatherAugmentedBackend` queries Open-Meteo at most once per UTC hour,
  caches the result thread-safely, and uses ``dataclasses.replace()`` to overwrite
  the weather fields in every snapshot before returning it.
* :class:`SeasonalMainsModel` computes T_mains purely from the calendar date
  (no network call) and is evaluated on every read.

Typical production setup::

    from home_optimizer.sensors import (
        HomeAssistantBackend, HAEntityConfig,
        OpenMeteoClient, SeasonalMainsModel, WeatherAugmentedBackend,
    )

    ha_backend = HomeAssistantBackend(
        room_temperature=HAEntityConfig("sensor.living_room_temperature"),
        ...
    )
    weather_client = OpenMeteoClient(
        latitude=52.37,
        longitude=4.90,
        tilt=90,        # south-facing windows (§4, Q_solar)
        azimuth=0,
        pv_tilt=35,     # south-facing PV panels
        pv_azimuth=0,
    )
    mains_model = SeasonalMainsModel.for_netherlands()

    backend = WeatherAugmentedBackend(ha_backend, weather_client, mains_model)
    readings = backend.read_all()
    # readings.gti_w_per_m2       ← current-hour GTI for south-facing windows [W/m²]
    # readings.gti_pv_w_per_m2    ← current-hour GTI for PV panels [W/m²]
    # readings.t_mains_estimated_c ← seasonal cold mains estimate [°C]
    # readings.hp_thermal_power_kw  ← derived: flow × λ × ΔT  [kW]  (property)
    # readings.household_elec_power_kw ← derived: P1 + PV - HP  [kW]  (property)

Units
-----
GTI   : W/m²
T_mains : °C
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

from .base import LiveReadings, SensorBackend
from .open_meteo import OpenMeteoClient, SeasonalMainsModel

#: Minimum horizon requested from Open-Meteo.  2 hours avoids edge cases near
#: the hour boundary where step 0 might fall in the previous hour.
_WEATHER_FETCH_HORIZON_HOURS: int = 2


@dataclass(frozen=True, slots=True)
class WeatherCurrentValues:
    """Cached current-hour weather values from Open-Meteo.

    Attributes
    ----------
    gti_w_per_m2:
        Global Tilted Irradiance for south-facing windows [W/m²].  Always ≥ 0.
        Corresponds to ``WeatherForecast.gti_w_per_m2[0]`` (step 0 = current hour).
    gti_pv_w_per_m2:
        Global Tilted Irradiance for PV panels [W/m²].  0.0 when ``pv_tilt``
        is not set on the :class:`OpenMeteoClient`.  Always ≥ 0.
    hour_utc:
        UTC hour (0–23) for which these values are valid.  The cache is
        invalidated when the UTC clock advances to a new hour.
    """

    gti_w_per_m2: float
    gti_pv_w_per_m2: float
    hour_utc: int


class WeatherAugmentedBackend(SensorBackend):
    """SensorBackend wrapper that injects Open-Meteo GTI and T_mains into readings.

    The wrapper calls the *inner* backend first, then replaces the three weather
    fields (``gti_w_per_m2``, ``gti_pv_w_per_m2``, ``t_mains_estimated_c``) using
    the current Open-Meteo forecast hour and the seasonal mains model.

    Open-Meteo is queried **at most once per UTC hour**; results are cached
    thread-safely so multiple sampler threads share one API call.
    T_mains is computed from :meth:`SeasonalMainsModel.estimate` on every read
    (pure function of date — no network call).

    Parameters
    ----------
    inner:
        Any :class:`SensorBackend` that populates all non-weather
        :class:`LiveReadings` fields.  Its ``gti_w_per_m2``,
        ``gti_pv_w_per_m2``, and ``t_mains_estimated_c`` values are
        discarded and replaced by this wrapper.
        Typically a :class:`HomeAssistantBackend` or :class:`LocalBackend`.
    weather_client:
        Configured :class:`OpenMeteoClient` for the site location and
        window / PV orientation.  Set ``pv_tilt`` to enable
        ``gti_pv_w_per_m2``; leave it ``None`` for ``gti_pv = 0.0``.
    mains_model:
        :class:`SeasonalMainsModel` instance.  Use
        ``SeasonalMainsModel.for_netherlands()`` for Dutch sites, or
        instantiate with custom parameters from the Pydantic config.
    """

    def __init__(
        self,
        inner: SensorBackend,
        weather_client: OpenMeteoClient,
        mains_model: SeasonalMainsModel,
    ) -> None:
        if not isinstance(inner, SensorBackend):
            raise TypeError(
                f"inner must be a SensorBackend instance, got {type(inner).__name__}."
            )
        self._inner = inner
        self._weather = weather_client
        self._mains = mains_model
        self._cache: WeatherCurrentValues | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Weather cache management
    # ------------------------------------------------------------------

    def _get_current_weather(self) -> WeatherCurrentValues:
        """Return cached GTI values, fetching from Open-Meteo only when stale.

        The cache is per UTC hour: the first call in any given UTC hour triggers
        an API request; subsequent calls within the same hour return the cached
        result without a network round-trip.

        Thread-safety: protected by ``self._lock``; the fetch is done inside the
        lock so concurrent callers wait for the result rather than each making
        their own API call.
        """
        hour_now = datetime.now(tz=timezone.utc).hour
        with self._lock:
            if self._cache is not None and self._cache.hour_utc == hour_now:
                # Cache hit — return without network call.
                return self._cache

            # Cache miss or new UTC hour: fetch from Open-Meteo.
            forecast = self._weather.get_forecast(
                horizon_hours=_WEATHER_FETCH_HORIZON_HOURS,
                dt_hours=1.0,
            )
            # Step 0 = current UTC hour.  Clamp to ≥ 0 (Open-Meteo sometimes
            # returns small negative values due to sensor calibration offsets).
            gti = max(float(forecast.gti_w_per_m2[0]), 0.0)
            gti_pv = (
                max(float(forecast.gti_pv_w_per_m2[0]), 0.0)
                if forecast.gti_pv_w_per_m2 is not None
                else 0.0
            )
            self._cache = WeatherCurrentValues(
                gti_w_per_m2=gti,
                gti_pv_w_per_m2=gti_pv,
                hour_utc=hour_now,
            )
            return self._cache

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def read_all(self) -> LiveReadings:
        """Read sensor snapshot from the inner backend and inject weather fields.

        Sequence:
        1. Fetch (or return cached) current-hour GTI from Open-Meteo.
        2. Compute T_mains from the seasonal model.
        3. Call ``inner.read_all()`` for all HA/local sensor values.
        4. Replace the three weather fields using ``dataclasses.replace()``.

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
        )

    def close(self) -> None:
        """Close the inner backend's resources.

        The HTTP cache is stateless; only the inner backend may hold connections.
        """
        self._inner.close()

