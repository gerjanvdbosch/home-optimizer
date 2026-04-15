"""Hourly Open-Meteo forecast persistence.

The :class:`ForecastPersister` runs on its own APScheduler interval (typically
once per UTC hour) and persists the full N-step forecast from
:class:`~home_optimizer.sensors.WeatherAugmentedBackend` into the
``forecast_snapshots`` table.

Design rationale
----------------
Forecast data has a fundamentally **different update cadence** than live sensor
telemetry:

* **Telemetry**: sampled every 30 s, flushed every 5 min.
* **Forecast**: updated at most once per UTC hour (Open-Meteo update cycle).

Keeping forecast persistence in a separate class with its own APScheduler job
avoids polluting the 30-second telemetry loop with hourly API calls and makes
the update frequencies explicit.

The ``ForecastPersister`` reads the cached :class:`~home_optimizer.sensors.WeatherForecast`
from :attr:`WeatherAugmentedBackend.latest_forecast` so **no extra API call**
is made — the data was already fetched by the backend when the first telemetry
sample of the current UTC hour was taken.

If the backend cache is not yet populated (e.g. at startup before the first
``read_all()``), :meth:`ForecastPersister.persist_once` calls
:meth:`WeatherAugmentedBackend.warm_up` to force an immediate fetch.

Database uniqueness
-------------------
Each ``(fetched_at_utc, step_k)`` pair is unique in ``forecast_snapshots``.
If the persister is called multiple times within the same UTC hour (e.g. after
a restart), the ``INSERT OR IGNORE`` / SQLAlchemy exception handling skips
already-persisted steps instead of raising an error.

Units
-----
Temperature : °C
Irradiance  : W/m²
Time        : UTC datetimes
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from ..sensors.weather_backend import WeatherAugmentedBackend
from .repository import TelemetryRepository

#: Default APScheduler job ID for the forecast persister.
DEFAULT_FORECAST_JOB_ID: str = "forecast-persist"

#: Number of seconds between forecast persistence runs.
#: Open-Meteo updates forecasts roughly every hour; re-fetching more often
#: wastes API calls.  3600 s = 1 h.
DEFAULT_FORECAST_INTERVAL_SECONDS: int = 3600


class ForecastPersister:
    """Persist the full Open-Meteo N-step forecast to ``forecast_snapshots`` hourly.

    Parameters
    ----------
    backend:
        :class:`~home_optimizer.sensors.WeatherAugmentedBackend` that holds the
        cached :class:`~home_optimizer.sensors.WeatherForecast`.  The persister
        reads :attr:`~WeatherAugmentedBackend.latest_forecast` and calls
        :meth:`~WeatherAugmentedBackend.warm_up` on the first run if the cache
        is empty.
    repository:
        :class:`TelemetryRepository` used to write to ``forecast_snapshots``.
    interval_seconds:
        APScheduler interval [s].  Default ``3600`` (one hour).
    job_id:
        APScheduler job identifier.  Override when running multiple persisters.
    """

    def __init__(
        self,
        backend: WeatherAugmentedBackend,
        repository: TelemetryRepository,
        interval_seconds: int = DEFAULT_FORECAST_INTERVAL_SECONDS,
        job_id: str = DEFAULT_FORECAST_JOB_ID,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be strictly positive.")
        self._backend = backend
        self._repository = repository
        self._interval_seconds = interval_seconds
        self._job_id = job_id

    # ------------------------------------------------------------------
    # Core persistence logic
    # ------------------------------------------------------------------

    def persist_once(self) -> int:
        """Persist the current forecast to ``forecast_snapshots``.

        Reads :attr:`WeatherAugmentedBackend.latest_forecast`.  If the cache is
        empty (first call before any ``read_all()``), calls
        :meth:`WeatherAugmentedBackend.warm_up` to populate it first.

        Duplicate rows (same ``fetched_at_utc`` + ``step_k``) are silently
        skipped via SQLAlchemy's ``INSERT OR IGNORE`` emulation — the repository
        raises :class:`sqlalchemy.exc.IntegrityError` which is caught here.

        Returns
        -------
        int
            Number of rows successfully inserted (0 if all were duplicates).
        """
        forecast = self._backend.latest_forecast
        if forecast is None:
            # Cache is empty — warm up to trigger the first fetch.
            forecast = self._backend.warm_up()

        rows: list[dict[str, Any]] = []
        for k in range(forecast.horizon_steps):
            valid_at = forecast.valid_from + timedelta(hours=float(k) * forecast.dt_hours)
            gti_pv = (
                max(float(forecast.gti_pv_w_per_m2[k]), 0.0)
                if forecast.gti_pv_w_per_m2 is not None
                else 0.0
            )
            rows.append(
                {
                    "fetched_at_utc": forecast.valid_from,
                    "valid_at_utc": valid_at,
                    "step_k": k,
                    "dt_hours": float(forecast.dt_hours),
                    "t_out_c": float(forecast.outdoor_temperature_c[k]),
                    "gti_w_per_m2": max(float(forecast.gti_w_per_m2[k]), 0.0),
                    "gti_pv_w_per_m2": gti_pv,
                }
            )

        return self._repository.bulk_add_forecast_snapshots(rows)

    # ------------------------------------------------------------------
    # APScheduler integration
    # ------------------------------------------------------------------

    def start(
        self,
        scheduler: BackgroundScheduler,
        *,
        run_immediately: bool = True,
    ) -> None:
        """Register the hourly persistence job on an APScheduler instance.

        Parameters
        ----------
        scheduler:
            A running or not-yet-started :class:`BackgroundScheduler`.
        run_immediately:
            If ``True`` (default), call :meth:`persist_once` immediately before
            registering the interval job.  This ensures forecast data is
            available from the very first second of operation rather than waiting
            for the first scheduled tick.
        """
        if run_immediately:
            self.persist_once()

        scheduler.add_job(
            self.persist_once,
            trigger="interval",
            seconds=self._interval_seconds,
            id=self._job_id,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    @property
    def job_id(self) -> str:
        """APScheduler job identifier for this persister."""
        return self._job_id

