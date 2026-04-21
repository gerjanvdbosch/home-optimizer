"""Hourly Open-Meteo forecast persistence.

The :class:`ForecastPersister` runs on its own APScheduler interval (typically
once per UTC hour) and persists the full N-step Open-Meteo forecast into the
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

The :class:`ForecastPersister` owns its own
:class:`~home_optimizer.sensors.OpenMeteoClient` instance and calls
:meth:`~home_optimizer.sensors.OpenMeteoClient.get_forecast` directly on each
scheduled tick.  This is architecturally clean: ``ForecastPersister`` is
responsible for *forecast data*; ``BufferedTelemetryCollector`` is responsible
for *sensor data*.  No shared state or cache between the two.

Database uniqueness
-------------------
Each ``(fetched_at_utc, step_k)`` pair is unique in ``forecast_snapshots``.
If the persister is called multiple times within the same UTC hour (e.g. after
a restart), the duplicate-skip logic in the repository skips already-persisted
steps instead of raising an error.

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

from ..sensors.open_meteo import OpenMeteoClient
from .repository import TelemetryRepository

#: Default APScheduler job ID for the forecast persister.
DEFAULT_FORECAST_JOB_ID: str = "forecast-persist"

#: Number of seconds between forecast persistence runs.
#: Open-Meteo updates forecasts roughly every hour; re-fetching more often
#: wastes API calls.  3600 s = 1 h.
DEFAULT_FORECAST_INTERVAL_SECONDS: int = 3600

#: Default forecast horizon fetched from Open-Meteo [h].
#: 48 h gives two days of lookahead — sufficient for MPC and forecast-error training.
DEFAULT_FORECAST_HORIZON_HOURS: int = 48


class ForecastPersister:
    """Persist the full Open-Meteo N-step forecast to ``forecast_snapshots`` hourly.

    Parameters
    ----------
    weather_client:
        Configured :class:`~home_optimizer.sensors.OpenMeteoClient` for the
        site location and window / PV orientation.  The persister calls
        :meth:`~OpenMeteoClient.get_forecast` directly on each scheduled tick.
    repository:
        :class:`TelemetryRepository` used to write to ``forecast_snapshots``.
    horizon_hours:
        Number of forecast hours to fetch and persist per run.
        Default ``48`` (two days).  Must be ≥ 1.
    interval_seconds:
        APScheduler interval [s].  Default ``3600`` (one hour).
    job_id:
        APScheduler job identifier.  Override when running multiple persisters.
    """

    def __init__(
        self,
        weather_client: OpenMeteoClient,
        repository: TelemetryRepository,
        horizon_hours: int = DEFAULT_FORECAST_HORIZON_HOURS,
        interval_seconds: int = DEFAULT_FORECAST_INTERVAL_SECONDS,
        job_id: str = DEFAULT_FORECAST_JOB_ID,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be strictly positive.")
        if horizon_hours < 1:
            raise ValueError("horizon_hours must be ≥ 1.")
        self._weather = weather_client
        self._repository = repository
        self._horizon_hours = horizon_hours
        self._interval_seconds = interval_seconds
        self._job_id = job_id

    # ------------------------------------------------------------------
    # Core persistence logic
    # ------------------------------------------------------------------

    def persist_once(self) -> int:
        """Fetch the current Open-Meteo forecast and persist it to ``forecast_snapshots``.

        Calls :meth:`~OpenMeteoClient.get_forecast` with the configured
        ``horizon_hours`` at 1-hour resolution, then inserts one row per step.

        Duplicate rows (same ``fetched_at_utc`` + ``step_k``) are silently
        skipped via the repository's duplicate-handling logic.

        Returns
        -------
        int
            Number of rows successfully inserted (0 if all were duplicates).
        """
        forecast = self._weather.get_forecast(
            horizon_hours=self._horizon_hours,
            dt_hours=1.0,
        )

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
