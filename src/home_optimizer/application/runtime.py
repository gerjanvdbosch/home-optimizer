"""Scheduled runtime orchestration for periodic optimizer execution."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from threading import Lock
from typing import TYPE_CHECKING

from .forecasting import build_repository_forecast_overrides
from .models import MPCStepResult, RunRequest, ScheduledRunSnapshot
from .request_handling import build_safe_calibration_overrides, merge_run_request_updates

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

    from ..sensors.base import SensorBackend
    from ..telemetry.repository import TelemetryRepository
    from .optimizer import Optimizer

log = logging.getLogger("home_optimizer.application.runtime")


class OptimizerRuntime:
    """Runtime helper that owns scheduled-input assembly and snapshot storage."""

    _latest_scheduled_snapshot: ScheduledRunSnapshot | None = None
    _snapshot_lock: Lock = Lock()

    @classmethod
    def run_scheduled_once(
        cls,
        *,
        optimizer: "Optimizer",
        base_input: RunRequest,
        backend: "SensorBackend | None" = None,
        repository: "TelemetryRepository | None" = None,
    ) -> MPCStepResult | None:
        """Execute one periodic MPC step and cache the successful result snapshot."""
        try:
            optimizer_input = cls.build_scheduled_input(
                base_input=base_input,
                backend=backend,
                repository=repository,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("MPC run skipped — sensor read failed: %s", exc)
            return None

        try:
            result = optimizer.solve(optimizer_input)
        except Exception as exc:  # noqa: BLE001
            log.error("MPC solve failed: %s", exc)
            return None

        solution = result.solution
        with cls._snapshot_lock:
            cls._latest_scheduled_snapshot = ScheduledRunSnapshot(
                solved_at_utc=datetime.now(tz=timezone.utc),
                request=optimizer_input,
                result=result,
            )
        log.info(
            "MPC step complete | status=%s | obj=%.3f | "
            "P_UFH[0]=%.2f kW | P_dhw[0]=%.2f kW | cost=%.4f EUR",
            solution.solver_status,
            solution.objective_value,
            result.p_ufh_kw[0],
            result.p_dhw_kw[0],
            result.total_cost_eur,
        )
        return result

    @classmethod
    def get_latest_scheduled_snapshot(cls) -> ScheduledRunSnapshot | None:
        """Return the most recent successful scheduled MPC snapshot."""
        with cls._snapshot_lock:
            return cls._latest_scheduled_snapshot

    @classmethod
    def clear_latest_scheduled_snapshot(cls) -> None:
        """Clear the cached scheduled snapshot."""
        with cls._snapshot_lock:
            cls._latest_scheduled_snapshot = None

    @classmethod
    def schedule_periodic(
        cls,
        *,
        optimizer: "Optimizer",
        base_input: RunRequest,
        scheduler: "BackgroundScheduler",
        interval_seconds: int,
        backend: "SensorBackend | None" = None,
        repository: "TelemetryRepository | None" = None,
        run_immediately: bool = True,
    ) -> None:
        """Register periodic MPC execution on an APScheduler scheduler."""
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be > 0, got {interval_seconds}. "
                "Pass 0 to disable MPC scheduling (do not call this method)."
            )

        if run_immediately:
            log.info("Running initial MPC step before scheduling periodic job...")
            cls.run_scheduled_once(
                optimizer=optimizer,
                base_input=base_input,
                backend=backend,
                repository=repository,
            )

        scheduler.add_job(
            cls.run_scheduled_once,
            trigger="interval",
            seconds=interval_seconds,
            kwargs={
                "optimizer": optimizer,
                "base_input": base_input,
                "backend": backend,
                "repository": repository,
            },
            id="mpc_periodic",
            replace_existing=True,
            misfire_grace_time=max(1, interval_seconds // 2),
        )
        log.info(
            "MPC periodic job scheduled: every %d s (%d min)",
            interval_seconds,
            interval_seconds // 60,
        )

    @staticmethod
    def build_scheduled_input(
        *,
        base_input: RunRequest,
        backend: "SensorBackend | None",
        repository: "TelemetryRepository | None" = None,
    ) -> RunRequest:
        """Build one solver input by applying calibration, live sensor, and forecast overrides."""
        overrides: dict = {}

        if repository is not None:
            try:
                overrides.update(build_safe_calibration_overrides(base_input, repository))
            except Exception as exc:  # noqa: BLE001
                log.warning("Calibration snapshot DB read failed: %s", exc)

        if backend is not None:
            effective_request = merge_run_request_updates(base_input, overrides)
            readings = backend.read_all()
            corrected_room_temperature_c = (
                readings.room_temperature_c + effective_request.room_temperature_bias_c
            )
            slab_temperature_estimate_c = (
                (readings.hp_supply_temperature_c + readings.hp_return_temperature_c) / 2.0
                if readings.hp_supply_temperature_c > corrected_room_temperature_c
                else corrected_room_temperature_c + 2.0
            )
            overrides.update(
                {
                    "T_r_init": corrected_room_temperature_c,
                    "T_b_init": slab_temperature_estimate_c,
                    "outdoor_temperature_c": readings.outdoor_temperature_c,
                    "shutter_living_room_pct": readings.shutter_living_room_pct,
                }
            )

            if base_input.dhw_enabled:
                overrides["dhw_T_top_init"] = (
                    readings.dhw_top_temperature_c + effective_request.dhw_top_temperature_bias_c
                )
                overrides["dhw_T_bot_init"] = (
                    readings.dhw_bottom_temperature_c + effective_request.dhw_bottom_temperature_bias_c
                )
                overrides["dhw_t_mains_c"] = readings.t_mains_estimated_c
                overrides["dhw_t_amb_c"] = (
                    readings.boiler_ambient_temp_c + effective_request.dhw_boiler_ambient_bias_c
                )

        if repository is not None:
            try:
                rows, forecast_overrides = build_repository_forecast_overrides(
                    base_input,
                    repository,
                    existing=overrides,
                )
                if rows:
                    overrides.update(forecast_overrides)
            except Exception as exc:  # noqa: BLE001
                log.warning("Forecast DB read failed; continuing with request fallbacks: %s", exc)

        return merge_run_request_updates(base_input, overrides)


__all__ = ["OptimizerRuntime"]
