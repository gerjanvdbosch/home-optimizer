"""Periodic MPC runner — schedules :func:`solve_mpc_step` via APScheduler.

This module bridges the gap between the scheduled world (APScheduler jobs
running in the background) and the MPC core (:func:`~home_optimizer.api.solve_mpc_step`).

Architecture
------------
:class:`MPCRunner` is the single integration point.  At each trigger it:

1. Optionally reads live sensor state from a
   :class:`~home_optimizer.sensors.base.SensorBackend` (addon mode).
   When no backend is provided it uses the configured fallback initial
   conditions (local-runner / testing mode).
2. Overrides the ``RunRequest`` initial conditions with live readings.
3. Calls :func:`~home_optimizer.api.solve_mpc_step` — the *same* core solver
   used by the ``POST /api/simulate`` HTTP endpoint.
4. Logs the first-step control actions so an operator can see what the MPC
   recommends at each interval.

Design decisions
----------------
* **No magic numbers**: all physical parameters come from the injected
  :class:`RunRequest` (which is fully validated by Pydantic on construction).
* **Fail-fast on backend errors**: a sensor read failure logs the exception
  but the MPC run is *skipped* for that interval.  The scheduler continues
  and retries at the next interval, so a single bad reading does not crash
  the process.
* **DRY**: :func:`~home_optimizer.api.solve_mpc_step` is called here, not
  re-implemented.  Chart generation is deliberately skipped to avoid the
  Plotly serialisation overhead in the background thread.
* **Thread-safety**: :class:`MPCRunner` is stateless between calls (no mutable
  instance attributes modified during a run), so APScheduler can safely call
  :meth:`run_once` from a background thread.

Units: power [kW], temperature [°C], time [h].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sensors.base import SensorBackend

log = logging.getLogger("home_optimizer.mpc_scheduler")


class MPCRunner:
    """Callable MPC job suitable for registration with APScheduler.

    Args:
        base_request:
            A fully-validated :class:`~home_optimizer.api.RunRequest` that
            provides all physical parameters and MPC weights.  Live sensor
            readings will *override* the initial-condition fields
            (``T_r_init``, ``T_b_init``, ``dhw_T_top_init``,
            ``dhw_T_bot_init``, ``outdoor_temperature_c``,
            ``dhw_t_mains_c``, ``dhw_t_amb_c``) when a backend is present.
        backend:
            Optional sensor backend.  When ``None`` the runner uses the
            initial conditions embedded in ``base_request`` (useful for the
            local development runner where no HA instance is available).
    """

    def __init__(
        self,
        base_request: "RunRequest",  # noqa: F821 – resolved at import time
        backend: "SensorBackend | None" = None,
    ) -> None:
        # Avoid a circular import: api.py imports mpc_scheduler indirectly
        # only at runtime via addon.py / local_runner.py.
        from .api import RunRequest  # noqa: PLC0415

        if not isinstance(base_request, RunRequest):
            raise TypeError(
                f"base_request must be a RunRequest instance, got {type(base_request)!r}"
            )
        self._base_request: RunRequest = base_request
        self._backend = backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """Execute one MPC step and log the recommended control actions.

        This method is designed to be called by APScheduler as a background
        job.  It is safe to call directly for testing.

        Workflow:
            1. Read live sensor state (if backend available).
            2. Override initial conditions in ``base_request`` with live data.
            3. Call :func:`~home_optimizer.api.solve_mpc_step`.
            4. Log the first-step UFH and DHW thermal powers [kW].

        Side effects:
            Writes INFO/WARNING log messages.  Does *not* mutate
            ``self._base_request``; instead it constructs a new
            ``RunRequest`` copy with updated fields via Pydantic's
            ``model_copy``.

        Raises:
            Nothing — all exceptions are caught and logged so that
            APScheduler does not reschedule the job with a failed state.
        """
        from .api import RunRequest, solve_mpc_step  # noqa: PLC0415

        try:
            req = self._build_request(RunRequest)
        except Exception as exc:  # noqa: BLE001
            log.warning("MPC run skipped — sensor read failed: %s", exc)
            return

        try:
            result = solve_mpc_step(req)
        except Exception as exc:  # noqa: BLE001
            log.error("MPC solve failed: %s", exc)
            return

        sol = result.solution
        log.info(
            "MPC step complete | status=%s | obj=%.3f | "
            "P_UFH[0]=%.2f kW | P_dhw[0]=%.2f kW | cost=%.4f €",
            sol.solver_status,
            sol.objective_value,
            result.p_ufh_kw[0],
            result.p_dhw_kw[0],
            result.total_cost_eur,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_request(self, RunRequest: type) -> object:
        """Build a :class:`RunRequest` with live sensor overrides where available.

        When no backend is configured the base request is returned unchanged
        (all initial conditions from the Pydantic model default or the
        caller-supplied values).

        Args:
            RunRequest: The ``RunRequest`` class (injected to avoid circular
                import at module level).

        Returns:
            A :class:`RunRequest` instance — either the original base request
            or a copy with live-sensor overrides applied.

        Raises:
            Exception: Re-raised from the sensor backend on read failure so
                that :meth:`run_once` can log and skip the interval.
        """
        if self._backend is None:
            # Local-runner mode: no live sensors — use config defaults.
            return self._base_request

        # Addon mode: override initial conditions with live sensor readings.
        readings = self._backend.read_all()

        # Estimate floor/slab temperature from supply/return average when
        # no floor sensor is present (common for UFH systems with the
        # Kalman filter disabled).  Falls back to T_r + 2 K as a safe
        # initial condition that avoids a cold-start bias.
        t_b_estimate = (
            (readings.hp_supply_temperature_c + readings.hp_return_temperature_c) / 2.0
            if readings.hp_supply_temperature_c > readings.room_temperature_c
            else readings.room_temperature_c + 2.0
        )

        overrides: dict = {
            "T_r_init": readings.room_temperature_c,
            "T_b_init": t_b_estimate,
            "outdoor_temperature_c": readings.outdoor_temperature_c,
        }

        if self._base_request.dhw_enabled:
            overrides["dhw_T_top_init"] = readings.dhw_top_temperature_c
            overrides["dhw_T_bot_init"] = readings.dhw_bottom_temperature_c
            overrides["dhw_t_mains_c"] = readings.t_mains_estimated_c
            overrides["dhw_t_amb_c"] = readings.boiler_ambient_temp_c

        return self._base_request.model_copy(update=overrides)


def schedule_mpc(
    runner: MPCRunner,
    scheduler: "BackgroundScheduler",  # type: ignore[name-defined]  # noqa: F821
    interval_seconds: int,
    run_immediately: bool = True,
) -> None:
    """Register the MPC runner as a recurring APScheduler job.

    Args:
        runner:           Configured :class:`MPCRunner` instance.
        scheduler:        Running APScheduler ``BackgroundScheduler``.
        interval_seconds: How often to trigger the MPC [s].  Must be > 0.
        run_immediately:  When ``True`` (default) the first run is triggered
                          synchronously *before* the scheduler takes over, so
                          an initial control action is available immediately
                          at start-up rather than after one full interval.

    Raises:
        ValueError: If ``interval_seconds`` ≤ 0.
    """
    if interval_seconds <= 0:
        raise ValueError(
            f"interval_seconds must be > 0, got {interval_seconds}.  "
            "Pass 0 to disable MPC scheduling (do not call this function)."
        )

    if run_immediately:
        log.info("Running initial MPC step before scheduling periodic job …")
        runner.run_once()

    scheduler.add_job(
        runner.run_once,
        trigger="interval",
        seconds=interval_seconds,
        id="mpc_periodic",
        replace_existing=True,
        misfire_grace_time=interval_seconds // 2,
    )
    log.info(
        "MPC periodic job scheduled: every %d s (%d min)",
        interval_seconds,
        interval_seconds // 60,
    )
