"""Periodic MPC runner — schedules :class:`~home_optimizer.optimizer.Optimizer` via APScheduler.

This module bridges the gap between the scheduled world (APScheduler jobs
running in the background) and the MPC core
(:class:`~home_optimizer.optimizer.Optimizer`).

Architecture
------------
:class:`MPCRunner` is the single integration point.  At each trigger it:

1. Optionally reads live sensor state from a
   :class:`~home_optimizer.sensors.base.SensorBackend` (addon mode).
   When no backend is provided it uses the configured fallback initial
   conditions (local-runner / testing mode).
2. Applies live sensor overrides to the base :class:`~home_optimizer.optimizer.OptimizerInput`
   via :func:`dataclasses.replace`.
3. Calls :meth:`~home_optimizer.optimizer.Optimizer.solve` directly — no API
   layer involved.
4. Logs the first-step control actions so an operator can see what the MPC
   recommends at each interval.

Design decisions
----------------
* **No API dependency**: the scheduler works exclusively with
  :class:`~home_optimizer.optimizer.OptimizerInput` and
  :class:`~home_optimizer.optimizer.Optimizer`.  ``api.py`` and ``RunRequest``
  are never imported here.
* **No magic numbers**: all physical parameters come from the injected
  :class:`~home_optimizer.optimizer.OptimizerInput`.
* **Fail-fast on backend errors**: a sensor read failure logs the exception
  but the MPC run is *skipped* for that interval.  The scheduler continues
  and retries at the next interval, so a single bad reading does not crash
  the process.
* **DRY**: :meth:`~home_optimizer.optimizer.Optimizer.solve` is called here,
  not re-implemented.  Chart generation is deliberately skipped to avoid the
  Plotly serialisation overhead in the background thread.
* **Thread-safety**: :class:`MPCRunner` is stateless between calls (no mutable
  instance attributes modified during a run), so APScheduler can safely call
  :meth:`run_once` from a background thread.

Units: power [kW], temperature [°C], time [h].
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from .optimizer import Optimizer, OptimizerInput

if TYPE_CHECKING:
    from .sensors.base import SensorBackend

log = logging.getLogger("home_optimizer.mpc_scheduler")


class MPCRunner:
    """Callable MPC job suitable for registration with APScheduler.

    Args:
        base_input:
            A fully-populated :class:`~home_optimizer.optimizer.OptimizerInput`
            that provides all physical parameters and MPC weights.  Live sensor
            readings will *override* the initial-condition fields
            (``T_r_init``, ``T_b_init``, ``dhw_T_top_init``,
            ``dhw_T_bot_init``, ``outdoor_temperature_c``,
            ``dhw_t_mains_c``, ``dhw_t_amb_c``) when a backend is present.
        backend:
            Optional sensor backend.  When ``None`` the runner uses the
            initial conditions embedded in ``base_input`` (useful for the
            local development runner where no HA instance is available).
    """

    def __init__(
        self,
        base_input: OptimizerInput,
        backend: "SensorBackend | None" = None,
    ) -> None:
        if not isinstance(base_input, OptimizerInput):
            raise TypeError(
                f"base_input must be an OptimizerInput instance, got {type(base_input)!r}"
            )
        self._base_input: OptimizerInput = base_input
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
            2. Apply sensor overrides to ``base_input`` via
               :func:`dataclasses.replace` (immutable — no mutation).
            3. Call :meth:`~home_optimizer.optimizer.Optimizer.solve`.
            4. Log the first-step UFH and DHW thermal powers [kW].

        Side effects:
            Writes INFO/WARNING log messages.  Does *not* mutate
            ``self._base_input``.

        Raises:
            Nothing — all exceptions are caught and logged so that
            APScheduler does not reschedule the job with a failed state.
        """
        try:
            optimizer_input = self._build_optimizer_input()
        except Exception as exc:  # noqa: BLE001
            log.warning("MPC run skipped — sensor read failed: %s", exc)
            return

        try:
            result = Optimizer().solve(optimizer_input)
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

    def _build_optimizer_input(self) -> OptimizerInput:
        """Return an :class:`~home_optimizer.optimizer.OptimizerInput` for this step.

        When no backend is configured the base input is returned unchanged
        (all initial conditions from the caller-supplied values).

        When a backend is available, live sensor readings override the
        initial-condition fields via :func:`dataclasses.replace`, leaving all
        physical parameters and MPC weights intact.

        Returns:
            :class:`~home_optimizer.optimizer.OptimizerInput` — either the
            original base input or an immutable copy with sensor overrides.

        Raises:
            Exception: Re-raised from the sensor backend on read failure so
                that :meth:`run_once` can log and skip the interval.
        """
        if self._backend is None:
            # Local-runner mode: no live sensors — use config defaults.
            return self._base_input

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

        if self._base_input.dhw_enabled:
            overrides["dhw_T_top_init"] = readings.dhw_top_temperature_c
            overrides["dhw_T_bot_init"] = readings.dhw_bottom_temperature_c
            overrides["dhw_t_mains_c"] = readings.t_mains_estimated_c
            overrides["dhw_t_amb_c"] = readings.boiler_ambient_temp_c

        return dataclasses.replace(self._base_input, **overrides)


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
