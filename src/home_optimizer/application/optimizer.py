"""Core MPC optimisation logic — the ``Optimizer`` class.

Architecture
------------
This module contains the **domain core** of the Home Optimizer.  It is
deliberately kept free of any web-framework or visualisation concerns; those
belong in ``api.py`` (FastAPI).  Periodic APScheduler orchestration is exposed
via methods on :class:`Optimizer` so API and background execution share one
validated path.

The public surface is intentionally small:

* :class:`MPCStepResult` — an immutable dataclass carrying all numerical
  outputs of one optimisation step (control sequences, COP arrays, energy
  summaries, forecasts).  No charts — chart generation is the API's concern.
* :class:`Optimizer` — a stateless service object that accepts a
  :class:`RunRequest`, orchestrates the full solve
  pipeline (thermal model → COP model → forecasts → MPC → summaries) and
  returns an :class:`MPCStepResult`.

Design decisions
----------------
* **No magic numbers** — every numeric constant originates from the
  validated ``RunRequest`` (Pydantic), never from literals here.
* **Fail-fast** — invalid physics (non-positive capacities, COP ≤ 1, etc.)
  are caught by Pydantic and the ``HeatPumpCOPModel`` *before* the CVXPY
  solver is invoked.
* **DRY** — both the HTTP endpoint (``api.py``) and periodic scheduler calls
  use :meth:`Optimizer.solve`, not a copy of the logic.

Units: power [kW], temperature [°C], energy [kWh], time [h].
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from .forecasting import (
    ForecastBuilder,
    _FORECAST_SERVICE,
    build_repository_forecast_overrides,
    inject_forecast_overrides,
)
from .models import MPCStepResult, RunRequest, ScheduledRunSnapshot
from .pipeline import OptimizerPipeline
from .request_handling import (
    build_safe_calibration_overrides,
    merge_run_request_updates,
    sanitize_calibration_overrides,
    validate_run_request_physics,
)
from .runtime import OptimizerRuntime
from ..domain.heat_pump.cop import HeatPumpCOPModel
from ..pricing import PriceConfig  # noqa: F401 — PriceConfig re-exported via RunRequest
from ..types.forecast import DHWForecastHorizon, ForecastHorizon

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

    from ..sensors.base import SensorBackend
    from ..telemetry.repository import TelemetryRepository


# ---------------------------------------------------------------------------
# Optimizer — the domain core
# ---------------------------------------------------------------------------


class Optimizer:
    """Stateless MPC optimisation service.

    Orchestrates the full solve pipeline:

    1. Build the UFH :class:`~home_optimizer.domain.ufh.model.ThermalModel`.
    2. Build the Carnot :class:`~home_optimizer.domain.heat_pump.cop.HeatPumpCOPModel` (§14.1).
    3. Compute time-varying COP arrays over the MPC horizon.
    4. Construct UFH and DHW :class:`~home_optimizer.types.ForecastHorizon` objects.
    5. Optionally build the :class:`~home_optimizer.domain.dhw.model.DHWModel`.
    6. Solve the QP via :class:`~home_optimizer.control.mpc.MPCController` (CVXPY only).
    7. Compute energy and electricity-cost summaries.

    The class is *stateless* between calls: no mutable attributes are
    modified by :meth:`solve`.  It is therefore safe to share a single
    instance across threads (e.g. from the FastAPI dependency-injection
    container and the APScheduler background thread).

    Example::

        optimizer = Optimizer()
        result = optimizer.solve(req)
        print(result.p_ufh_kw[0])  # first-step UFH power [kW]
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, req: RunRequest) -> MPCStepResult:
        """Run one full MPC optimisation step.

        Args:
            req: :class:`RunRequest` with all physical and MPC parameters.
                 Callers are responsible for validating the parameters before
                 passing them (e.g. via Pydantic in the API layer).

        Returns:
            :class:`MPCStepResult` with the full solution and numerical
            summaries.  No Plotly charts — visualisation is the API's concern.

        Raises:
            ValueError: If any derived parameter violates a physical constraint
                (e.g. COP ≤ 1 after Carnot pre-calculation, or no convex solver
                reaches an optimal MPC solution).
        """
        start_hour = datetime.now(tz=timezone.utc).hour
        return OptimizerPipeline.solve(req, start_hour=start_hour)

    def run_scheduled_once(
        self,
        base_input: RunRequest,
        backend: "SensorBackend | None" = None,
        repository: "TelemetryRepository | None" = None,
    ) -> MPCStepResult | None:
        """Execute one periodic MPC step and log first-step actions.

        This method is intended for APScheduler jobs. It keeps the legacy
        scheduler behavior: sensor-read failures and solver failures are logged
        and skipped for the current interval.

        Args:
            base_input:  Fully populated request containing all physical and MPC
                parameters; live sensors override selected initial conditions.
            backend:     Optional sensor backend providing live readings.
            repository:  Optional telemetry repository.  When provided, the most
                recently persisted Open-Meteo forecast is injected as real
                ``t_out_forecast``, ``gti_window_forecast``, and
                ``gti_pv_forecast`` arrays, replacing the proxy sine-curves.

        Returns:
            :class:`MPCStepResult` on success; ``None`` when the step is
            skipped due to a recoverable read/solve failure.
        """
        return OptimizerRuntime.run_scheduled_once(
            optimizer=self,
            base_input=base_input,
            backend=backend,
            repository=repository,
        )

    @classmethod
    def get_latest_scheduled_snapshot(cls) -> ScheduledRunSnapshot | None:
        """Return the most recent successful scheduled MPC snapshot.

        Returns:
            :class:`ScheduledRunSnapshot` when at least one periodic run
            succeeded in this process; otherwise ``None``.
        """
        return OptimizerRuntime.get_latest_scheduled_snapshot()

    @classmethod
    def clear_latest_scheduled_snapshot(cls) -> None:
        """Clear cached scheduled snapshot (mainly used by tests)."""
        OptimizerRuntime.clear_latest_scheduled_snapshot()

    def schedule_periodic(
        self,
        base_input: RunRequest,
        scheduler: "BackgroundScheduler",
        interval_seconds: int,
        backend: "SensorBackend | None" = None,
        repository: "TelemetryRepository | None" = None,
        run_immediately: bool = True,
    ) -> None:
        """Register periodic MPC execution on an APScheduler scheduler.

        Args:
            base_input:       Immutable request template for periodic solves.
            scheduler:        Running APScheduler background scheduler.
            interval_seconds: Period between MPC runs [s], must be > 0.
            backend:          Optional sensor backend used for live overrides.
            repository:       Optional telemetry repository used to inject the
                latest Open-Meteo forecast arrays (t_out, GTI window, GTI PV)
                into each scheduled run.  When ``None`` the proxy sine-curves
                are used as fallback.
            run_immediately:  Run one synchronous step before scheduling.

        Raises:
            ValueError: If ``interval_seconds`` is not strictly positive.
        """
        OptimizerRuntime.schedule_periodic(
            optimizer=self,
            base_input=base_input,
            scheduler=scheduler,
            interval_seconds=interval_seconds,
            backend=backend,
            repository=repository,
            run_immediately=run_immediately,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _materialize_horizon_array(
        *,
        name: str,
        horizon_steps: int,
        values: list[float] | None,
        fallback_scalar: float | None = None,
    ) -> np.ndarray:
        """Return a full-horizon forecast array with fail-fast length validation.

        Args:
            name: Human-readable forecast name for error messages.
            horizon_steps: Required horizon length ``N`` [-].
            values: Optional explicit forecast samples. When provided, the array
                must contain at least ``N`` values; only the first ``N`` are used.
            fallback_scalar: Optional scalar value that is broadcast to length ``N``
                when ``values`` is absent. This is reserved for request fields that
                are explicitly scalar by design (e.g. current outdoor temperature).

        Returns:
            NumPy array with shape ``(N,)`` and dtype ``float``.

        Raises:
            ValueError: If the provided forecast is not one-dimensional or does not
                contain enough samples for the MPC horizon.
        """
        return ForecastBuilder.materialize_horizon_array(
            name=name,
            horizon_steps=horizon_steps,
            values=values,
            fallback_scalar=fallback_scalar,
        )

    @staticmethod
    def _build_cop_model(req: RunRequest) -> HeatPumpCOPModel:
        """Construct the Carnot COP model from the user request.

        Assembles a ``HeatPumpCOPParameters`` dataclass from the request fields
        and wraps it in a ``HeatPumpCOPModel``.  Validation (e.g. cop_min > 1,
        ``eta_carnot_ufh`` / ``eta_carnot_dhw`` ∈ (0,1]) is handled by
        ``HeatPumpCOPParameters.__post_init__``.

        Args:
            req: Validated Pydantic request containing all COP model parameters.

        Returns:
            Constructed and validated :class:`~home_optimizer.domain.heat_pump.cop.HeatPumpCOPModel`.

        Raises:
            ValueError: If any parameter violates a physical constraint.
        """
        return OptimizerPipeline.build_cop_model(req)

    @staticmethod
    def _build_ufh_forecast(
        req: RunRequest,
        start_hour: int,
        cop_model: HeatPumpCOPModel,
    ) -> ForecastHorizon:
        """Build the UFH disturbance and price forecast over the horizon.

        Injects the time-varying UFH COP array (computed from the Carnot model
        and outdoor temperature) so the MPC can use physical electricity costs.
        The electricity price array is constructed by the configured
        :class:`~home_optimizer.pricing.BasePriceModel` subclass
        (flat / dual-tariff / Nordpool), ensuring no magic numbers appear here.

        Args:
            req:        Validated request with all forecast parameters.
            start_hour: Current hour of the day (0–23), used to index the price
                        pattern and solar profile.
            cop_model:  Carnot COP model; used to compute ``cop_ufh_k`` array.

        Returns:
            :class:`~home_optimizer.types.ForecastHorizon` with N steps,
            including the COP array.
        """
        return ForecastBuilder.build_ufh_forecast(
            req,
            start_hour=start_hour,
            cop_model=cop_model,
        )

    @staticmethod
    def _build_internal_gains_profile(req: RunRequest, *, horizon_steps: int) -> np.ndarray:
        """Return the internal-gains horizon using explicit forecast or physical baseload mapping.

        Precedence:
        1. explicit ``internal_gains_forecast``
        2. ``max(internal_gains_kw, internal_gains_heat_fraction × baseload_forecast)``
        3. scalar ``internal_gains_kw`` broadcast across the horizon

        The baseload mapping expresses that only a fraction of electrical household
        demand becomes useful sensible heat in the conditioned zone, while the
        scalar baseline captures occupant/metabolic and other non-forecast gains.
        To avoid a second free tuning knob, the electrical baseload reference is
        derived implicitly from ``internal_gains_kw`` and
        ``internal_gains_heat_fraction`` instead of being exposed as a
        separate user parameter.
        """

        return ForecastBuilder.build_internal_gains_profile(
            req,
            horizon_steps=horizon_steps,
        )

    @staticmethod
    def _map_baseload_to_internal_gains_increment(
        *,
        baseload_profile_kw: np.ndarray,
        baseline_internal_gains_kw: float,
        heat_fraction: float,
    ) -> np.ndarray:
        """Return the incremental sensible heat gain implied by the electrical baseload.

        Implements the optimizer-side mapping

        ``Q_int[k] = max(Q_int,baseline, heat_fraction * P_baseload[k])``

        from the project-specific grey-box interpretation of internal gains.  The
        implicit electrical baseload reference is derived as

        ``P_reference = Q_int,baseline / heat_fraction``

        for ``heat_fraction > 0``. This keeps only one tuned baseline term in the
        request model and avoids separate calibration of two overlapping background
        offsets. The returned increment therefore equals

        ``max(heat_fraction * P_baseload[k] - Q_int,baseline, 0)``.

        Args:
            baseload_profile_kw: Forecast non-heat-pump household electrical demand
                over the horizon [kW], shape ``(N,)``.
            baseline_internal_gains_kw: Scalar background internal-gains level
                already present in the UFH disturbance model [kW].
            heat_fraction: Fraction of the excess electrical demand that appears
                indoors as useful sensible heat [-].

        Returns:
            Incremental internal-gains profile [kW], shape ``(N,)``.

        Raises:
            ValueError: If the baseload profile contains negative values or when
                ``heat_fraction`` is negative.
        """

        return ForecastBuilder.map_baseload_to_internal_gains_increment(
            baseload_profile_kw=baseload_profile_kw,
            baseline_internal_gains_kw=baseline_internal_gains_kw,
            heat_fraction=heat_fraction,
        )

    @staticmethod
    def _build_dhw_forecast(
        req: RunRequest,
        N: int,
        cop_model: HeatPumpCOPModel,
    ) -> DHWForecastHorizon:
        """Build the DHW disturbance forecast over the horizon.

        The DHW COP depends on the outdoor temperature (evaporator side) and the
        hot-water supply temperature (condenser side ≈ T_dhw_min for normal
        operation).  Time-varying COP is injected into the forecast.

        Args:
            req:       Validated request with all DHW parameters.
            N:         Horizon length [steps].
            cop_model: Carnot COP model; used to compute ``cop_dhw_k`` array.

        Returns:
            :class:`~home_optimizer.types.DHWForecastHorizon` with N steps,
            including the COP array.
        """
        return ForecastBuilder.build_dhw_forecast(
            req,
            horizon_steps=N,
            cop_model=cop_model,
        )

    @staticmethod
    def _build_scheduled_input(
        base_input: RunRequest,
        backend: "SensorBackend | None",
        repository: "TelemetryRepository | None" = None,
    ) -> RunRequest:
        """Build one solver input by applying optional live sensor and forecast overrides.

        Overrides applied (in order):

        1. **Persisted calibration snapshot** (via ``repository``): latest
           calibrated physical/COP parameters stored by the addon.
        2. **Live sensor readings** (via ``backend``): room temperature,
           slab temperature estimate, outdoor temperature, DHW temperatures.
        3. **Open-Meteo forecast arrays** (via ``repository``): fetches the most
           recently persisted :class:`~home_optimizer.telemetry.models.ForecastSnapshot`
           batch and injects ``t_out_forecast``, ``gti_window_forecast``, and
           ``gti_pv_forecast`` so the MPC uses real forecast data instead of
           the proxy sine-curves.

        Args:
            base_input:  Baseline validated request for periodic execution.
            backend:     Optional sensor backend. When ``None``, initial-state
                overrides are skipped.
            repository:  Optional telemetry repository. When ``None`` or when
                the forecast table is empty, forecast arrays remain ``None``
                and the proxy fallback in ``_build_ufh_forecast`` is used.

        Returns:
            Immutable :class:`RunRequest` for one optimizer step.

        Raises:
            Exception: Re-raised backend read failures for fail-fast scheduling.
        """
        return OptimizerRuntime.build_scheduled_input(
            base_input=base_input,
            backend=backend,
            repository=repository,
        )
