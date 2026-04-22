"""Runtime pipeline builders that keep the optimizer orchestration compact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .forecasting import ForecastBuilder
from ..control.mpc import MPCController
from ..domain.dhw.model import DHWModel
from ..domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters
from ..domain.ufh.model import ThermalModel
from ..types.control import CombinedMPCParameters, DHWMPCParameters, MPCParameters
from ..types.forecast import DHWForecastHorizon, ForecastHorizon
from ..types.physical import DHWParameters, ThermalParameters

if TYPE_CHECKING:
    from .optimizer import MPCStepResult, RunRequest


@dataclass(frozen=True, slots=True)
class OptimizerSolveContext:
    """Fully assembled runtime context required for one optimizer solve."""

    thermal_parameters: ThermalParameters
    ufh_model: ThermalModel
    cop_model: HeatPumpCOPModel
    ufh_forecast: ForecastHorizon
    mpc_parameters: MPCParameters
    controller_parameters: MPCParameters | CombinedMPCParameters
    dhw_model: DHWModel | None
    dhw_forecast: DHWForecastHorizon | None
    initial_ufh_state_c: np.ndarray
    initial_dhw_state_c: np.ndarray | None
    cop_ufh_scalar: float
    cop_dhw_scalar: float


class OptimizerPipeline:
    """Build runtime solve context and summarize the resulting MPC solution."""

    @staticmethod
    def build_cop_model(req: "RunRequest") -> HeatPumpCOPModel:
        """Construct the Carnot COP model from the user request."""
        params = HeatPumpCOPParameters(
            eta_carnot=req.eta_carnot,
            delta_T_cond=req.delta_T_cond,
            delta_T_evap=req.delta_T_evap,
            T_supply_min=req.T_supply_min,
            T_ref_outdoor=req.T_ref_outdoor_curve,
            heating_curve_slope=req.heating_curve_slope,
            cop_min=req.cop_min,
            cop_max=req.cop_max,
        )
        return HeatPumpCOPModel(params)

    @staticmethod
    def build_solve_context(req: "RunRequest", *, start_hour: int) -> OptimizerSolveContext:
        """Assemble all models, forecasts, and controller parameters for one solve."""
        horizon_steps = req.horizon_hours
        thermal_parameters = ThermalParameters(
            dt_hours=req.dt_hours,
            C_r=req.C_r,
            C_b=req.C_b,
            R_br=req.R_br,
            R_ro=req.R_ro,
            alpha=req.alpha,
            eta=req.eta,
            A_glass=req.A_glass,
        )
        ufh_model = ThermalModel(thermal_parameters)
        cop_model = OptimizerPipeline.build_cop_model(req)

        representative_outdoor = np.array([req.outdoor_temperature_c], dtype=float)
        cop_ufh_scalar = float(cop_model.cop_ufh(representative_outdoor)[0])
        cop_dhw_scalar = float(cop_model.cop_dhw(representative_outdoor, req.dhw_T_min)[0])

        mpc_parameters = MPCParameters(
            horizon_steps=horizon_steps,
            Q_c=req.Q_c,
            R_c=req.R_c,
            Q_N=req.Q_N,
            P_max=req.P_max,
            delta_P_max=req.delta_P_max,
            T_min=req.T_min,
            T_max=req.T_max,
            cop_ufh=cop_ufh_scalar,
            cop_max=req.cop_max,
        )
        ufh_forecast = ForecastBuilder.build_ufh_forecast(
            req,
            start_hour=start_hour,
            cop_model=cop_model,
        )

        dhw_model: DHWModel | None = None
        dhw_forecast: DHWForecastHorizon | None = None
        controller_parameters: MPCParameters | CombinedMPCParameters = mpc_parameters
        initial_dhw_state_c: np.ndarray | None = None

        if req.dhw_enabled:
            dhw_parameters = DHWParameters(
                dt_hours=req.dt_hours,
                C_top=req.dhw_C_top,
                C_bot=req.dhw_C_bot,
                R_strat=req.dhw_R_strat,
                R_loss=req.dhw_R_loss,
                lambda_water=req.dhw_lambda_water_kwh_per_m3k,
            )
            dhw_mpc_parameters = DHWMPCParameters(
                P_dhw_max=req.dhw_P_max,
                delta_P_dhw_max=req.dhw_delta_P_max,
                T_dhw_min=req.dhw_T_min,
                T_legionella=req.dhw_T_legionella,
                legionella_period_steps=req.dhw_legionella_period_steps,
                legionella_duration_steps=req.dhw_legionella_duration_steps,
                cop_dhw=cop_dhw_scalar,
                cop_max=req.cop_max,
            )
            controller_parameters = CombinedMPCParameters(
                ufh=mpc_parameters,
                dhw=dhw_mpc_parameters,
                P_hp_max_elec=req.P_hp_max_elec,
            )
            dhw_model = DHWModel(dhw_parameters)
            dhw_forecast = ForecastBuilder.build_dhw_forecast(
                req,
                horizon_steps=horizon_steps,
                cop_model=cop_model,
            )
            dhw_forecast.assert_compatible_with_parameters(dhw_parameters)
            initial_dhw_state_c = np.array([req.dhw_T_top_init, req.dhw_T_bot_init], dtype=float)

        return OptimizerSolveContext(
            thermal_parameters=thermal_parameters,
            ufh_model=ufh_model,
            cop_model=cop_model,
            ufh_forecast=ufh_forecast,
            mpc_parameters=mpc_parameters,
            controller_parameters=controller_parameters,
            dhw_model=dhw_model,
            dhw_forecast=dhw_forecast,
            initial_ufh_state_c=np.array([req.T_r_init, req.T_b_init], dtype=float),
            initial_dhw_state_c=initial_dhw_state_c,
            cop_ufh_scalar=cop_ufh_scalar,
            cop_dhw_scalar=cop_dhw_scalar,
        )

    @staticmethod
    def solve(req: "RunRequest", *, start_hour: int) -> "MPCStepResult":
        """Run the fully assembled optimizer pipeline for one request."""
        from .optimizer import MPCStepResult

        context = OptimizerPipeline.build_solve_context(req, start_hour=start_hour)
        controller = MPCController(
            ufh_model=context.ufh_model,
            params=context.controller_parameters,
            dhw_model=context.dhw_model,
        )
        solution = controller.solve(
            initial_ufh_state_c=context.initial_ufh_state_c,
            ufh_forecast=context.ufh_forecast,
            initial_dhw_state_c=context.initial_dhw_state_c,
            dhw_forecast=context.dhw_forecast,
            previous_p_ufh_kw=req.previous_power_kw,
        )

        ufh_power_kw = np.maximum(solution.ufh_control_sequence_kw, 0.0)
        dhw_power_kw = np.maximum(solution.dhw_control_sequence_kw, 0.0)
        horizon_steps = req.horizon_hours
        dt_hours = context.thermal_parameters.dt_hours
        cop_ufh_arr = context.ufh_forecast.cop_ufh_k
        assert cop_ufh_arr is not None, "UFH COP horizon must be present in the assembled forecast."
        cop_dhw_arr = (
            context.dhw_forecast.cop_dhw_k
            if context.dhw_forecast is not None and context.dhw_forecast.cop_dhw_k is not None
            else np.ones(horizon_steps)
        )
        pv_kw = context.ufh_forecast.pv_kw
        assert pv_kw is not None, "PV profile must be present on the assembled UFH forecast."
        prices = context.ufh_forecast.price_eur_per_kwh
        feed_in_prices = context.ufh_forecast.feed_in_price_eur_per_kwh
        ufh_electrical_kw = ufh_power_kw / cop_ufh_arr
        dhw_electrical_kw = dhw_power_kw / cop_dhw_arr
        net_grid_power_kw = ufh_electrical_kw + dhw_electrical_kw - pv_kw
        grid_import_kw = np.maximum(net_grid_power_kw, 0.0)
        grid_export_kw = np.maximum(-net_grid_power_kw, 0.0)

        return MPCStepResult(
            solution=solution,
            ufh_forecast=context.ufh_forecast,
            dhw_forecast=context.dhw_forecast,
            p_ufh_kw=ufh_power_kw,
            p_dhw_kw=dhw_power_kw,
            cop_ufh_arr=cop_ufh_arr,
            cop_dhw_arr=cop_dhw_arr,
            pv_kw=pv_kw,
            total_cost_eur=float(
                np.sum((grid_import_kw * prices - grid_export_kw * feed_in_prices) * dt_hours)
            ),
            ufh_energy_kwh=float(np.sum(ufh_power_kw) * dt_hours),
            dhw_energy_kwh=float(np.sum(dhw_power_kw) * dt_hours),
            start_hour=start_hour,
        )


__all__ = [
    "OptimizerPipeline",
    "OptimizerSolveContext",
]
