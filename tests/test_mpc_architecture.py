"""Focused architecture tests for MPC supervisors and builder boundaries."""

from __future__ import annotations

import cvxpy as cp

from home_optimizer.control.problem_builder import MpcProblemBuilder
from home_optimizer.control.supervisors import HeatPumpTopologySupervisor, LegionellaSupervisor
from home_optimizer.types import DHWMPCParameters, MPCParameters


def _ufh_params() -> MPCParameters:
    return MPCParameters(
        horizon_steps=4,
        Q_c=10.0,
        R_c=0.05,
        Q_N=15.0,
        P_max=4.0,
        delta_P_max=1.0,
        T_min=19.0,
        T_max=22.5,
        cop_ufh=3.5,
        cop_max=7.0,
    )


def _dhw_params() -> DHWMPCParameters:
    return DHWMPCParameters(
        P_dhw_max=3.0,
        delta_P_dhw_max=1.0,
        T_dhw_min=50.0,
        T_legionella=60.0,
        legionella_period_steps=168,
        legionella_duration_steps=1,
        cop_dhw=3.0,
        cop_max=7.0,
    )


def test_legionella_supervisor_switches_target_temperature() -> None:
    """LegionellaSupervisor must expose the active DHW target from the requirement flag."""
    supervisor = LegionellaSupervisor(_dhw_params())

    assert supervisor.target_temperature_c(False) == 50.0
    assert supervisor.target_temperature_c(True) == 60.0


def test_problem_builder_allocates_only_ufh_variables_in_ufh_only_mode() -> None:
    """UFH-only MPC must not allocate DHW slack or control variables."""
    builder = MpcProblemBuilder(
        ufh_parameters=_ufh_params(),
        dhw_parameters=None,
        topology_supervisor=HeatPumpTopologySupervisor(
            ufh_parameters=_ufh_params(),
            dhw_parameters=None,
            shared_hp_max_elec_kw=2.0,
        ),
        legionella_supervisor=None,
        dhw_enabled=False,
    )

    variables = builder._create_variables(n_horizon=4, n_states=2, has_pv=False)

    assert variables.u_dhw is None
    assert variables.s_dhw_lo is None
    assert variables.s_dhw_hi is None
    assert variables.s_leg is None


def test_problem_builder_allocates_dhw_variables_in_combined_mode() -> None:
    """Combined MPC must allocate DHW controls and DHW slack variables."""
    dhw_parameters = _dhw_params()
    builder = MpcProblemBuilder(
        ufh_parameters=_ufh_params(),
        dhw_parameters=dhw_parameters,
        topology_supervisor=HeatPumpTopologySupervisor(
            ufh_parameters=_ufh_params(),
            dhw_parameters=dhw_parameters,
            shared_hp_max_elec_kw=3.0,
        ),
        legionella_supervisor=LegionellaSupervisor(dhw_parameters),
        dhw_enabled=True,
    )

    variables = builder._create_variables(n_horizon=4, n_states=4, has_pv=True)

    assert variables.u_dhw is not None
    assert variables.s_dhw_lo is not None
    assert variables.s_dhw_hi is not None
    assert variables.s_leg is not None
    assert variables.p_import is not None
    assert variables.p_export is not None


def test_legionella_supervisor_constrains_both_dhw_nodes() -> None:
    """Legionella surrogate must constrain both top and bottom DHW nodes."""
    dhw_parameters = _dhw_params()
    supervisor = LegionellaSupervisor(dhw_parameters)
    x = cp.Variable((4, 3))
    lower = cp.Variable(2, nonneg=True)
    upper = cp.Variable(2, nonneg=True)
    leg = cp.Variable(2, nonneg=True)
    constraints: list = []

    supervisor.apply_step_constraints(
        constraints=constraints,
        predicted_state_matrix=x,
        step_index=0,
        lower_slack=lower,
        upper_slack=upper,
        legionella_slack=leg,
        legionella_required=True,
    )

    rendered = "\n".join(str(constraint) for constraint in constraints)
    assert "[2, 1]" in rendered
    assert "[3, 1]" in rendered
    assert "60.0" in rendered


def test_topology_supervisor_supports_exclusive_modes() -> None:
    """Exclusive topology must disable one heat channel per supervisor mode."""
    supervisor = HeatPumpTopologySupervisor(
        ufh_parameters=_ufh_params(),
        dhw_parameters=_dhw_params(),
        shared_hp_max_elec_kw=3.0,
        heat_pump_topology="exclusive_dhw",
    )
    u_ufh = cp.Variable(2)
    u_dhw = cp.Variable(2)
    constraints: list = []

    supervisor.apply_step_constraints(
        constraints=constraints,
        step_index=0,
        u_ufh=u_ufh,
        previous_u_ufh=0.5,
        total_electrical_power_kw=u_ufh[0] / 3.5 + u_dhw[0] / 3.0,
        u_dhw=u_dhw,
        previous_u_dhw=0.0,
    )

    rendered = "\n".join(str(constraint) for constraint in constraints)
    assert "[0] == 0.0" in rendered
