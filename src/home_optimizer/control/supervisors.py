"""Supervisors that inject physically admissible MPC constraints before solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..types.control import DHWMPCParameters, MPCParameters

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class HeatPumpTopologySupervisor:
    """Apply actuator and shared heat-pump topology constraints to the MPC problem.

    The current implementation represents the existing combined architecture:

    * UFH and DHW are independent thermal actuators.
    * In combined mode they share one electrical heat-pump budget.
    * Both channels remain non-negative because the heat pump cannot actively cool.
    """

    ufh_parameters: MPCParameters
    dhw_parameters: DHWMPCParameters | None
    shared_hp_max_elec_kw: float
    heat_pump_topology: str = "shared"
    exclusive_active_mode: Literal["ufh", "dhw"] | None = None

    def ufh_available_power_kw(self) -> float:
        if self.heat_pump_topology == "shared":
            return self.ufh_parameters.P_max
        if self.heat_pump_topology != "exclusive":
            raise ValueError(f"Unsupported heat-pump topology: {self.heat_pump_topology}.")
        if self.exclusive_active_mode in {None, "ufh"}:
            return self.ufh_parameters.P_max
        return 0.0

    def dhw_available_power_kw(self) -> float:
        if self.dhw_parameters is None:
            return 0.0
        if self.heat_pump_topology == "shared":
            return self.dhw_parameters.P_dhw_max
        if self.heat_pump_topology != "exclusive":
            raise ValueError(f"Unsupported heat-pump topology: {self.heat_pump_topology}.")
        if self.exclusive_active_mode in {None, "dhw"}:
            return self.dhw_parameters.P_dhw_max
        return 0.0

    def apply_step_constraints(
        self,
        *,
        constraints: list,
        step_index: int,
        u_ufh,
        previous_u_ufh,
        total_electrical_power_kw,
        u_dhw=None,
        previous_u_dhw=None,
        ufh_on=None,
        dhw_on=None,
        ufh_switch=None,
        dhw_switch=None,
    ) -> None:
        """Append per-step actuator and shared electrical constraints."""
        topology = self.heat_pump_topology
        ufh_available_power_kw = self.ufh_available_power_kw()
        if ufh_on is None:
            constraints.extend(
                [
                    u_ufh[step_index] >= 0.0,
                    u_ufh[step_index] <= ufh_available_power_kw,
                ]
            )
            constraints.extend(
                [
                    u_ufh[step_index] - previous_u_ufh <= self.ufh_parameters.delta_P_max,
                    previous_u_ufh - u_ufh[step_index] <= self.ufh_parameters.delta_P_max,
                ]
            )
        else:
            if ufh_switch is None:
                raise ValueError("ufh_switch is required when ufh_on is provided.")
            if step_index == 0:
                on_reference_ufh = (
                    1.0
                    if float(previous_u_ufh) >= max(self.ufh_parameters.P_min, 1e-9)
                    else 0.0
                )
            else:
                on_reference_ufh = ufh_on[step_index - 1]
            constraints.extend(
                [
                    u_ufh[step_index] >= self.ufh_parameters.P_min * ufh_on[step_index],
                    u_ufh[step_index] <= ufh_available_power_kw * ufh_on[step_index],
                    ufh_switch[step_index] >= ufh_on[step_index] - on_reference_ufh,
                    ufh_switch[step_index] >= on_reference_ufh - ufh_on[step_index],
                    u_ufh[step_index] - previous_u_ufh
                    <= self.ufh_parameters.delta_P_max + self.ufh_parameters.P_max * ufh_switch[step_index],
                    previous_u_ufh - u_ufh[step_index]
                    <= self.ufh_parameters.delta_P_max + self.ufh_parameters.P_max * ufh_switch[step_index],
                ]
            )
        if u_dhw is None:
            return
        if self.dhw_parameters is None:
            raise ValueError("DHW constraints requested without DHW parameters.")
        if previous_u_dhw is None:
            raise ValueError("previous_u_dhw is required when DHW control is active.")
        dhw_available_power_kw = self.dhw_available_power_kw()
        if dhw_on is None:
            constraints.extend(
                [
                    u_dhw[step_index] >= 0.0,
                    u_dhw[step_index] <= dhw_available_power_kw,
                    u_dhw[step_index] - previous_u_dhw <= self.dhw_parameters.delta_P_dhw_max,
                    previous_u_dhw - u_dhw[step_index] <= self.dhw_parameters.delta_P_dhw_max,
                ]
            )
        else:
            if dhw_switch is None:
                raise ValueError("dhw_switch is required when dhw_on is provided.")
            if step_index == 0:
                on_reference_dhw = (
                    1.0
                    if float(previous_u_dhw) >= max(self.dhw_parameters.P_dhw_min, 1e-9)
                    else 0.0
                )
            else:
                on_reference_dhw = dhw_on[step_index - 1]
            constraints.extend(
                [
                    u_dhw[step_index] >= self.dhw_parameters.P_dhw_min * dhw_on[step_index],
                    u_dhw[step_index] <= dhw_available_power_kw * dhw_on[step_index],
                    dhw_switch[step_index] >= dhw_on[step_index] - on_reference_dhw,
                    dhw_switch[step_index] >= on_reference_dhw - dhw_on[step_index],
                    u_dhw[step_index] - previous_u_dhw
                    <= self.dhw_parameters.delta_P_dhw_max + self.dhw_parameters.P_dhw_max * dhw_switch[step_index],
                    previous_u_dhw - u_dhw[step_index]
                    <= self.dhw_parameters.delta_P_dhw_max + self.dhw_parameters.P_dhw_max * dhw_switch[step_index],
                ]
            )
        if topology == "shared":
            constraints.append(total_electrical_power_kw <= self.shared_hp_max_elec_kw)
        elif topology == "exclusive":
            if ufh_on is not None and dhw_on is not None:
                if self.exclusive_active_mode == "ufh":
                    constraints.append(dhw_on[step_index] == 0.0)
                elif self.exclusive_active_mode == "dhw":
                    constraints.append(ufh_on[step_index] == 0.0)
                else:
                    constraints.append(ufh_on[step_index] + dhw_on[step_index] <= 1.0)
        else:
            raise ValueError(f"Unsupported heat-pump topology: {topology}.")


@dataclass(frozen=True, slots=True)
class LegionellaSupervisor:
    """Apply DHW comfort and legionella surrogate constraints to one MPC step."""

    dhw_parameters: DHWMPCParameters

    def target_temperature_c(self, legionella_required: bool) -> float:
        """Return the active upper-band target for the DHW top layer [°C]."""
        return (
            self.dhw_parameters.T_legionella if legionella_required else self.dhw_parameters.T_dhw_target
        )

    def apply_step_constraints(
        self,
        *,
        constraints: list,
        predicted_state_matrix,
        step_index: int,
        lower_slack,
        upper_slack,
        legionella_slack,
        legionella_required: bool,
    ) -> None:
        """Append DHW comfort and legionella surrogate constraints for one step."""
        target_temperature_c = self.target_temperature_c(legionella_required)
        constraints.extend(
            [
                predicted_state_matrix[2, step_index + 1]
                >= self.dhw_parameters.T_dhw_min - lower_slack[step_index],
                predicted_state_matrix[2, step_index + 1]
                <= target_temperature_c + upper_slack[step_index],
            ]
        )
        if legionella_required:
            constraints.extend(
                [
                    predicted_state_matrix[2, step_index + 1]
                    >= self.dhw_parameters.T_legionella - legionella_slack[step_index],
                    predicted_state_matrix[3, step_index + 1]
                    >= self.dhw_parameters.T_legionella - legionella_slack[step_index],
                ]
            )
