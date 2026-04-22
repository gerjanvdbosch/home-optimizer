"""Supervisors that inject physically admissible MPC constraints before solve."""

from __future__ import annotations

from dataclasses import dataclass

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
    ) -> None:
        """Append per-step actuator and shared electrical constraints."""
        topology = self.heat_pump_topology
        constraints.extend(
            [
                u_ufh[step_index] >= 0.0,
                u_ufh[step_index] <= self.ufh_parameters.P_max,
            ]
        )
        if not (topology == "exclusive_dhw" and step_index == 0):
            constraints.append(
                cp.abs(u_ufh[step_index] - previous_u_ufh) <= self.ufh_parameters.delta_P_max
            )
        if u_dhw is None:
            return
        if self.dhw_parameters is None:
            raise ValueError("DHW constraints requested without DHW parameters.")
        if previous_u_dhw is None:
            raise ValueError("previous_u_dhw is required when DHW control is active.")
        constraints.extend(
            [
                u_dhw[step_index] >= 0.0,
                u_dhw[step_index] <= self.dhw_parameters.P_dhw_max,
            ]
        )
        if not (topology == "exclusive_ufh" and step_index == 0):
            constraints.append(
                cp.abs(u_dhw[step_index] - previous_u_dhw) <= self.dhw_parameters.delta_P_dhw_max
            )
        if topology == "shared":
            constraints.append(total_electrical_power_kw <= self.shared_hp_max_elec_kw)
        elif topology == "exclusive_ufh":
            constraints.append(u_dhw[step_index] == 0.0)
        elif topology == "exclusive_dhw":
            constraints.append(u_ufh[step_index] == 0.0)
        else:
            raise ValueError(f"Unsupported heat-pump topology: {topology}.")


@dataclass(frozen=True, slots=True)
class LegionellaSupervisor:
    """Apply DHW comfort and legionella surrogate constraints to one MPC step."""

    dhw_parameters: DHWMPCParameters

    def target_temperature_c(self, legionella_required: bool) -> float:
        """Return the active upper-band target for the DHW top layer [°C]."""
        return (
            self.dhw_parameters.T_legionella if legionella_required else self.dhw_parameters.T_dhw_min
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
