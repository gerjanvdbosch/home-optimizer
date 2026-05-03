from __future__ import annotations

from home_optimizer.domain import (
    DEFAULT_FLOOR_HEAT_STATE_ALPHA,
    FLOOR_HEAT_STATE,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    IdentifiedModel,
    OUTDOOR_TEMPERATURE,
    ROOM_TEMPERATURE,
)
from home_optimizer.domain.models import DomainModel


class StateSpaceThermalState(DomainModel):
    room_temperature: float
    floor_heat_state: float


class StateSpaceThermalControlInput(DomainModel):
    thermal_output: float


class StateSpaceThermalDisturbance(DomainModel):
    outdoor_temperature: float
    solar_gain: float


class StateSpaceThermalModel(DomainModel):
    model_name: str
    interval_minutes: int
    room_temperature_coefficient: float
    floor_to_room_coefficient: float
    floor_heat_state_alpha: float
    outdoor_temperature_coefficient: float
    solar_gain_coefficient: float
    intercept: float

    @classmethod
    def from_identified_model(
        cls,
        model: IdentifiedModel,
        *,
        floor_heat_state_alpha: float = DEFAULT_FLOOR_HEAT_STATE_ALPHA,
    ) -> "StateSpaceThermalModel":
        if model.target_name != ROOM_TEMPERATURE:
            raise ValueError("identified model target must be room_temperature")
        required = {
            "previous_room_temperature",
            OUTDOOR_TEMPERATURE,
            GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
            FLOOR_HEAT_STATE,
        }
        if not required.issubset(model.coefficients):
            raise ValueError("identified model is missing 2-state room-temperature coefficients")
        if not 0.0 <= floor_heat_state_alpha <= 1.0:
            raise ValueError("floor_heat_state_alpha must be between 0 and 1")

        return cls(
            model_name=model.model_name,
            interval_minutes=model.interval_minutes,
            room_temperature_coefficient=float(model.coefficients["previous_room_temperature"]),
            floor_to_room_coefficient=float(model.coefficients[FLOOR_HEAT_STATE]),
            floor_heat_state_alpha=float(floor_heat_state_alpha),
            outdoor_temperature_coefficient=float(model.coefficients[OUTDOOR_TEMPERATURE]),
            solar_gain_coefficient=float(model.coefficients[GTI_LIVING_ROOM_WINDOWS_ADJUSTED]),
            intercept=float(model.intercept),
        )

    @property
    def floor_heat_input_gain(self) -> float:
        return 1.0 - self.floor_heat_state_alpha

    @property
    def state_transition_matrix(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (
                self.room_temperature_coefficient,
                self.floor_to_room_coefficient * self.floor_heat_state_alpha,
            ),
            (0.0, self.floor_heat_state_alpha),
        )

    @property
    def input_matrix(self) -> tuple[tuple[float], tuple[float]]:
        gain = self.floor_heat_input_gain
        return (
            (self.floor_to_room_coefficient * gain,),
            (gain,),
        )

    @property
    def disturbance_matrix(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (
                self.outdoor_temperature_coefficient,
                self.solar_gain_coefficient,
            ),
            (0.0, 0.0),
        )

    @property
    def affine_offset(self) -> tuple[float, float]:
        return (self.intercept, 0.0)

    def step(
        self,
        state: StateSpaceThermalState,
        control_input: StateSpaceThermalControlInput,
        disturbance: StateSpaceThermalDisturbance,
    ) -> StateSpaceThermalState:
        next_floor_heat_state = (
            self.floor_heat_state_alpha * state.floor_heat_state
            + self.floor_heat_input_gain * control_input.thermal_output
        )
        next_room_temperature = (
            self.intercept
            + self.room_temperature_coefficient * state.room_temperature
            + self.outdoor_temperature_coefficient * disturbance.outdoor_temperature
            + self.solar_gain_coefficient * disturbance.solar_gain
            + self.floor_to_room_coefficient * next_floor_heat_state
        )
        return StateSpaceThermalState(
            room_temperature=next_room_temperature,
            floor_heat_state=next_floor_heat_state,
        )

    def simulate(
        self,
        *,
        initial_state: StateSpaceThermalState,
        control_inputs: list[StateSpaceThermalControlInput],
        disturbances: list[StateSpaceThermalDisturbance],
    ) -> list[StateSpaceThermalState]:
        if len(control_inputs) != len(disturbances):
            raise ValueError("control_inputs and disturbances must have the same length")

        states: list[StateSpaceThermalState] = []
        current_state = initial_state
        for control_input, disturbance in zip(control_inputs, disturbances, strict=True):
            current_state = self.step(
                current_state,
                control_input=control_input,
                disturbance=disturbance,
            )
            states.append(current_state)
        return states
