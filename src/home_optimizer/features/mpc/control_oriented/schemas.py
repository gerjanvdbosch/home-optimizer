from __future__ import annotations

from datetime import datetime

from home_optimizer.domain import DomainModel, IdentifiedModel, NumericSeries

from .model import StateSpaceThermalState

OPTIMIZED_CONTROL_PLAN_NAME = "optimized_control_plan"


class StateSpaceThermalPredictionRequest(DomainModel):
    start_time: datetime
    end_time: datetime
    initial_state: StateSpaceThermalState
    thermal_output_schedule: NumericSeries
    outdoor_temperature_series: NumericSeries
    solar_gain_series: NumericSeries


class StateSpaceThermalPredictionResult(DomainModel):
    model_name: str
    interval_minutes: int
    room_temperature: NumericSeries
    floor_heat_state: NumericSeries


class StateSpaceThermalPlanEvaluation(DomainModel):
    plan_name: str
    thermal_output_schedule: NumericSeries
    predicted_room_temperature: NumericSeries
    predicted_floor_heat_state: NumericSeries
    total_cost: float
    comfort_violation_cost: float
    thermal_output_usage_cost: float
    thermal_output_change_cost: float
    minimum_predicted_temperature: float | None
    maximum_predicted_temperature: float | None


class StateSpaceThermalMpcPlanRequest(DomainModel):
    start_time: datetime
    end_time: datetime
    initial_state: StateSpaceThermalState
    allowed_thermal_outputs: list[float]
    move_block_times: list[datetime]
    outdoor_temperature_series: NumericSeries
    solar_gain_series: NumericSeries
    comfort_min_temperature: float = 19.0
    comfort_max_temperature: float = 21.0
    comfort_undershoot_penalty: float = 1.0
    comfort_overshoot_penalty: float = 2.0
    thermal_output_usage_penalty: float = 0.0
    thermal_output_change_penalty: float = 0.01
    previous_applied_thermal_output: float | None = None


class StateSpaceThermalMpcPlanResult(DomainModel):
    model_name: str
    interval_minutes: int
    plan_results: list[StateSpaceThermalPlanEvaluation]
    recommended_plan: StateSpaceThermalPlanEvaluation


class StateSpaceSetpointPredictionRequest(DomainModel):
    start_time: datetime
    end_time: datetime
    initial_state: StateSpaceThermalState
    initial_thermal_output: float
    thermostat_setpoint_schedule: NumericSeries
    outdoor_temperature_series: NumericSeries
    solar_gain_series: NumericSeries
    supply_target_temperature_series: NumericSeries


class StateSpaceSetpointPredictionResult(DomainModel):
    model_name: str
    interval_minutes: int
    thermal_output_model_name: str
    room_temperature: NumericSeries
    floor_heat_state: NumericSeries
    thermal_output: NumericSeries


class StateSpaceSetpointPlanEvaluation(DomainModel):
    plan_name: str
    thermostat_setpoint_schedule: NumericSeries
    target_thermal_output_schedule: NumericSeries
    predicted_thermal_output: NumericSeries
    predicted_room_temperature: NumericSeries
    predicted_floor_heat_state: NumericSeries
    total_cost: float
    comfort_violation_cost: float
    thermal_output_tracking_cost: float
    setpoint_change_cost: float
    minimum_predicted_temperature: float | None
    maximum_predicted_temperature: float | None


class StateSpaceSetpointMpcPlanRequest(DomainModel):
    start_time: datetime
    end_time: datetime
    initial_state: StateSpaceThermalState
    initial_thermal_output: float
    allowed_setpoints: list[float]
    allowed_thermal_outputs: list[float]
    move_block_times: list[datetime]
    outdoor_temperature_series: NumericSeries
    solar_gain_series: NumericSeries
    supply_target_temperature_series: NumericSeries
    comfort_min_temperature: float = 19.0
    comfort_max_temperature: float = 21.0
    comfort_undershoot_penalty: float = 1.0
    comfort_overshoot_penalty: float = 2.0
    thermal_output_usage_penalty: float = 0.0
    thermal_output_change_penalty: float = 0.01
    thermal_output_tracking_penalty: float = 0.1
    setpoint_change_penalty: float = 0.1
    previous_applied_thermal_output: float | None = None
    previous_applied_setpoint: float | None = None


class StateSpaceSetpointMpcPlanResult(DomainModel):
    model_name: str
    interval_minutes: int
    thermal_output_model_name: str
    thermal_plan: StateSpaceThermalPlanEvaluation
    plan_results: list[StateSpaceSetpointPlanEvaluation]
    recommended_plan: StateSpaceSetpointPlanEvaluation


class StateSpaceActuatorSensitivityRow(DomainModel):
    thermostat_setpoint: float
    initial_heating_demand: float
    first_predicted_thermal_output: float | None
    peak_predicted_thermal_output: float | None
    average_predicted_thermal_output: float | None
    final_room_temperature: float | None
    maximum_room_temperature: float | None


class StateSpaceActuatorSensitivityResult(DomainModel):
    model_name: str
    thermal_output_model_name: str
    interval_minutes: int
    start_time: datetime
    end_time: datetime
    initial_room_temperature: float
    initial_floor_heat_state: float
    initial_thermal_output: float
    first_supply_target_temperature: float | None
    rows: list[StateSpaceActuatorSensitivityRow]
