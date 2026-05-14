from __future__ import annotations

import numpy as np

from home_optimizer.features.modeling import RoomRcModel, TrainedLinearRoomModel
from home_optimizer.features.modeling.room_2r2c import (
    RoomRC2StateParams,
    RoomRC2StatePhysicalModel,
)
from home_optimizer.features.mpc.models import (
    ControlModelKind,
    ControlModelConversionOptions,
    LinearThermalControlModel,
    Rc2StateThermalControlModel,
)


def to_control_model(
    model: TrainedLinearRoomModel | RoomRcModel,
    *,
    options: ControlModelConversionOptions | None = None,
    control_model_kind: ControlModelKind = "linear_1state",
) -> LinearThermalControlModel | Rc2StateThermalControlModel:
    resolved_options = options or ControlModelConversionOptions()
    model_kind = getattr(model, "model_kind", None)
    if model_kind == "room_arx":
        if control_model_kind != "linear_1state":
            raise ValueError("room_arx models only support linear_1state MPC control conversion")
        return _arx_to_control_model(model, resolved_options)
    if model_kind == "room_2r2c":
        if control_model_kind == "rc_2state":
            return _rc_to_2state_control_model(model, resolved_options)
        return _rc_to_linear_control_model(model, resolved_options)
    raise ValueError(f"Unsupported model kind for MPC control conversion: {model_kind}")


def _arx_to_control_model(
    model: TrainedLinearRoomModel,
    options: ControlModelConversionOptions,
) -> LinearThermalControlModel:
    weights_by_feature = dict(zip(model.feature_names, model.coefficients, strict=True))
    solar_scale = options.solar_gain_input_scale
    return LinearThermalControlModel(
        a=_sum_matching(weights_by_feature, "room_temperature_lag_"),
        b_out=_sum_matching(weights_by_feature, "outdoor_temperature_lag_"),
        b_solar=_sum_matching(weights_by_feature, "solar_gain_lag_") / solar_scale,
        b_heat=_sum_matching(weights_by_feature, "thermal_output_lag_"),
        b_occ=_sum_matching(weights_by_feature, "occupied_flag_lag_"),
        c=float(model.intercept),
        notes=(
            "Approximate 1-state control model derived from ARX coefficients by aggregating "
            "same-signal lag weights into single-step MPC coefficients."
        ),
    )


def _rc_to_linear_control_model(
    model: RoomRcModel,
    options: ControlModelConversionOptions,
) -> LinearThermalControlModel:
    physical = RoomRC2StatePhysicalModel(model.config.with_interval_minutes(model.interval_minutes))
    params = RoomRC2StateParams.from_dict(model.params)
    solar_scale = options.solar_gain_input_scale
    _, _, a_discrete, b_discrete = physical.params_to_matrices(params)
    dominant_eigenvalue, dominant_left, dominant_right = _dominant_room_mode(a_discrete)
    reduced_inputs = dominant_left @ b_discrete

    return LinearThermalControlModel(
        a=dominant_eigenvalue,
        b_out=float(reduced_inputs[0]),
        b_solar=float((reduced_inputs[2] + reduced_inputs[3]) / solar_scale),
        b_heat=float(reduced_inputs[1]),
        b_occ=float(reduced_inputs[4]),
        c=0.0,
        notes=(
            "Approximate 1-state control model projected from the exact discrete 2R2C room "
            "dynamics onto the dominant room-temperature mode."
        ),
    )


def _rc_to_2state_control_model(
    model: RoomRcModel,
    options: ControlModelConversionOptions,
) -> Rc2StateThermalControlModel:
    physical = RoomRC2StatePhysicalModel(model.config.with_interval_minutes(model.interval_minutes))
    params = RoomRC2StateParams.from_dict(model.params)
    solar_scale = options.solar_gain_input_scale
    _, _, a_discrete, b_discrete = physical.params_to_matrices(params)
    return Rc2StateThermalControlModel(
        a11=float(a_discrete[0, 0]),
        a12=float(a_discrete[0, 1]),
        a21=float(a_discrete[1, 0]),
        a22=float(a_discrete[1, 1]),
        b_out_room=float(b_discrete[0, 0]),
        b_out_mass=float(b_discrete[1, 0]),
        b_heat_room=float(b_discrete[0, 1]),
        b_heat_mass=float(b_discrete[1, 1]),
        b_solar_room=float((b_discrete[0, 2] + b_discrete[0, 3]) / solar_scale),
        b_solar_mass=float((b_discrete[1, 2] + b_discrete[1, 3]) / solar_scale),
        b_occ_room=float(b_discrete[0, 4]),
        b_occ_mass=float(b_discrete[1, 4]),
        c_room=0.0,
        c_mass=0.0,
        notes=(
            "Control-oriented 2-state model taken directly from the exact discrete 2R2C room "
            "dynamics used during physical forecasting."
        ),
    )


def _sum_matching(weights_by_feature: dict[str, float], prefix: str) -> float:
    return float(
        sum(
            weight
            for feature_name, weight in weights_by_feature.items()
            if feature_name.startswith(prefix)
        )
    )


def _dominant_room_mode(a_discrete: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    eigenvalues, right_eigenvectors = np.linalg.eig(a_discrete)
    dominant_index = int(np.argmax(np.abs(eigenvalues)))
    dominant_eigenvalue = np.real_if_close(eigenvalues[dominant_index]).item()

    dominant_right = np.real_if_close(right_eigenvectors[:, dominant_index]).astype(float)
    room_projection = float(dominant_right[0])
    if abs(room_projection) < 1e-9:
        raise ValueError("Cannot reduce 2R2C model for MPC because the dominant mode is unobservable")
    dominant_right = dominant_right / room_projection

    left_eigenvalues, left_eigenvectors = np.linalg.eig(a_discrete.T)
    left_index = int(np.argmin(np.abs(left_eigenvalues - eigenvalues[dominant_index])))
    dominant_left = np.real_if_close(left_eigenvectors[:, left_index]).astype(float)
    normalization = float(dominant_left @ dominant_right)
    if abs(normalization) < 1e-9:
        raise ValueError("Cannot normalize dominant 2R2C mode for MPC reduction")
    dominant_left = dominant_left / normalization

    return float(dominant_eigenvalue), dominant_left, dominant_right
