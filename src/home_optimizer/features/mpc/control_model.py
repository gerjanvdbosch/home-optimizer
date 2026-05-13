from __future__ import annotations

import numpy as np

from home_optimizer.features.modeling import RoomRcModel, TrainedLinearRoomModel
from home_optimizer.features.modeling.room_2r2c import (
    RoomRC2StateParams,
    RoomRC2StatePhysicalModel,
)
from home_optimizer.features.mpc.models import (
    ControlModelConversionOptions,
    LinearThermalControlModel,
)


def to_control_model(
    model: TrainedLinearRoomModel | RoomRcModel,
    *,
    options: ControlModelConversionOptions | None = None,
) -> LinearThermalControlModel:
    resolved_options = options or ControlModelConversionOptions()
    model_kind = getattr(model, "model_kind", None)
    if model_kind == "room_arx":
        return _arx_to_control_model(model, resolved_options)
    if model_kind == "room_2r2c":
        return _rc_to_control_model(model, resolved_options)
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


def _rc_to_control_model(
    model: RoomRcModel,
    options: ControlModelConversionOptions,
) -> LinearThermalControlModel:
    physical = RoomRC2StatePhysicalModel(model.config.with_interval_minutes(model.interval_minutes))
    params = RoomRC2StateParams.from_dict(model.params)
    solar_scale = options.solar_gain_input_scale
    f_matrix, g_matrix, a_discrete, _ = physical.params_to_matrices(params)
    steady_state_gain = -np.linalg.inv(f_matrix) @ g_matrix
    dominant_eigenvalue = float(np.max(np.abs(np.linalg.eigvals(a_discrete))))
    one_minus_a = max(1e-6, 1.0 - dominant_eigenvalue)

    return LinearThermalControlModel(
        a=dominant_eigenvalue,
        b_out=float(steady_state_gain[0, 0] * one_minus_a),
        b_solar=float(
            (steady_state_gain[0, 2] + steady_state_gain[0, 3]) * one_minus_a / solar_scale
        ),
        b_heat=float(steady_state_gain[0, 1] * one_minus_a),
        b_occ=float(steady_state_gain[0, 4] * one_minus_a),
        c=0.0,
        notes=(
            "Approximate 1-state control model reduced from 2R2C using the dominant discrete "
            "time constant and steady-state gains of the physical state-space model."
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
