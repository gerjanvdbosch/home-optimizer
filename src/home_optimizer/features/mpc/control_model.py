from __future__ import annotations

from home_optimizer.features.modeling import RoomRcModel
from home_optimizer.features.modeling.room_2r2c import (
    RoomRC2StateParams,
    RoomRC2StatePhysicalModel,
)
from home_optimizer.features.mpc.models import (
    ControlModelConversionOptions,
    Rc2StateThermalControlModel,
)


def to_control_model(
    model: RoomRcModel,
    *,
    options: ControlModelConversionOptions | None = None,
) -> Rc2StateThermalControlModel:
    resolved_options = options or ControlModelConversionOptions()
    model_kind = getattr(model, "model_kind", None)
    if model_kind == "room_2r2c":
        return _rc_to_2state_control_model(model, resolved_options)
    raise ValueError(f"Unsupported model kind for MPC control conversion: {model_kind}")


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
        b_solar_direct_room=float(b_discrete[0, 2] / solar_scale),
        b_solar_filtered_room=float(b_discrete[0, 3] / solar_scale),
        b_solar_direct_mass=float(b_discrete[1, 2] / solar_scale),
        b_solar_filtered_mass=float(b_discrete[1, 3] / solar_scale),
        b_occ_room=float(b_discrete[0, 4]),
        b_occ_mass=float(b_discrete[1, 4]),
        b_hour_sin_room=float(b_discrete[0, 5]),
        b_hour_cos_room=float(b_discrete[0, 6]),
        b_hour_sin_mass=float(b_discrete[1, 5]),
        b_hour_cos_mass=float(b_discrete[1, 6]),
        actuator_alpha=float(model.config.alpha_heat),
        c_room=0.0,
        c_mass=0.0,
        notes=(
            "Control-oriented 2-state model taken directly from the exact discrete 2R2C room "
            "dynamics, including separate direct/filtered solar inputs and diurnal terms."
        ),
    )
