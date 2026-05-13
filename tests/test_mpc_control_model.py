from __future__ import annotations

from datetime import datetime, timezone

import pytest

from home_optimizer.features.modeling import RoomRcConfig, RoomRcModel
from home_optimizer.features.modeling.room_2r2c import (
    RoomRC2StateParams,
    RoomRC2StatePhysicalModel,
)
from home_optimizer.features.modeling.room_arx import RoomArxConfig, RoomArxModel
from home_optimizer.features.mpc import ControlModelConversionOptions, to_control_model


def test_to_control_model_from_arx_aggregates_matching_coefficients() -> None:
    model = RoomArxModel(
        trained_from_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomArxConfig(),
        feature_names=[
            "room_temperature_lag_0",
            "room_temperature_lag_1",
            "outdoor_temperature_lag_0",
            "thermal_output_lag_0",
            "solar_gain_lag_0",
            "occupied_flag_lag_0",
        ],
        intercept=1.25,
        coefficients=[0.6, 0.1, 0.05, 0.4, 0.02, 0.03],
        sample_count=100,
    )

    control_model = to_control_model(
        model,
        options=ControlModelConversionOptions(solar_gain_input_scale=0.5),
    )

    assert control_model.a == pytest.approx(0.7)
    assert control_model.b_out == pytest.approx(0.05)
    assert control_model.b_heat == pytest.approx(0.4)
    assert control_model.b_solar == pytest.approx(0.04)
    assert control_model.b_occ == pytest.approx(0.03)
    assert control_model.c == pytest.approx(1.25)


def test_to_control_model_from_room_rc_matches_reduced_discrete_terms() -> None:
    room_rc_model = RoomRcModel(
        trained_from_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomRcConfig(),
        params=RoomRC2StateParams().to_dict(),
        sample_count=100,
    )

    control_model = to_control_model(room_rc_model)
    physical = RoomRC2StatePhysicalModel(RoomRcConfig(interval_minutes=10))
    _, _, a_discrete, b_discrete = physical.params_to_matrices(RoomRC2StateParams())

    assert control_model.a == pytest.approx(float(a_discrete[0, 0] + a_discrete[0, 1]))
    assert control_model.b_out == pytest.approx(float(b_discrete[0, 0]))
    assert control_model.b_heat == pytest.approx(float(b_discrete[0, 1]))
    assert control_model.b_solar == pytest.approx(float(b_discrete[0, 2] + b_discrete[0, 3]))
    assert control_model.b_occ == pytest.approx(float(b_discrete[0, 4]))
