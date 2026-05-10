from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from home_optimizer.features.modeling.room_rc import (
    RoomRC2StateConfig,
    RoomRC2StateParams,
    RoomRC2StatePhysicalModel,
)


def build_dataframe(row_count: int = 72) -> pd.DataFrame:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows: list[dict[str, float | datetime]] = []
    room_temp = 20.0
    for index in range(row_count):
        outdoor = 5.0 + 5.0 * np.sin(2 * np.pi * index / 144.0)
        heating = 1.5 if index % 18 < 6 else 0.2
        irradiance = 500.0 if 36 <= (index % 144) <= 72 else 0.0
        shutter = 75.0 if irradiance > 0 else 10.0
        occupied = 1.0 if 42 <= (index % 144) <= 120 else 0.0
        room_temp = 0.92 * room_temp + 0.03 * outdoor + 0.12 * heating + 0.0005 * irradiance
        rows.append(
            {
                "timestamp": start + timedelta(minutes=10 * index),
                "room_temp_c": room_temp,
                "outdoor_temp_c": outdoor,
                "heating_kw": heating,
                "irradiance_wm2": irradiance,
                "shutter_position": shutter,
                "occupied_flag": occupied,
            }
        )
    return pd.DataFrame(rows)


def test_solar_calculation_uses_8m2_glass_and_g_value() -> None:
    model = RoomRC2StatePhysicalModel(RoomRC2StateConfig())
    df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T12:00:00Z",
                "room_temp_c": 20.0,
                "outdoor_temp_c": 5.0,
                "heating_kw": 0.0,
                "irradiance_wm2": 500.0,
                "shutter_position": 75.0,
                "occupied_flag": 0.0,
            }
        ]
    )
    prepared = model.prepare_features(df)
    assert prepared.loc[0, "solar_glass_kw"] == 1.5


def test_shutter_open_percent_maps_100_to_one_and_0_to_zero() -> None:
    model = RoomRC2StatePhysicalModel(RoomRC2StateConfig(shutter_mode="open_percent"))
    assert model._shutter_factor_from_position(100.0) == 1.0
    assert model._shutter_factor_from_position(0.0) == 0.0


def test_shutter_closed_percent_maps_100_to_zero_and_0_to_one() -> None:
    model = RoomRC2StatePhysicalModel(RoomRC2StateConfig(shutter_mode="closed_percent"))
    assert model._shutter_factor_from_position(100.0) == 0.0
    assert model._shutter_factor_from_position(0.0) == 1.0


def test_shutter_factor_is_clamped() -> None:
    model_open = RoomRC2StatePhysicalModel(RoomRC2StateConfig(shutter_mode="open_percent"))
    model_closed = RoomRC2StatePhysicalModel(RoomRC2StateConfig(shutter_mode="closed_percent"))
    assert model_open._shutter_factor_from_position(-10.0) == 0.0
    assert model_open._shutter_factor_from_position(150.0) == 1.0
    assert model_closed._shutter_factor_from_position(-10.0) == 1.0
    assert model_closed._shutter_factor_from_position(150.0) == 0.0


def test_matrix_discretization_shapes_and_stability() -> None:
    model = RoomRC2StatePhysicalModel(RoomRC2StateConfig())
    _, _, A_d, B_d = model.params_to_matrices(RoomRC2StateParams())
    assert A_d.shape == (2, 2)
    assert B_d.shape == (2, 7)
    assert max(abs(np.linalg.eigvals(A_d))) < 1.0


def test_predict_one_step_returns_expected_columns() -> None:
    model = RoomRC2StatePhysicalModel(RoomRC2StateConfig())
    df = build_dataframe(24)
    prediction = model.predict_one_step(df)
    assert {
        "timestamp",
        "room_temp_c",
        "predicted_room_temp_c",
        "predicted_mass_temp_c",
        "heating_kw_eff",
        "solar_glass_kw",
        "solar_glass_kw_filtered",
    }.issubset(prediction.columns)
    assert len(prediction) == len(df)


def test_evaluate_returns_json_serializable_dict() -> None:
    model = RoomRC2StatePhysicalModel(RoomRC2StateConfig())
    df = build_dataframe(80)
    metrics = model.evaluate(df, horizons=(1, 6))
    json.dumps(metrics)
    assert [entry["horizon_steps"] for entry in metrics["aggregate_metrics"]] == [1, 6]


def test_save_load_preserves_parameters_and_predictions(tmp_path) -> None:
    config = RoomRC2StateConfig()
    model = RoomRC2StatePhysicalModel(config)
    params = RoomRC2StateParams(
        R_air_out=4.0,
        R_air_mass=1.2,
        R_mass_out=18.0,
        C_air=0.7,
        C_mass=42.0,
        eta_heat=0.8,
        eta_solar_air=0.3,
        eta_solar_mass=0.6,
        eta_internal=0.1,
        b_hour_sin_air=0.01,
        b_hour_cos_air=-0.02,
        b_hour_sin_mass=0.02,
        b_hour_cos_mass=-0.01,
        initial_mass_offset_c=0.3,
    )
    model.set_params(params)
    df = build_dataframe(30)
    before = model.predict_one_step(df)

    path = tmp_path / "room_rc_2state.json"
    model.save(str(path))
    loaded = RoomRC2StatePhysicalModel.load(str(path))
    after = loaded.predict_one_step(df)

    assert loaded.get_params() == params
    assert np.allclose(before["predicted_room_temp_c"], after["predicted_room_temp_c"])
    assert np.allclose(before["predicted_mass_temp_c"], after["predicted_mass_temp_c"])

