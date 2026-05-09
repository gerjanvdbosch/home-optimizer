from __future__ import annotations

from datetime import datetime, timezone
import json

from home_optimizer.features.modeling import (
    HorizonMetric,
    ROOM_ARX_MODEL_KIND,
    ROOM_GREYBOX_MODEL_KIND,
    RoomArxConfig,
    RoomGreyBoxConfig,
    RoomGreyBoxModel,
    RoomModelValidationReport,
    StoredModelVersion,
    TrainedLinearRoomModel,
    ValidationFoldResult,
)
from home_optimizer.infrastructure.database.model_version_repository import (
    ModelVersionRepository,
)
from home_optimizer.infrastructure.database.session import Database


def build_model() -> TrainedLinearRoomModel:
    return TrainedLinearRoomModel(
        trained_from_utc=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomArxConfig(
            room_temperature_lags=[0],
            outdoor_temperature_lags=[0],
            thermal_output_lags=[0],
            solar_gain_lags=[0],
            occupied_flag_lags=[0],
            validation_horizons_steps=[6, 36, 72, 144],
            min_train_rows=10,
            validation_window_rows=10,
        ),
        feature_names=[
            "room_temperature_lag_0",
            "outdoor_temperature_lag_0",
            "thermal_output_lag_0",
            "solar_gain_lag_0",
            "occupied_flag_lag_0",
        ],
        intercept=1.23,
        coefficients=[0.8, 0.1, 0.2, 0.01, 0.05],
        sample_count=200,
    )


def build_validation_report() -> RoomModelValidationReport:
    metrics = [
        HorizonMetric(
            horizon_steps=6,
            horizon_minutes=60,
            sample_count=12,
            mae_c=0.2,
            rmse_c=0.25,
            bias_c=-0.01,
            p95_abs_error_c=0.4,
        ),
        HorizonMetric(
            horizon_steps=36,
            horizon_minutes=360,
            sample_count=12,
            mae_c=0.6,
            rmse_c=0.7,
            bias_c=0.05,
            p95_abs_error_c=1.1,
        ),
        HorizonMetric(
            horizon_steps=72,
            horizon_minutes=720,
            sample_count=12,
            mae_c=0.9,
            rmse_c=1.0,
            bias_c=0.08,
            p95_abs_error_c=1.5,
        ),
        HorizonMetric(
            horizon_steps=144,
            horizon_minutes=1440,
            sample_count=12,
            mae_c=1.2,
            rmse_c=1.35,
            bias_c=0.12,
            p95_abs_error_c=2.0,
        ),
    ]
    return RoomModelValidationReport(
        interval_minutes=10,
        config=build_model().config,
        folds=[
            ValidationFoldResult(
                train_start_utc=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
                train_end_utc=datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc),
                validate_start_utc=datetime(2026, 5, 5, 0, 10, tzinfo=timezone.utc),
                validate_end_utc=datetime(2026, 5, 6, 0, 0, tzinfo=timezone.utc),
                training_sample_count=120,
                metrics=metrics,
            )
        ],
        aggregate_metrics=metrics,
    )
def build_greybox_model() -> RoomGreyBoxModel:
    return RoomGreyBoxModel(
        trained_from_utc=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomGreyBoxConfig(
            validation_horizons_steps=[6, 36, 72, 144],
            min_train_rows=10,
            validation_window_rows=10,
            history_warmup_rows=24,
        ),
        feature_names=[
            "room_temperature_c",
            "thermal_mass_state",
            "outdoor_temperature_c",
            "thermal_output_energy_kwh",
            "solar_effective_exposure",
        ],
        intercept=0.0,
        coefficients=[0.85, 0.10, 0.05, 0.04, 0.002],
        sample_count=220,
        k_out=0.05,
        k_mass=0.10,
        k_air_mass=0.04,
        g_heat_mass=0.03,
        g_solar_mass=0.001,
        observer_gain=0.1,
    )


def test_model_version_repository_round_trips_room_model_versions(tmp_path) -> None:
    database = Database(str(tmp_path / "model_versions.sqlite"))
    database.init_schema()
    repository = ModelVersionRepository(database)

    version = StoredModelVersion(
        model_id="room-model-v1",
        model_type=ROOM_ARX_MODEL_KIND,
        created_at_utc=datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc),
        is_active=True,
        model=build_model(),
        validation_report=build_validation_report(),
    )

    repository.save_room_model_version(version)

    loaded = repository.get_room_model_version("room-model-v1")
    active = repository.get_active_room_model_version()
    summaries = repository.list_room_model_versions()

    assert loaded is not None
    assert loaded.model_id == "room-model-v1"
    assert loaded.is_active is True
    assert loaded.model.feature_names == version.model.feature_names
    assert loaded.validation_report is not None
    assert loaded.validation_report.aggregate_metrics[1].mae_c == 0.6
    assert active is not None
    assert active.model_id == "room-model-v1"
    assert len(summaries) == 1
    assert summaries[0].validation_mae_1h_c == 0.2
    assert summaries[0].validation_mae_6h_c == 0.6
    assert summaries[0].validation_mae_12h_c == 0.9
    assert summaries[0].validation_mae_24h_c == 1.2
    assert summaries[0].validation_bias_6h_c == 0.05
    assert summaries[0].validation_p95_12h_c == 1.5


def test_model_version_repository_switches_active_room_model(tmp_path) -> None:
    database = Database(str(tmp_path / "model_versions.sqlite"))
    database.init_schema()
    repository = ModelVersionRepository(database)

    repository.save_room_model_version(
        StoredModelVersion(
            model_id="room-model-v1",
            model_type=ROOM_ARX_MODEL_KIND,
            created_at_utc=datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc),
            is_active=True,
            model=build_model(),
            validation_report=build_validation_report(),
        )
    )
    repository.save_room_model_version(
        StoredModelVersion(
            model_id="room-model-v2",
            model_type=ROOM_ARX_MODEL_KIND,
            created_at_utc=datetime(2026, 5, 11, 10, 0, tzinfo=timezone.utc),
            is_active=False,
            model=build_model().model_copy(
                update={
                    "trained_to_utc": datetime(2026, 5, 11, 0, 0, tzinfo=timezone.utc),
                    "sample_count": 240,
                }
            ),
            validation_report=build_validation_report(),
        )
    )

    repository.activate_room_model_version("room-model-v2")

    active = repository.get_active_room_model_version()
    summaries = repository.list_room_model_versions()

    assert active is not None
    assert active.model_id == "room-model-v2"
    summary_by_id = {summary.model_id: summary for summary in summaries}
    assert summary_by_id["room-model-v1"].is_active is False
    assert summary_by_id["room-model-v2"].is_active is True
def test_model_version_repository_keeps_one_active_model_per_type(tmp_path) -> None:
    database = Database(str(tmp_path / "model_versions.sqlite"))
    database.init_schema()
    repository = ModelVersionRepository(database)

    repository.save_room_model_version(
        StoredModelVersion(
            model_id="room-model-arx",
            model_type=ROOM_ARX_MODEL_KIND,
            created_at_utc=datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc),
            is_active=True,
            model=build_model(),
            validation_report=build_validation_report(),
        )
    )
    repository.save_room_model_version(
        StoredModelVersion(
            model_id="room-model-greybox",
            model_type=ROOM_GREYBOX_MODEL_KIND,
            created_at_utc=datetime(2026, 5, 11, 10, 0, tzinfo=timezone.utc),
            is_active=True,
            model=build_greybox_model(),
            validation_report=build_validation_report(),
        )
    )

    summaries = repository.list_room_model_versions()
    summary_by_id = {summary.model_id: summary for summary in summaries}

    assert summary_by_id["room-model-arx"].is_active is True
    assert summary_by_id["room-model-greybox"].is_active is True


def test_model_version_repository_round_trips_greybox_room_model_versions(tmp_path) -> None:
    database = Database(str(tmp_path / "model_versions.sqlite"))
    database.init_schema()
    repository = ModelVersionRepository(database)

    version = StoredModelVersion(
        model_id="room-model-greybox",
        model_type=ROOM_GREYBOX_MODEL_KIND,
        created_at_utc=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
        is_active=True,
        model=build_greybox_model(),
        validation_report=build_validation_report(),
    )

    repository.save_room_model_version(version)

    loaded = repository.get_room_model_version("room-model-greybox")
    active = repository.get_active_room_model_version()

    assert loaded is not None
    assert loaded.model_type == ROOM_GREYBOX_MODEL_KIND
    assert isinstance(loaded.model, RoomGreyBoxModel)
    assert loaded.model.k_out == 0.05
    assert loaded.model.k_mass == 0.10
    assert loaded.model.k_air_mass == 0.04
    assert loaded.model.g_heat_air == 0.04
    assert loaded.model.g_solar_air == 0.002
    assert "Trained grey-box room state-space model" in loaded.model.notes
    assert active is not None
    assert active.model_id == "room-model-greybox"


def test_greybox_model_accepts_legacy_serialized_fields() -> None:
    legacy_payload = {
        "model_kind": ROOM_GREYBOX_MODEL_KIND,
        "trained_from_utc": "2026-05-01T00:00:00+00:00",
        "trained_to_utc": "2026-05-10T00:00:00+00:00",
        "interval_minutes": 10,
        "config": {
            "model_kind": ROOM_GREYBOX_MODEL_KIND,
            "min_train_rows": 10,
            "validation_window_rows": 10,
            "validation_horizons_steps": [6, 36, 72],
            "history_warmup_rows": 24,
        },
        "feature_names": [
            "room_temperature_c",
            "thermal_mass_state",
            "outdoor_temperature_c",
            "thermal_output_energy_kwh",
            "solar_effective_exposure",
        ],
        "intercept": 0.0,
        "coefficients": [0.84, 0.11, 0.05, 0.04, 0.002],
        "sample_count": 200,
        "a21": 0.04,
        "a22": 0.96,
        "heat_to_mass": 0.03,
        "solar_to_mass": 0.001,
        "observer_gain": 0.1,
    }

    model = RoomGreyBoxModel.model_validate_json(json.dumps(legacy_payload))

    assert model.k_out == 0.05
    assert model.k_mass == 0.11
    assert model.k_air_mass == 0.04
    assert model.g_heat_air == 0.04
    assert model.g_solar_air == 0.002
    assert model.g_heat_mass == 0.03
    assert model.g_solar_mass == 0.001

