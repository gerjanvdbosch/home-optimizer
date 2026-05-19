from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.features.modeling import (
    HorizonMetric,
    ROOM_RC_MODEL_KIND,
    RoomRcConfig,
    RoomRcModel,
    RoomModelValidationReport,
    StoredModelVersion,
    ValidationFoldResult,
)
from home_optimizer.infrastructure.database.model_version_repository import (
    ModelVersionRepository,
)
from home_optimizer.infrastructure.database.session import Database


def build_model() -> RoomRcModel:
    return RoomRcModel(
        trained_from_utc=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomRcConfig(
            min_train_rows=10,
            validation_window_rows=10,
            validation_horizons_steps=[6, 36, 72, 144],
        ),
        params={
            "R_air_out": 5.0,
            "R_air_mass": 1.0,
            "R_mass_out": 20.0,
            "C_air": 0.5,
            "C_mass": 50.0,
            "eta_heat": 0.9,
            "eta_solar_air": 0.2,
            "eta_solar_mass": 0.5,
            "eta_internal": 0.1,
            "b_hour_sin_air": 0.0,
            "b_hour_cos_air": 0.0,
            "b_hour_sin_mass": 0.0,
            "b_hour_cos_mass": 0.0,
            "initial_mass_offset_c": 0.0,
        },
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


def build_rc_model() -> RoomRcModel:
    return RoomRcModel(
        trained_from_utc=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        trained_to_utc=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        interval_minutes=10,
        config=RoomRcConfig(
            min_train_rows=10,
            validation_window_rows=10,
            validation_horizons_steps=[1, 6, 36, 72],
        ),
        params={
            "R_air_out": 5.0,
            "R_air_mass": 1.0,
            "R_mass_out": 20.0,
            "C_air": 0.5,
            "C_mass": 50.0,
            "eta_heat": 0.9,
            "eta_solar_air": 0.2,
            "eta_solar_mass": 0.5,
            "eta_internal": 0.1,
            "b_hour_sin_air": 0.0,
            "b_hour_cos_air": 0.0,
            "b_hour_sin_mass": 0.0,
            "b_hour_cos_mass": 0.0,
            "initial_mass_offset_c": 0.0,
        },
        sample_count=180,
        training_metadata={"final_train_loss": 0.12},
        feature_schema={"input_columns": ["outdoor_temp_c"]},
    )


def test_model_version_repository_round_trips_room_model_versions(tmp_path) -> None:
    database = Database(str(tmp_path / "model_versions.sqlite"))
    database.init_schema()
    repository = ModelVersionRepository(database)

    version = StoredModelVersion(
        model_id="room-model-v1",
        model_type=ROOM_RC_MODEL_KIND,
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
    assert isinstance(loaded.model, RoomRcModel)
    assert loaded.model.params["C_mass"] == version.model.params["C_mass"]
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
            model_type=ROOM_RC_MODEL_KIND,
            created_at_utc=datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc),
            is_active=True,
            model=build_model(),
            validation_report=build_validation_report(),
        )
    )
    repository.save_room_model_version(
        StoredModelVersion(
            model_id="room-model-v2",
            model_type=ROOM_RC_MODEL_KIND,
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


def test_model_version_repository_round_trips_room_rc_model_versions(tmp_path) -> None:
    database = Database(str(tmp_path / "model_versions_rc.sqlite"))
    database.init_schema()
    repository = ModelVersionRepository(database)

    version = StoredModelVersion(
        model_id="room-rc-v1",
        model_type=ROOM_RC_MODEL_KIND,
        created_at_utc=datetime(2026, 5, 12, 9, 0, tzinfo=timezone.utc),
        is_active=True,
        model=build_rc_model(),
        validation_report=build_validation_report(),
    )

    repository.save_room_model_version(version)

    loaded = repository.get_room_model_version("room-rc-v1")
    active = repository.get_active_room_model_version()
    summaries = repository.list_room_model_versions()

    assert loaded is not None
    assert loaded.model_id == "room-rc-v1"
    assert loaded.model_type == ROOM_RC_MODEL_KIND
    assert isinstance(loaded.model, RoomRcModel)
    assert loaded.model.params["C_mass"] == 50.0
    assert active is not None
    assert active.model_id == "room-rc-v1"
    summary_by_id = {summary.model_id: summary for summary in summaries}
    assert summary_by_id["room-rc-v1"].model_type == ROOM_RC_MODEL_KIND
