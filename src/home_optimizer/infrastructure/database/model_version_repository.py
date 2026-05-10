from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert

from home_optimizer.domain.time import normalize_utc_timestamp, parse_datetime
from home_optimizer.features.modeling import (
    ROOM_RC_MODEL_KIND,
    RoomModelValidationReport,
    ROOM_ARX_MODEL_KIND,
    RoomArxModel,
    RoomRcModel,
    StoredModelVersion,
    StoredModelVersionSummary,
)
from home_optimizer.infrastructure.database.orm_models import ModelVersion
from home_optimizer.infrastructure.database.session import Database



def _metric_by_minutes(
    report: RoomModelValidationReport | None,
    horizon_minutes: int,
):
    if report is None:
        return None
    for metric in report.aggregate_metrics:
        if metric.horizon_minutes == horizon_minutes:
            return metric
    return None


class ModelVersionRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save_room_model_version(self, version: StoredModelVersion) -> None:
        if version.model_type not in {ROOM_ARX_MODEL_KIND, ROOM_RC_MODEL_KIND}:
            raise ValueError(f"unsupported room model type: {version.model_type}")

        metric_1h = _metric_by_minutes(version.validation_report, 60)
        metric_6h = _metric_by_minutes(version.validation_report, 360)
        metric_12h = _metric_by_minutes(version.validation_report, 720)
        metric_24h = _metric_by_minutes(version.validation_report, 1440)

        row = {
            "model_id": version.model_id,
            "model_type": version.model_type,
            "created_at_utc": normalize_utc_timestamp(version.created_at_utc),
            "trained_from_utc": normalize_utc_timestamp(version.model.trained_from_utc),
            "trained_to_utc": normalize_utc_timestamp(version.model.trained_to_utc),
            "interval_minutes": version.model.interval_minutes,
            "sample_count": version.model.sample_count,
            "is_active": int(version.is_active),
            "validation_mae_1h_c": metric_1h.mae_c if metric_1h else None,
            "validation_mae_6h_c": metric_6h.mae_c if metric_6h else None,
            "validation_mae_12h_c": metric_12h.mae_c if metric_12h else None,
            "validation_mae_24h_c": metric_24h.mae_c if metric_24h else None,
            "validation_bias_6h_c": metric_6h.bias_c if metric_6h else None,
            "validation_p95_12h_c": metric_12h.p95_abs_error_c if metric_12h else None,
            "model_json": version.model.model_dump_json(),
            "validation_report_json": (
                version.validation_report.model_dump_json()
                if version.validation_report is not None
                else None
            ),
        }

        with self.database.session() as session:
            if version.is_active:
                session.execute(
                    update(ModelVersion)
                    .where(ModelVersion.model_type.in_([ROOM_ARX_MODEL_KIND, ROOM_RC_MODEL_KIND]))
                    .values(is_active=0)
                )

            statement = insert(ModelVersion).values(row)
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[ModelVersion.model_id],
                    set_={
                        "model_type": statement.excluded.model_type,
                        "created_at_utc": statement.excluded.created_at_utc,
                        "trained_from_utc": statement.excluded.trained_from_utc,
                        "trained_to_utc": statement.excluded.trained_to_utc,
                        "interval_minutes": statement.excluded.interval_minutes,
                        "sample_count": statement.excluded.sample_count,
                        "is_active": statement.excluded.is_active,
                        "validation_mae_1h_c": statement.excluded.validation_mae_1h_c,
                        "validation_mae_6h_c": statement.excluded.validation_mae_6h_c,
                        "validation_mae_12h_c": statement.excluded.validation_mae_12h_c,
                        "validation_mae_24h_c": statement.excluded.validation_mae_24h_c,
                        "validation_bias_6h_c": statement.excluded.validation_bias_6h_c,
                        "validation_p95_12h_c": statement.excluded.validation_p95_12h_c,
                        "model_json": statement.excluded.model_json,
                        "validation_report_json": statement.excluded.validation_report_json,
                    },
                )
            )
            session.commit()

    def get_room_model_version(self, model_id: str) -> StoredModelVersion | None:
        with self.database.session() as session:
            row = session.execute(
                select(ModelVersion).where(
                    ModelVersion.model_id == model_id,
                    ModelVersion.model_type.in_([ROOM_ARX_MODEL_KIND, ROOM_RC_MODEL_KIND]),
                )
            ).scalar_one_or_none()

        if row is None:
            return None
        return self._to_room_model_version(row)

    def get_active_room_model_version(self) -> StoredModelVersion | None:
        with self.database.session() as session:
            row = session.execute(
                select(ModelVersion)
                .where(
                    ModelVersion.model_type.in_([ROOM_ARX_MODEL_KIND, ROOM_RC_MODEL_KIND]),
                    ModelVersion.is_active == 1,
                )
                .order_by(ModelVersion.created_at_utc.desc())
                .limit(1)
            ).scalar_one_or_none()

        if row is None:
            return None
        return self._to_room_model_version(row)

    def list_room_model_versions(self) -> list[StoredModelVersionSummary]:
        with self.database.session() as session:
            rows = session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_type.in_([ROOM_ARX_MODEL_KIND, ROOM_RC_MODEL_KIND]))
                .order_by(ModelVersion.created_at_utc.desc())
            ).scalars().all()

        return [self._to_room_model_version_summary(row) for row in rows]

    def activate_room_model_version(self, model_id: str) -> None:
        with self.database.session() as session:
            row = session.execute(
                select(ModelVersion).where(
                    ModelVersion.model_id == model_id,
                    ModelVersion.model_type.in_([ROOM_ARX_MODEL_KIND, ROOM_RC_MODEL_KIND]),
                )
            ).scalar_one_or_none()
            if row is None:
                raise ValueError(f"unknown room model version: {model_id}")

            session.execute(
                update(ModelVersion)
                .where(ModelVersion.model_type.in_([ROOM_ARX_MODEL_KIND, ROOM_RC_MODEL_KIND]))
                .values(is_active=0)
            )
            session.execute(
                update(ModelVersion)
                .where(ModelVersion.model_id == model_id)
                .values(is_active=1)
            )
            session.commit()

    def _to_room_model_version(self, row: ModelVersion) -> StoredModelVersion:
        if row.model_type == ROOM_ARX_MODEL_KIND:
            model = RoomArxModel.model_validate_json(row.model_json)
        elif row.model_type == ROOM_RC_MODEL_KIND:
            model = RoomRcModel.model_validate_json(row.model_json)
        else:
            raise ValueError(f"unsupported room model type: {row.model_type}")

        return StoredModelVersion(
            model_id=row.model_id,
            model_type=row.model_type,
            created_at_utc=parse_datetime(row.created_at_utc),
            is_active=bool(row.is_active),
            model=model,
            validation_report=(
                RoomModelValidationReport.model_validate_json(row.validation_report_json)
                if row.validation_report_json is not None
                else None
            ),
        )

    def _to_room_model_version_summary(
        self,
        row: ModelVersion,
    ) -> StoredModelVersionSummary:
        return StoredModelVersionSummary(
            model_id=row.model_id,
            model_type=row.model_type,
            created_at_utc=parse_datetime(row.created_at_utc),
            trained_from_utc=parse_datetime(row.trained_from_utc),
            trained_to_utc=parse_datetime(row.trained_to_utc),
            interval_minutes=row.interval_minutes,
            sample_count=row.sample_count,
            is_active=bool(row.is_active),
            validation_mae_1h_c=row.validation_mae_1h_c,
            validation_mae_6h_c=row.validation_mae_6h_c,
            validation_mae_12h_c=row.validation_mae_12h_c,
            validation_mae_24h_c=row.validation_mae_24h_c,
            validation_bias_6h_c=row.validation_bias_6h_c,
            validation_p95_12h_c=row.validation_p95_12h_c,
        )
