from __future__ import annotations

import json

from sqlalchemy import select

from home_optimizer.domain import IdentifiedModel, normalize_utc_timestamp
from home_optimizer.domain.time import parse_datetime
from home_optimizer.infrastructure.database.orm_models import IdentifiedModelRecord
from home_optimizer.infrastructure.database.session import Database


class IdentifiedModelRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, model: IdentifiedModel) -> None:
        with self.database.session() as session:
            session.merge(self._to_record(model))
            session.commit()

    def latest(
        self,
        *,
        model_kind: str,
        model_name: str | None = None,
    ) -> IdentifiedModel | None:
        with self.database.session() as session:
            stmt = select(IdentifiedModelRecord).where(
                IdentifiedModelRecord.model_kind == model_kind
            )
            if model_name is not None:
                stmt = stmt.where(IdentifiedModelRecord.model_name == model_name)
            stmt = stmt.order_by(IdentifiedModelRecord.trained_at_utc.desc()).limit(1)
            row = session.execute(stmt).scalar_one_or_none()

        if row is None:
            return None
        return self._to_model(row)

    @staticmethod
    def _to_record(model: IdentifiedModel) -> IdentifiedModelRecord:
        return IdentifiedModelRecord(
            model_kind=model.model_kind,
            trained_at_utc=normalize_utc_timestamp(model.trained_at_utc),
            model_name=model.model_name,
            training_start_time_utc=normalize_utc_timestamp(model.training_start_time_utc),
            training_end_time_utc=normalize_utc_timestamp(model.training_end_time_utc),
            interval_minutes=model.interval_minutes,
            sample_count=model.sample_count,
            train_sample_count=model.train_sample_count,
            test_sample_count=model.test_sample_count,
            coefficients_json=json.dumps(model.coefficients, sort_keys=True),
            intercept=model.intercept,
            train_rmse=model.train_rmse,
            test_rmse=model.test_rmse,
            test_rmse_recursive=model.test_rmse_recursive,
            target_name=model.target_name,
        )

    @staticmethod
    def _to_model(row: IdentifiedModelRecord) -> IdentifiedModel:
        return IdentifiedModel(
            model_kind=row.model_kind,
            model_name=row.model_name,
            trained_at_utc=parse_datetime(row.trained_at_utc),
            training_start_time_utc=parse_datetime(row.training_start_time_utc),
            training_end_time_utc=parse_datetime(row.training_end_time_utc),
            interval_minutes=row.interval_minutes,
            sample_count=row.sample_count,
            train_sample_count=row.train_sample_count,
            test_sample_count=row.test_sample_count,
            coefficients={
                name: float(value)
                for name, value in json.loads(row.coefficients_json).items()
            },
            intercept=row.intercept,
            train_rmse=row.train_rmse,
            test_rmse=row.test_rmse,
            test_rmse_recursive=row.test_rmse_recursive,
            target_name=row.target_name,
        )
