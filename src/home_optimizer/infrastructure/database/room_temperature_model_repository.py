from __future__ import annotations

import json

from sqlalchemy import select

from home_optimizer.domain import RoomTemperatureModel, normalize_utc_timestamp
from home_optimizer.domain.time import parse_datetime
from home_optimizer.infrastructure.database.orm_models import RoomTemperatureModelRecord
from home_optimizer.infrastructure.database.session import Database


class RoomTemperatureModelRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, model: RoomTemperatureModel) -> None:
        with self.database.session() as session:
            session.merge(self._to_record(model))
            session.commit()

    def latest(self) -> RoomTemperatureModel | None:
        with self.database.session() as session:
            stmt = (
                select(RoomTemperatureModelRecord)
                .order_by(RoomTemperatureModelRecord.trained_at_utc.desc())
                .limit(1)
            )
            row = session.execute(stmt).scalar_one_or_none()

        if row is None:
            return None
        return self._to_model(row)

    @staticmethod
    def _to_record(model: RoomTemperatureModel) -> RoomTemperatureModelRecord:
        return RoomTemperatureModelRecord(
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
            target_name=model.target_name,
        )

    @staticmethod
    def _to_model(row: RoomTemperatureModelRecord) -> RoomTemperatureModel:
        return RoomTemperatureModel(
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
            target_name=row.target_name,
        )
