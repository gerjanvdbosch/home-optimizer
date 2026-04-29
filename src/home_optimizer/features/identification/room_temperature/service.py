from __future__ import annotations

from datetime import datetime

from home_optimizer.domain import IdentifiedModel, utc_now

from ..ports import IdentificationDataReader, IdentifiedModelRepository
from ..schemas import IdentificationDataset, IdentificationResult
from .dataset import RoomTemperatureDatasetBuilder
from .model import MODEL_KIND, RoomTemperatureModelIdentifier


class RoomTemperatureModelIdentificationService:
    """Builds a baseline autoregressive room-temperature dataset and fits a linear model."""

    def __init__(
        self,
        reader: IdentificationDataReader,
        model_repository: IdentifiedModelRepository | None = None,
    ) -> None:
        self.reader = reader
        self.model_repository = model_repository
        self.dataset_builder = RoomTemperatureDatasetBuilder(reader)
        self.identifier = RoomTemperatureModelIdentifier()

    def build_dataset(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
    ) -> IdentificationDataset:
        return self.dataset_builder.build(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )

    def identify(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult:
        dataset = self.build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        return self.identifier.identify(
            dataset,
            interval_minutes=interval_minutes,
            train_fraction=train_fraction,
        )

    def identify_and_store(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentifiedModel:
        if self.model_repository is None:
            raise ValueError("no identified model repository configured")

        result = self.identify(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            train_fraction=train_fraction,
        )
        model = IdentifiedModel(
            model_kind=MODEL_KIND,
            model_name=result.model_name,
            trained_at_utc=utc_now(),
            training_start_time_utc=start_time,
            training_end_time_utc=end_time,
            interval_minutes=result.interval_minutes,
            sample_count=result.sample_count,
            train_sample_count=result.train_sample_count,
            test_sample_count=result.test_sample_count,
            coefficients=result.coefficients,
            intercept=result.intercept,
            train_rmse=result.train_rmse,
            test_rmse=result.test_rmse,
            target_name=result.target_name,
        )
        self.model_repository.save(model)
        return model
