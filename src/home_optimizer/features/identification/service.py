from __future__ import annotations

from datetime import datetime

from home_optimizer.app.settings import AppSettings
from home_optimizer.features.dataset import MpcDatasetService
from home_optimizer.features.identification.models import (
    IdentificationDataset,
    IdentificationDatasetRow,
    IdentificationDatasetSummary,
)
from home_optimizer.features.identification.ports import IdentificationDataReader


class IdentificationDatasetService:
    def __init__(self, reader: IdentificationDataReader, settings: AppSettings) -> None:
        self.dataset_service = MpcDatasetService(reader, settings)

    def build_dataset(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15,
    ) -> IdentificationDataset:
        dataset = self.dataset_service.build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        return IdentificationDataset(
            interval_minutes=dataset.interval_minutes,
            start_time_utc=dataset.start_time_utc,
            end_time_utc=dataset.end_time_utc,
            rows=[
                IdentificationDatasetRow.model_validate(row.model_dump())
                for row in dataset.rows
            ],
        )

    def summarize_dataset(
        self,
        dataset: IdentificationDataset,
    ) -> IdentificationDatasetSummary:
        summary = self.dataset_service.summarize_dataset(dataset)
        return IdentificationDatasetSummary(
            total_rows=summary.total_rows,
            mode_space_rows=summary.mode_space_rows,
            mode_dhw_rows=summary.mode_dhw_rows,
            mode_off_rows=summary.mode_off_rows,
            defrost_rows=summary.defrost_rows,
            booster_rows=summary.booster_rows,
            valid_room_rows=summary.valid_room_rows,
            valid_dhw_rows=summary.valid_dhw_rows,
            valid_cop_rows=summary.valid_cop_rows,
            exclusion_reason_counts=summary.exclusion_reason_counts,
        )
