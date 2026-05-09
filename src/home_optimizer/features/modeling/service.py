from __future__ import annotations

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.rolling_validation import rolling_validate_room_model
from home_optimizer.features.modeling.models import (
    RoomModelValidationReport,
    TrainedLinearRoomModel,
)
from home_optimizer.features.modeling.room.arx import RoomArxConfig, RoomArxTrainer


class RoomModelingService:
    def __init__(self, trainer: RoomArxTrainer | None = None) -> None:
        self.trainer = trainer or RoomArxTrainer()

    def fit_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomArxConfig | None = None,
    ) -> TrainedLinearRoomModel:
        config = config or RoomArxConfig()
        return self.trainer.fit(dataset, config)

    def predict_next_room_temperature(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        return self.trainer.predict_next(
            model,
            rows,
            source_index=source_index,
            predicted_room_temperatures=predicted_room_temperatures,
            prediction_origin_index=prediction_origin_index,
        )

    def simulate_horizon(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        return self.trainer.simulate_horizon(
            model,
            rows,
            start_index=start_index,
            horizon_steps=horizon_steps,
        )

    def rolling_validate_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomArxConfig | None = None,
    ) -> RoomModelValidationReport:
        config = config or RoomArxConfig()
        return rolling_validate_room_model(
            dataset,
            config=config,
            trainer=self.trainer,
        )
