from __future__ import annotations

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.common.rolling_validation import rolling_validate_room_model
from home_optimizer.features.modeling.models import (
    RoomModelConfig,
    RoomModelValidationReport,
    TrainedLinearRoomModel,
)
from home_optimizer.features.modeling.room.arx import (
    fit_room_arx_model,
    predict_next_room_arx_temperature,
    row_segments,
    segment_definitions,
    simulate_room_arx_horizon,
    validation_stride_rows,
)


class RoomModelingService:
    def fit_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomModelConfig | None = None,
    ) -> TrainedLinearRoomModel:
        config = config or RoomModelConfig()
        return fit_room_arx_model(dataset, config)

    def predict_next_room_temperature(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        return predict_next_room_arx_temperature(
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
        return simulate_room_arx_horizon(
            model,
            rows,
            start_index=start_index,
            horizon_steps=horizon_steps,
        )

    def rolling_validate_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomModelConfig | None = None,
    ) -> RoomModelValidationReport:
        config = config or RoomModelConfig()
        return rolling_validate_room_model(
            dataset,
            config=config,
            fit_model=fit_room_arx_model,
            simulate_horizon=lambda model, rows, start_index, horizon_steps: simulate_room_arx_horizon(
                model,
                rows,
                start_index=start_index,
                horizon_steps=horizon_steps,
            ),
            row_segments=row_segments,
            segment_definitions=segment_definitions,
            validation_stride_rows=validation_stride_rows,
        )
