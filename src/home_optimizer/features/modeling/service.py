from __future__ import annotations

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.rolling_validation import rolling_validate_room_model
from home_optimizer.features.modeling.models import (
    RoomModelValidationReport,
    TrainedLinearRoomModel,
)
from home_optimizer.features.modeling.room_greybox import (
    ROOM_GREYBOX_MODEL_KIND,
    RoomGreyBoxConfig,
    RoomGreyBoxModel,
    RoomGreyBoxTrainer,
)
from home_optimizer.features.modeling.room_arx import (
    ROOM_ARX_MODEL_KIND,
    RoomArxConfig,
    RoomArxModel,
    RoomArxTrainer,
)


class RoomModelingService:
    def __init__(
        self,
        arx_trainer: RoomArxTrainer | None = None,
        room_greybox_trainer: RoomGreyBoxTrainer | None = None,
    ) -> None:
        self.arx_trainer = arx_trainer or RoomArxTrainer()
        self.room_greybox_trainer = room_greybox_trainer or RoomGreyBoxTrainer()

    def trainer_for_config(self, config) -> RoomArxTrainer | RoomGreyBoxTrainer:
        if config.model_kind == ROOM_ARX_MODEL_KIND:
            return self.arx_trainer
        if config.model_kind == ROOM_GREYBOX_MODEL_KIND:
            return self.room_greybox_trainer
        raise ValueError(f"unsupported room model kind: {config.model_kind}")

    def trainer_for_model(
        self,
        model: TrainedLinearRoomModel,
    ) -> RoomArxTrainer | RoomGreyBoxTrainer:
        model_kind = getattr(model, "model_kind", None)
        if (
            isinstance(model, RoomArxModel)
            or model_kind == ROOM_ARX_MODEL_KIND
            or isinstance(model.config, RoomArxConfig)
        ):
            return self.arx_trainer
        if (
            isinstance(model, RoomGreyBoxModel)
            or model_kind == ROOM_GREYBOX_MODEL_KIND
            or isinstance(model.config, RoomGreyBoxConfig)
        ):
            return self.room_greybox_trainer
        raise ValueError(f"unsupported room model kind: {model_kind}")

    def max_history_rows(self, model: TrainedLinearRoomModel) -> int:
        trainer = self.trainer_for_model(model)
        return trainer.max_history_rows(model.config)

    def fit_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomArxConfig | RoomGreyBoxConfig | None = None,
    ) -> TrainedLinearRoomModel:
        config = config or RoomArxConfig()
        return self.trainer_for_config(config).fit(dataset, config)

    def predict_next_room_temperature(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        return self.trainer_for_model(model).predict_next(
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
        return self.trainer_for_model(model).simulate_horizon(
            model,
            rows,
            start_index=start_index,
            horizon_steps=horizon_steps,
        )

    def rolling_validate_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomArxConfig | RoomGreyBoxConfig | None = None,
    ) -> RoomModelValidationReport:
        config = config or RoomArxConfig()
        return rolling_validate_room_model(
            dataset,
            config=config,
            trainer=self.trainer_for_config(config),
        )
