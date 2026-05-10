from __future__ import annotations

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.rolling_validation import rolling_validate_room_model
from home_optimizer.features.modeling.models import (
    RoomModelValidationReport,
    TrainedLinearRoomModel,
)
from home_optimizer.features.modeling.room_arx import (
    ROOM_ARX_MODEL_KIND,
    RoomArxConfig,
    RoomArxModel,
    RoomArxTrainer,
)
from home_optimizer.features.modeling.room_rc import (
    ROOM_RC_MODEL_KIND,
    RoomRcConfig,
    RoomRcModel,
    RoomRcTrainer,
    room_rc_validation_report_from_metrics,
)


class RoomModelingService:
    def __init__(
        self,
        arx_trainer: RoomArxTrainer | None = None,
        rc_trainer: RoomRcTrainer | None = None,
    ) -> None:
        self.arx_trainer = arx_trainer or RoomArxTrainer()
        self.rc_trainer = rc_trainer or RoomRcTrainer()

    def trainer_for_config(self, config) -> RoomArxTrainer | RoomRcTrainer:
        if config.model_kind == ROOM_ARX_MODEL_KIND:
            return self.arx_trainer
        if config.model_kind == ROOM_RC_MODEL_KIND:
            return self.rc_trainer
        raise ValueError(f"unsupported room model kind: {config.model_kind}")

    def trainer_for_model(
        self,
        model: TrainedLinearRoomModel | RoomRcModel,
    ) -> RoomArxTrainer | RoomRcTrainer:
        model_kind = getattr(model, "model_kind", None)
        if (
            isinstance(model, RoomArxModel)
            or model_kind == ROOM_ARX_MODEL_KIND
            or isinstance(model.config, RoomArxConfig)
        ):
            return self.arx_trainer
        if (
            isinstance(model, RoomRcModel)
            or model_kind == ROOM_RC_MODEL_KIND
            or isinstance(model.config, RoomRcConfig)
        ):
            return self.rc_trainer
        raise ValueError(f"unsupported room model kind: {model_kind}")

    def max_history_rows(self, model: TrainedLinearRoomModel) -> int:
        trainer = self.trainer_for_model(model)
        return trainer.max_history_rows(model.config)

    def fit_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomArxConfig | RoomRcConfig | None = None,
    ) -> TrainedLinearRoomModel | RoomRcModel:
        config = config or RoomArxConfig()
        return self.trainer_for_config(config).fit(dataset, config)

    def predict_next_room_temperature(
        self,
        model: TrainedLinearRoomModel | RoomRcModel,
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
        model: TrainedLinearRoomModel | RoomRcModel,
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
        config: RoomArxConfig | RoomRcConfig | None = None,
    ) -> RoomModelValidationReport:
        config = config or RoomArxConfig()
        if isinstance(config, RoomRcConfig):
            model = self.rc_trainer.fit(dataset, config)
            prepared = self.rc_trainer.prepare(dataset.rows, config)
            physical = self.rc_trainer._physical_from_model(model)
            metrics = physical.evaluate(prepared.frame, horizons=config.validation_horizons_steps)
            segment_metrics = physical.evaluate_segments(
                prepared.frame,
                horizons=config.validation_horizons_steps,
            )
            return room_rc_validation_report_from_metrics(
                model=model,
                metrics={**metrics, **segment_metrics},
            )
        return rolling_validate_room_model(
            dataset,
            config=config,
            trainer=self.trainer_for_config(config),
        )

    def validation_report_for_model(
        self,
        dataset: MpcDataset,
        *,
        model: TrainedLinearRoomModel | RoomRcModel,
    ) -> RoomModelValidationReport:
        if isinstance(model, RoomRcModel):
            prepared = self.rc_trainer.prepare(dataset.rows, model.config)
            physical = self.rc_trainer._physical_from_model(model)
            metrics = physical.evaluate(
                prepared.frame,
                horizons=model.config.validation_horizons_steps,
            )
            segment_metrics = physical.evaluate_segments(
                prepared.frame,
                horizons=model.config.validation_horizons_steps,
            )
            return room_rc_validation_report_from_metrics(
                model=model,
                metrics={**metrics, **segment_metrics},
            )
        return self.rolling_validate_room_model(dataset, config=model.config)
