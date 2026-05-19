from __future__ import annotations

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.models import RoomModelValidationReport
from home_optimizer.features.modeling.room_2r2c import (
    ROOM_RC_MODEL_KIND,
    RoomRcConfig,
    RoomRcModel,
    RoomRcTrainer,
    room_rc_validation_report_from_metrics,
)


class RoomModelingService:
    def __init__(
        self,
        rc_trainer: RoomRcTrainer | None = None,
    ) -> None:
        self.rc_trainer = rc_trainer or RoomRcTrainer()

    def trainer_for_config(self, config) -> RoomRcTrainer:
        if config.model_kind == ROOM_RC_MODEL_KIND:
            return self.rc_trainer
        raise ValueError(f"unsupported room model kind: {config.model_kind}")

    def trainer_for_model(
        self,
        model: RoomRcModel,
    ) -> RoomRcTrainer:
        if isinstance(model, RoomRcModel):
            return self.rc_trainer
        raise ValueError(f"unsupported room model type: {type(model)}")

    def max_history_rows(self, model: RoomRcModel) -> int:
        trainer = self.trainer_for_model(model)
        return trainer.max_history_rows(model.config)

    def fit_room_model(
        self,
        dataset: MpcDataset,
        *,
        config: RoomRcConfig | None = None,
    ) -> RoomRcModel:
        config = config or RoomRcConfig()
        return self.trainer_for_config(config).fit(dataset, config)

    def predict_next_room_temperature(
        self,
        model: RoomRcModel,
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
        model: RoomRcModel,
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
        config: RoomRcConfig | None = None,
    ) -> RoomModelValidationReport:
        config = config or RoomRcConfig()
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

    def validation_report_for_model(
        self,
        dataset: MpcDataset,
        *,
        model: RoomRcModel,
    ) -> RoomModelValidationReport:
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
