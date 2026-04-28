from __future__ import annotations

from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd

from home_optimizer.domain import (
    BuildingTemperatureModel,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    NumericSeries,
    OUTDOOR_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    adjusted_gti_with_shutter,
    latest_value_at,
    utc_now,
)

from .ports import BuildingTemperatureModelRepository, IdentificationDataReader
from .schemas import IdentificationDataset, IdentificationResult


class BuildingModelIdentificationService:
    """Builds a baseline autoregressive dataset and fits a linear grey-box model."""

    def __init__(
        self,
        reader: IdentificationDataReader,
        model_repository: BuildingTemperatureModelRepository | None = None,
    ) -> None:
        self.reader = reader
        self.model_repository = model_repository

    def build_dataset(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
    ) -> IdentificationDataset:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")

        series = self.reader.read_series(
            names=[
                ROOM_TEMPERATURE,
                OUTDOOR_TEMPERATURE,
                THERMOSTAT_SETPOINT,
                SHUTTER_LIVING_ROOM,
            ],
            start_time=start_time,
            end_time=end_time,
        )
        forecast_series = self.reader.read_forecast_series(
            names=[GTI_LIVING_ROOM_WINDOWS],
            start_time=start_time,
            end_time=end_time,
        )

        series_by_name = {item.name: item for item in series}
        forecast_by_name = {item.name: item for item in forecast_series}

        room_temperature = series_by_name[ROOM_TEMPERATURE]
        if not room_temperature.points:
            raise ValueError("room_temperature series is empty")

        empty_gti_series = NumericSeries(name=GTI_LIVING_ROOM_WINDOWS, unit="Wm2", points=[])
        adjusted_gti = adjusted_gti_with_shutter(
            forecast_by_name.get(GTI_LIVING_ROOM_WINDOWS, empty_gti_series),
            series_by_name.get(
                SHUTTER_LIVING_ROOM,
                NumericSeries(name=SHUTTER_LIVING_ROOM, unit="percent", points=[]),
            ),
        )

        raw_rows: list[dict[str, float | str]] = []
        for room_point in room_temperature.points:
            outdoor_temperature = latest_value_at(
                series_by_name.get(
                    OUTDOOR_TEMPERATURE,
                    NumericSeries(name=OUTDOOR_TEMPERATURE, unit="degC", points=[]),
                ).points,
                room_point.timestamp,
            )
            thermostat_setpoint = latest_value_at(
                series_by_name.get(
                    THERMOSTAT_SETPOINT,
                    NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=[]),
                ).points,
                room_point.timestamp,
            )
            solar_gain = latest_value_at(adjusted_gti.points, room_point.timestamp)

            if None in (
                outdoor_temperature,
                thermostat_setpoint,
                solar_gain,
            ):
                continue

            raw_rows.append(
                {
                    "timestamp": room_point.timestamp,
                    OUTDOOR_TEMPERATURE: float(outdoor_temperature),
                    THERMOSTAT_SETPOINT: float(thermostat_setpoint),
                    GTI_LIVING_ROOM_WINDOWS_ADJUSTED: float(solar_gain),
                    ROOM_TEMPERATURE: room_point.value,
                }
            )

        frame = pd.DataFrame(raw_rows)
        if frame.empty:
            raise ValueError("identification dataset is empty after alignment")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()

        resampled = frame.resample(f"{interval_minutes}min").agg(
            {
                OUTDOOR_TEMPERATURE: "mean",
                THERMOSTAT_SETPOINT: "last",
                GTI_LIVING_ROOM_WINDOWS_ADJUSTED: "mean",
                ROOM_TEMPERATURE: "last",
            }
        )
        resampled["previous_room_temperature"] = resampled[ROOM_TEMPERATURE].shift(1)
        resampled = resampled.dropna()

        feature_names = [
            "previous_room_temperature",
            OUTDOOR_TEMPERATURE,
            THERMOSTAT_SETPOINT,
            GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
        ]
        if len(resampled) < 3:
            raise ValueError("not enough aligned samples to identify a model")

        return IdentificationDataset(
            timestamps=[timestamp.isoformat() for timestamp in resampled.index.to_pydatetime()],
            feature_names=feature_names,
            target_name=ROOM_TEMPERATURE,
            features=resampled[feature_names].to_numpy(dtype=float).tolist(),
            targets=resampled[ROOM_TEMPERATURE].to_numpy(dtype=float).tolist(),
        )

    def identify(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> IdentificationResult:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

        dataset = self.build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        features = np.asarray(dataset.features, dtype=float)
        targets = np.asarray(dataset.targets, dtype=float)

        sample_count = len(targets)
        split_index = max(1, min(sample_count - 1, int(sample_count * train_fraction)))

        train_features = features[:split_index]
        test_features = features[split_index:]
        train_targets = targets[:split_index]
        test_targets = targets[split_index:]

        train_design = np.column_stack([np.ones(len(train_features)), train_features])
        coefficients, _, _, _ = np.linalg.lstsq(train_design, train_targets, rcond=None)

        intercept = float(coefficients[0])
        feature_coefficients = {
            name: float(value)
            for name, value in zip(dataset.feature_names, coefficients[1:], strict=True)
        }

        train_predictions = train_design @ coefficients
        test_design = np.column_stack([np.ones(len(test_features)), test_features])
        test_predictions = test_design @ coefficients

        return IdentificationResult(
            model_name="linear_1step_room_temperature",
            interval_minutes=interval_minutes,
            sample_count=sample_count,
            train_sample_count=len(train_targets),
            test_sample_count=len(test_targets),
            coefficients=feature_coefficients,
            intercept=intercept,
            train_rmse=_rmse(train_targets, train_predictions),
            test_rmse=_rmse(test_targets, test_predictions),
            target_name=dataset.target_name,
        )

    def identify_and_store(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
        train_fraction: float = 0.8,
    ) -> BuildingTemperatureModel:
        if self.model_repository is None:
            raise ValueError("no building temperature model repository configured")

        result = self.identify(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            train_fraction=train_fraction,
        )
        model = BuildingTemperatureModel(
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


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sqrt(float(np.mean(np.square(actual - predicted))))
