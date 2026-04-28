from __future__ import annotations

from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd

from home_optimizer.domain import (
    GTI_LIVING_ROOM_WINDOWS,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    HP_ELECTRIC_POWER,
    HP_FLOW,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    NumericSeries,
    OUTDOOR_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMAL_OUTPUT,
    THERMOSTAT_SETPOINT,
    adjusted_gti_with_shutter,
    build_thermal_output_series,
    latest_value_at,
)

from .ports import IdentificationDataReader
from .schemas import IdentificationDataset, IdentificationResult


class BuildingModelIdentificationService:
    """Builds a baseline autoregressive dataset and fits a linear grey-box model."""

    def __init__(self, reader: IdentificationDataReader) -> None:
        self.reader = reader

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
                HP_FLOW,
                HP_SUPPLY_TEMPERATURE,
                HP_RETURN_TEMPERATURE,
                HP_ELECTRIC_POWER,
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
        thermal_output = build_thermal_output_series(
            series_by_name.get(HP_FLOW),
            series_by_name.get(HP_SUPPLY_TEMPERATURE),
            series_by_name.get(HP_RETURN_TEMPERATURE),
        )

        raw_rows: list[dict[str, float | str]] = []
        previous_temperature: float | None = None
        for room_point in room_temperature.points:
            if previous_temperature is None:
                previous_temperature = room_point.value
                continue

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
            hp_electric_power = latest_value_at(
                series_by_name.get(
                    HP_ELECTRIC_POWER,
                    NumericSeries(name=HP_ELECTRIC_POWER, unit="kW", points=[]),
                ).points,
                room_point.timestamp,
            )
            solar_gain = latest_value_at(adjusted_gti.points, room_point.timestamp)
            delivered_heat = latest_value_at(thermal_output.points, room_point.timestamp)

            if None in (
                outdoor_temperature,
                thermostat_setpoint,
                hp_electric_power,
                solar_gain,
                delivered_heat,
            ):
                previous_temperature = room_point.value
                continue

            raw_rows.append(
                {
                    "timestamp": room_point.timestamp,
                    "previous_room_temperature": previous_temperature,
                    OUTDOOR_TEMPERATURE: float(outdoor_temperature),
                    THERMOSTAT_SETPOINT: float(thermostat_setpoint),
                    HP_ELECTRIC_POWER: float(hp_electric_power),
                    GTI_LIVING_ROOM_WINDOWS_ADJUSTED: float(solar_gain),
                    THERMAL_OUTPUT: float(delivered_heat),
                    ROOM_TEMPERATURE: room_point.value,
                }
            )
            previous_temperature = room_point.value

        frame = pd.DataFrame(raw_rows)
        if frame.empty:
            raise ValueError("identification dataset is empty after alignment")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()

        resampled = frame.resample(f"{interval_minutes}min").agg(
            {
                "previous_room_temperature": "first",
                OUTDOOR_TEMPERATURE: "mean",
                THERMOSTAT_SETPOINT: "last",
                HP_ELECTRIC_POWER: "mean",
                GTI_LIVING_ROOM_WINDOWS_ADJUSTED: "mean",
                THERMAL_OUTPUT: "mean",
                ROOM_TEMPERATURE: "last",
            }
        )
        resampled = resampled.dropna()

        feature_names = [
            "previous_room_temperature",
            OUTDOOR_TEMPERATURE,
            THERMOSTAT_SETPOINT,
            HP_ELECTRIC_POWER,
            GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
            THERMAL_OUTPUT,
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


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sqrt(float(np.mean(np.square(actual - predicted))))
