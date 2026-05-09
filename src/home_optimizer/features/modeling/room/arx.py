from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from pydantic import Field, field_validator

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.models import TrainedLinearRoomModel, ValidationConfig

ROOM_ARX_MODEL_KIND = "room_arx"


class RoomArxConfig(ValidationConfig):
    model_kind: str = ROOM_ARX_MODEL_KIND
    room_temperature_lags: list[int] = Field(default_factory=lambda: [0, 1])
    outdoor_temperature_lags: list[int] = Field(default_factory=lambda: [0])
    thermal_output_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6])
    solar_gain_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6, 12, 18])
    shutter_position_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6])
    solar_shutter_interaction_lags: list[int] = Field(default_factory=lambda: [0, 1, 3, 6, 12])
    occupied_flag_lags: list[int] = Field(default_factory=lambda: [0])
    ridge_alpha: float = Field(default=0.0, ge=0.0)
    sunny_irradiance_threshold_w_m2: float = Field(default=150.0, ge=0.0)
    heating_active_threshold_kw: float = Field(default=0.1, ge=0.0)
    shutters_open_min_pct: float = Field(default=75.0, ge=0.0, le=100.0)
    shutters_closed_max_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    cold_outdoor_temperature_max_c: float = Field(default=5.0)
    freezing_outdoor_temperature_max_c: float = Field(default=0.0)
    mild_outdoor_temperature_min_c: float = Field(default=12.0)
    night_start_hour: int = Field(default=22, ge=0, le=23)
    night_end_hour: int = Field(default=6, ge=0, le=23)
    sunny_midday_start_hour: int = Field(default=11, ge=0, le=23)
    sunny_midday_end_hour: int = Field(default=16, ge=1, le=24)
    notes: str = Field(
        default="Autoregressive room model with exogenous thermal, solar, shutter, and occupancy inputs."
    )

    @field_validator(
        "room_temperature_lags",
        "outdoor_temperature_lags",
        "thermal_output_lags",
        "solar_gain_lags",
        "shutter_position_lags",
        "solar_shutter_interaction_lags",
        "occupied_flag_lags",
    )
    @classmethod
    def _validate_non_negative_int_list(cls, value: list[int]) -> list[int]:
        ordered = sorted(set(value))
        if not ordered:
            raise ValueError("lag lists cannot be empty")
        if ordered[0] < 0:
            raise ValueError("lag values must be non-negative")
        return ordered


class RoomArxModel(TrainedLinearRoomModel):
    model_kind: str = ROOM_ARX_MODEL_KIND
    config: RoomArxConfig
    notes: str = Field(
        default="Trained ARX room model with linear coefficients over lagged room and exogenous features."
    )


@dataclass(frozen=True)
class PreparedRoomArxData:
    timestamps_utc: list[datetime]
    feature_specs: list[tuple[str, str, int]]
    feature_names: list[str]
    feature_matrix: np.ndarray
    target_values: np.ndarray
    valid_training_mask: np.ndarray
    field_arrays: dict[str, np.ndarray]
    segment_masks: dict[str, np.ndarray]


class RoomArxTrainer:
    def max_lag(self, config: RoomArxConfig) -> int:
        return max(
            config.room_temperature_lags
            + config.outdoor_temperature_lags
            + config.thermal_output_lags
            + config.solar_gain_lags
            + config.shutter_position_lags
            + config.solar_shutter_interaction_lags
            + config.occupied_flag_lags
        )

    def row_value(self, row: MpcDatasetRow, field_name: str) -> float | None:
        if field_name == "solar_shutter_interaction":
            solar_irradiance = row.solar_irradiance_w_m2
            shutter_position = row.shutter_position_pct
            if solar_irradiance is None and shutter_position is None:
                return None
            resolved_irradiance = float(solar_irradiance or 0.0)
            resolved_shutter_fraction = (
                max(0.0, min(float(shutter_position or 0.0), 100.0)) / 100.0
            )
            return resolved_irradiance * resolved_shutter_fraction

        value = getattr(row, field_name)
        if value is None:
            return None
        return float(value)

    def feature_specs(self, config: RoomArxConfig) -> list[tuple[str, str, int]]:
        specs: list[tuple[str, str, int]] = []
        specs.extend(
            ("room_temperature_c", f"room_temperature_lag_{lag}", lag)
            for lag in config.room_temperature_lags
        )
        specs.extend(
            ("outdoor_temperature_c", f"outdoor_temperature_lag_{lag}", lag)
            for lag in config.outdoor_temperature_lags
        )
        specs.extend(
            ("thermal_output_estimate_kw", f"thermal_output_lag_{lag}", lag)
            for lag in config.thermal_output_lags
        )
        specs.extend(
            ("solar_gain_proxy_w_m2", f"solar_gain_lag_{lag}", lag)
            for lag in config.solar_gain_lags
        )
        specs.extend(
            ("shutter_position_pct", f"shutter_position_lag_{lag}", lag)
            for lag in config.shutter_position_lags
        )
        specs.extend(
            ("solar_shutter_interaction", f"solar_shutter_interaction_lag_{lag}", lag)
            for lag in config.solar_shutter_interaction_lags
        )
        specs.extend(
            ("occupied_flag", f"occupied_flag_lag_{lag}", lag)
            for lag in config.occupied_flag_lags
        )
        return specs

    def default_feature_value(self, field_name: str) -> float | None:
        if field_name in {
            "outdoor_temperature_c",
            "thermal_output_estimate_kw",
            "solar_gain_proxy_w_m2",
            "shutter_position_pct",
            "solar_shutter_interaction",
            "occupied_flag",
        }:
            return 0.0
        return None

    def validation_stride_rows(self, config: RoomArxConfig, interval_minutes: int) -> int:
        if config.validation_stride_rows is not None:
            return config.validation_stride_rows
        return max(1, 60 // interval_minutes)

    def prepare(self, rows: list[MpcDatasetRow], config: RoomArxConfig) -> PreparedRoomArxData:
        timestamps_utc = [row.timestamp_utc for row in rows]
        local_hours = np.asarray(
            [timestamp_utc.astimezone().hour for timestamp_utc in timestamps_utc],
            dtype=int,
        )
        room_temperature = np.asarray(
            [
                np.nan if row.room_temperature_c is None else float(row.room_temperature_c)
                for row in rows
            ],
            dtype=float,
        )
        outdoor_temperature = np.asarray(
            [
                np.nan if row.outdoor_temperature_c is None else float(row.outdoor_temperature_c)
                for row in rows
            ],
            dtype=float,
        )
        thermal_output = np.asarray(
            [
                0.0
                if row.thermal_output_estimate_kw is None
                else float(row.thermal_output_estimate_kw)
                for row in rows
            ],
            dtype=float,
        )
        solar_gain = np.asarray(
            [0.0 if row.solar_gain_proxy_w_m2 is None else float(row.solar_gain_proxy_w_m2) for row in rows],
            dtype=float,
        )
        solar_irradiance = np.asarray(
            [0.0 if row.solar_irradiance_w_m2 is None else float(row.solar_irradiance_w_m2) for row in rows],
            dtype=float,
        )
        shutter_position = np.asarray(
            [0.0 if row.shutter_position_pct is None else float(row.shutter_position_pct) for row in rows],
            dtype=float,
        )
        occupied_flag = np.asarray(
            [0.0 if row.occupied_flag is None else float(row.occupied_flag) for row in rows],
            dtype=float,
        )
        solar_shutter_interaction = solar_irradiance * np.clip(shutter_position, 0.0, 100.0) / 100.0

        frame = pd.DataFrame(
            {
                "room_temperature_c": room_temperature,
                "outdoor_temperature_c": outdoor_temperature,
                "thermal_output_estimate_kw": thermal_output,
                "solar_gain_proxy_w_m2": solar_gain,
                "shutter_position_pct": shutter_position,
                "solar_shutter_interaction": solar_shutter_interaction,
                "occupied_flag": occupied_flag,
            }
        )

        specs = self.feature_specs(config)
        feature_names = [feature_name for _, feature_name, _ in specs]
        for field_name, feature_name, lag in specs:
            frame[feature_name] = frame[field_name].shift(lag)

        feature_matrix = frame[feature_names].to_numpy(dtype=float)
        target_values = pd.Series(room_temperature).shift(-1).to_numpy(dtype=float)

        source_indices = np.arange(len(rows), dtype=int)
        valid_training_mask = (
            (source_indices >= self.max_lag(config))
            & (source_indices < max(0, len(rows) - 1))
            & ~np.isnan(target_values)
        )
        for column_index, (field_name, _, _) in enumerate(specs):
            if self.default_feature_value(field_name) is None:
                valid_training_mask &= ~np.isnan(feature_matrix[:, column_index])

        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

        is_sunny = solar_irradiance >= config.sunny_irradiance_threshold_w_m2
        is_heating_active = thermal_output >= config.heating_active_threshold_kw
        is_shutters_open = shutter_position >= config.shutters_open_min_pct
        is_shutters_closed = shutter_position <= config.shutters_closed_max_pct
        is_night = (local_hours >= config.night_start_hour) | (local_hours < config.night_end_hour)
        is_freezing = outdoor_temperature <= config.freezing_outdoor_temperature_max_c

        segment_masks = {
            "sunny": is_sunny,
            "heating_active": is_heating_active,
            "cold_weather": outdoor_temperature <= config.cold_outdoor_temperature_max_c,
            "freezing_weather": is_freezing,
            "mild_weather": outdoor_temperature >= config.mild_outdoor_temperature_min_c,
            "freezing_and_heating": is_freezing & is_heating_active,
            "freezing_night": is_freezing & is_night,
            "shutters_open": is_shutters_open,
            "shutters_closed": is_shutters_closed,
            "sunny_shutters_open": is_sunny & is_shutters_open,
            "sunny_shutters_closed": is_sunny & is_shutters_closed,
            "heating_and_sunny": is_sunny & is_heating_active,
            "night": is_night,
            "occupied": occupied_flag >= 1.0,
            "sunny_midday": is_sunny
            & (local_hours >= config.sunny_midday_start_hour)
            & (local_hours < config.sunny_midday_end_hour),
        }

        return PreparedRoomArxData(
            timestamps_utc=timestamps_utc,
            feature_specs=specs,
            feature_names=feature_names,
            feature_matrix=feature_matrix,
            target_values=target_values,
            valid_training_mask=valid_training_mask,
            field_arrays={
                "room_temperature_c": room_temperature,
                "outdoor_temperature_c": outdoor_temperature,
                "thermal_output_estimate_kw": thermal_output,
                "solar_gain_proxy_w_m2": solar_gain,
                "shutter_position_pct": shutter_position,
                "solar_shutter_interaction": solar_shutter_interaction,
                "occupied_flag": occupied_flag,
            },
            segment_masks=segment_masks,
        )

    def segment_definitions(self, config: RoomArxConfig) -> list[tuple[str, str]]:
        return [
            ("sunny", f"solar irradiance >= {config.sunny_irradiance_threshold_w_m2:.0f} W/m2"),
            ("heating_active", f"thermal output >= {config.heating_active_threshold_kw:.2f} kW"),
            ("cold_weather", f"outdoor temperature <= {config.cold_outdoor_temperature_max_c:.1f} C"),
            (
                "freezing_weather",
                f"outdoor temperature <= {config.freezing_outdoor_temperature_max_c:.1f} C",
            ),
            ("mild_weather", f"outdoor temperature >= {config.mild_outdoor_temperature_min_c:.1f} C"),
            ("freezing_and_heating", "freezing weather with heating active"),
            ("freezing_night", "freezing weather during night hours"),
            ("shutters_open", f"shutter position >= {config.shutters_open_min_pct:.0f}%"),
            (
                "shutters_closed",
                f"shutter position <= {config.shutters_closed_max_pct:.0f}%",
            ),
            ("sunny_shutters_open", "sunny with shutters open"),
            ("sunny_shutters_closed", "sunny with shutters closed"),
            ("heating_and_sunny", "heating active with sunny conditions"),
            (
                "night",
                (
                    f"local hour in "
                    f"[{config.night_start_hour:02d}:00, 24:00) or [00:00, {config.night_end_hour:02d}:00)"
                ),
            ),
            ("occupied", "occupied_flag == 1"),
            (
                "sunny_midday",
                (
                    "sunny and local hour in "
                    f"[{config.sunny_midday_start_hour:02d}:00, {config.sunny_midday_end_hour:02d}:00)"
                ),
            ),
        ]

    def row_segments(self, row: MpcDatasetRow, config: RoomArxConfig) -> set[str]:
        segments: set[str] = set()
        solar_irradiance = row.solar_irradiance_w_m2 or 0.0
        thermal_output = row.thermal_output_estimate_kw or 0.0
        outdoor_temperature = row.outdoor_temperature_c
        shutter_position = row.shutter_position_pct
        local_hour = row.timestamp_utc.astimezone().hour
        occupied_flag = row.occupied_flag

        is_sunny = solar_irradiance >= config.sunny_irradiance_threshold_w_m2
        is_heating_active = thermal_output >= config.heating_active_threshold_kw
        is_shutters_open = (
            shutter_position is not None and shutter_position >= config.shutters_open_min_pct
        )
        is_shutters_closed = (
            shutter_position is not None and shutter_position <= config.shutters_closed_max_pct
        )
        is_night = local_hour >= config.night_start_hour or local_hour < config.night_end_hour
        is_freezing = (
            outdoor_temperature is not None
            and outdoor_temperature <= config.freezing_outdoor_temperature_max_c
        )

        if outdoor_temperature is not None and outdoor_temperature <= config.cold_outdoor_temperature_max_c:
            segments.add("cold_weather")
        if is_freezing:
            segments.add("freezing_weather")
        if outdoor_temperature is not None and outdoor_temperature >= config.mild_outdoor_temperature_min_c:
            segments.add("mild_weather")
        if is_sunny:
            segments.add("sunny")
            if config.sunny_midday_start_hour <= local_hour < config.sunny_midday_end_hour:
                segments.add("sunny_midday")
        if is_heating_active:
            segments.add("heating_active")
        if is_shutters_open:
            segments.add("shutters_open")
        if is_shutters_closed:
            segments.add("shutters_closed")
        if is_sunny and is_shutters_open:
            segments.add("sunny_shutters_open")
        if is_sunny and is_shutters_closed:
            segments.add("sunny_shutters_closed")
        if is_sunny and is_heating_active:
            segments.add("heating_and_sunny")
        if is_night:
            segments.add("night")
        if is_freezing and is_heating_active:
            segments.add("freezing_and_heating")
        if is_freezing and is_night:
            segments.add("freezing_night")
        if occupied_flag:
            segments.add("occupied")

        return segments

    def training_slice(
        self,
        prepared: PreparedRoomArxData,
        *,
        start_index: int = 0,
        end_exclusive: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        resolved_end_exclusive = (
            len(prepared.timestamps_utc) if end_exclusive is None else end_exclusive
        )
        mask = prepared.valid_training_mask.copy()
        mask[:start_index] = False
        mask[resolved_end_exclusive:] = False
        return prepared.feature_matrix[mask], prepared.target_values[mask]

    def solve_ridge_regression(
        self,
        x_matrix: np.ndarray,
        y_values: np.ndarray,
        *,
        ridge_alpha: float,
    ) -> tuple[float, list[float]]:
        if x_matrix.size == 0 or y_values.size == 0:
            raise ValueError("not enough valid rows to fit room model")

        design_matrix = np.column_stack([np.ones(len(x_matrix)), x_matrix])
        if ridge_alpha == 0.0:
            coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y_values, rcond=None)
            return float(coefficients[0]), [float(value) for value in coefficients[1:]]

        penalty = np.eye(design_matrix.shape[1]) * ridge_alpha
        penalty[0, 0] = 0.0

        coefficients = np.linalg.solve(
            design_matrix.T @ design_matrix + penalty,
            design_matrix.T @ y_values,
        )
        return float(coefficients[0]), [float(value) for value in coefficients[1:]]

    def fit(
        self,
        dataset: MpcDataset,
        config: RoomArxConfig,
    ) -> RoomArxModel:
        prepared = self.prepare(dataset.rows, config)
        x_matrix, y_values = self.training_slice(prepared)
        intercept, coefficients = self.solve_ridge_regression(
            x_matrix,
            y_values,
            ridge_alpha=config.ridge_alpha,
        )

        return RoomArxModel(
            trained_from_utc=dataset.start_time_utc,
            trained_to_utc=dataset.end_time_utc,
            interval_minutes=dataset.interval_minutes,
            config=config,
            feature_names=prepared.feature_names,
            intercept=intercept,
            coefficients=coefficients,
            sample_count=len(y_values),
        )

    def fit_prepared(
        self,
        prepared: PreparedRoomArxData,
        *,
        config: RoomArxConfig,
        interval_minutes: int,
        train_start_index: int = 0,
        train_end_exclusive: int | None = None,
    ) -> RoomArxModel:
        resolved_train_end_exclusive = (
            len(prepared.timestamps_utc) if train_end_exclusive is None else train_end_exclusive
        )
        x_matrix, y_values = self.training_slice(
            prepared,
            start_index=train_start_index,
            end_exclusive=resolved_train_end_exclusive,
        )
        intercept, coefficients = self.solve_ridge_regression(
            x_matrix,
            y_values,
            ridge_alpha=config.ridge_alpha,
        )
        trained_to_index = max(train_start_index, resolved_train_end_exclusive - 1)
        return RoomArxModel(
            trained_from_utc=prepared.timestamps_utc[train_start_index],
            trained_to_utc=prepared.timestamps_utc[trained_to_index],
            interval_minutes=interval_minutes,
            config=config,
            feature_names=prepared.feature_names,
            intercept=intercept,
            coefficients=coefficients,
            sample_count=len(y_values),
        )

    def predict_next_prepared(
        self,
        model: TrainedLinearRoomModel,
        prepared: PreparedRoomArxData,
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        if len(model.coefficients) != len(prepared.feature_specs):
            raise ValueError(
                "model coefficient count does not match the current ARX feature layout; "
                "retrain or reload a compatible model"
            )

        predicted_room_temperatures = predicted_room_temperatures or {}
        prediction_origin_index = (
            source_index if prediction_origin_index is None else prediction_origin_index
        )

        feature_values: list[float] = []
        for index, (field_name, _, lag) in enumerate(prepared.feature_specs):
            lagged_index = source_index - lag

            if field_name == "room_temperature_c" and lagged_index > prediction_origin_index:
                value = predicted_room_temperatures.get(lagged_index)
            else:
                value = float(prepared.field_arrays[field_name][lagged_index])
                if np.isnan(value):
                    value = None

            if value is None:
                value = self.default_feature_value(field_name)
            if value is None:
                return None

            feature_values.append(value)

        return model.intercept + sum(
            value * model.coefficients[index]
            for index, value in enumerate(feature_values)
        )

    def predict_next(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        predicted_room_temperatures = predicted_room_temperatures or {}
        prediction_origin_index = (
            source_index if prediction_origin_index is None else prediction_origin_index
        )
        prepared = self.prepare(rows, model.config)
        return self.predict_next_prepared(
            model,
            prepared,
            source_index=source_index,
            predicted_room_temperatures=predicted_room_temperatures,
            prediction_origin_index=prediction_origin_index,
        )

    def simulate_horizon_prepared(
        self,
        model: TrainedLinearRoomModel,
        prepared: PreparedRoomArxData,
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero")

        predictions: dict[int, float] = {}
        for step in range(1, horizon_steps + 1):
            source_index = start_index + step - 1
            target_index = start_index + step
            prediction = self.predict_next_prepared(
                model,
                prepared,
                source_index=source_index,
                predicted_room_temperatures=predictions,
                prediction_origin_index=start_index,
            )
            if prediction is None:
                return []
            predictions[target_index] = prediction

        return [predictions[start_index + step] for step in range(1, horizon_steps + 1)]

    def simulate_horizon(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        prepared = self.prepare(rows, model.config)
        return self.simulate_horizon_prepared(
            model,
            prepared,
            start_index=start_index,
            horizon_steps=horizon_steps,
        )
