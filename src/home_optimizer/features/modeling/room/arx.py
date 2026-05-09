from __future__ import annotations

import numpy as np

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
    notes: str = Field(
        default="Trained ARX room model with linear coefficients over lagged room and exogenous features."
    )


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

    def segment_definitions(self, config: RoomArxConfig) -> list[tuple[str, str]]:
        return [
            ("sunny", f"solar irradiance >= {config.sunny_irradiance_threshold_w_m2:.0f} W/m2"),
            ("heating_active", f"thermal output >= {config.heating_active_threshold_kw:.2f} kW"),
            ("shutters_open", f"shutter position >= {config.shutters_open_min_pct:.0f}%"),
            (
                "shutters_closed",
                f"shutter position <= {config.shutters_closed_max_pct:.0f}%",
            ),
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
        shutter_position = row.shutter_position_pct
        local_hour = row.timestamp_utc.astimezone().hour

        if solar_irradiance >= config.sunny_irradiance_threshold_w_m2:
            segments.add("sunny")
            if config.sunny_midday_start_hour <= local_hour < config.sunny_midday_end_hour:
                segments.add("sunny_midday")
        if thermal_output >= config.heating_active_threshold_kw:
            segments.add("heating_active")
        if shutter_position is not None and shutter_position >= config.shutters_open_min_pct:
            segments.add("shutters_open")
        if shutter_position is not None and shutter_position <= config.shutters_closed_max_pct:
            segments.add("shutters_closed")

        return segments

    def build_training_matrix(
        self,
        rows: list[MpcDatasetRow],
        config: RoomArxConfig,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        max_lag = self.max_lag(config)
        specs = self.feature_specs(config)
        feature_names = [feature_name for _, feature_name, _ in specs]

        x_rows: list[list[float]] = []
        y_values: list[float] = []

        for source_index in range(max_lag, len(rows) - 1):
            next_room_temperature = rows[source_index + 1].room_temperature_c
            if next_room_temperature is None:
                continue

            feature_values: list[float] = []
            valid = True
            for field_name, _, lag in specs:
                lagged_index = source_index - lag
                value = self.row_value(rows[lagged_index], field_name)
                if value is None:
                    value = self.default_feature_value(field_name)
                if value is None:
                    valid = False
                    break
                feature_values.append(value)

            if not valid:
                continue

            x_rows.append(feature_values)
            y_values.append(float(next_room_temperature))

        if not x_rows:
            return np.zeros((0, len(feature_names))), np.zeros((0,)), feature_names

        return np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float), feature_names

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
        x_matrix, y_values, feature_names = self.build_training_matrix(dataset.rows, config)
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
            feature_names=feature_names,
            intercept=intercept,
            coefficients=coefficients,
            sample_count=len(y_values),
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

        feature_values: list[float] = []
        for field_name, _, lag in self.feature_specs(model.config):
            lagged_index = source_index - lag

            if field_name == "room_temperature_c" and lagged_index > prediction_origin_index:
                value = predicted_room_temperatures.get(lagged_index)
            else:
                value = self.row_value(rows[lagged_index], field_name)

            if value is None:
                value = self.default_feature_value(field_name)
            if value is None:
                return None

            feature_values.append(value)

        return model.intercept + sum(
            value * model.coefficients[index]
            for index, value in enumerate(feature_values)
        )

    def simulate_horizon(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
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
            prediction = self.predict_next(
                model,
                rows,
                source_index=source_index,
                predicted_room_temperatures=predictions,
                prediction_origin_index=start_index,
            )
            if prediction is None:
                return []
            predictions[target_index] = prediction

        return [predictions[start_index + step] for step in range(1, horizon_steps + 1)]


ROOM_ARX_TRAINER = RoomArxTrainer()
