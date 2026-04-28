from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from home_optimizer.features.system_identification.dataset import (
    IdentificationRow,
    SeriesLookup,
    numeric_series,
    points_by_timestamp,
    timed_values,
)
from home_optimizer.features.system_identification.metrics import regression_metrics
from home_optimizer.features.system_identification.schemas import (
    NumericSeries,
    RoomTemperatureModelResult,
    TextSeries,
    TrainTestMetrics,
)
from home_optimizer.features.system_identification.state_detection import StateMask

ROOM_TEMPERATURE_FEATURES = [
    "intercept",
    "room_temperature",
    "room_outdoor_delta",
    "thermal_output_lag_15m",
    "thermal_output_lag_30m",
    "thermal_output_lag_60m",
    "solar_gain_lag_15m",
    "solar_gain_lag_30m",
]


@dataclass(frozen=True)
class RoomTemperatureModelInputs:
    room_temperature: NumericSeries
    outdoor_temperature: NumericSeries
    thermal_output: NumericSeries
    solar_gain: NumericSeries | None = None
    defrost_active: NumericSeries | None = None
    booster_heater_active: NumericSeries | None = None
    hp_mode: TextSeries | None = None


class RoomTemperatureModelIdentifier:
    def __init__(
        self,
        *,
        sample_interval_minutes: int = 15,
        train_fraction: float = 0.7,
        max_input_age_minutes: int = 20,
        min_samples: int = 12,
    ) -> None:
        if sample_interval_minutes <= 0:
            raise ValueError("sample_interval_minutes must be positive")
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")

        self.sample_interval = timedelta(minutes=sample_interval_minutes)
        self.sample_interval_minutes = sample_interval_minutes
        self.train_fraction = train_fraction
        self.max_input_age = timedelta(minutes=max_input_age_minutes)
        self.min_samples = min_samples

    def identify(self, inputs: RoomTemperatureModelInputs) -> RoomTemperatureModelResult:
        rows = self._build_rows(inputs)
        if len(rows) < self.min_samples:
            raise ValueError("not enough valid samples to identify a room-temperature model")

        train_count = int(len(rows) * self.train_fraction)
        if train_count == 0 or train_count == len(rows):
            raise ValueError("train/test split produced an empty partition")

        x = np.array(
            [
                [row.features[feature_name] for feature_name in ROOM_TEMPERATURE_FEATURES]
                for row in rows
            ],
            dtype=float,
        )
        y = np.array([row.target for row in rows], dtype=float)

        train_x = x[:train_count]
        train_y = y[:train_count]
        test_y = y[train_count:]

        coefficients, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
        predictions = x @ coefficients
        residuals = y - predictions

        return RoomTemperatureModelResult(
            target_name=f"{inputs.room_temperature.name}_next",
            input_names=ROOM_TEMPERATURE_FEATURES,
            sample_interval_minutes=self.sample_interval_minutes,
            train_fraction=self.train_fraction,
            coefficients={
                feature_name: float(coefficients[index])
                for index, feature_name in enumerate(ROOM_TEMPERATURE_FEATURES)
            },
            metrics=TrainTestMetrics(
                train=regression_metrics(train_y, predictions[:train_count]),
                test=regression_metrics(test_y, predictions[train_count:]),
            ),
            actual_series=numeric_series(
                "room_temperature_actual",
                inputs.room_temperature.unit,
                [(row.timestamp, row.target) for row in rows],
            ),
            predicted_series=numeric_series(
                "room_temperature_predicted",
                inputs.room_temperature.unit,
                [
                    (row.timestamp, float(predictions[index]))
                    for index, row in enumerate(rows)
                ],
            ),
            residual_series=numeric_series(
                "room_temperature_residual",
                inputs.room_temperature.unit,
                [
                    (row.timestamp, float(residuals[index]))
                    for index, row in enumerate(rows)
                ],
            ),
        )

    def _build_rows(self, inputs: RoomTemperatureModelInputs) -> list[IdentificationRow]:
        room_points = timed_values(inputs.room_temperature)
        next_room_by_timestamp = points_by_timestamp(inputs.room_temperature)
        outdoor_lookup = SeriesLookup(timed_values(inputs.outdoor_temperature))
        thermal_lookup = SeriesLookup(timed_values(inputs.thermal_output))
        solar_lookup = SeriesLookup(timed_values(inputs.solar_gain)) if inputs.solar_gain else None
        state_mask = StateMask(
            defrost_active=inputs.defrost_active,
            booster_heater_active=inputs.booster_heater_active,
            hp_mode=inputs.hp_mode,
            max_state_age=self.max_input_age,
        )

        rows: list[IdentificationRow] = []
        for point in room_points:
            target_timestamp = point.timestamp + self.sample_interval
            target = next_room_by_timestamp.get(target_timestamp)
            if target is None or not state_mask.is_valid_room_model_state(point.timestamp):
                continue

            outdoor = outdoor_lookup.latest_at(point.timestamp, self.max_input_age)
            thermal_lag_15 = thermal_lookup.latest_at(
                point.timestamp - timedelta(minutes=15),
                self.max_input_age,
            )
            thermal_lag_30 = thermal_lookup.latest_at(
                point.timestamp - timedelta(minutes=30),
                self.max_input_age,
            )
            thermal_lag_60 = thermal_lookup.latest_at(
                point.timestamp - timedelta(minutes=60),
                self.max_input_age,
            )
            if (
                outdoor is None
                or thermal_lag_15 is None
                or thermal_lag_30 is None
                or thermal_lag_60 is None
            ):
                continue

            solar_lag_15 = self._lagged_optional_value(solar_lookup, point.timestamp, 15)
            solar_lag_30 = self._lagged_optional_value(solar_lookup, point.timestamp, 30)
            rows.append(
                IdentificationRow(
                    timestamp=target_timestamp,
                    target=target,
                    features={
                        "intercept": 1.0,
                        "room_temperature": point.value,
                        "room_outdoor_delta": point.value - outdoor,
                        "thermal_output_lag_15m": thermal_lag_15,
                        "thermal_output_lag_30m": thermal_lag_30,
                        "thermal_output_lag_60m": thermal_lag_60,
                        "solar_gain_lag_15m": solar_lag_15,
                        "solar_gain_lag_30m": solar_lag_30,
                    },
                )
            )

        return rows

    def _lagged_optional_value(
        self,
        lookup: SeriesLookup | None,
        timestamp: datetime,
        lag_minutes: int,
    ) -> float:
        if lookup is None:
            return 0.0
        value = lookup.latest_at(timestamp - timedelta(minutes=lag_minutes), self.max_input_age)
        return value if value is not None else 0.0
