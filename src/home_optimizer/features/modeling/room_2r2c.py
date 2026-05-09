from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydantic import Field, field_validator

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.models import TrainedLinearRoomModel, ValidationConfig
from home_optimizer.features.modeling.common import (
    PreparedRoomData,
    prepare_room_data,
    row_segments,
    segment_definitions,
)

ROOM_2R2C_MODEL_KIND = "room_2r2c"


class Room2R2CConfig(ValidationConfig):
    model_kind: str = ROOM_2R2C_MODEL_KIND
    mass_decay_candidates: list[float] = Field(
        default_factory=lambda: [0.90, 0.94, 0.97, 0.985, 0.992]
    )
    thermal_to_mass_candidates: list[float] = Field(
        default_factory=lambda: [0.0, 0.02, 0.05, 0.08, 0.12]
    )
    solar_to_mass_candidates: list[float] = Field(
        default_factory=lambda: [0.0, 0.0005, 0.001, 0.002]
    )
    ridge_alpha: float = Field(default=1e-3, ge=0.0)
    history_warmup_rows: int = Field(default=144, ge=1)
    candidate_scoring_horizons_steps: list[int] = Field(default_factory=lambda: [36, 72])
    candidate_scoring_stride_rows: int = Field(default=12, ge=1)
    candidate_scoring_max_origins: int = Field(default=64, ge=1)
    recursive_candidate_top_n: int = Field(default=8, ge=1)
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
        default="Two-state room / mass model with fitted mass persistence and linear room dynamics."
    )

    @field_validator(
        "mass_decay_candidates",
        "thermal_to_mass_candidates",
        "solar_to_mass_candidates",
        "candidate_scoring_horizons_steps",
    )
    @classmethod
    def _validate_candidates(cls, value: list[float]) -> list[float]:
        ordered = sorted(set(value))
        if not ordered:
            raise ValueError("candidate lists cannot be empty")
        return ordered


class Room2R2CModel(TrainedLinearRoomModel):
    model_kind: str = ROOM_2R2C_MODEL_KIND
    config: Room2R2CConfig
    mass_decay: float
    thermal_to_mass: float
    solar_to_mass: float
    notes: str = Field(
        default="Trained two-state room model with explicit room and effective thermal-mass states."
    )


@dataclass(frozen=True)
class PreparedRoom2R2CData:
    common: PreparedRoomData
    target_values: np.ndarray
    valid_training_mask: np.ndarray
    candidate_mass_states: dict[tuple[float, float, float], np.ndarray]
    thermal_output_energy_kwh: np.ndarray
    solar_effective_energy: np.ndarray

    @property
    def segment_masks(self) -> dict[str, np.ndarray]:
        return self.common.segment_masks

    @property
    def timestamps_utc(self) -> list:
        return self.common.timestamps_utc


@dataclass(frozen=True)
class CandidateFit:
    candidate: tuple[float, float, float]
    intercept: float
    coefficients: list[float]
    sample_count: int
    one_step_rmse: float


class Room2R2CTrainer:
    feature_names = [
        "room_temperature_c",
        "thermal_mass_state",
        "outdoor_temperature_c",
        "thermal_output_energy_kwh",
        "solar_effective_energy",
    ]

    def max_history_rows(self, config: Room2R2CConfig) -> int:
        return config.history_warmup_rows

    def validation_stride_rows(self, config: Room2R2CConfig, interval_minutes: int) -> int:
        if config.validation_stride_rows is not None:
            return config.validation_stride_rows
        return max(1, 60 // interval_minutes)

    def prepare(self, rows: list[MpcDatasetRow], config: Room2R2CConfig) -> PreparedRoom2R2CData:
        common = prepare_room_data(rows, config)
        room_temperature = common.field_arrays["room_temperature_c"]
        interval_hours = (
            (rows[1].timestamp_utc - rows[0].timestamp_utc).total_seconds() / 3600.0
            if len(rows) > 1
            else 0.0
        )
        if interval_hours <= 0.0:
            interval_hours = 10.0 / 60.0
        thermal_output_energy_kwh = (
            common.field_arrays["thermal_output_estimate_kw"] * interval_hours
        )
        solar_effective_energy = (
            common.field_arrays["solar_gain_proxy_w_m2"] * interval_hours
        )
        target_values = np.roll(room_temperature, -1)
        valid_training_mask = (
            ~np.isnan(room_temperature)
            & ~np.isnan(common.field_arrays["outdoor_temperature_c"])
            & ~np.isnan(target_values)
        )
        if len(valid_training_mask) > 0:
            valid_training_mask[-1] = False

        candidate_mass_states = {
            (mass_decay, thermal_to_mass, solar_to_mass): self._build_mass_state(
                room_temperature=room_temperature,
                thermal_energy=thermal_output_energy_kwh,
                solar_energy=solar_effective_energy,
                mass_decay=mass_decay,
                thermal_to_mass=thermal_to_mass,
                solar_to_mass=solar_to_mass,
            )
            for mass_decay in config.mass_decay_candidates
            for thermal_to_mass in config.thermal_to_mass_candidates
            for solar_to_mass in config.solar_to_mass_candidates
        }

        return PreparedRoom2R2CData(
            common=common,
            target_values=target_values,
            valid_training_mask=valid_training_mask,
            candidate_mass_states=candidate_mass_states,
            thermal_output_energy_kwh=thermal_output_energy_kwh,
            solar_effective_energy=solar_effective_energy,
        )

    def segment_definitions(self, config: Room2R2CConfig) -> list[tuple[str, str]]:
        return segment_definitions(config)

    def row_segments(self, row: MpcDatasetRow, config: Room2R2CConfig) -> set[str]:
        return row_segments(row, config)

    def _build_mass_state(
        self,
        *,
        room_temperature: np.ndarray,
        thermal_energy: np.ndarray,
        solar_energy: np.ndarray,
        mass_decay: float,
        thermal_to_mass: float,
        solar_to_mass: float,
    ) -> np.ndarray:
        mass_state = np.zeros(len(room_temperature), dtype=float)
        if len(room_temperature) == 0:
            return mass_state

        initial_room_temperature = room_temperature[0]
        mass_state[0] = 20.0 if np.isnan(initial_room_temperature) else float(initial_room_temperature)
        room_to_mass = 1.0 - mass_decay

        for index in range(len(room_temperature) - 1):
            current_room_temperature = mass_state[index]
            if not np.isnan(room_temperature[index]):
                current_room_temperature = float(room_temperature[index])
            mass_state[index + 1] = (
                mass_decay * mass_state[index]
                + room_to_mass * current_room_temperature
                + thermal_to_mass * thermal_energy[index]
                + solar_to_mass * solar_energy[index]
            )

        return mass_state

    def _design_matrix(
        self,
        prepared: PreparedRoom2R2CData,
        *,
        mass_state: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        common = prepared.common
        return np.column_stack(
            [
                common.field_arrays["room_temperature_c"][mask],
                mass_state[mask],
                common.field_arrays["outdoor_temperature_c"][mask],
                prepared.thermal_output_energy_kwh[mask],
                prepared.solar_effective_energy[mask],
            ]
        )

    def _training_mask(
        self,
        prepared: PreparedRoom2R2CData,
        *,
        start_index: int,
        end_exclusive: int | None,
    ) -> np.ndarray:
        resolved_end_exclusive = (
            len(prepared.common.timestamps_utc) if end_exclusive is None else end_exclusive
        )
        mask = prepared.valid_training_mask.copy()
        mask[:start_index] = False
        mask[resolved_end_exclusive:] = False
        return mask

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

    def _candidate_recursive_score(
        self,
        *,
        prepared: PreparedRoom2R2CData,
        intercept: float,
        coefficients: list[float],
        candidate: tuple[float, float, float],
        config: Room2R2CConfig,
        train_start_index: int,
        train_end_exclusive: int,
    ) -> float:
        horizons = [
            horizon
            for horizon in config.candidate_scoring_horizons_steps
            if horizon > 0 and train_start_index + horizon < train_end_exclusive
        ]
        if not horizons:
            return float("inf")

        max_horizon = max(horizons)
        origin_start = max(train_start_index, config.history_warmup_rows)
        origin_end = train_end_exclusive - max_horizon - 1
        if origin_end < origin_start:
            return float("inf")

        origins = list(range(origin_start, origin_end + 1, config.candidate_scoring_stride_rows))
        if len(origins) > config.candidate_scoring_max_origins:
            origins = origins[-config.candidate_scoring_max_origins :]
        if not origins:
            return float("inf")

        errors_by_horizon: dict[int, list[float]] = {horizon: [] for horizon in horizons}
        abs_bias_penalties: list[float] = []

        for origin_index in origins:
            simulated = self._simulate_candidate_horizon_prepared(
                prepared=prepared,
                intercept=intercept,
                coefficients=coefficients,
                candidate=candidate,
                start_index=origin_index,
                horizon_steps=max_horizon,
            )
            if len(simulated) != max_horizon:
                continue
            for horizon in horizons:
                actual_room_temperature = prepared.common.field_arrays["room_temperature_c"][
                    origin_index + horizon
                ]
                if np.isnan(actual_room_temperature):
                    continue
                error = simulated[horizon - 1] - float(actual_room_temperature)
                errors_by_horizon[horizon].append(error)

        if any(not errors_by_horizon[horizon] for horizon in horizons):
            return float("inf")

        score = 0.0
        for horizon in horizons:
            errors = np.asarray(errors_by_horizon[horizon], dtype=float)
            mae = float(np.mean(np.abs(errors)))
            bias = float(np.mean(errors))
            weight = 2.0 if horizon >= 36 else 1.0
            score += weight * mae
            abs_bias_penalties.append(weight * abs(bias))

        return score + sum(abs_bias_penalties)

    def fit(
        self,
        dataset: MpcDataset,
        config: Room2R2CConfig,
    ) -> Room2R2CModel:
        prepared = self.prepare(dataset.rows, config)
        return self.fit_prepared(
            prepared,
            config=config,
            interval_minutes=dataset.interval_minutes,
            train_start_index=0,
            train_end_exclusive=len(dataset.rows),
        )

    def fit_prepared(
        self,
        prepared: PreparedRoom2R2CData,
        *,
        config: Room2R2CConfig,
        interval_minutes: int,
        train_start_index: int = 0,
        train_end_exclusive: int | None = None,
    ) -> Room2R2CModel:
        mask = self._training_mask(
            prepared,
            start_index=train_start_index,
            end_exclusive=train_end_exclusive,
        )
        y_values = prepared.target_values[mask]
        if y_values.size == 0:
            raise ValueError("not enough valid rows to fit room model")

        best_candidate: tuple[float, float, float] | None = None
        best_intercept = 0.0
        best_coefficients: list[float] = []
        best_sample_count = len(y_values)
        best_score = float("inf")
        resolved_train_end_exclusive = (
            len(prepared.common.timestamps_utc) if train_end_exclusive is None else train_end_exclusive
        )

        candidate_fits: list[CandidateFit] = []
        for candidate, mass_state in prepared.candidate_mass_states.items():
            x_matrix = self._design_matrix(prepared, mass_state=mass_state, mask=mask)
            intercept, coefficients = self.solve_ridge_regression(
                x_matrix,
                y_values,
                ridge_alpha=config.ridge_alpha,
            )
            predictions = intercept + (x_matrix @ np.asarray(coefficients, dtype=float))
            one_step_rmse = float(np.sqrt(np.mean(np.square(predictions - y_values))))
            candidate_fits.append(
                CandidateFit(
                    candidate=candidate,
                    intercept=intercept,
                    coefficients=coefficients,
                    sample_count=len(y_values),
                    one_step_rmse=one_step_rmse,
                )
            )

        # Rank cheaply on one-step fit first, then spend recursive scoring budget
        # only on the most plausible candidates.
        candidate_fits.sort(key=lambda fit: fit.one_step_rmse)
        shortlisted_fits = candidate_fits[: config.recursive_candidate_top_n]

        for candidate_fit in shortlisted_fits:
            score = self._candidate_recursive_score(
                prepared=prepared,
                intercept=candidate_fit.intercept,
                coefficients=candidate_fit.coefficients,
                candidate=candidate_fit.candidate,
                config=config,
                train_start_index=train_start_index,
                train_end_exclusive=resolved_train_end_exclusive,
            )
            if not np.isfinite(score):
                score = candidate_fit.one_step_rmse
            if score < best_score:
                best_score = score
                best_candidate = candidate_fit.candidate
                best_intercept = candidate_fit.intercept
                best_coefficients = candidate_fit.coefficients
                best_sample_count = candidate_fit.sample_count

        if best_candidate is None and candidate_fits:
            fallback_fit = candidate_fits[0]
            best_candidate = fallback_fit.candidate
            best_intercept = fallback_fit.intercept
            best_coefficients = fallback_fit.coefficients
            best_sample_count = fallback_fit.sample_count

        if best_candidate is None:
            raise ValueError("not enough valid rows to fit room model")

        trained_to_index = max(train_start_index, resolved_train_end_exclusive - 1)
        mass_decay, thermal_to_mass, solar_to_mass = best_candidate
        return Room2R2CModel(
            trained_from_utc=prepared.common.timestamps_utc[train_start_index],
            trained_to_utc=prepared.common.timestamps_utc[trained_to_index],
            interval_minutes=interval_minutes,
            config=config,
            feature_names=list(self.feature_names),
            intercept=best_intercept,
            coefficients=best_coefficients,
            sample_count=best_sample_count,
            mass_decay=mass_decay,
            thermal_to_mass=thermal_to_mass,
            solar_to_mass=solar_to_mass,
        )

    def predict_next_prepared(
        self,
        model: Room2R2CModel,
        prepared: PreparedRoom2R2CData,
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        predicted_mass_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> tuple[float, float] | None:
        predicted_room_temperatures = predicted_room_temperatures or {}
        predicted_mass_temperatures = predicted_mass_temperatures or {}
        prediction_origin_index = (
            source_index if prediction_origin_index is None else prediction_origin_index
        )

        common = prepared.common
        room_state = predicted_room_temperatures.get(source_index)
        if room_state is None or source_index <= prediction_origin_index:
            room_state = common.field_arrays["room_temperature_c"][source_index]
        if np.isnan(room_state):
            return None

        mass_state = predicted_mass_temperatures.get(source_index)
        if mass_state is None or source_index <= prediction_origin_index:
            mass_state = prepared.candidate_mass_states[
                (model.mass_decay, model.thermal_to_mass, model.solar_to_mass)
            ][
                source_index
            ]

        outdoor_temperature = common.field_arrays["outdoor_temperature_c"][source_index]
        if np.isnan(outdoor_temperature):
            return None

        thermal_output = common.field_arrays["thermal_output_estimate_kw"][source_index]
        thermal_output_energy = prepared.thermal_output_energy_kwh[source_index]
        solar_effective_energy = prepared.solar_effective_energy[source_index]

        next_mass_state = (
            model.mass_decay * mass_state
            + (1.0 - model.mass_decay) * room_state
            + model.thermal_to_mass * thermal_output_energy
            + model.solar_to_mass * solar_effective_energy
        )
        features = np.asarray(
            [
                room_state,
                mass_state,
                outdoor_temperature,
                thermal_output_energy,
                solar_effective_energy,
            ],
            dtype=float,
        )
        next_room_state = float(model.intercept + (features @ np.asarray(model.coefficients)))
        return next_room_state, next_mass_state

    def _simulate_candidate_horizon_prepared(
        self,
        *,
        prepared: PreparedRoom2R2CData,
        intercept: float,
        coefficients: list[float],
        candidate: tuple[float, float, float],
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        predicted_room_temperatures: dict[int, float] = {}
        predicted_mass_states: dict[int, float] = {}
        mass_decay, thermal_to_mass, solar_to_mass = candidate
        predicted_mass_states[start_index] = prepared.candidate_mass_states[candidate][start_index]

        for step in range(1, horizon_steps + 1):
            source_index = start_index + step - 1
            target_index = start_index + step
            room_state = predicted_room_temperatures.get(source_index)
            if room_state is None or source_index <= start_index:
                room_state = prepared.common.field_arrays["room_temperature_c"][source_index]
            if np.isnan(room_state):
                return []
            mass_state = predicted_mass_states.get(source_index)
            if mass_state is None or source_index <= start_index:
                mass_state = prepared.candidate_mass_states[candidate][source_index]
            outdoor_temperature = prepared.common.field_arrays["outdoor_temperature_c"][source_index]
            if np.isnan(outdoor_temperature):
                return []
            thermal_output_energy = prepared.thermal_output_energy_kwh[source_index]
            solar_effective_energy = prepared.solar_effective_energy[source_index]

            next_mass_state = (
                mass_decay * mass_state
                + (1.0 - mass_decay) * room_state
                + thermal_to_mass * thermal_output_energy
                + solar_to_mass * solar_effective_energy
            )
            feature_values = np.asarray(
                [
                    room_state,
                    mass_state,
                    outdoor_temperature,
                    thermal_output_energy,
                    solar_effective_energy,
                ],
                dtype=float,
            )
            next_room_state = float(intercept + (feature_values @ np.asarray(coefficients)))
            predicted_room_temperatures[target_index] = next_room_state
            predicted_mass_states[target_index] = next_mass_state

        return [
            predicted_room_temperatures[start_index + step] for step in range(1, horizon_steps + 1)
        ]

    def predict_next(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        if not isinstance(model, Room2R2CModel):
            raise ValueError("room_2r2c trainer requires a Room2R2CModel")
        prepared = self.prepare(rows, model.config)
        prediction = self.predict_next_prepared(
            model,
            prepared,
            source_index=source_index,
            predicted_room_temperatures=predicted_room_temperatures,
            prediction_origin_index=prediction_origin_index,
        )
        if prediction is None:
            return None
        return prediction[0]

    def simulate_horizon_prepared(
        self,
        model: Room2R2CModel,
        prepared: PreparedRoom2R2CData,
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero")

        predicted_room_temperatures: dict[int, float] = {}
        predicted_mass_temperatures: dict[int, float] = {}
        initial_mass_state = prepared.candidate_mass_states[
            (model.mass_decay, model.thermal_to_mass, model.solar_to_mass)
        ][start_index]
        predicted_mass_temperatures[start_index] = initial_mass_state

        for step in range(1, horizon_steps + 1):
            source_index = start_index + step - 1
            target_index = start_index + step
            prediction = self.predict_next_prepared(
                model,
                prepared,
                source_index=source_index,
                predicted_room_temperatures=predicted_room_temperatures,
                predicted_mass_temperatures=predicted_mass_temperatures,
                prediction_origin_index=start_index,
            )
            if prediction is None:
                return []
            predicted_room_temperatures[target_index] = prediction[0]
            predicted_mass_temperatures[target_index] = prediction[1]

        return [
            predicted_room_temperatures[start_index + step] for step in range(1, horizon_steps + 1)
        ]

    def simulate_horizon(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        if not isinstance(model, Room2R2CModel):
            raise ValueError("room_2r2c trainer requires a Room2R2CModel")
        prepared = self.prepare(rows, model.config)
        return self.simulate_horizon_prepared(
            model,
            prepared,
            start_index=start_index,
            horizon_steps=horizon_steps,
        )
