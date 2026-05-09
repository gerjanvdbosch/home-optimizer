from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydantic import Field, field_validator

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.common import (
    PreparedRoomData,
    prepare_room_data,
    row_segments,
    segment_definitions,
)
from home_optimizer.features.modeling.models import TrainedLinearRoomModel, ValidationConfig

ROOM_2R2C_MODEL_KIND = "room_2r2c"


class Room2R2CConfig(ValidationConfig):
    model_kind: str = ROOM_2R2C_MODEL_KIND
    mass_decay_candidates: list[float] = Field(
        default_factory=lambda: [0.94, 0.97, 0.985]
    )
    thermal_to_mass_candidates: list[float] = Field(
        default_factory=lambda: [0.0, 0.05, 0.1]
    )
    solar_to_mass_candidates: list[float] = Field(
        default_factory=lambda: [0.0, 0.001]
    )
    observer_gain_candidates: list[float] = Field(
        default_factory=lambda: [0.0, 0.1]
    )
    state_refinement_passes: int = Field(default=1, ge=1, le=6)
    ridge_alpha: float = Field(default=1e-3, ge=0.0)
    history_warmup_rows: int = Field(default=144, ge=1)
    candidate_scoring_horizons_steps: list[int] = Field(default_factory=lambda: [36])
    candidate_scoring_stride_rows: int = Field(default=24, ge=1)
    candidate_scoring_max_origins: int = Field(default=16, ge=1)
    structural_candidate_top_n: int = Field(default=8, ge=1)
    recursive_candidate_top_n: int = Field(default=8, ge=1)
    max_stability_radius: float = Field(default=0.999, gt=0.0, lt=1.0)
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
        default=(
            "Grey-box two-state room model with latent thermal mass state, "
            "observer correction, and recursive candidate selection."
        )
    )

    @field_validator(
        "mass_decay_candidates",
        "thermal_to_mass_candidates",
        "solar_to_mass_candidates",
        "observer_gain_candidates",
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
    observer_gain: float
    notes: str = Field(
        default=(
            "Trained grey-box two-state room model with latent thermal-mass "
            "state estimated through observer correction."
        )
    )


@dataclass(frozen=True)
class PreparedRoom2R2CData:
    common: PreparedRoomData
    target_values: np.ndarray
    valid_training_mask: np.ndarray
    thermal_output_energy_kwh: np.ndarray
    solar_effective_exposure: np.ndarray
    estimated_mass_state_cache: dict[tuple, np.ndarray]

    @property
    def segment_masks(self) -> dict[str, np.ndarray]:
        return self.common.segment_masks

    @property
    def timestamps_utc(self) -> list:
        return self.common.timestamps_utc


@dataclass(frozen=True)
class CandidateFit:
    candidate: tuple[float, float, float, float]
    intercept: float
    coefficients: list[float]
    sample_count: int
    one_step_rmse: float
    mass_state: np.ndarray


@dataclass(frozen=True)
class StructuralCandidateFit:
    candidate: tuple[float, float, float]
    one_step_rmse: float


class Room2R2CTrainer:
    feature_names = [
        "room_temperature_c",
        "thermal_mass_state",
        "outdoor_temperature_c",
        "thermal_output_energy_kwh",
        "solar_effective_exposure",
    ]

    def max_history_rows(self, config: Room2R2CConfig) -> int:
        return config.history_warmup_rows

    def validation_stride_rows(self, config: Room2R2CConfig, interval_minutes: int) -> int:
        if config.validation_stride_rows is not None:
            return config.validation_stride_rows
        # Grey-box validation is materially more expensive than ARX, so sample
        # validation origins more sparsely by default.
        return max(1, 180 // interval_minutes)

    def prepare(self, rows: list[MpcDatasetRow], config: Room2R2CConfig) -> PreparedRoom2R2CData:
        common = prepare_room_data(rows, config)
        room_temperature = common.field_arrays["room_temperature_c"]
        interval_hours = (
            (rows[1].timestamp_utc - rows[0].timestamp_utc).total_seconds() / 3600.0
            if len(rows) > 1
            else 10.0 / 60.0
        )
        if interval_hours <= 0.0:
            interval_hours = 10.0 / 60.0

        thermal_output_energy_kwh = (
            common.field_arrays["thermal_output_estimate_kw"] * interval_hours
        )
        solar_effective_exposure = common.field_arrays["solar_gain_proxy_w_m2"] * interval_hours
        target_values = np.roll(room_temperature, -1)
        valid_training_mask = (
            ~np.isnan(room_temperature)
            & ~np.isnan(common.field_arrays["outdoor_temperature_c"])
            & ~np.isnan(target_values)
        )
        if len(valid_training_mask) > 0:
            valid_training_mask[-1] = False

        return PreparedRoom2R2CData(
            common=common,
            target_values=target_values,
            valid_training_mask=valid_training_mask,
            thermal_output_energy_kwh=thermal_output_energy_kwh,
            solar_effective_exposure=solar_effective_exposure,
            estimated_mass_state_cache={},
        )

    def segment_definitions(self, config: Room2R2CConfig) -> list[tuple[str, str]]:
        return segment_definitions(config)

    def row_segments(self, row: MpcDatasetRow, config: Room2R2CConfig) -> set[str]:
        return row_segments(row, config)

    def _initial_mass_state(self, room_temperature: np.ndarray) -> float:
        if len(room_temperature) == 0:
            return 20.0
        initial_room_temperature = room_temperature[0]
        return 20.0 if np.isnan(initial_room_temperature) else float(initial_room_temperature)

    def _build_open_loop_mass_state(
        self,
        *,
        room_temperature: np.ndarray,
        thermal_energy: np.ndarray,
        solar_energy: np.ndarray,
        mass_decay: float,
        thermal_to_mass: float,
        solar_to_mass: float,
        end_exclusive: int | None = None,
    ) -> np.ndarray:
        resolved_end_exclusive = len(room_temperature) if end_exclusive is None else end_exclusive
        mass_state = np.zeros(len(room_temperature), dtype=float)
        if resolved_end_exclusive <= 0:
            return mass_state

        mass_state[0] = self._initial_mass_state(room_temperature)
        for index in range(min(len(room_temperature) - 1, resolved_end_exclusive - 1)):
            current_room_temperature = (
                mass_state[index]
                if np.isnan(room_temperature[index])
                else float(room_temperature[index])
            )
            mass_state[index + 1] = (
                mass_decay * mass_state[index]
                + (1.0 - mass_decay) * current_room_temperature
                + thermal_to_mass * thermal_energy[index]
                + solar_to_mass * solar_energy[index]
            )
        return mass_state

    def _estimate_mass_state_series(
        self,
        prepared: PreparedRoom2R2CData,
        *,
        intercept: float,
        coefficients: list[float],
        candidate: tuple[float, float, float, float],
        end_exclusive: int | None = None,
    ) -> np.ndarray:
        resolved_end_exclusive = (
            len(prepared.common.timestamps_utc) if end_exclusive is None else end_exclusive
        )
        room_temperature = prepared.common.field_arrays["room_temperature_c"]
        outdoor_temperature = prepared.common.field_arrays["outdoor_temperature_c"]
        mass_decay, thermal_to_mass, solar_to_mass, observer_gain = candidate
        coefficient_vector = np.asarray(coefficients, dtype=float)

        mass_state = np.zeros(len(room_temperature), dtype=float)
        if resolved_end_exclusive <= 0:
            return mass_state

        mass_state[0] = self._initial_mass_state(room_temperature)
        room_state = mass_state[0]

        for index in range(min(len(room_temperature) - 1, resolved_end_exclusive - 1)):
            measured_room = room_temperature[index]
            if not np.isnan(measured_room):
                room_state = float(measured_room)

            outdoor_state = outdoor_temperature[index]
            if np.isnan(outdoor_state):
                outdoor_state = room_state if not np.isnan(room_state) else mass_state[index]

            features = np.asarray(
                [
                    room_state,
                    mass_state[index],
                    outdoor_state,
                    prepared.thermal_output_energy_kwh[index],
                    prepared.solar_effective_exposure[index],
                ],
                dtype=float,
            )
            predicted_room_next = float(intercept + (features @ coefficient_vector))
            predicted_mass_next = (
                mass_decay * mass_state[index]
                + (1.0 - mass_decay) * room_state
                + thermal_to_mass * prepared.thermal_output_energy_kwh[index]
                + solar_to_mass * prepared.solar_effective_exposure[index]
            )

            measured_room_next = room_temperature[index + 1]
            if not np.isnan(measured_room_next):
                innovation = float(measured_room_next) - predicted_room_next
                mass_state[index + 1] = predicted_mass_next + (observer_gain * innovation)
                room_state = float(measured_room_next)
            else:
                mass_state[index + 1] = predicted_mass_next
                room_state = predicted_room_next

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
                prepared.solar_effective_exposure[mask],
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

    def _is_physically_plausible_candidate(
        self,
        *,
        coefficients: list[float],
        mass_decay: float,
        max_stability_radius: float,
    ) -> bool:
        if len(coefficients) < 5:
            return False

        mass_gain = coefficients[1]
        thermal_room_gain = coefficients[3]
        solar_room_gain = coefficients[4]
        if mass_gain < 0.0:
            return False
        if thermal_room_gain < -1e-6 or solar_room_gain < -1e-6:
            return False

        a_matrix = np.asarray(
            [
                [coefficients[0], mass_gain],
                [1.0 - mass_decay, mass_decay],
            ],
            dtype=float,
        )
        eigvals = np.linalg.eigvals(a_matrix)
        return bool(np.all(np.abs(eigvals) < max_stability_radius))

    def _fit_candidate(
        self,
        prepared: PreparedRoom2R2CData,
        *,
        candidate: tuple[float, float, float, float],
        config: Room2R2CConfig,
        mask: np.ndarray,
        train_end_exclusive: int,
        enforce_physical_checks: bool = True,
    ) -> CandidateFit:
        y_values = prepared.target_values[mask]
        mass_decay, thermal_to_mass, solar_to_mass, _ = candidate
        mass_state = self._build_open_loop_mass_state(
            room_temperature=prepared.common.field_arrays["room_temperature_c"],
            thermal_energy=prepared.thermal_output_energy_kwh,
            solar_energy=prepared.solar_effective_exposure,
            mass_decay=mass_decay,
            thermal_to_mass=thermal_to_mass,
            solar_to_mass=solar_to_mass,
            end_exclusive=train_end_exclusive,
        )
        intercept = 0.0
        coefficients: list[float] = [0.0] * len(self.feature_names)

        for _ in range(config.state_refinement_passes):
            x_matrix = self._design_matrix(prepared, mass_state=mass_state, mask=mask)
            intercept, coefficients = self.solve_ridge_regression(
                x_matrix,
                y_values,
                ridge_alpha=config.ridge_alpha,
            )
            mass_state = self._estimate_mass_state_series(
                prepared,
                intercept=intercept,
                coefficients=coefficients,
                candidate=candidate,
                end_exclusive=train_end_exclusive,
            )

        if enforce_physical_checks and not self._is_physically_plausible_candidate(
            coefficients=coefficients,
            mass_decay=mass_decay,
            max_stability_radius=config.max_stability_radius,
        ):
            return CandidateFit(
                candidate=candidate,
                intercept=intercept,
                coefficients=coefficients,
                sample_count=len(y_values),
                one_step_rmse=float("inf"),
                mass_state=mass_state,
            )

        x_matrix = self._design_matrix(prepared, mass_state=mass_state, mask=mask)
        predictions = intercept + (x_matrix @ np.asarray(coefficients, dtype=float))
        one_step_rmse = float(np.sqrt(np.mean(np.square(predictions - y_values))))
        return CandidateFit(
            candidate=candidate,
            intercept=intercept,
            coefficients=coefficients,
            sample_count=len(y_values),
            one_step_rmse=one_step_rmse,
            mass_state=mass_state,
        )

    def _score_structural_candidate(
        self,
        prepared: PreparedRoom2R2CData,
        *,
        structural_candidate: tuple[float, float, float],
        config: Room2R2CConfig,
        mask: np.ndarray,
        train_end_exclusive: int,
    ) -> StructuralCandidateFit:
        candidate_fit = self._fit_candidate(
            prepared,
            candidate=(*structural_candidate, 0.0),
            config=config,
            mask=mask,
            train_end_exclusive=train_end_exclusive,
            enforce_physical_checks=False,
        )
        return StructuralCandidateFit(
            candidate=structural_candidate,
            one_step_rmse=candidate_fit.one_step_rmse,
        )

    def _candidate_recursive_score(
        self,
        *,
        prepared: PreparedRoom2R2CData,
        candidate_fit: CandidateFit,
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
                candidate_fit=candidate_fit,
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
        train_start_index = 0
        if config.training_window_rows is not None:
            train_start_index = max(0, len(dataset.rows) - config.training_window_rows)
        return self.fit_prepared(
            prepared,
            config=config,
            interval_minutes=dataset.interval_minutes,
            train_start_index=train_start_index,
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

        resolved_train_end_exclusive = (
            len(prepared.common.timestamps_utc) if train_end_exclusive is None else train_end_exclusive
        )

        structural_fits: list[StructuralCandidateFit] = []
        for mass_decay in config.mass_decay_candidates:
            for thermal_to_mass in config.thermal_to_mass_candidates:
                for solar_to_mass in config.solar_to_mass_candidates:
                    structural_fits.append(
                        self._score_structural_candidate(
                            prepared,
                            structural_candidate=(
                                mass_decay,
                                thermal_to_mass,
                                solar_to_mass,
                            ),
                            config=config,
                            mask=mask,
                            train_end_exclusive=resolved_train_end_exclusive,
                        )
                    )

        structural_fits = [
            fit for fit in structural_fits if np.isfinite(fit.one_step_rmse)
        ]
        if not structural_fits:
            raise ValueError("no physically plausible room_2r2c structural candidates found")

        structural_fits.sort(key=lambda fit: fit.one_step_rmse)
        shortlisted_structural_fits = structural_fits[: config.structural_candidate_top_n]

        candidate_fits: list[CandidateFit] = []
        for structural_fit in shortlisted_structural_fits:
            mass_decay, thermal_to_mass, solar_to_mass = structural_fit.candidate
            for observer_gain in config.observer_gain_candidates:
                candidate_fits.append(
                    self._fit_candidate(
                        prepared,
                        candidate=(
                            mass_decay,
                            thermal_to_mass,
                            solar_to_mass,
                            observer_gain,
                        ),
                        config=config,
                        mask=mask,
                        train_end_exclusive=resolved_train_end_exclusive,
                    )
                )

        all_candidate_fits = candidate_fits
        candidate_fits = [fit for fit in candidate_fits if np.isfinite(fit.one_step_rmse)]
        if not candidate_fits:
            candidate_fits = sorted(
                all_candidate_fits,
                key=lambda fit: fit.one_step_rmse,
            )[: config.recursive_candidate_top_n]
            if not candidate_fits:
                raise ValueError("no room_2r2c candidates found")

        candidate_fits.sort(key=lambda fit: fit.one_step_rmse)
        shortlisted_fits = candidate_fits[: config.recursive_candidate_top_n]

        best_fit: CandidateFit | None = None
        best_score = float("inf")
        for candidate_fit in shortlisted_fits:
            score = self._candidate_recursive_score(
                prepared=prepared,
                candidate_fit=candidate_fit,
                config=config,
                train_start_index=train_start_index,
                train_end_exclusive=resolved_train_end_exclusive,
            )
            if not np.isfinite(score):
                score = candidate_fit.one_step_rmse
            if score < best_score:
                best_score = score
                best_fit = candidate_fit

        if best_fit is None:
            best_fit = candidate_fits[0]
        if best_fit is None:
            raise ValueError("not enough valid rows to fit room model")

        trained_to_index = max(train_start_index, resolved_train_end_exclusive - 1)
        mass_decay, thermal_to_mass, solar_to_mass, observer_gain = best_fit.candidate
        return Room2R2CModel(
            trained_from_utc=prepared.common.timestamps_utc[train_start_index],
            trained_to_utc=prepared.common.timestamps_utc[trained_to_index],
            interval_minutes=interval_minutes,
            config=config,
            feature_names=list(self.feature_names),
            intercept=best_fit.intercept,
            coefficients=best_fit.coefficients,
            sample_count=best_fit.sample_count,
            mass_decay=mass_decay,
            thermal_to_mass=thermal_to_mass,
            solar_to_mass=solar_to_mass,
            observer_gain=observer_gain,
        )

    def _model_cache_key(self, model: Room2R2CModel) -> tuple:
        return (
            model.trained_from_utc,
            model.trained_to_utc,
            round(model.intercept, 10),
            tuple(round(value, 10) for value in model.coefficients),
            round(model.mass_decay, 10),
            round(model.thermal_to_mass, 10),
            round(model.solar_to_mass, 10),
            round(model.observer_gain, 10),
        )

    def _estimated_mass_state_series_for_model(
        self,
        prepared: PreparedRoom2R2CData,
        model: Room2R2CModel,
    ) -> np.ndarray:
        cache_key = self._model_cache_key(model)
        cached = prepared.estimated_mass_state_cache.get(cache_key)
        if cached is not None:
            return cached
        estimated = self._estimate_mass_state_series(
            prepared,
            intercept=model.intercept,
            coefficients=model.coefficients,
            candidate=(
                model.mass_decay,
                model.thermal_to_mass,
                model.solar_to_mass,
                model.observer_gain,
            ),
        )
        prepared.estimated_mass_state_cache[cache_key] = estimated
        return estimated

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
            mass_state = self._estimated_mass_state_series_for_model(prepared, model)[source_index]

        outdoor_temperature = common.field_arrays["outdoor_temperature_c"][source_index]
        if np.isnan(outdoor_temperature):
            return None

        thermal_output_energy = prepared.thermal_output_energy_kwh[source_index]
        solar_effective_exposure = prepared.solar_effective_exposure[source_index]

        next_mass_state = (
            model.mass_decay * mass_state
            + (1.0 - model.mass_decay) * float(room_state)
            + model.thermal_to_mass * thermal_output_energy
            + model.solar_to_mass * solar_effective_exposure
        )
        features = np.asarray(
            [
                float(room_state),
                mass_state,
                outdoor_temperature,
                thermal_output_energy,
                solar_effective_exposure,
            ],
            dtype=float,
        )
        next_room_state = float(model.intercept + (features @ np.asarray(model.coefficients)))
        return next_room_state, next_mass_state

    def _simulate_candidate_horizon_prepared(
        self,
        *,
        prepared: PreparedRoom2R2CData,
        candidate_fit: CandidateFit,
        start_index: int,
        horizon_steps: int,
    ) -> list[float]:
        predicted_room_temperatures: dict[int, float] = {}
        predicted_mass_states: dict[int, float] = {start_index: candidate_fit.mass_state[start_index]}
        mass_decay, thermal_to_mass, solar_to_mass, _ = candidate_fit.candidate
        coefficient_vector = np.asarray(candidate_fit.coefficients, dtype=float)

        for step in range(1, horizon_steps + 1):
            source_index = start_index + step - 1
            target_index = start_index + step
            room_state = predicted_room_temperatures.get(source_index)
            if room_state is None or source_index <= start_index:
                room_state = prepared.common.field_arrays["room_temperature_c"][source_index]
            if np.isnan(room_state):
                return []

            mass_state = predicted_mass_states[source_index]
            outdoor_temperature = prepared.common.field_arrays["outdoor_temperature_c"][source_index]
            if np.isnan(outdoor_temperature):
                return []

            thermal_output_energy = prepared.thermal_output_energy_kwh[source_index]
            solar_effective_exposure = prepared.solar_effective_exposure[source_index]
            next_mass_state = (
                mass_decay * mass_state
                + (1.0 - mass_decay) * float(room_state)
                + thermal_to_mass * thermal_output_energy
                + solar_to_mass * solar_effective_exposure
            )
            feature_values = np.asarray(
                [
                    float(room_state),
                    mass_state,
                    outdoor_temperature,
                    thermal_output_energy,
                    solar_effective_exposure,
                ],
                dtype=float,
            )
            next_room_state = float(candidate_fit.intercept + (feature_values @ coefficient_vector))
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
        predicted_mass_temperatures: dict[int, float] = {
            start_index: self._estimated_mass_state_series_for_model(prepared, model)[start_index]
        }

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
