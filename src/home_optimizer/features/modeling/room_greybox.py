from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import least_squares

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling.common import (
    PreparedRoomData,
    prepare_room_data,
    row_segments,
    room_identification_mask,
    segment_definitions,
)
from home_optimizer.features.modeling.models import TrainedLinearRoomModel, ValidationConfig

ROOM_GREYBOX_MODEL_KIND = "room_greybox"

PARAM_K_OUT = 0
PARAM_K_MASS = 1
PARAM_G_HEAT_AIR = 2
PARAM_G_SOLAR_AIR = 3
PARAM_K_AIR_MASS = 4
PARAM_G_HEAT_MASS = 5
PARAM_G_SOLAR_MASS = 6
PARAM_OBSERVER_GAIN = 7
PARAMETER_NAMES = (
    "k_out",
    "k_mass",
    "g_heat_air",
    "g_solar_air",
    "k_air_mass",
    "g_heat_mass",
    "g_solar_mass",
    "observer_gain",
)


class RoomGreyBoxConfig(ValidationConfig):
    model_kind: str = ROOM_GREYBOX_MODEL_KIND
    history_warmup_rows: int = Field(default=144, ge=1)
    optimization_max_nfev: int = Field(default=80, ge=10, le=1000)
    optimization_loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = Field(
        default="soft_l1"
    )
    training_window_rows: int | None = Field(default=None, gt=1)
    validation_stride_rows: int | None = Field(default=None, gt=0)
    validation_window_rows: int = Field(default=144, gt=1)
    validation_horizons_steps: list[int] = Field(default_factory=lambda: [1, 6, 36, 72])
    max_stability_radius: float = Field(default=0.999, gt=0.0, lt=1.0)
    observer_gain_initial: float = Field(default=0.1, ge=0.0, le=0.5)
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
            "Bound-constrained grey-box room model with delta-form air/mass state "
            "updates and observer-corrected latent mass state."
        )
    )


class RoomGreyBoxModel(TrainedLinearRoomModel):
    model_kind: str = ROOM_GREYBOX_MODEL_KIND
    config: RoomGreyBoxConfig
    k_out: float
    k_mass: float
    k_air_mass: float
    g_heat_mass: float
    g_solar_mass: float
    observer_gain: float
    notes: str = Field(
        default=(
            "Trained grey-box room state-space model with delta-form thermal "
            "couplings and observer-estimated latent mass state."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_fields(cls, value):
        if not isinstance(value, dict):
            return value

        coefficients = list(value.get("coefficients") or [])
        updates: dict[str, float] = {}
        if "k_out" not in value and len(coefficients) > 2:
            updates["k_out"] = float(coefficients[2])
        if "k_mass" not in value and len(coefficients) > 1:
            updates["k_mass"] = float(coefficients[1])
        if "k_air_mass" not in value:
            if value.get("a21") is not None:
                updates["k_air_mass"] = float(value["a21"])
            elif value.get("a22") is not None:
                updates["k_air_mass"] = max(0.0, 1.0 - float(value["a22"]))
        if "g_heat_mass" not in value and value.get("heat_to_mass") is not None:
            updates["g_heat_mass"] = float(value["heat_to_mass"])
        if "g_solar_mass" not in value and value.get("solar_to_mass") is not None:
            updates["g_solar_mass"] = float(value["solar_to_mass"])
        return {**value, **updates}

    @property
    def a11(self) -> float:
        return float(self.coefficients[0])

    @property
    def a12(self) -> float:
        return float(self.coefficients[1])

    @property
    def b_out(self) -> float:
        return float(self.coefficients[2])

    @property
    def g_heat_air(self) -> float:
        return float(self.coefficients[3])

    @property
    def g_solar_air(self) -> float:
        return float(self.coefficients[4])

    @property
    def a21(self) -> float:
        return float(self.k_air_mass)

    @property
    def a22(self) -> float:
        return float(1.0 - self.k_air_mass)

    @property
    def heat_to_mass(self) -> float:
        return float(self.g_heat_mass)

    @property
    def solar_to_mass(self) -> float:
        return float(self.g_solar_mass)


@dataclass(frozen=True)
class PreparedRoomGreyBoxData:
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


class RoomGreyBoxTrainer:
    feature_names = [
        "room_temperature_c",
        "thermal_mass_state",
        "outdoor_temperature_c",
        "thermal_output_energy_kwh",
        "solar_effective_exposure",
    ]

    def max_history_rows(self, config: RoomGreyBoxConfig) -> int:
        return config.history_warmup_rows

    def validation_stride_rows(self, config: RoomGreyBoxConfig, interval_minutes: int) -> int:
        if config.validation_stride_rows is not None:
            return config.validation_stride_rows
        return max(1, 180 // interval_minutes)

    def prepare(self, rows: list[MpcDatasetRow], config: RoomGreyBoxConfig) -> PreparedRoomGreyBoxData:
        common = prepare_room_data(rows, config)
        room_temperature = common.field_arrays["room_temperature_c"]
        valid_identification_rows = room_identification_mask(rows)
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
        thermal_output_energy_kwh = np.nan_to_num(
            thermal_output_energy_kwh,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        solar_effective_exposure = np.nan_to_num(
            solar_effective_exposure,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        target_values = np.roll(room_temperature, -1)
        valid_training_mask = (
            valid_identification_rows
            & np.isfinite(room_temperature)
            & np.isfinite(common.field_arrays["outdoor_temperature_c"])
            & np.isfinite(target_values)
        )
        if len(valid_training_mask) > 0:
            valid_training_mask[-1] = False

        return PreparedRoomGreyBoxData(
            common=common,
            target_values=target_values,
            valid_training_mask=valid_training_mask,
            thermal_output_energy_kwh=thermal_output_energy_kwh,
            solar_effective_exposure=solar_effective_exposure,
            estimated_mass_state_cache={},
        )

    def segment_definitions(self, config: RoomGreyBoxConfig) -> list[tuple[str, str]]:
        return segment_definitions(config)

    def row_segments(self, row: MpcDatasetRow, config: RoomGreyBoxConfig) -> set[str]:
        return row_segments(row, config)

    def _training_mask(
        self,
        prepared: PreparedRoomGreyBoxData,
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
        if resolved_end_exclusive - 1 >= start_index:
            mask[resolved_end_exclusive - 1] = False
        return mask

    def _initial_air_state(self, room_temperature: np.ndarray, start_index: int) -> float:
        value = room_temperature[start_index]
        return 20.0 if not np.isfinite(value) else float(value)

    def _air_coefficients_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        k_out = float(parameters[PARAM_K_OUT])
        k_mass = float(parameters[PARAM_K_MASS])
        g_heat_air = float(parameters[PARAM_G_HEAT_AIR])
        g_solar_air = float(parameters[PARAM_G_SOLAR_AIR])
        return np.asarray(
            [1.0 - k_out - k_mass, k_mass, k_out, g_heat_air, g_solar_air],
            dtype=float,
        )

    def _mass_parameters_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        k_air_mass = float(parameters[PARAM_K_AIR_MASS])
        g_heat_mass = float(parameters[PARAM_G_HEAT_MASS])
        g_solar_mass = float(parameters[PARAM_G_SOLAR_MASS])
        return np.asarray([k_air_mass, g_heat_mass, g_solar_mass], dtype=float)

    def _observer_gain_from_parameters(self, parameters: np.ndarray) -> float:
        return float(parameters[PARAM_OBSERVER_GAIN])

    def _state_matrix_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        k_out = float(parameters[PARAM_K_OUT])
        k_mass = float(parameters[PARAM_K_MASS])
        k_air_mass = float(parameters[PARAM_K_AIR_MASS])
        return np.asarray(
            [
                [1.0 - k_out - k_mass, k_mass],
                [k_air_mass, 1.0 - k_air_mass],
            ],
            dtype=float,
        )

    def _optimizer_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bounds = np.asarray([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        upper_bounds = np.asarray([0.20, 0.40, 4.00, 0.10, 0.40, 4.00, 0.10, 0.50])
        return lower_bounds, upper_bounds

    def _bound_hits(
        self,
        parameters: np.ndarray,
        *,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        tolerance: float = 1e-4,
    ) -> list[str]:
        hits: list[str] = []
        for index, name in enumerate(PARAMETER_NAMES):
            value = float(parameters[index])
            if abs(value - float(lower_bounds[index])) <= tolerance:
                hits.append(f"{name}@lower")
            elif abs(value - float(upper_bounds[index])) <= tolerance:
                hits.append(f"{name}@upper")
        return hits

    def _build_model_notes(
        self,
        parameters: np.ndarray,
        *,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> str:
        parameter_summary = ", ".join(
            f"{name}={float(parameters[index]):.4f}"
            for index, name in enumerate(PARAMETER_NAMES)
        )
        bound_hits = self._bound_hits(
            parameters,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        bound_summary = ", ".join(bound_hits) if bound_hits else "none"
        return (
            "Trained grey-box room state-space model with delta-form thermal "
            "couplings and observer-estimated latent mass state. "
            f"Parameters: {parameter_summary}. Bound hits: {bound_summary}."
        )

    def _simulate_with_observer(
        self,
        prepared: PreparedRoomGreyBoxData,
        *,
        air_coefficients: np.ndarray,
        mass_parameters: np.ndarray,
        observer_gain: float,
        start_index: int,
        end_exclusive: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        room_temperature = prepared.common.field_arrays["room_temperature_c"]
        outdoor_temperature = prepared.common.field_arrays["outdoor_temperature_c"]
        predicted_next = np.full(len(room_temperature), np.nan, dtype=float)
        estimated_mass_state = np.zeros(len(room_temperature), dtype=float)

        a11, a12, b_out, g_heat_air, g_solar_air = air_coefficients
        k_air_mass, g_heat_mass, g_solar_mass = mass_parameters

        air_state = self._initial_air_state(room_temperature, start_index)
        mass_state = air_state
        estimated_mass_state[start_index] = mass_state

        for index in range(start_index, min(len(room_temperature) - 1, end_exclusive - 1)):
            outdoor_state = outdoor_temperature[index]
            if not np.isfinite(outdoor_state):
                outdoor_state = air_state

            heat_input = prepared.thermal_output_energy_kwh[index]
            solar_input = prepared.solar_effective_exposure[index]
            if not np.isfinite(heat_input):
                heat_input = 0.0
            if not np.isfinite(solar_input):
                solar_input = 0.0

            predicted_room_next = (
                a11 * air_state
                + a12 * mass_state
                + b_out * outdoor_state
                + g_heat_air * heat_input
                + g_solar_air * solar_input
            )
            predicted_mass_next = (
                mass_state
                + k_air_mass * (air_state - mass_state)
                + g_heat_mass * heat_input
                + g_solar_mass * solar_input
            )
            predicted_next[index] = predicted_room_next

            measured_room_next = room_temperature[index + 1]
            if np.isfinite(measured_room_next):
                innovation = float(measured_room_next) - predicted_room_next
                if not np.isfinite(innovation):
                    innovation = 0.0
                mass_state = predicted_mass_next + (observer_gain * innovation)
                air_state = float(measured_room_next)
            else:
                mass_state = predicted_mass_next
                air_state = predicted_room_next
            estimated_mass_state[index + 1] = mass_state

        return predicted_next, estimated_mass_state

    def _parameter_residuals(
        self,
        parameters: np.ndarray,
        prepared: PreparedRoomGreyBoxData,
        *,
        mask: np.ndarray,
        start_index: int,
        end_exclusive: int,
        max_stability_radius: float,
    ) -> np.ndarray:
        air_coefficients = self._air_coefficients_from_parameters(parameters)
        mass_parameters = self._mass_parameters_from_parameters(parameters)
        predicted_next, _ = self._simulate_with_observer(
            prepared,
            air_coefficients=air_coefficients,
            mass_parameters=mass_parameters,
            observer_gain=self._observer_gain_from_parameters(parameters),
            start_index=start_index,
            end_exclusive=end_exclusive,
        )
        errors = predicted_next[mask] - prepared.target_values[mask]
        errors = np.nan_to_num(errors, nan=1e6, posinf=1e6, neginf=-1e6)
        a_matrix = self._state_matrix_from_parameters(parameters)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a_matrix))))
        if not np.isfinite(spectral_radius):
            spectral_radius = max_stability_radius + 10.0
        stability_penalty = max(0.0, spectral_radius - max_stability_radius) * 100.0
        return np.concatenate([errors, np.asarray([stability_penalty], dtype=float)])

    def _is_physically_plausible_parameter_set(
        self,
        parameters: np.ndarray,
        *,
        max_stability_radius: float,
    ) -> bool:
        k_out = float(parameters[PARAM_K_OUT])
        k_mass = float(parameters[PARAM_K_MASS])
        g_heat_air = float(parameters[PARAM_G_HEAT_AIR])
        g_solar_air = float(parameters[PARAM_G_SOLAR_AIR])
        k_air_mass = float(parameters[PARAM_K_AIR_MASS])
        g_heat_mass = float(parameters[PARAM_G_HEAT_MASS])
        g_solar_mass = float(parameters[PARAM_G_SOLAR_MASS])
        air_retention = 1.0 - k_out - k_mass
        a_matrix = self._state_matrix_from_parameters(parameters)
        eigvals = np.linalg.eigvals(a_matrix)
        spectral_radius = float(np.max(np.abs(eigvals)))
        if not np.isfinite(spectral_radius):
            return False
        if k_out < 0.0 or k_mass < 0.0 or k_air_mass < 0.0:
            return False
        if air_retention < 0.0:
            return False
        if g_heat_air < -1e-6 or g_solar_air < -1e-6:
            return False
        if g_heat_mass < -1e-6 or g_solar_mass < -1e-6:
            return False
        if k_air_mass >= 1.0:
            return False
        if spectral_radius >= max_stability_radius:
            return False
        return True

    def _has_finite_rollout(
        self,
        prepared: PreparedRoomGreyBoxData,
        *,
        parameters: np.ndarray,
        mask: np.ndarray,
        start_index: int,
        end_exclusive: int,
    ) -> bool:
        air_coefficients = self._air_coefficients_from_parameters(parameters)
        mass_parameters = self._mass_parameters_from_parameters(parameters)
        predicted_next, estimated_mass_state = self._simulate_with_observer(
            prepared,
            air_coefficients=air_coefficients,
            mass_parameters=mass_parameters,
            observer_gain=self._observer_gain_from_parameters(parameters),
            start_index=start_index,
            end_exclusive=end_exclusive,
        )
        return bool(
            np.isfinite(predicted_next[mask]).all()
            and np.isfinite(estimated_mass_state[start_index:end_exclusive]).all()
        )

    def _fit_parameters(
        self,
        prepared: PreparedRoomGreyBoxData,
        *,
        config: RoomGreyBoxConfig,
        mask: np.ndarray,
        start_index: int,
        end_exclusive: int,
    ) -> np.ndarray:
        lower_bounds, upper_bounds = self._optimizer_bounds()
        initial_guesses = [
            np.asarray([0.03, 0.12, 0.90, 0.008, 0.04, 0.25, 0.002, config.observer_gain_initial]),
            np.asarray([0.05, 0.18, 1.20, 0.010, 0.06, 0.35, 0.003, 0.05]),
            np.asarray([0.02, 0.08, 0.70, 0.006, 0.02, 0.20, 0.001, 0.15]),
        ]

        best_parameters = None
        best_cost = float("inf")
        fallback_parameters = None
        fallback_cost = float("inf")
        for initial_guess in initial_guesses:
            def objective(parameters: np.ndarray) -> np.ndarray:
                return self._parameter_residuals(
                    parameters,
                    prepared,
                    mask=mask,
                    start_index=start_index,
                    end_exclusive=end_exclusive,
                    max_stability_radius=config.max_stability_radius,
                )

            result = least_squares(
                objective,
                x0=np.clip(initial_guess, lower_bounds, upper_bounds),
                bounds=(lower_bounds, upper_bounds),
                loss=config.optimization_loss,
                max_nfev=config.optimization_max_nfev,
            )
            candidate_parameters = np.asarray(result.x, dtype=float)
            has_finite_rollout = self._has_finite_rollout(
                prepared,
                parameters=candidate_parameters,
                mask=mask,
                start_index=start_index,
                end_exclusive=end_exclusive,
            )
            if not has_finite_rollout:
                continue
            if result.cost < fallback_cost:
                fallback_cost = float(result.cost)
                fallback_parameters = candidate_parameters
            if not self._is_physically_plausible_parameter_set(
                candidate_parameters,
                max_stability_radius=config.max_stability_radius,
            ):
                continue
            if result.cost < best_cost:
                best_cost = float(result.cost)
                best_parameters = candidate_parameters

        if best_parameters is not None:
            return best_parameters
        if fallback_parameters is not None:
            return fallback_parameters
        for initial_guess in initial_guesses:
            candidate_parameters = np.clip(initial_guess, lower_bounds, upper_bounds).astype(float)
            if self._has_finite_rollout(
                prepared,
                parameters=candidate_parameters,
                mask=mask,
                start_index=start_index,
                end_exclusive=end_exclusive,
            ):
                return candidate_parameters
        raise ValueError("grey-box optimizer found no finite rollout solution")

    def fit(
        self,
        dataset: MpcDataset,
        config: RoomGreyBoxConfig,
    ) -> RoomGreyBoxModel:
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
        prepared: PreparedRoomGreyBoxData,
        *,
        config: RoomGreyBoxConfig,
        interval_minutes: int,
        train_start_index: int = 0,
        train_end_exclusive: int | None = None,
    ) -> RoomGreyBoxModel:
        mask = self._training_mask(
            prepared,
            start_index=train_start_index,
            end_exclusive=train_end_exclusive,
        )
        y_values = prepared.target_values[mask]
        if y_values.size == 0:
            raise ValueError("not enough valid rows to fit grey-box room model")

        resolved_train_end_exclusive = (
            len(prepared.common.timestamps_utc) if train_end_exclusive is None else train_end_exclusive
        )
        parameters = self._fit_parameters(
            prepared,
            config=config,
            mask=mask,
            start_index=train_start_index,
            end_exclusive=resolved_train_end_exclusive,
        )
        lower_bounds, upper_bounds = self._optimizer_bounds()
        air_coefficients = self._air_coefficients_from_parameters(parameters)
        mass_parameters = self._mass_parameters_from_parameters(parameters)
        predicted_next, estimated_mass_state = self._simulate_with_observer(
            prepared,
            air_coefficients=air_coefficients,
            mass_parameters=mass_parameters,
            observer_gain=self._observer_gain_from_parameters(parameters),
            start_index=train_start_index,
            end_exclusive=resolved_train_end_exclusive,
        )

        model = RoomGreyBoxModel(
            trained_from_utc=prepared.common.timestamps_utc[train_start_index],
            trained_to_utc=prepared.common.timestamps_utc[resolved_train_end_exclusive - 1],
            interval_minutes=interval_minutes,
            config=config,
            feature_names=list(self.feature_names),
            intercept=0.0,
            coefficients=[float(value) for value in air_coefficients],
            sample_count=len(y_values),
            k_out=float(parameters[PARAM_K_OUT]),
            k_mass=float(parameters[PARAM_K_MASS]),
            k_air_mass=float(parameters[PARAM_K_AIR_MASS]),
            g_heat_mass=float(parameters[PARAM_G_HEAT_MASS]),
            g_solar_mass=float(parameters[PARAM_G_SOLAR_MASS]),
            observer_gain=self._observer_gain_from_parameters(parameters),
            notes=self._build_model_notes(
                parameters,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            ),
        )
        prepared.estimated_mass_state_cache[self._model_cache_key(model)] = estimated_mass_state
        if not np.isfinite(predicted_next[mask]).all():
            raise ValueError("grey-box fit produced non-finite predictions")
        return model

    def _model_cache_key(self, model: RoomGreyBoxModel) -> tuple:
        return (
            model.trained_from_utc,
            model.trained_to_utc,
            tuple(round(value, 10) for value in model.coefficients),
            round(model.k_out, 10),
            round(model.k_mass, 10),
            round(model.k_air_mass, 10),
            round(model.g_heat_mass, 10),
            round(model.g_solar_mass, 10),
            round(model.observer_gain, 10),
        )

    def _estimated_mass_state_series_for_model(
        self,
        prepared: PreparedRoomGreyBoxData,
        model: RoomGreyBoxModel,
    ) -> np.ndarray:
        cache_key = self._model_cache_key(model)
        cached = prepared.estimated_mass_state_cache.get(cache_key)
        if cached is not None:
            return cached
        _, estimated_mass_state = self._simulate_with_observer(
            prepared,
            air_coefficients=np.asarray(model.coefficients, dtype=float),
            mass_parameters=np.asarray(
                [model.k_air_mass, model.g_heat_mass, model.g_solar_mass],
                dtype=float,
            ),
            observer_gain=model.observer_gain,
            start_index=0,
            end_exclusive=len(prepared.common.timestamps_utc),
        )
        prepared.estimated_mass_state_cache[cache_key] = estimated_mass_state
        return estimated_mass_state

    def predict_next_prepared(
        self,
        model: RoomGreyBoxModel,
        prepared: PreparedRoomGreyBoxData,
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
        if not np.isfinite(room_state):
            return None

        mass_state = predicted_mass_temperatures.get(source_index)
        if mass_state is None or source_index <= prediction_origin_index:
            mass_state = self._estimated_mass_state_series_for_model(prepared, model)[source_index]

        outdoor_temperature = common.field_arrays["outdoor_temperature_c"][source_index]
        if not np.isfinite(outdoor_temperature):
            outdoor_temperature = float(room_state)

        thermal_output_energy = prepared.thermal_output_energy_kwh[source_index]
        solar_effective_exposure = prepared.solar_effective_exposure[source_index]
        if not np.isfinite(thermal_output_energy):
            thermal_output_energy = 0.0
        if not np.isfinite(solar_effective_exposure):
            solar_effective_exposure = 0.0

        next_room_state = (
            model.coefficients[0] * float(room_state)
            + model.coefficients[1] * mass_state
            + model.coefficients[2] * outdoor_temperature
            + model.coefficients[3] * thermal_output_energy
            + model.coefficients[4] * solar_effective_exposure
        )
        next_mass_state = (
            mass_state
            + model.k_air_mass * (float(room_state) - mass_state)
            + model.g_heat_mass * thermal_output_energy
            + model.g_solar_mass * solar_effective_exposure
        )
        return float(next_room_state), float(next_mass_state)

    def predict_next(
        self,
        model: TrainedLinearRoomModel,
        rows: list[MpcDatasetRow],
        *,
        source_index: int,
        predicted_room_temperatures: dict[int, float] | None = None,
        prediction_origin_index: int | None = None,
    ) -> float | None:
        if not isinstance(model, RoomGreyBoxModel):
            raise ValueError("room_greybox trainer requires a RoomGreyBoxModel")
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
        model: RoomGreyBoxModel,
        prepared: PreparedRoomGreyBoxData,
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
        if not isinstance(model, RoomGreyBoxModel):
            raise ValueError("room_greybox trainer requires a RoomGreyBoxModel")
        prepared = self.prepare(rows, model.config)
        return self.simulate_horizon_prepared(
            model,
            prepared,
            start_index=start_index,
            horizon_steps=horizon_steps,
        )
