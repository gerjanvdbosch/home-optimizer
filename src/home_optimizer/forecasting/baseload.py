"""Persisted scikit-learn baseload forecast provider.

The baseload target is the household non-heat-pump electrical demand proxy that
already exists in telemetry as ``household_elec_power_*``. This signal is useful
both as an electrical forecast in its own right and as a proxy for time-varying
internal heat gains in the UFH thermal model.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .common import (
    PersistedRegressorArtifactMetadata,
    cyclical_time_features,
    load_regressor_artifact,
    persist_regressor_artifact,
)
from .models import BaseloadForecastSettings

if TYPE_CHECKING:
    from ..telemetry.repository import TelemetryRepository


_ARTIFACT_NAME: str = "baseload_model"
_ARTIFACT_VERSION: int = 1

log = logging.getLogger("home_optimizer.forecasting.baseload")


@dataclass(frozen=True, slots=True)
class _BaseloadSample:
    """One historical training sample for the baseload regressor."""

    valid_at_utc: datetime
    outdoor_temperature_c: float
    gti_w_per_m2: float
    previous_baseload_kw: float
    baseload_kw: float


@dataclass(frozen=True, slots=True)
class _CachedBaseloadModel:
    """Cached fitted regressor or missing-artifact marker."""

    regressor: RandomForestRegressor | None
    sample_count: int
    artifact_mtime_ns: int | None = None


def _build_feature_row(
    *,
    valid_at_utc: datetime,
    outdoor_temperature_c: float,
    gti_w_per_m2: float,
    previous_baseload_kw: float,
) -> np.ndarray:
    """Assemble one numerical feature vector for the baseload model."""

    hour_sin, hour_cos, weekday_sin, weekday_cos, weekend_flag = cyclical_time_features(valid_at_utc)
    return np.array(
        [
            hour_sin,
            hour_cos,
            weekday_sin,
            weekday_cos,
            weekend_flag,
            outdoor_temperature_c,
            gti_w_per_m2,
            previous_baseload_kw,
        ],
        dtype=float,
    )


class BaseloadForecaster:
    """Predict a horizon-wide household baseload profile with scikit-learn."""

    _model_cache: dict[tuple[object, ...], _CachedBaseloadModel] = {}
    _cache_lock: Lock = Lock()

    def __init__(self, settings: BaseloadForecastSettings | None = None) -> None:
        self._settings = settings or BaseloadForecastSettings()

    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear all cached loaded baseload models in the current process."""
        with cls._cache_lock:
            cls._model_cache.clear()

    def train_and_persist_from_repository(
        self,
        *,
        repository: "TelemetryRepository",
    ) -> PersistedRegressorArtifactMetadata | None:
        """Fit a baseload model from repository history and persist it to disk."""

        training_samples = self._build_training_samples(repository)
        if len(training_samples) < self._settings.min_training_samples:
            log.info(
                "Skipping baseload-model training: need at least %d samples, found %d.",
                self._settings.min_training_samples,
                len(training_samples),
            )
            return None

        X = np.vstack(
            [
                _build_feature_row(
                    valid_at_utc=sample.valid_at_utc,
                    outdoor_temperature_c=sample.outdoor_temperature_c,
                    gti_w_per_m2=sample.gti_w_per_m2,
                    previous_baseload_kw=sample.previous_baseload_kw,
                )
                for sample in training_samples
            ]
        )
        y = np.array([sample.baseload_kw for sample in training_samples], dtype=float)
        regressor = self._fit_regressor(X, y)
        metadata = PersistedRegressorArtifactMetadata(
            model_version=_ARTIFACT_VERSION,
            trained_at_utc=datetime.now(tz=timezone.utc),
            sample_count=len(training_samples),
            settings_signature=self._settings_signature(),
            training_fingerprint=repository.forecast_training_fingerprint(),
        )
        artifact_path = repository.forecast_artifact_path(_ARTIFACT_NAME)
        self._persist_artifact(artifact_path=artifact_path, regressor=regressor, metadata=metadata)
        return metadata

    def predict_from_repository(
        self,
        *,
        repository: "TelemetryRepository",
        weather_rows: Sequence[Any],
        horizon_steps: int,
    ) -> np.ndarray | None:
        """Load the persisted baseload artifact and predict the upcoming horizon."""

        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be strictly positive.")
        if len(weather_rows) < horizon_steps:
            return None

        cached_model = self._load_cached_artifact(repository)
        if cached_model.regressor is None:
            return None

        previous_baseload_kw = self._latest_baseload_kw(repository)
        predictions = np.zeros(horizon_steps, dtype=float)
        for index, row in enumerate(weather_rows[:horizon_steps]):
            features = _build_feature_row(
                valid_at_utc=row.valid_at_utc,
                outdoor_temperature_c=float(row.t_out_c),
                gti_w_per_m2=float(row.gti_w_per_m2),
                previous_baseload_kw=previous_baseload_kw,
            )
            predicted_kw = float(cached_model.regressor.predict(features.reshape(1, -1))[0])
            clipped_kw = float(np.clip(predicted_kw, 0.0, None))
            predictions[index] = clipped_kw
            previous_baseload_kw = clipped_kw
        return predictions

    def _get_ordered_aggregates(self, repository: "TelemetryRepository") -> list[Any]:
        return sorted(repository.list_aggregates(), key=lambda row: row.bucket_end_utc)

    def _latest_baseload_kw(self, repository: "TelemetryRepository") -> float:
        """Return the latest measured baseload proxy [kW] from telemetry history."""

        aggregates = self._get_ordered_aggregates(repository)
        if not aggregates:
            return 0.0
        return max(0.0, float(aggregates[-1].household_elec_power_last_kw))

    def _build_training_samples(self, repository: "TelemetryRepository") -> list[_BaseloadSample]:
        """Construct sequential training samples from telemetry and forecast history."""

        aggregates = self._get_ordered_aggregates(repository)
        if len(aggregates) < 2:
            return []

        forecast_rows = repository.list_forecast_snapshots()
        ordered_forecast_rows = sorted(forecast_rows, key=lambda row: row.valid_at_utc)
        forecast_times = [row.valid_at_utc for row in ordered_forecast_rows]
        forecast_lookup = self._average_weather_by_timestamp(ordered_forecast_rows)
        samples: list[_BaseloadSample] = []

        for previous_row, current_row in zip(aggregates, aggregates[1:]):
            gap_hours = (current_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / 3600.0
            if gap_hours <= 0.0 or gap_hours > self._settings.max_history_gap_hours:
                continue

            matched_weather = self._nearest_weather_features(
                valid_at_utc=current_row.bucket_end_utc,
                sorted_valid_at_utc=forecast_times,
                weather_by_valid_at=forecast_lookup,
            )
            outdoor_temperature_c = float(current_row.outdoor_temperature_last_c)
            gti_w_per_m2 = 0.0
            if matched_weather is not None:
                outdoor_temperature_c, gti_w_per_m2 = matched_weather

            samples.append(
                _BaseloadSample(
                    valid_at_utc=current_row.bucket_end_utc,
                    outdoor_temperature_c=outdoor_temperature_c,
                    gti_w_per_m2=gti_w_per_m2,
                    previous_baseload_kw=max(0.0, float(previous_row.household_elec_power_last_kw)),
                    baseload_kw=max(0.0, float(current_row.household_elec_power_last_kw)),
                )
            )
        return samples

    @staticmethod
    def _average_weather_by_timestamp(forecast_rows: Sequence[Any]) -> dict[datetime, tuple[float, float]]:
        grouped: dict[datetime, list[tuple[float, float]]] = {}
        for row in forecast_rows:
            grouped.setdefault(row.valid_at_utc, []).append((float(row.t_out_c), float(row.gti_w_per_m2)))
        return {
            valid_at_utc: (
                float(np.mean([item[0] for item in items])),
                float(np.mean([item[1] for item in items])),
            )
            for valid_at_utc, items in grouped.items()
        }

    def _nearest_weather_features(
        self,
        *,
        valid_at_utc: datetime,
        sorted_valid_at_utc: Sequence[datetime],
        weather_by_valid_at: dict[datetime, tuple[float, float]],
    ) -> tuple[float, float] | None:
        if not sorted_valid_at_utc:
            return None

        insert_index = bisect_left(sorted_valid_at_utc, valid_at_utc)
        candidate_indices = [
            index for index in (insert_index - 1, insert_index) if 0 <= index < len(sorted_valid_at_utc)
        ]
        if not candidate_indices:
            return None

        best_index = min(
            candidate_indices,
            key=lambda index: abs((sorted_valid_at_utc[index] - valid_at_utc).total_seconds()),
        )
        best_valid_at_utc = sorted_valid_at_utc[best_index]
        time_error_hours = abs((best_valid_at_utc - valid_at_utc).total_seconds()) / 3600.0
        if time_error_hours > self._settings.alignment_tolerance_hours:
            return None
        return weather_by_valid_at[best_valid_at_utc]

    def _load_cached_artifact(self, repository: "TelemetryRepository") -> _CachedBaseloadModel:
        """Load the persisted baseload artifact, reusing an in-process cache."""

        artifact_path = repository.forecast_artifact_path(_ARTIFACT_NAME)
        if not artifact_path.exists():
            return _CachedBaseloadModel(regressor=None, sample_count=0, artifact_mtime_ns=None)

        artifact_mtime_ns = artifact_path.stat().st_mtime_ns
        cache_key = (str(artifact_path), self._settings_signature())
        with type(self)._cache_lock:
            cached = type(self)._model_cache.get(cache_key)
        if cached is not None and cached.artifact_mtime_ns == artifact_mtime_ns:
            return cached

        loaded = self._load_artifact_from_disk(artifact_path)
        if loaded is None:
            return _CachedBaseloadModel(regressor=None, sample_count=0, artifact_mtime_ns=None)

        metadata, regressor = loaded
        if metadata.model_version != _ARTIFACT_VERSION:
            log.warning(
                "Ignoring baseload-model artifact %s with version %s (expected %s).",
                artifact_path,
                metadata.model_version,
                _ARTIFACT_VERSION,
            )
            return _CachedBaseloadModel(regressor=None, sample_count=0, artifact_mtime_ns=None)
        if metadata.settings_signature != self._settings_signature():
            log.info(
                "Ignoring baseload-model artifact %s because the hyperparameter signature changed.",
                artifact_path,
            )
            return _CachedBaseloadModel(regressor=None, sample_count=0, artifact_mtime_ns=None)
        if not isinstance(regressor, RandomForestRegressor):
            log.warning("Ignoring baseload-model artifact %s with unexpected estimator type.", artifact_path)
            return _CachedBaseloadModel(regressor=None, sample_count=0, artifact_mtime_ns=None)

        cached = _CachedBaseloadModel(
            regressor=regressor,
            sample_count=metadata.sample_count,
            artifact_mtime_ns=artifact_mtime_ns,
        )
        with type(self)._cache_lock:
            type(self)._model_cache[cache_key] = cached
        return cached

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        regressor = RandomForestRegressor(
            n_estimators=self._settings.tree_count,
            max_depth=self._settings.tree_max_depth,
            min_samples_leaf=self._settings.min_samples_leaf,
            random_state=self._settings.random_state,
        )
        regressor.fit(X, y)
        return regressor

    def _settings_signature(self) -> tuple[object, ...]:
        return (
            self._settings.min_training_samples,
            self._settings.max_history_gap_hours,
            self._settings.alignment_tolerance_hours,
            self._settings.random_state,
            self._settings.tree_count,
            self._settings.tree_max_depth,
            self._settings.min_samples_leaf,
        )

    def _persist_artifact(
        self,
        *,
        artifact_path,
        regressor: RandomForestRegressor,
        metadata: PersistedRegressorArtifactMetadata,
    ) -> None:
        persist_regressor_artifact(artifact_path=artifact_path, regressor=regressor, metadata=metadata)

        cached = _CachedBaseloadModel(
            regressor=regressor,
            sample_count=metadata.sample_count,
            artifact_mtime_ns=artifact_path.stat().st_mtime_ns,
        )
        cache_key = (str(artifact_path), self._settings_signature())
        with type(self)._cache_lock:
            type(self)._model_cache[cache_key] = cached

    def _load_artifact_from_disk(self, artifact_path) -> tuple[PersistedRegressorArtifactMetadata, Any] | None:
        return load_regressor_artifact(artifact_path)
