"""Scikit-learn shutter forecast provider.

This module implements a lightweight autoregressive forecaster for the living-
room shutters. The predicted horizon array is used to attenuate solar gains on
the south-facing glazing in the UFH model.

Why this model exists
---------------------
The shutter position is not purely exogenous weather data: it depends on human
behavior, time of day, and recent shutter state. A small tree-based regressor is
therefore a pragmatic grey-box predictor: weather enters explicitly through GTI
and outdoor temperature, while occupancy/behavior patterns are learned from the
historical shutter sequence.
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
from .models import ShutterForecastSettings

if TYPE_CHECKING:
    from ..telemetry.repository import TelemetryRepository


_ARTIFACT_NAME: str = "shutter_model"
_ARTIFACT_VERSION: int = 1

log = logging.getLogger("home_optimizer.forecasting.shutter")


@dataclass(frozen=True, slots=True)
class _ShutterSample:
    """One historical training sample for the shutter regressor.

    Attributes:
        valid_at_utc: Timestamp represented by this sample [UTC].
        outdoor_temperature_c: Outdoor air temperature [°C].
        gti_w_per_m2: Irradiance on the south-facing windows [W/m²].
        previous_shutter_pct: Previous shutter position [% open].
        shutter_pct: Current shutter target [% open].
    """

    valid_at_utc: datetime
    outdoor_temperature_c: float
    gti_w_per_m2: float
    previous_shutter_pct: float
    shutter_pct: float


@dataclass(frozen=True, slots=True)
class _CachedShutterModel:
    """Cached fitted regressor or cached sparse-history miss.

    Attributes:
        regressor: Trained random forest, or ``None`` when history was too sparse.
        sample_count: Number of training samples that were available when this
            cache entry was created [-].
    """

    regressor: RandomForestRegressor | None
    sample_count: int
    artifact_mtime_ns: int | None = None


def _build_feature_row(
    *,
    valid_at_utc: datetime,
    outdoor_temperature_c: float,
    gti_w_per_m2: float,
    previous_shutter_pct: float,
) -> np.ndarray:
    """Assemble one numerical feature vector for the shutter model.

    Args:
        valid_at_utc: Prediction timestamp [UTC].
        outdoor_temperature_c: Outdoor air temperature [°C].
        gti_w_per_m2: South-window irradiance [W/m²].
        previous_shutter_pct: Previous shutter state [% open].

    Returns:
        Feature vector with shape ``(8,)``.
    """

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
            previous_shutter_pct,
        ],
        dtype=float,
    )


class ShutterForecaster:
    """Predict a horizon-wide living-room shutter profile with scikit-learn.

    The model is trained on historical telemetry buckets and forecast-aligned GTI
    values. Prediction is autoregressive in the previous shutter position, so the
    current measured/manual shutter state acts as the initial condition.
    """

    _model_cache: dict[tuple[object, ...], _CachedShutterModel] = {}
    _cache_lock: Lock = Lock()

    def __init__(self, settings: ShutterForecastSettings | None = None) -> None:
        self._settings = settings or ShutterForecastSettings()

    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear all cached trained shutter models in the current process."""
        with cls._cache_lock:
            cls._model_cache.clear()

    def train_and_persist_from_repository(
        self,
        *,
        repository: "TelemetryRepository",
    ) -> PersistedRegressorArtifactMetadata | None:
        """Fit a shutter model from repository history and persist it to disk.

        Args:
            repository: Telemetry repository that supplies historical telemetry and
                the canonical artifact storage path next to its SQLite database.

        Returns:
            Persisted artifact metadata, or ``None`` when the repository still
            contains too little history to fit a stable model.
        """

        training_samples = self._build_training_samples(repository)
        if len(training_samples) < self._settings.min_training_samples:
            log.info(
                "Skipping shutter-model training: need at least %d samples, found %d.",
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
                    previous_shutter_pct=sample.previous_shutter_pct,
                )
                for sample in training_samples
            ]
        )
        y = np.array([sample.shutter_pct for sample in training_samples], dtype=float)
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
        initial_shutter_pct: float,
    ) -> np.ndarray | None:
        """Train on repository history and predict the upcoming shutter horizon.

        Args:
            repository: Telemetry repository containing historical training data.
            weather_rows: Future weather rows ordered by horizon step. Each row must
                expose ``valid_at_utc``, ``t_out_c`` and ``gti_w_per_m2``.
            horizon_steps: Required horizon length ``N`` [-].
            initial_shutter_pct: Latest known shutter position [% open].

        Returns:
            Predicted shutter array [% open], shape ``(N,)``, or ``None`` when not
            enough history is available to fit a reliable model.
        """

        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be strictly positive.")
        if not 0.0 <= initial_shutter_pct <= 100.0:
            raise ValueError("initial_shutter_pct must be in [0, 100].")
        if len(weather_rows) < horizon_steps:
            return None

        cached_model = self._load_cached_artifact(repository)
        if cached_model.regressor is None:
            return None

        predictions = np.zeros(horizon_steps, dtype=float)
        previous_shutter_pct = initial_shutter_pct
        for index, row in enumerate(weather_rows[:horizon_steps]):
            features = _build_feature_row(
                valid_at_utc=row.valid_at_utc,
                outdoor_temperature_c=float(row.t_out_c),
                gti_w_per_m2=float(row.gti_w_per_m2),
                previous_shutter_pct=previous_shutter_pct,
            )
            predicted_pct = float(cached_model.regressor.predict(features.reshape(1, -1))[0])
            clipped_pct = float(np.clip(predicted_pct, 0.0, 100.0))
            predictions[index] = clipped_pct
            previous_shutter_pct = clipped_pct
        return predictions

    def _load_cached_artifact(self, repository: "TelemetryRepository") -> _CachedShutterModel:
        """Load the persisted shutter artifact, reusing an in-process cache.

        The disk artifact is the source of truth. The in-process cache only avoids
        repeated deserialisation while the artifact file remains unchanged.
        """

        artifact_path = repository.forecast_artifact_path(_ARTIFACT_NAME)
        if not artifact_path.exists():
            return _CachedShutterModel(regressor=None, sample_count=0, artifact_mtime_ns=None)

        artifact_mtime_ns = artifact_path.stat().st_mtime_ns
        cache_key = (str(artifact_path), self._settings_signature())
        with type(self)._cache_lock:
            cached = type(self)._model_cache.get(cache_key)
        if cached is not None and cached.artifact_mtime_ns == artifact_mtime_ns:
            return cached

        loaded = self._load_artifact_from_disk(artifact_path)
        if loaded is None:
            return _CachedShutterModel(regressor=None, sample_count=0, artifact_mtime_ns=None)

        metadata, regressor = loaded
        if metadata.model_version != _ARTIFACT_VERSION:
            log.warning(
                "Ignoring shutter-model artifact %s with version %s (expected %s).",
                artifact_path,
                metadata.model_version,
                _ARTIFACT_VERSION,
            )
            return _CachedShutterModel(regressor=None, sample_count=0, artifact_mtime_ns=None)
        if metadata.settings_signature != self._settings_signature():
            log.info(
                "Ignoring shutter-model artifact %s because the hyperparameter signature changed.",
                artifact_path,
            )
            return _CachedShutterModel(regressor=None, sample_count=0, artifact_mtime_ns=None)

        cached = _CachedShutterModel(
            regressor=regressor,
            sample_count=metadata.sample_count,
            artifact_mtime_ns=artifact_mtime_ns,
        )
        with type(self)._cache_lock:
            type(self)._model_cache[cache_key] = cached
        return cached

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """Fit one shutter regressor from tabular training data.

        Args:
            X: Feature matrix with shape ``(n_samples, 8)``.
            y: Target shutter positions [% open], shape ``(n_samples,)``.

        Returns:
            Trained scikit-learn random forest regressor.
        """

        # A shallow random forest is robust to non-linear occupant behavior while
        # requiring little feature engineering. This keeps the runtime model small
        # and deterministic enough for the MPC path.
        regressor = RandomForestRegressor(
            n_estimators=self._settings.tree_count,
            max_depth=self._settings.tree_max_depth,
            min_samples_leaf=self._settings.min_samples_leaf,
            random_state=self._settings.random_state,
        )
        regressor.fit(X, y)
        return regressor

    def _settings_signature(self) -> tuple[object, ...]:
        """Return a stable hyperparameter signature for artifact compatibility checks."""

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
        """Persist one trained shutter model atomically to disk.

        Atomic replacement ensures runtime inference never observes a partially
        written model file while the nightly training job is updating it.
        """

        persist_regressor_artifact(artifact_path=artifact_path, regressor=regressor, metadata=metadata)

        cached = _CachedShutterModel(
            regressor=regressor,
            sample_count=metadata.sample_count,
            artifact_mtime_ns=artifact_path.stat().st_mtime_ns,
        )
        cache_key = (str(artifact_path), self._settings_signature())
        with type(self)._cache_lock:
            type(self)._model_cache[cache_key] = cached

    def _load_artifact_from_disk(
        self,
        artifact_path,
    ) -> tuple[ShutterModelArtifactMetadata, RandomForestRegressor] | None:
        """Deserialize one persisted shutter model artifact from disk."""

        loaded = load_regressor_artifact(artifact_path)
        if loaded is None:
            return None
        metadata, regressor = loaded
        if not isinstance(regressor, RandomForestRegressor):
            log.warning("Ignoring shutter-model artifact %s with unexpected estimator type.", artifact_path)
            return None
        return metadata, regressor

    def _build_training_samples(self, repository: "TelemetryRepository") -> list[_ShutterSample]:
        """Construct sequential training samples from telemetry and forecast history.

        The target is the current bucket's shutter position, while the previous
        bucket's shutter position becomes an autoregressive feature. GTI is aligned
        by nearest hourly forecast timestamp when available; otherwise 0 W/m² is
        used as an explicit "no GTI information" fallback.
        """

        aggregates = sorted(repository.list_aggregates(), key=lambda row: row.bucket_end_utc)
        if len(aggregates) < 2:
            return []

        forecast_rows = repository.list_forecast_snapshots()
        ordered_forecast_rows = sorted(forecast_rows, key=lambda row: row.valid_at_utc)
        forecast_times = [row.valid_at_utc for row in ordered_forecast_rows]
        forecast_lookup = self._average_weather_by_timestamp(ordered_forecast_rows)
        samples: list[_ShutterSample] = []

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
                _ShutterSample(
                    valid_at_utc=current_row.bucket_end_utc,
                    outdoor_temperature_c=outdoor_temperature_c,
                    gti_w_per_m2=gti_w_per_m2,
                    previous_shutter_pct=float(previous_row.shutter_living_room_last_pct),
                    shutter_pct=float(current_row.shutter_living_room_last_pct),
                )
            )
        return samples

    @staticmethod
    def _average_weather_by_timestamp(forecast_rows: Sequence[Any]) -> dict[datetime, tuple[float, float]]:
        """Average weather features over duplicate forecast rows with equal valid time.

        Multiple Open-Meteo fetches may exist for the same ``valid_at_utc``. For the
        behavior model we only need one representative weather vector per timestamp,
        so duplicates are averaged.
        """

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
        """Return the nearest available weather vector for one telemetry timestamp.

        Args:
            valid_at_utc: Target timestamp [UTC].
            sorted_valid_at_utc: Sorted forecast timestamps [UTC].
            weather_by_valid_at: Mapping ``valid_at_utc -> (T_out [°C], GTI [W/m²])``.

        Returns:
            Matched weather tuple or ``None`` when no forecast row lies within the
            configured alignment tolerance.
        """

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
