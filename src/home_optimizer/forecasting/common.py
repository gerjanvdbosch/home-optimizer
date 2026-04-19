"""Shared utilities for persisted ML forecast providers.

This module contains the small amount of logic that is identical across the
forecast providers: cyclic time encoding and joblib-based model artifact I/O.
Keeping that here avoids duplicating persistence logic in the individual
forecasters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from math import cos, pi, sin
from pathlib import Path
from typing import Any

import joblib

_ARTIFACT_TMP_SUFFIX: str = ".tmp"

log = logging.getLogger("home_optimizer.forecasting.common")


@dataclass(frozen=True, slots=True)
class PersistedRegressorArtifactMetadata:
    """Metadata stored alongside one persisted forecast regressor artifact.

    Attributes:
        model_version: Artifact schema version [-].
        trained_at_utc: UTC timestamp when the model was trained.
        sample_count: Number of training samples used for fitting [-].
        settings_signature: Tuple of hyperparameters that must match the current
            provider settings before the artifact is accepted.
        training_fingerprint: Compact repository fingerprint captured during
            training, used for observability and traceability.
    """

    model_version: int
    trained_at_utc: datetime
    sample_count: int
    settings_signature: tuple[object, ...]
    training_fingerprint: tuple[object, ...]


def cyclical_time_features(valid_at_utc: datetime) -> tuple[float, float, float, float, float]:
    """Return periodic time features for one UTC timestamp.

    Args:
        valid_at_utc: Timestamp [UTC].

    Returns:
        Tuple containing:
            hour_sin [-], hour_cos [-], weekday_sin [-], weekday_cos [-],
            weekend_flag [-].
    """

    hour_fraction = (valid_at_utc.hour + valid_at_utc.minute / 60.0) / 24.0
    weekday_fraction = valid_at_utc.weekday() / 7.0
    weekend_flag = 1.0 if valid_at_utc.weekday() >= 5 else 0.0
    return (
        sin(2.0 * pi * hour_fraction),
        cos(2.0 * pi * hour_fraction),
        sin(2.0 * pi * weekday_fraction),
        cos(2.0 * pi * weekday_fraction),
        weekend_flag,
    )


def persist_regressor_artifact(
    *,
    artifact_path: Path,
    regressor: Any,
    metadata: PersistedRegressorArtifactMetadata,
) -> None:
    """Persist one trained ML regressor atomically to disk.

    Args:
        artifact_path: Final artifact path on disk.
        regressor: Trained scikit-learn estimator.
        metadata: Persisted metadata describing the artifact.
    """

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_suffix(f"{artifact_path.suffix}{_ARTIFACT_TMP_SUFFIX}")
    payload = {
        "metadata": {
            "model_version": metadata.model_version,
            "trained_at_utc": metadata.trained_at_utc.isoformat(),
            "sample_count": metadata.sample_count,
            "settings_signature": metadata.settings_signature,
            "training_fingerprint": metadata.training_fingerprint,
        },
        "model": regressor,
    }
    joblib.dump(payload, temporary_path)
    temporary_path.replace(artifact_path)


def load_regressor_artifact(
    artifact_path: Path,
) -> tuple[PersistedRegressorArtifactMetadata, Any] | None:
    """Deserialize one persisted forecast regressor artifact from disk.

    Args:
        artifact_path: Artifact path on disk.

    Returns:
        Tuple ``(metadata, model)`` or ``None`` when the artifact is unreadable or
        malformed.
    """

    try:
        payload = joblib.load(artifact_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load forecast-model artifact %s: %s", artifact_path, exc)
        return None

    metadata_payload = payload.get("metadata") if isinstance(payload, dict) else None
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(metadata_payload, dict) or model is None:
        log.warning("Ignoring malformed forecast-model artifact at %s.", artifact_path)
        return None

    try:
        metadata = PersistedRegressorArtifactMetadata(
            model_version=int(metadata_payload["model_version"]),
            trained_at_utc=datetime.fromisoformat(str(metadata_payload["trained_at_utc"])),
            sample_count=int(metadata_payload["sample_count"]),
            settings_signature=tuple(metadata_payload["settings_signature"]),
            training_fingerprint=tuple(metadata_payload["training_fingerprint"]),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Ignoring forecast-model artifact with invalid metadata at %s: %s", artifact_path, exc)
        return None
    return metadata, model
