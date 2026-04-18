"""Configuration models for machine-learning forecast providers.

The forecasting layer is intentionally provider-oriented: each provider predicts
one or more optional horizon arrays that can enrich a runtime ``RunRequest``.
Today this package predicts ``shutter_forecast``; later the same service can add
baseload or internal-gains forecasts without changing the Optimizer pipeline.

Units
-----
Time        : h
Temperature : °C
Irradiance  : W/m²
Shutter     : % open
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SHUTTER_MIN_TRAINING_SAMPLES: int = 24
DEFAULT_SHUTTER_MAX_HISTORY_GAP_HOURS: float = 2.0
DEFAULT_SHUTTER_FORECAST_ALIGNMENT_TOLERANCE_HOURS: float = 0.5
DEFAULT_SHUTTER_RANDOM_STATE: int = 7
DEFAULT_SHUTTER_TREE_COUNT: int = 64
DEFAULT_SHUTTER_TREE_MAX_DEPTH: int = 6
DEFAULT_SHUTTER_MIN_SAMPLES_LEAF: int = 2


@dataclass(frozen=True, slots=True)
class ShutterForecastSettings:
    """Hyperparameters and fail-fast limits for the ML shutter forecaster.

    Attributes:
        min_training_samples: Minimum number of sequential historical samples [-]
            required before fitting a model. Fewer samples return no forecast so
            the optimizer safely falls back to the scalar shutter position.
        max_history_gap_hours: Maximum allowed time gap [h] between two telemetry
            buckets that still counts as one continuous sequence.
        alignment_tolerance_hours: Maximum absolute time difference [h] allowed
            when matching hourly weather forecast rows to telemetry buckets.
        random_state: Deterministic scikit-learn seed [-] for reproducible tests.
        tree_count: Number of trees in the random forest [-].
        tree_max_depth: Maximum tree depth [-].
        min_samples_leaf: Minimum number of samples in a leaf node [-].
    """

    min_training_samples: int = DEFAULT_SHUTTER_MIN_TRAINING_SAMPLES
    max_history_gap_hours: float = DEFAULT_SHUTTER_MAX_HISTORY_GAP_HOURS
    alignment_tolerance_hours: float = DEFAULT_SHUTTER_FORECAST_ALIGNMENT_TOLERANCE_HOURS
    random_state: int = DEFAULT_SHUTTER_RANDOM_STATE
    tree_count: int = DEFAULT_SHUTTER_TREE_COUNT
    tree_max_depth: int = DEFAULT_SHUTTER_TREE_MAX_DEPTH
    min_samples_leaf: int = DEFAULT_SHUTTER_MIN_SAMPLES_LEAF

    def __post_init__(self) -> None:
        if self.min_training_samples <= 0:
            raise ValueError("min_training_samples must be strictly positive.")
        if self.max_history_gap_hours <= 0.0:
            raise ValueError("max_history_gap_hours must be strictly positive.")
        if self.alignment_tolerance_hours < 0.0:
            raise ValueError("alignment_tolerance_hours must be non-negative.")
        if self.tree_count <= 0:
            raise ValueError("tree_count must be strictly positive.")
        if self.tree_max_depth <= 0:
            raise ValueError("tree_max_depth must be strictly positive.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be strictly positive.")


@dataclass(frozen=True, slots=True)
class ForecastServiceSettings:
    """Top-level settings for the runtime ML forecasting service.

    Attributes:
        shutter: Settings for the current shutter-forecast provider.
    """

    shutter: ShutterForecastSettings = field(default_factory=ShutterForecastSettings)
