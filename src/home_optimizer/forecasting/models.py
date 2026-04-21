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
DEFAULT_BASELOAD_MIN_TRAINING_SAMPLES: int = 24
DEFAULT_BASELOAD_MAX_HISTORY_GAP_HOURS: float = 2.0
DEFAULT_BASELOAD_FORECAST_ALIGNMENT_TOLERANCE_HOURS: float = 0.5
DEFAULT_BASELOAD_RANDOM_STATE: int = 11
DEFAULT_BASELOAD_TREE_COUNT: int = 64
DEFAULT_BASELOAD_TREE_MAX_DEPTH: int = 6
DEFAULT_BASELOAD_MIN_SAMPLES_LEAF: int = 2
DEFAULT_DHW_TAP_MIN_TRAINING_SAMPLES: int = 24
DEFAULT_DHW_TAP_MIN_SAMPLES_PER_HOUR: int = 2
DEFAULT_DHW_TAP_MAX_HISTORY_GAP_HOURS: float = 2.0
DEFAULT_DHW_TAP_MAX_IMPLIED_TAP_M3_PER_H: float = 0.2
DEFAULT_DHW_TAP_MIN_PEAK_TO_BACKGROUND_RATIO: float = 2.0
DEFAULT_DHW_TAP_MIN_TAP_TEMP_DIFF_K: float = 5.0


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
        dhw_tap: Settings for the recurring DHW tap-forecast provider.
        shutter: Settings for the current shutter-forecast provider.
        baseload: Settings for the baseload forecast provider.
    """

    dhw_tap: "DHWTapForecastSettings" = field(default_factory=lambda: DHWTapForecastSettings())
    shutter: ShutterForecastSettings = field(default_factory=ShutterForecastSettings)
    baseload: "BaseloadForecastSettings" = field(default_factory=lambda: BaseloadForecastSettings())


@dataclass(frozen=True, slots=True)
class DHWTapForecastSettings:
    """Hyperparameters and fail-fast limits for the history-based DHW tap forecaster.

    Attributes:
        min_training_samples: Minimum total inferred ``V_tap`` samples [-] required
            before a recurring profile is accepted for training.
        min_samples_per_hour: Minimum samples per UTC hour [-] required before that
            hour's mean is used in the forecast; under-sampled hours return 0.0.
        max_history_gap_hours: Maximum allowed time gap [h] between consecutive
            telemetry buckets that still counts as one continuous sequence.
        max_implied_tap_m3_per_h: Physical upper clamp [m³/h] applied to each
            individually inferred ``V_tap`` sample before hourly aggregation.
        min_peak_to_background_ratio: Minimum ratio of the hourly-profile peak to the
            median of all sampled hours [-]. Profiles whose peak is not at least this
            factor above the median background are treated as flat calibration
            artifacts (caused by ``R_loss`` or sensor bias errors) and rejected.
            A value of 2.0 means the peak hour must have at least twice the flow
            of the typical background hour.  A truly flat uniform bias yields a
            ratio of 1.0 and is always rejected; a sparse evening shower
            (e.g. 22:00 with 0.10 m³/h against a 0.02 m³/h background) yields
            a ratio of ~5.0 and is always accepted.
        min_tap_temp_diff_k: Minimum required temperature difference
            ``T_top - T_mains`` [K] for the energy-balance inference to be
            considered reliable. When ``T_top`` is nearly at mains temperature
            the denominator ``λ·(T_top - T_mains)`` is very small, making a tiny
            numerator error (e.g. 0.1 kW) produce an unrealistically large inferred
            ``V_tap``. Below this threshold the inference sample is skipped (0.0) —
            physically the tank is nearly cold and no hot-tap demand is detectable.
            Corresponds to the EKF observability condition from §12.5:
            observability of ``V_tap`` degrades when ``T_top ≈ T_mains``.
    """

    min_training_samples: int = DEFAULT_DHW_TAP_MIN_TRAINING_SAMPLES
    min_samples_per_hour: int = DEFAULT_DHW_TAP_MIN_SAMPLES_PER_HOUR
    max_history_gap_hours: float = DEFAULT_DHW_TAP_MAX_HISTORY_GAP_HOURS
    max_implied_tap_m3_per_h: float = DEFAULT_DHW_TAP_MAX_IMPLIED_TAP_M3_PER_H
    min_peak_to_background_ratio: float = DEFAULT_DHW_TAP_MIN_PEAK_TO_BACKGROUND_RATIO
    min_tap_temp_diff_k: float = DEFAULT_DHW_TAP_MIN_TAP_TEMP_DIFF_K

    def __post_init__(self) -> None:
        if self.min_training_samples <= 0:
            raise ValueError("min_training_samples must be strictly positive.")
        if self.min_samples_per_hour <= 0:
            raise ValueError("min_samples_per_hour must be strictly positive.")
        if self.max_history_gap_hours <= 0.0:
            raise ValueError("max_history_gap_hours must be strictly positive.")
        if self.max_implied_tap_m3_per_h <= 0.0:
            raise ValueError("max_implied_tap_m3_per_h must be strictly positive.")
        if self.min_peak_to_background_ratio < 1.0:
            raise ValueError("min_peak_to_background_ratio must be >= 1.0.")
        if self.min_tap_temp_diff_k <= 0.0:
            raise ValueError("min_tap_temp_diff_k must be strictly positive [K].")


@dataclass(frozen=True, slots=True)
class BaseloadForecastSettings:
    """Hyperparameters and fail-fast limits for the ML baseload forecaster.

    The target is the electrical household baseload proxy from telemetry,
    expressed in kW. This signal can be propagated to the UFH model as a
    time-varying ``internal_gains_forecast`` when no explicit thermal-gains
    forecast is supplied.
    """

    min_training_samples: int = DEFAULT_BASELOAD_MIN_TRAINING_SAMPLES
    max_history_gap_hours: float = DEFAULT_BASELOAD_MAX_HISTORY_GAP_HOURS
    alignment_tolerance_hours: float = DEFAULT_BASELOAD_FORECAST_ALIGNMENT_TOLERANCE_HOURS
    random_state: int = DEFAULT_BASELOAD_RANDOM_STATE
    tree_count: int = DEFAULT_BASELOAD_TREE_COUNT
    tree_max_depth: int = DEFAULT_BASELOAD_TREE_MAX_DEPTH
    min_samples_leaf: int = DEFAULT_BASELOAD_MIN_SAMPLES_LEAF

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
