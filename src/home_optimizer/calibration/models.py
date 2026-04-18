"""Calibration data models and validated settings for offline thermal-parameter learning.

This module contains the small immutable objects used by the first automatic
calibration path: fitting an effective UFH envelope model from historical
``off`` telemetry windows.

Units
-----
Temperature : °C
Capacity    : kWh/K
Resistance  : K/kW
Power       : kW
Time        : h / UTC datetimes
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

DEFAULT_OFF_MODE_NAME: str = "off"
DEFAULT_UFH_MODE_NAME: str = "ufh"
DEFAULT_DHW_MODE_NAME: str = "dhw"
DEFAULT_MAX_PAIR_DT_HOURS: float = 0.25
DEFAULT_MAX_GTI_W_PER_M2: float = 25.0
DEFAULT_ACTIVE_MAX_GTI_W_PER_M2: float = 1_500.0
DEFAULT_MAX_DEFROST_ACTIVE_FRACTION: float = 0.0
DEFAULT_MAX_BOOSTER_ACTIVE_FRACTION: float = 0.0
DEFAULT_MIN_SAMPLE_COUNT: int = 12
DEFAULT_INITIAL_TAU_HOURS: float = 24.0
DEFAULT_MIN_TAU_HOURS: float = 0.5
DEFAULT_MAX_TAU_HOURS: float = 500.0
DEFAULT_FORECAST_ALIGNMENT_TOLERANCE_HOURS: float = 1.0
DEFAULT_MIN_UFH_POWER_KW: float = 0.1
DEFAULT_DT_COMPATIBILITY_TOLERANCE_HOURS: float = 1.0 / 3600.0
DEFAULT_INITIAL_FLOOR_TEMPERATURE_OFFSET_C: float = 1.0
DEFAULT_INITIAL_ROOM_COVARIANCE_K2: float = 0.05
DEFAULT_INITIAL_FLOOR_COVARIANCE_K2: float = 4.0
DEFAULT_PROCESS_NOISE_ROOM_K2: float = 1e-4
DEFAULT_PROCESS_NOISE_FLOOR_K2: float = 1e-4
DEFAULT_MEASUREMENT_VARIANCE_K2: float = 1e-4
DEFAULT_MIN_PARAMETER_RATIO: float = 0.25
DEFAULT_MAX_PARAMETER_RATIO: float = 4.0
DEFAULT_REGULARIZATION_WEIGHT: float = 0.0
DEFAULT_REGULARIZATION_SCALE_RATIO: float = 0.25
DEFAULT_MIN_SEGMENT_SAMPLES: int = 4
DEFAULT_MIN_SEGMENT_UFH_POWER_SPAN_KW: float = 0.05
DEFAULT_MIN_SEGMENT_ROOM_TEMPERATURE_SPAN_C: float = 0.05
DEFAULT_MIN_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C: float = 0.1
DEFAULT_MIN_SEGMENT_SCORE: float = 0.0
DEFAULT_SEGMENT_SCORE_WEIGHT_SAMPLE_COUNT: float = 1.0
DEFAULT_SEGMENT_SCORE_WEIGHT_UFH_POWER_SPAN: float = 1.0
DEFAULT_SEGMENT_SCORE_WEIGHT_ROOM_TEMPERATURE_SPAN: float = 1.0
DEFAULT_SEGMENT_SCORE_WEIGHT_OUTDOOR_TEMPERATURE_SPAN: float = 1.0
DEFAULT_MIN_INITIAL_FLOOR_OFFSET_C: float = -15.0
DEFAULT_MAX_INITIAL_FLOOR_OFFSET_C: float = 15.0
DEFAULT_INITIAL_FLOOR_OFFSET_REGULARIZATION_WEIGHT: float = 0.0
DEFAULT_INITIAL_FLOOR_OFFSET_SCALE_C: float = 2.0
DEFAULT_MAX_DHW_LAYER_TEMPERATURE_SPREAD_C: float = 2.0
DEFAULT_MIN_DHW_POWER_KW: float = 0.25
DEFAULT_MIN_DHW_LAYER_TEMPERATURE_SPREAD_C: float = 3.0
DEFAULT_MAX_DHW_IMPLIED_TAP_M3_PER_H: float = 0.002
DEFAULT_MIN_DHW_SEGMENT_SAMPLES: int = 4
DEFAULT_MIN_DHW_SEGMENT_DELIVERED_ENERGY_KWH: float = 0.6
DEFAULT_MIN_DHW_SEGMENT_MEAN_LAYER_SPREAD_C: float = 3.5
DEFAULT_MIN_DHW_SEGMENT_LAYER_SPREAD_SPAN_C: float = 0.5
DEFAULT_MIN_DHW_SEGMENT_BOTTOM_TEMPERATURE_RISE_C: float = 0.5
DEFAULT_MIN_DHW_SEGMENT_TOP_TEMPERATURE_RISE_C: float = 0.1
DEFAULT_MIN_DHW_SEGMENT_SCORE: float = 0.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_SAMPLE_COUNT: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_DELIVERED_ENERGY: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_MEAN_LAYER_SPREAD: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_LAYER_SPREAD_SPAN: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_BOTTOM_TEMPERATURE_RISE: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_TOP_TEMPERATURE_RISE: float = 0.5
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_TAP_MARGIN: float = 0.5

from ..types import DHWParameters, ThermalParameters


@dataclass(frozen=True, slots=True)
class UFHCalibrationSample:
    """One identified off-mode transition sample for envelope calibration.

    Attributes:
        interval_start_utc: Start timestamp of the fitted transition [UTC].
        interval_end_utc: End timestamp of the fitted transition [UTC].
        dt_hours: Transition duration Δt [h].
        room_temperature_start_c: Room temperature at k [°C].
        room_temperature_end_c: Room temperature at k+1 [°C].
        outdoor_temperature_mean_c: Mean outdoor temperature over the interval [°C].
        gti_w_per_m2: Window GTI proxy used only for filtering low-solar windows [W/m²].
        household_elec_power_mean_kw: Mean non-HP household electric power proxy [kW].
    """

    interval_start_utc: datetime
    interval_end_utc: datetime
    dt_hours: float
    room_temperature_start_c: float
    room_temperature_end_c: float
    outdoor_temperature_mean_c: float
    gti_w_per_m2: float
    household_elec_power_mean_kw: float

    def __post_init__(self) -> None:
        if self.dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")


@dataclass(frozen=True, slots=True)
class UFHCalibrationDataset:
    """Collection of off-mode UFH samples used for batch parameter learning."""

    samples: tuple[UFHCalibrationSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("UFHCalibrationDataset requires at least one sample.")

    @property
    def sample_count(self) -> int:
        """Number of transition samples in the dataset [-]."""
        return len(self.samples)

    @property
    def start_utc(self) -> datetime:
        """Earliest timestamp in the dataset [UTC]."""
        return self.samples[0].interval_start_utc

    @property
    def end_utc(self) -> datetime:
        """Latest timestamp in the dataset [UTC]."""
        return self.samples[-1].interval_end_utc


@dataclass(frozen=True, slots=True)
class UFHOffCalibrationSettings:
    """Validated settings for the first offline UFH envelope calibrator.

    This first-stage calibrator intentionally fits only the *identifiable*
    one-state cool-down time constant during ``off`` windows:

        dT_r/dt = -(T_r - T_out) / tau_house

    With passive cool-down data alone, only the product ``tau_house = C_eff · R_ro``
    is structurally identifiable. A separate estimate of ``R_ro`` can be derived
    only if a reference effective capacity is injected.
    """

    off_mode_name: str = DEFAULT_OFF_MODE_NAME
    max_pair_dt_hours: float = DEFAULT_MAX_PAIR_DT_HOURS
    max_gti_w_per_m2: float = DEFAULT_ACTIVE_MAX_GTI_W_PER_M2
    max_defrost_active_fraction: float = DEFAULT_MAX_DEFROST_ACTIVE_FRACTION
    max_booster_active_fraction: float = DEFAULT_MAX_BOOSTER_ACTIVE_FRACTION
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT
    forecast_alignment_tolerance_hours: float = DEFAULT_FORECAST_ALIGNMENT_TOLERANCE_HOURS
    initial_tau_hours: float = DEFAULT_INITIAL_TAU_HOURS
    min_tau_hours: float = DEFAULT_MIN_TAU_HOURS
    max_tau_hours: float = DEFAULT_MAX_TAU_HOURS
    reference_c_eff_kwh_per_k: float | None = None

    def __post_init__(self) -> None:
        if not self.off_mode_name.strip():
            raise ValueError("off_mode_name must not be blank.")
        for name in (
            "max_pair_dt_hours",
            "forecast_alignment_tolerance_hours",
            "initial_tau_hours",
            "min_tau_hours",
            "max_tau_hours",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive.")
        if self.max_gti_w_per_m2 < 0.0:
            raise ValueError("max_gti_w_per_m2 must be non-negative.")
        if self.max_defrost_active_fraction < 0.0 or self.max_defrost_active_fraction > 1.0:
            raise ValueError("max_defrost_active_fraction must be in [0, 1].")
        if self.max_booster_active_fraction < 0.0 or self.max_booster_active_fraction > 1.0:
            raise ValueError("max_booster_active_fraction must be in [0, 1].")
        if self.min_sample_count < 2:
            raise ValueError("min_sample_count must be at least 2.")
        if self.min_tau_hours >= self.max_tau_hours:
            raise ValueError("min_tau_hours must be < max_tau_hours.")
        if not (self.min_tau_hours <= self.initial_tau_hours <= self.max_tau_hours):
            raise ValueError("initial_tau_hours must lie within its bounds.")
        if self.reference_c_eff_kwh_per_k is not None and self.reference_c_eff_kwh_per_k <= 0.0:
            raise ValueError("reference_c_eff_kwh_per_k must be strictly positive when provided.")


@dataclass(frozen=True, slots=True)
class UFHOffCalibrationResult:
    """Result of fitting the identifiable off-mode UFH envelope time constant."""

    tau_house_hours: float
    suggested_r_ro_k_per_kw: float | None
    reference_c_eff_kwh_per_k: float | None
    rmse_room_temperature_c: float
    max_abs_residual_c: float
    sample_count: int
    dataset_start_utc: datetime
    dataset_end_utc: datetime
    optimizer_status: str
    optimizer_cost: float

    def __post_init__(self) -> None:
        if self.tau_house_hours <= 0.0:
            raise ValueError("tau_house_hours must be strictly positive.")
        if self.suggested_r_ro_k_per_kw is not None and self.suggested_r_ro_k_per_kw <= 0.0:
            raise ValueError("suggested_r_ro_k_per_kw must be strictly positive when present.")
        if self.reference_c_eff_kwh_per_k is not None and self.reference_c_eff_kwh_per_k <= 0.0:
            raise ValueError("reference_c_eff_kwh_per_k must be strictly positive when present.")
        if self.rmse_room_temperature_c < 0.0:
            raise ValueError("rmse_room_temperature_c must be non-negative.")
        if self.max_abs_residual_c < 0.0:
            raise ValueError("max_abs_residual_c must be non-negative.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")


@dataclass(frozen=True, slots=True)
class DHWStandbyCalibrationSample:
    """One quasi-mixed standby DHW transition sample for loss calibration.

    This conservative first DHW stage uses only windows where the tank is not in
    ``dhw`` mode and the top/bottom layers remain close enough that the 2-node
    tank can be approximated as a single mixed node for standby-loss fitting.

    Attributes:
        interval_start_utc: Start timestamp of the fitted transition [UTC].
        interval_end_utc: End timestamp of the fitted transition [UTC].
        dt_hours: Transition duration Δt [h].
        t_top_start_c: Top-layer temperature at k [°C].
        t_top_end_c: Top-layer temperature at k+1 [°C].
        t_bot_start_c: Bottom-layer temperature at k [°C].
        t_bot_end_c: Bottom-layer temperature at k+1 [°C].
        boiler_ambient_mean_c: Mean boiler ambient temperature over the interval [°C].
    """

    interval_start_utc: datetime
    interval_end_utc: datetime
    dt_hours: float
    t_top_start_c: float
    t_top_end_c: float
    t_bot_start_c: float
    t_bot_end_c: float
    boiler_ambient_mean_c: float

    def __post_init__(self) -> None:
        if self.dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")


@dataclass(frozen=True, slots=True)
class DHWStandbyCalibrationDataset:
    """Collection of quasi-mixed standby DHW samples used for batch loss learning."""

    samples: tuple[DHWStandbyCalibrationSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("DHWStandbyCalibrationDataset requires at least one sample.")

    @property
    def sample_count(self) -> int:
        """Number of standby transition samples in the dataset [-]."""
        return len(self.samples)

    @property
    def start_utc(self) -> datetime:
        """Earliest timestamp in the standby dataset [UTC]."""
        return self.samples[0].interval_start_utc

    @property
    def end_utc(self) -> datetime:
        """Latest timestamp in the standby dataset [UTC]."""
        return self.samples[-1].interval_end_utc


@dataclass(frozen=True, slots=True)
class DHWStandbyCalibrationSettings:
    """Validated settings for the first identifiable DHW standby-loss calibrator.

    The fitted model is derived from the full 2-node DHW energy balance (§9.5)
    under the conservative assumptions used by this stage:

    * no DHW heating input over the interval (``hp_mode_last != dhw_mode_name``)
    * quasi-mixed tank: ``T_top ≈ T_bot`` at both interval boundaries
    * no explicit tap-flow fit in this stage

    Under these assumptions the total tank energy follows the identifiable
    one-state standby envelope:

        dT_dhw/dt = -(T_dhw - T_amb) / tau_standby

    with ``tau_standby = (C_top + C_bot) · R_loss / 2``.  Therefore this first
    stage fits only ``tau_standby`` and derives ``R_loss`` from the injected
    layer capacities.
    """

    dt_hours: float
    reference_c_top_kwh_per_k: float
    reference_c_bot_kwh_per_k: float
    dhw_mode_name: str = DEFAULT_DHW_MODE_NAME
    max_pair_dt_hours: float = DEFAULT_MAX_PAIR_DT_HOURS
    dt_compatibility_tolerance_hours: float = DEFAULT_DT_COMPATIBILITY_TOLERANCE_HOURS
    max_defrost_active_fraction: float = DEFAULT_MAX_DEFROST_ACTIVE_FRACTION
    max_booster_active_fraction: float = DEFAULT_MAX_BOOSTER_ACTIVE_FRACTION
    max_layer_temperature_spread_c: float = DEFAULT_MAX_DHW_LAYER_TEMPERATURE_SPREAD_C
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT
    initial_tau_hours: float = DEFAULT_INITIAL_TAU_HOURS
    min_tau_hours: float = DEFAULT_MIN_TAU_HOURS
    max_tau_hours: float = DEFAULT_MAX_TAU_HOURS

    def __post_init__(self) -> None:
        if not self.dhw_mode_name.strip():
            raise ValueError("dhw_mode_name must not be blank.")
        for name in (
            "dt_hours",
            "reference_c_top_kwh_per_k",
            "reference_c_bot_kwh_per_k",
            "max_pair_dt_hours",
            "dt_compatibility_tolerance_hours",
            "initial_tau_hours",
            "min_tau_hours",
            "max_tau_hours",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive.")
        if self.max_defrost_active_fraction < 0.0 or self.max_defrost_active_fraction > 1.0:
            raise ValueError("max_defrost_active_fraction must be in [0, 1].")
        if self.max_booster_active_fraction < 0.0 or self.max_booster_active_fraction > 1.0:
            raise ValueError("max_booster_active_fraction must be in [0, 1].")
        if self.max_layer_temperature_spread_c < 0.0:
            raise ValueError("max_layer_temperature_spread_c must be non-negative.")
        if self.min_sample_count < 2:
            raise ValueError("min_sample_count must be at least 2.")
        if self.min_tau_hours >= self.max_tau_hours:
            raise ValueError("min_tau_hours must be < max_tau_hours.")
        if not (self.min_tau_hours <= self.initial_tau_hours <= self.max_tau_hours):
            raise ValueError("initial_tau_hours must lie within its bounds.")

    @property
    def reference_c_total_kwh_per_k(self) -> float:
        """Injected total DHW heat capacity ``C_top + C_bot`` [kWh/K]."""
        return self.reference_c_top_kwh_per_k + self.reference_c_bot_kwh_per_k


@dataclass(frozen=True, slots=True)
class DHWStandbyCalibrationResult:
    """Result of fitting the identifiable DHW standby time constant."""

    tau_standby_hours: float
    suggested_r_loss_k_per_kw: float
    reference_c_total_kwh_per_k: float
    rmse_mean_tank_temperature_c: float
    max_abs_residual_c: float
    sample_count: int
    dataset_start_utc: datetime
    dataset_end_utc: datetime
    optimizer_status: str
    optimizer_cost: float

    def __post_init__(self) -> None:
        if self.tau_standby_hours <= 0.0:
            raise ValueError("tau_standby_hours must be strictly positive.")
        if self.suggested_r_loss_k_per_kw <= 0.0:
            raise ValueError("suggested_r_loss_k_per_kw must be strictly positive.")
        if self.reference_c_total_kwh_per_k <= 0.0:
            raise ValueError("reference_c_total_kwh_per_k must be strictly positive.")
        if self.rmse_mean_tank_temperature_c < 0.0:
            raise ValueError("rmse_mean_tank_temperature_c must be non-negative.")
        if self.max_abs_residual_c < 0.0:
            raise ValueError("max_abs_residual_c must be non-negative.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")


@dataclass(frozen=True, slots=True)
class DHWActiveCalibrationSample:
    """One active DHW no-draw transition sample for stratification calibration.

    Attributes:
        interval_start_utc: Start timestamp of the replay interval [UTC].
        interval_end_utc: End timestamp of the replay interval [UTC].
        dt_hours: Replay interval duration Δt [h].
        t_top_start_c: Measured top-layer temperature at k [°C].
        t_top_end_c: Measured top-layer temperature at k+1 [°C].
        t_bot_start_c: Measured bottom-layer temperature at k [°C].
        t_bot_end_c: Measured bottom-layer temperature at k+1 [°C].
        p_dhw_mean_kw: Mean DHW thermal charging power over the interval [kW].
        t_mains_c: Mean mains-water temperature over the interval [°C].
        t_amb_c: Mean boiler ambient temperature over the interval [°C].
        implied_v_tap_m3_per_h: Tap draw implied by the total-energy balance [m³/h],
            clamped to the physically feasible set ``≥ 0`` before storage.
        segment_index: Index of the contiguous active DHW replay run this sample belongs to [-].
    """

    interval_start_utc: datetime
    interval_end_utc: datetime
    dt_hours: float
    t_top_start_c: float
    t_top_end_c: float
    t_bot_start_c: float
    t_bot_end_c: float
    p_dhw_mean_kw: float
    t_mains_c: float
    t_amb_c: float
    implied_v_tap_m3_per_h: float
    segment_index: int

    def __post_init__(self) -> None:
        if self.dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")
        if self.p_dhw_mean_kw < 0.0:
            raise ValueError("p_dhw_mean_kw must be non-negative.")
        if self.implied_v_tap_m3_per_h < 0.0:
            raise ValueError("implied_v_tap_m3_per_h must be non-negative.")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class DHWActiveCalibrationSegmentQuality:
    """Quality diagnostics for one raw contiguous active-DHW replay segment.

    Attributes:
        raw_segment_index: Zero-based index before selection/reranking [-].
        selected_segment_index: Zero-based index after selection/reranking, or
            ``None`` if the segment was dropped [-].
        selected: Whether this segment survives the quality filter [-].
        sample_count: Number of replay transitions in the raw segment [-].
        duration_hours: Total segment duration [h].
        delivered_energy_kwh: Integrated DHW charging energy over the segment [kWh].
        mean_layer_spread_c: Mean ``|T_top - T_bot|`` over the segment [°C].
        layer_spread_span_c: Range of ``|T_top - T_bot|`` over the segment [°C].
        bottom_temperature_rise_c: Net bottom-layer temperature rise over the segment [°C].
        top_temperature_rise_c: Net top-layer temperature rise over the segment [°C].
        p95_implied_v_tap_m3_per_h: 95th percentile of implied tap flow over the segment [m³/h].
        max_implied_v_tap_m3_per_h: Maximum implied tap flow over the segment [m³/h].
        score: Dimensionless quality score used for ranking [-].
    """

    raw_segment_index: int
    selected_segment_index: int | None
    selected: bool
    sample_count: int
    duration_hours: float
    delivered_energy_kwh: float
    mean_layer_spread_c: float
    layer_spread_span_c: float
    bottom_temperature_rise_c: float
    top_temperature_rise_c: float
    p95_implied_v_tap_m3_per_h: float
    max_implied_v_tap_m3_per_h: float
    score: float

    def __post_init__(self) -> None:
        if self.raw_segment_index < 0:
            raise ValueError("raw_segment_index must be non-negative.")
        if self.selected_segment_index is not None and self.selected_segment_index < 0:
            raise ValueError("selected_segment_index must be non-negative when present.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")
        if self.duration_hours <= 0.0:
            raise ValueError("duration_hours must be strictly positive.")
        for name in (
            "delivered_energy_kwh",
            "mean_layer_spread_c",
            "layer_spread_span_c",
            "p95_implied_v_tap_m3_per_h",
            "max_implied_v_tap_m3_per_h",
            "score",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.selected and self.selected_segment_index is None:
            raise ValueError("selected segments must expose selected_segment_index.")
        if not self.selected and self.selected_segment_index is not None:
            raise ValueError("Dropped segments may not expose selected_segment_index.")


@dataclass(frozen=True, slots=True)
class DHWActiveCalibrationDataset:
    """Collection of active DHW no-draw replay samples used for R_strat fitting."""

    samples: tuple[DHWActiveCalibrationSample, ...]
    segment_qualities: tuple[DHWActiveCalibrationSegmentQuality, ...] = ()

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("DHWActiveCalibrationDataset requires at least one sample.")
        segment_indices = [sample.segment_index for sample in self.samples]
        if segment_indices[0] != 0:
            raise ValueError("DHWActiveCalibrationDataset segment_index must start at 0.")
        for previous_index, next_index in zip(segment_indices, segment_indices[1:]):
            if next_index < previous_index:
                raise ValueError("segment_index must be non-decreasing across the dataset.")
            if next_index - previous_index > 1:
                raise ValueError("segment_index increments may not skip values.")
        if self.segment_qualities:
            raw_indices = [quality.raw_segment_index for quality in self.segment_qualities]
            if raw_indices != list(range(len(raw_indices))):
                raise ValueError("segment_qualities raw_segment_index values must be contiguous from 0.")
            selected_indices_from_samples = sorted(set(segment_indices))
            selected_indices_from_quality = [
                quality.selected_segment_index for quality in self.segment_qualities if quality.selected
            ]
            if selected_indices_from_quality != selected_indices_from_samples:
                raise ValueError(
                    "Selected segment indices in segment_qualities must match dataset sample.segment_index values."
                )

    @property
    def sample_count(self) -> int:
        """Number of replay samples available for active DHW calibration [-]."""
        return len(self.samples)

    @property
    def start_utc(self) -> datetime:
        """Earliest timestamp in the active DHW replay dataset [UTC]."""
        return self.samples[0].interval_start_utc

    @property
    def end_utc(self) -> datetime:
        """Latest timestamp in the active DHW replay dataset [UTC]."""
        return self.samples[-1].interval_end_utc

    @property
    def segment_count(self) -> int:
        """Number of contiguous active DHW replay runs represented in the dataset [-]."""
        return self.samples[-1].segment_index + 1

    @property
    def raw_segment_count(self) -> int:
        """Number of raw contiguous active-DHW runs evaluated before quality selection [-]."""
        return len(self.segment_qualities) if self.segment_qualities else self.segment_count

    @property
    def dropped_segment_count(self) -> int:
        """Number of raw active-DHW segments rejected by the quality-selection policy [-]."""
        return self.raw_segment_count - self.segment_count


@dataclass(frozen=True, slots=True)
class DHWActiveCalibrationSettings:
    """Validated settings for active DHW stratification identification.

    This stage fits only ``R_strat`` from contiguous DHW charging windows that are
    filtered to look like no-draw events.  ``C_top``, ``C_bot``, ``R_loss`` and
    ``lambda_water`` remain fixed from the injected reference parameter object.

    The fitter replays the 2-state DHW model with ``V_tap = 0`` and minimises the
    one-step residuals on both ``T_top`` and ``T_bot``.
    """

    reference_parameters: DHWParameters
    active_mode_name: str = DEFAULT_DHW_MODE_NAME
    max_pair_dt_hours: float = DEFAULT_MAX_PAIR_DT_HOURS
    dt_compatibility_tolerance_hours: float = DEFAULT_DT_COMPATIBILITY_TOLERANCE_HOURS
    max_defrost_active_fraction: float = DEFAULT_MAX_DEFROST_ACTIVE_FRACTION
    max_booster_active_fraction: float = DEFAULT_MAX_BOOSTER_ACTIVE_FRACTION
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT
    min_segment_samples: int = DEFAULT_MIN_DHW_SEGMENT_SAMPLES
    min_dhw_power_kw: float = DEFAULT_MIN_DHW_POWER_KW
    min_layer_temperature_spread_c: float = DEFAULT_MIN_DHW_LAYER_TEMPERATURE_SPREAD_C
    max_implied_tap_m3_per_h: float = DEFAULT_MAX_DHW_IMPLIED_TAP_M3_PER_H
    min_segment_delivered_energy_kwh: float = DEFAULT_MIN_DHW_SEGMENT_DELIVERED_ENERGY_KWH
    min_segment_mean_layer_spread_c: float = DEFAULT_MIN_DHW_SEGMENT_MEAN_LAYER_SPREAD_C
    min_segment_layer_spread_span_c: float = DEFAULT_MIN_DHW_SEGMENT_LAYER_SPREAD_SPAN_C
    min_segment_bottom_temperature_rise_c: float = DEFAULT_MIN_DHW_SEGMENT_BOTTOM_TEMPERATURE_RISE_C
    min_segment_top_temperature_rise_c: float = DEFAULT_MIN_DHW_SEGMENT_TOP_TEMPERATURE_RISE_C
    min_segment_score: float = DEFAULT_MIN_DHW_SEGMENT_SCORE
    max_selected_segments: int | None = None
    segment_score_weight_sample_count: float = DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_SAMPLE_COUNT
    segment_score_weight_delivered_energy: float = DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_DELIVERED_ENERGY
    segment_score_weight_mean_layer_spread: float = DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_MEAN_LAYER_SPREAD
    segment_score_weight_layer_spread_span: float = DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_LAYER_SPREAD_SPAN
    segment_score_weight_bottom_temperature_rise: float = DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_BOTTOM_TEMPERATURE_RISE
    segment_score_weight_top_temperature_rise: float = DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_TOP_TEMPERATURE_RISE
    segment_score_weight_tap_margin: float = DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_TAP_MARGIN
    min_parameter_ratio: float = DEFAULT_MIN_PARAMETER_RATIO
    max_parameter_ratio: float = DEFAULT_MAX_PARAMETER_RATIO
    regularization_weight: float = DEFAULT_REGULARIZATION_WEIGHT
    regularization_scale_ratio: float = DEFAULT_REGULARIZATION_SCALE_RATIO

    def __post_init__(self) -> None:
        if not self.active_mode_name.strip():
            raise ValueError("active_mode_name must not be blank.")
        for name in (
            "max_pair_dt_hours",
            "dt_compatibility_tolerance_hours",
            "min_dhw_power_kw",
            "min_layer_temperature_spread_c",
            "max_implied_tap_m3_per_h",
            "min_segment_delivered_energy_kwh",
            "min_segment_mean_layer_spread_c",
            "min_segment_layer_spread_span_c",
            "min_segment_bottom_temperature_rise_c",
            "min_segment_top_temperature_rise_c",
            "min_parameter_ratio",
            "max_parameter_ratio",
            "regularization_scale_ratio",
            "segment_score_weight_sample_count",
            "segment_score_weight_delivered_energy",
            "segment_score_weight_mean_layer_spread",
            "segment_score_weight_layer_spread_span",
            "segment_score_weight_bottom_temperature_rise",
            "segment_score_weight_top_temperature_rise",
            "segment_score_weight_tap_margin",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive.")
        if self.min_segment_score < 0.0:
            raise ValueError("min_segment_score must be non-negative.")
        if self.max_defrost_active_fraction < 0.0 or self.max_defrost_active_fraction > 1.0:
            raise ValueError("max_defrost_active_fraction must be in [0, 1].")
        if self.max_booster_active_fraction < 0.0 or self.max_booster_active_fraction > 1.0:
            raise ValueError("max_booster_active_fraction must be in [0, 1].")
        if self.min_sample_count < 2:
            raise ValueError("min_sample_count must be at least 2.")
        if self.min_segment_samples < 2:
            raise ValueError("min_segment_samples must be at least 2.")
        if self.min_segment_samples > self.min_sample_count:
            raise ValueError("min_segment_samples must be <= min_sample_count.")
        if self.max_selected_segments is not None and self.max_selected_segments <= 0:
            raise ValueError("max_selected_segments must be strictly positive when provided.")
        if self.min_parameter_ratio >= self.max_parameter_ratio:
            raise ValueError("min_parameter_ratio must be < max_parameter_ratio.")
        if self.regularization_weight < 0.0:
            raise ValueError("regularization_weight must be non-negative.")


@dataclass(frozen=True, slots=True)
class DHWActiveCalibrationResult:
    """Result of fitting active DHW stratification parameters from no-draw charging runs."""

    fitted_parameters: DHWParameters
    rmse_t_top_c: float
    rmse_t_bot_c: float
    max_abs_residual_c: float
    sample_count: int
    segment_count: int
    dataset_start_utc: datetime
    dataset_end_utc: datetime
    optimizer_status: str
    optimizer_cost: float

    def __post_init__(self) -> None:
        if self.rmse_t_top_c < 0.0:
            raise ValueError("rmse_t_top_c must be non-negative.")
        if self.rmse_t_bot_c < 0.0:
            raise ValueError("rmse_t_bot_c must be non-negative.")
        if self.max_abs_residual_c < 0.0:
            raise ValueError("max_abs_residual_c must be non-negative.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")
        if self.segment_count <= 0:
            raise ValueError("segment_count must be strictly positive.")


@dataclass(frozen=True, slots=True)
class UFHActiveCalibrationSegmentQuality:
    """Quality diagnostics for one raw contiguous UFH replay segment.

    Attributes:
        raw_segment_index: Zero-based index before selection/reranking [-].
        selected_segment_index: Zero-based index after selection/reranking, or
            ``None`` if the segment was dropped [-].
        selected: Whether this segment survives the quality filter [-].
        sample_count: Number of replay transitions in the raw segment [-].
        duration_hours: Total segment duration [h].
        ufh_power_span_kw: Range of UFH thermal power over the segment [kW].
        room_temperature_span_c: Range of room temperature over the segment [°C].
        outdoor_temperature_span_c: Range of outdoor temperature over the segment [°C].
        mean_gti_w_per_m2: Mean GTI over the segment [W/m²].
        score: Dimensionless quality score used for ranking [-].
    """

    raw_segment_index: int
    selected_segment_index: int | None
    selected: bool
    sample_count: int
    duration_hours: float
    ufh_power_span_kw: float
    room_temperature_span_c: float
    outdoor_temperature_span_c: float
    mean_gti_w_per_m2: float
    score: float

    def __post_init__(self) -> None:
        if self.raw_segment_index < 0:
            raise ValueError("raw_segment_index must be non-negative.")
        if self.selected_segment_index is not None and self.selected_segment_index < 0:
            raise ValueError("selected_segment_index must be non-negative when present.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")
        if self.duration_hours <= 0.0:
            raise ValueError("duration_hours must be strictly positive.")
        if self.ufh_power_span_kw < 0.0:
            raise ValueError("ufh_power_span_kw must be non-negative.")
        if self.room_temperature_span_c < 0.0:
            raise ValueError("room_temperature_span_c must be non-negative.")
        if self.outdoor_temperature_span_c < 0.0:
            raise ValueError("outdoor_temperature_span_c must be non-negative.")
        if self.mean_gti_w_per_m2 < 0.0:
            raise ValueError("mean_gti_w_per_m2 must be non-negative.")
        if self.score < 0.0:
            raise ValueError("score must be non-negative.")
        if self.selected and self.selected_segment_index is None:
            raise ValueError("selected segments must expose selected_segment_index.")
        if not self.selected and self.selected_segment_index is not None:
            raise ValueError("Dropped segments may not expose selected_segment_index.")


@dataclass(frozen=True, slots=True)
class UFHActiveCalibrationSample:
    """One active UFH transition sample for batch RC calibration.

    Attributes:
        interval_start_utc: Start timestamp of the replay interval [UTC].
        interval_end_utc: End timestamp of the replay interval [UTC].
        dt_hours: Replay interval duration Δt [h].
        room_temperature_start_c: Measured room temperature at k [°C].
        room_temperature_end_c: Measured room temperature at k+1 [°C].
        outdoor_temperature_mean_c: Mean outdoor temperature over the interval [°C].
        gti_w_per_m2: Forecast GTI aligned to the interval end [W/m²].
        internal_gain_proxy_kw: Mean household electric power proxy for Q_int [kW].
        ufh_power_mean_kw: Mean UFH thermal power over the interval [kW].
        segment_index: Index of the contiguous UFH replay run this sample belongs to [-].
    """

    interval_start_utc: datetime
    interval_end_utc: datetime
    dt_hours: float
    room_temperature_start_c: float
    room_temperature_end_c: float
    outdoor_temperature_mean_c: float
    gti_w_per_m2: float
    internal_gain_proxy_kw: float
    ufh_power_mean_kw: float
    segment_index: int

    def __post_init__(self) -> None:
        if self.dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")
        if self.ufh_power_mean_kw < 0.0:
            raise ValueError("ufh_power_mean_kw must be non-negative.")
        if self.gti_w_per_m2 < 0.0:
            raise ValueError("gti_w_per_m2 must be non-negative.")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class UFHActiveCalibrationDataset:
    """Collection of active UFH replay samples used for 2-state RC fitting."""

    samples: tuple[UFHActiveCalibrationSample, ...]
    segment_qualities: tuple[UFHActiveCalibrationSegmentQuality, ...] = ()

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("UFHActiveCalibrationDataset requires at least one sample.")
        segment_indices = [sample.segment_index for sample in self.samples]
        if segment_indices[0] != 0:
            raise ValueError("UFHActiveCalibrationDataset segment_index must start at 0.")
        for previous_index, next_index in zip(segment_indices, segment_indices[1:]):
            if next_index < previous_index:
                raise ValueError("segment_index must be non-decreasing across the dataset.")
            if next_index - previous_index > 1:
                raise ValueError("segment_index increments may not skip values.")
        if self.segment_qualities:
            raw_indices = [quality.raw_segment_index for quality in self.segment_qualities]
            if raw_indices != list(range(len(raw_indices))):
                raise ValueError("segment_qualities raw_segment_index values must be contiguous from 0.")
            selected_indices_from_samples = sorted(set(segment_indices))
            selected_indices_from_quality = [
                quality.selected_segment_index for quality in self.segment_qualities if quality.selected
            ]
            if selected_indices_from_quality != selected_indices_from_samples:
                raise ValueError(
                    "Selected segment indices in segment_qualities must match dataset sample.segment_index values."
                )

    @property
    def sample_count(self) -> int:
        """Number of replay samples available for active UFH calibration [-]."""
        return len(self.samples)

    @property
    def start_utc(self) -> datetime:
        """Earliest timestamp in the active replay dataset [UTC]."""
        return self.samples[0].interval_start_utc

    @property
    def end_utc(self) -> datetime:
        """Latest timestamp in the active replay dataset [UTC]."""
        return self.samples[-1].interval_end_utc

    @property
    def segment_count(self) -> int:
        """Number of contiguous UFH replay runs represented in the dataset [-]."""
        return self.samples[-1].segment_index + 1

    @property
    def raw_segment_count(self) -> int:
        """Number of raw contiguous UFH runs evaluated before quality selection [-]."""
        return len(self.segment_qualities) if self.segment_qualities else self.segment_count

    @property
    def dropped_segment_count(self) -> int:
        """Number of raw segments rejected by the quality-selection policy [-]."""
        return self.raw_segment_count - self.segment_count


@dataclass(frozen=True, slots=True)
class UFHActiveCalibrationSettings:
    """Validated settings for active UFH RC identification.

    The active stage replays the full 2-state UFH dynamics with the existing
    :class:`home_optimizer.kalman.UFHKalmanFilter` and learns the physical
    parameters from measured room-temperature innovations.

    Attributes:
        reference_parameters: Reference UFH parameters. ``dt_hours``, ``alpha``,
            ``eta`` and ``A_glass`` are held fixed; ``C_r`` is reused unless
            ``fit_c_r`` is enabled. Units: see :class:`ThermalParameters`.
        active_mode_name: Telemetry mode name indicating active UFH operation [-].
        max_pair_dt_hours: Maximum allowed spacing between consecutive telemetry
            rows used as a replay sample [h].
        dt_compatibility_tolerance_hours: Allowed mismatch between telemetry Δt and
            ``reference_parameters.dt_hours`` [h].
        max_defrost_active_fraction: Maximum tolerated defrost fraction per bucket [-].
        max_booster_active_fraction: Maximum tolerated booster fraction per bucket [-].
        forecast_alignment_tolerance_hours: Max UTC mismatch between forecast and
            telemetry timestamps for GTI lookup [h].
        min_sample_count: Minimum replay samples required before fitting [-].
        min_segment_samples: Minimum replay samples per contiguous UFH run [-].
        min_segment_ufh_power_span_kw: Minimum UFH power range inside a segment [kW].
        min_segment_room_temperature_span_c: Minimum room-temperature range inside a segment [°C].
        min_segment_outdoor_temperature_span_c: Minimum outdoor-temperature range inside a segment [°C].
        min_segment_score: Minimum dimensionless quality score required for selection [-].
        max_selected_segments: Optional cap on the number of retained segments [-].
            ``None`` keeps all segments that pass the score thresholds.
        segment_score_weight_sample_count: Weight on segment length contribution [-].
        segment_score_weight_ufh_power_span: Weight on UFH power variation contribution [-].
        segment_score_weight_room_temperature_span: Weight on room-temperature variation [-].
        segment_score_weight_outdoor_temperature_span: Weight on outdoor-temperature variation [-].
        min_ufh_power_kw: Minimum mean UFH power to keep a bucket in the active set [kW].
        fit_c_r: Whether to fit ``C_r`` in addition to ``C_b``, ``R_br`` and ``R_ro`` [-].
        fit_initial_floor_temperature_offset: Whether to fit a global nuisance
            offset applied to every segment start for the hidden floor state [-].
        initial_floor_temperature_offset_c: Initial floor-state guess relative to the
            first measured room temperature of each replay segment [°C].
        min_initial_floor_temperature_offset_c: Lower bound on the fitted or fixed
            initial floor offset [°C].
        max_initial_floor_temperature_offset_c: Upper bound on the fitted or fixed
            initial floor offset [°C].
        initial_room_covariance_k2: Initial Kalman variance for ``T_r`` [K²].
        initial_floor_covariance_k2: Initial Kalman variance for ``T_b`` [K²].
        process_noise_room_k2: Kalman process noise variance for ``T_r`` [K²].
        process_noise_floor_k2: Kalman process noise variance for ``T_b`` [K²].
        measurement_variance_k2: Room-sensor measurement variance [K²].
        min_parameter_ratio: Lower bound for fitted parameters relative to the
            reference values [-].
        max_parameter_ratio: Upper bound for fitted parameters relative to the
            reference values [-].
        regularization_weight: Optional Tikhonov weight on deviations from the
            reference parameter set [-].
        regularization_scale_ratio: Relative scale used to normalise the optional
            regularisation residuals [-].
        initial_floor_offset_regularization_weight: Optional Tikhonov weight on the
            nuisance floor-offset parameter [-].
        initial_floor_offset_scale_c: Scale used to normalise floor-offset
            regularisation residuals [°C].
    """

    reference_parameters: ThermalParameters
    active_mode_name: str = DEFAULT_UFH_MODE_NAME
    max_pair_dt_hours: float = DEFAULT_MAX_PAIR_DT_HOURS
    dt_compatibility_tolerance_hours: float = DEFAULT_DT_COMPATIBILITY_TOLERANCE_HOURS
    max_gti_w_per_m2: float = DEFAULT_MAX_GTI_W_PER_M2
    max_defrost_active_fraction: float = DEFAULT_MAX_DEFROST_ACTIVE_FRACTION
    max_booster_active_fraction: float = DEFAULT_MAX_BOOSTER_ACTIVE_FRACTION
    forecast_alignment_tolerance_hours: float = DEFAULT_FORECAST_ALIGNMENT_TOLERANCE_HOURS
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT
    min_segment_samples: int = DEFAULT_MIN_SEGMENT_SAMPLES
    min_segment_ufh_power_span_kw: float = DEFAULT_MIN_SEGMENT_UFH_POWER_SPAN_KW
    min_segment_room_temperature_span_c: float = DEFAULT_MIN_SEGMENT_ROOM_TEMPERATURE_SPAN_C
    min_segment_outdoor_temperature_span_c: float = DEFAULT_MIN_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C
    min_segment_score: float = DEFAULT_MIN_SEGMENT_SCORE
    max_selected_segments: int | None = None
    segment_score_weight_sample_count: float = DEFAULT_SEGMENT_SCORE_WEIGHT_SAMPLE_COUNT
    segment_score_weight_ufh_power_span: float = DEFAULT_SEGMENT_SCORE_WEIGHT_UFH_POWER_SPAN
    segment_score_weight_room_temperature_span: float = DEFAULT_SEGMENT_SCORE_WEIGHT_ROOM_TEMPERATURE_SPAN
    segment_score_weight_outdoor_temperature_span: float = DEFAULT_SEGMENT_SCORE_WEIGHT_OUTDOOR_TEMPERATURE_SPAN
    min_ufh_power_kw: float = DEFAULT_MIN_UFH_POWER_KW
    fit_c_r: bool = False
    fit_initial_floor_temperature_offset: bool = False
    initial_floor_temperature_offset_c: float = DEFAULT_INITIAL_FLOOR_TEMPERATURE_OFFSET_C
    min_initial_floor_temperature_offset_c: float = DEFAULT_MIN_INITIAL_FLOOR_OFFSET_C
    max_initial_floor_temperature_offset_c: float = DEFAULT_MAX_INITIAL_FLOOR_OFFSET_C
    initial_room_covariance_k2: float = DEFAULT_INITIAL_ROOM_COVARIANCE_K2
    initial_floor_covariance_k2: float = DEFAULT_INITIAL_FLOOR_COVARIANCE_K2
    process_noise_room_k2: float = DEFAULT_PROCESS_NOISE_ROOM_K2
    process_noise_floor_k2: float = DEFAULT_PROCESS_NOISE_FLOOR_K2
    measurement_variance_k2: float = DEFAULT_MEASUREMENT_VARIANCE_K2
    min_parameter_ratio: float = DEFAULT_MIN_PARAMETER_RATIO
    max_parameter_ratio: float = DEFAULT_MAX_PARAMETER_RATIO
    regularization_weight: float = DEFAULT_REGULARIZATION_WEIGHT
    regularization_scale_ratio: float = DEFAULT_REGULARIZATION_SCALE_RATIO
    initial_floor_offset_regularization_weight: float = DEFAULT_INITIAL_FLOOR_OFFSET_REGULARIZATION_WEIGHT
    initial_floor_offset_scale_c: float = DEFAULT_INITIAL_FLOOR_OFFSET_SCALE_C

    def __post_init__(self) -> None:
        if not self.active_mode_name.strip():
            raise ValueError("active_mode_name must not be blank.")
        for name in (
            "max_pair_dt_hours",
            "dt_compatibility_tolerance_hours",
            "forecast_alignment_tolerance_hours",
            "min_ufh_power_kw",
            "min_segment_ufh_power_span_kw",
            "min_segment_room_temperature_span_c",
            "min_segment_outdoor_temperature_span_c",
            "max_initial_floor_temperature_offset_c",
            "initial_room_covariance_k2",
            "initial_floor_covariance_k2",
            "process_noise_room_k2",
            "process_noise_floor_k2",
            "measurement_variance_k2",
            "min_parameter_ratio",
            "max_parameter_ratio",
            "regularization_scale_ratio",
            "initial_floor_offset_scale_c",
            "segment_score_weight_sample_count",
            "segment_score_weight_ufh_power_span",
            "segment_score_weight_room_temperature_span",
            "segment_score_weight_outdoor_temperature_span",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive.")
        if self.max_gti_w_per_m2 < 0.0:
            raise ValueError("max_gti_w_per_m2 must be non-negative.")
        if self.min_segment_score < 0.0:
            raise ValueError("min_segment_score must be non-negative.")
        if self.max_defrost_active_fraction < 0.0 or self.max_defrost_active_fraction > 1.0:
            raise ValueError("max_defrost_active_fraction must be in [0, 1].")
        if self.max_booster_active_fraction < 0.0 or self.max_booster_active_fraction > 1.0:
            raise ValueError("max_booster_active_fraction must be in [0, 1].")
        if self.min_sample_count < 2:
            raise ValueError("min_sample_count must be at least 2.")
        if self.min_segment_samples < 2:
            raise ValueError("min_segment_samples must be at least 2.")
        if self.min_segment_samples > self.min_sample_count:
            raise ValueError("min_segment_samples must be <= min_sample_count.")
        if self.max_selected_segments is not None and self.max_selected_segments <= 0:
            raise ValueError("max_selected_segments must be strictly positive when provided.")
        if self.min_parameter_ratio >= self.max_parameter_ratio:
            raise ValueError("min_parameter_ratio must be < max_parameter_ratio.")
        if self.regularization_weight < 0.0:
            raise ValueError("regularization_weight must be non-negative.")
        if self.initial_floor_temperature_offset_c < self.min_initial_floor_temperature_offset_c:
            raise ValueError(
                "initial_floor_temperature_offset_c must be >= min_initial_floor_temperature_offset_c."
            )
        if self.initial_floor_temperature_offset_c > self.max_initial_floor_temperature_offset_c:
            raise ValueError(
                "initial_floor_temperature_offset_c must be <= max_initial_floor_temperature_offset_c."
            )
        if self.min_initial_floor_temperature_offset_c >= self.max_initial_floor_temperature_offset_c:
            raise ValueError(
                "min_initial_floor_temperature_offset_c must be < max_initial_floor_temperature_offset_c."
            )
        if self.initial_floor_offset_regularization_weight < 0.0:
            raise ValueError("initial_floor_offset_regularization_weight must be non-negative.")


@dataclass(frozen=True, slots=True)
class UFHActiveCalibrationResult:
    """Result of fitting active UFH RC parameters with Kalman replay."""

    fitted_parameters: ThermalParameters
    fit_c_r: bool
    fit_initial_floor_temperature_offset: bool
    fitted_initial_floor_temperature_offset_c: float
    rmse_room_temperature_c: float
    max_abs_innovation_c: float
    sample_count: int
    segment_count: int
    dataset_start_utc: datetime
    dataset_end_utc: datetime
    optimizer_status: str
    optimizer_cost: float

    def __post_init__(self) -> None:
        if self.segment_count <= 0:
            raise ValueError("segment_count must be strictly positive.")
        if self.rmse_room_temperature_c < 0.0:
            raise ValueError("rmse_room_temperature_c must be non-negative.")
        if self.max_abs_innovation_c < 0.0:
            raise ValueError("max_abs_innovation_c must be non-negative.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")


