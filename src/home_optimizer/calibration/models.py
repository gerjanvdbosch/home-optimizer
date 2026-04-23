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
DEFAULT_FIT_ETA: bool = False
DEFAULT_MIN_ETA: float = 0.0
DEFAULT_MAX_ETA: float = 1.0
DEFAULT_INITIAL_INTERNAL_GAINS_HEAT_FRACTION: float = 0.7
DEFAULT_MIN_INTERNAL_GAINS_HEAT_FRACTION: float = 0.0
DEFAULT_MAX_INTERNAL_GAINS_HEAT_FRACTION: float = 1.0
DEFAULT_FIT_INTERNAL_GAINS_HEAT_FRACTION: bool = False
DEFAULT_FIT_ROOM_TEMPERATURE_BIAS: bool = False
DEFAULT_INITIAL_ROOM_TEMPERATURE_BIAS_C: float = 0.0
DEFAULT_MIN_ROOM_TEMPERATURE_BIAS_C: float = -5.0
DEFAULT_MAX_ROOM_TEMPERATURE_BIAS_C: float = 5.0
DEFAULT_MIN_INITIAL_FLOOR_OFFSET_C: float = -15.0
DEFAULT_MAX_INITIAL_FLOOR_OFFSET_C: float = 15.0
DEFAULT_INITIAL_FLOOR_OFFSET_REGULARIZATION_WEIGHT: float = 0.0
DEFAULT_INITIAL_FLOOR_OFFSET_SCALE_C: float = 2.0
DEFAULT_MAX_DHW_LAYER_TEMPERATURE_SPREAD_C: float = 2.0
DEFAULT_MIN_DHW_POWER_KW: float = 0.25
DEFAULT_MIN_DHW_LAYER_TEMPERATURE_SPREAD_C: float = 0.5
DEFAULT_MAX_DHW_IMPLIED_TAP_M3_PER_H: float = 0.1
DEFAULT_MIN_DHW_ACTIVE_SAMPLE_COUNT: int = 8
DEFAULT_MIN_DHW_SEGMENT_SAMPLES: int = 3
DEFAULT_MIN_DHW_SEGMENT_DELIVERED_ENERGY_KWH: float = 0.6
DEFAULT_MIN_DHW_SEGMENT_MEAN_LAYER_SPREAD_C: float = 0.75
DEFAULT_MIN_DHW_SEGMENT_LAYER_SPREAD_SPAN_C: float = 0.5
DEFAULT_MIN_DHW_SEGMENT_BOTTOM_TEMPERATURE_RISE_C: float = 0.5
DEFAULT_MIN_DHW_SEGMENT_TOP_TEMPERATURE_RISE_C: float = 0.1
DEFAULT_MIN_DHW_SEGMENT_SCORE: float = 0.0
DEFAULT_MIN_DHW_R_STRAT_K_PER_KW: float = 1e-3
DEFAULT_MAX_DHW_R_STRAT_K_PER_KW: float = 50.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_SAMPLE_COUNT: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_DELIVERED_ENERGY: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_MEAN_LAYER_SPREAD: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_LAYER_SPREAD_SPAN: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_BOTTOM_TEMPERATURE_RISE: float = 1.0
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_TOP_TEMPERATURE_RISE: float = 0.5
DEFAULT_DHW_SEGMENT_SCORE_WEIGHT_TAP_MARGIN: float = 0.5
DEFAULT_FIT_DHW_AMBIENT_TEMPERATURE_BIAS: bool = False
DEFAULT_INITIAL_DHW_AMBIENT_TEMPERATURE_BIAS_C: float = 0.0
DEFAULT_MIN_DHW_AMBIENT_TEMPERATURE_BIAS_C: float = -5.0
DEFAULT_MAX_DHW_AMBIENT_TEMPERATURE_BIAS_C: float = 5.0
DEFAULT_FIT_DHW_CAPACITY_SPLIT: bool = False
DEFAULT_FIT_DHW_TOTAL_CAPACITY: bool = False
DEFAULT_MIN_DHW_C_TOP_FRACTION: float = 0.1
DEFAULT_MAX_DHW_C_TOP_FRACTION: float = 0.9
DEFAULT_MIN_DHW_C_TOTAL_SCALE: float = 0.25
DEFAULT_MAX_DHW_C_TOTAL_SCALE: float = 4.0
DEFAULT_FIT_DHW_TEMPERATURE_BIASES: bool = False
DEFAULT_INITIAL_DHW_TOP_TEMPERATURE_BIAS_C: float = 0.0
DEFAULT_INITIAL_DHW_BOTTOM_TEMPERATURE_BIAS_C: float = 0.0
DEFAULT_MIN_DHW_TEMPERATURE_BIAS_C: float = -5.0
DEFAULT_MAX_DHW_TEMPERATURE_BIAS_C: float = 5.0

from ..domain.heat_pump.cop import HeatPumpCOPParameters
from ..types import DHWParameters, ThermalParameters

DEFAULT_INITIAL_ETA_CARNOT: float = 0.45
DEFAULT_MIN_ETA_CARNOT: float = 0.1
DEFAULT_MAX_ETA_CARNOT: float = 0.99
DEFAULT_INITIAL_T_SUPPLY_MIN_C: float = 25.0
DEFAULT_MIN_T_SUPPLY_MIN_C: float = 15.0
DEFAULT_MAX_T_SUPPLY_MIN_C: float = 45.0
DEFAULT_INITIAL_HEATING_CURVE_SLOPE: float = 1.0
DEFAULT_MIN_HEATING_CURVE_SLOPE: float = 0.0
DEFAULT_MAX_HEATING_CURVE_SLOPE: float = 3.0
DEFAULT_MIN_T_REF_OUTDOOR_C: float = 5.0
DEFAULT_MAX_T_REF_OUTDOOR_C: float = 25.0
DEFAULT_FIT_T_REF_OUTDOOR: bool = True
DEFAULT_MIN_UFH_T_REF_SIDE_SAMPLE_COUNT: int = 2
DEFAULT_MIN_THERMAL_ENERGY_KWH: float = 0.05
DEFAULT_MIN_ELECTRIC_ENERGY_KWH: float = 0.01
DEFAULT_MIN_UFH_CURVE_SAMPLE_COUNT: int = 8
DEFAULT_T_REF_OUTDOOR_C: float = 18.0
DEFAULT_DELTA_T_COND_K: float = 5.0
DEFAULT_DELTA_T_EVAP_K: float = 5.0
DEFAULT_COP_MIN: float = 1.5
DEFAULT_COP_MAX: float = 7.0
DEFAULT_MIN_COP_SEGMENT_SAMPLES: int = 3
DEFAULT_MIN_COP_SEGMENT_THERMAL_ENERGY_KWH: float = 0.3
DEFAULT_MIN_COP_SEGMENT_ACTUAL_COP_SPAN: float = 0.2
DEFAULT_MAX_COP_SEGMENT_SUPPLY_TRACKING_RMSE_C: float = 4.0
DEFAULT_MIN_UFH_COP_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C: float = 1.0
DEFAULT_MIN_UFH_COP_SEGMENT_SUPPLY_TARGET_SPAN_C: float = 1.0
DEFAULT_MIN_COP_SEGMENT_SCORE: float = 0.0
DEFAULT_COP_SEGMENT_SCORE_WEIGHT_SAMPLE_COUNT: float = 1.0
DEFAULT_COP_SEGMENT_SCORE_WEIGHT_THERMAL_ENERGY: float = 1.0
DEFAULT_COP_SEGMENT_SCORE_WEIGHT_ACTUAL_COP_SPAN: float = 1.0
DEFAULT_COP_SEGMENT_SCORE_WEIGHT_OUTDOOR_TEMPERATURE_SPAN: float = 1.0
DEFAULT_COP_SEGMENT_SCORE_WEIGHT_SUPPLY_TARGET_SPAN: float = 1.0
DEFAULT_COP_SEGMENT_SCORE_WEIGHT_SUPPLY_TRACKING: float = 0.5
DEFAULT_COP_REAGGREGATE_MIN_ELECTRIC_ENERGY_KWH: float = 0.3
DEFAULT_COP_REAGGREGATE_MIN_BUCKET_COUNT: int = 2
DEFAULT_COP_MAX_SEGMENT_BOUNDARY_GAP_RATIO: float = 0.1
COP_LEAST_SQUARES_LOSS_CHOICES: tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy", "arctan")
DEFAULT_COP_HEATING_CURVE_LOSS_NAME: str = "soft_l1"
DEFAULT_COP_ETA_LOSS_NAME: str = "soft_l1"
DEFAULT_COP_HEATING_CURVE_LOSS_SCALE_C: float = 1.0
DEFAULT_COP_ETA_LOSS_SCALE_KWH: float = 0.05
DEFAULT_AUTOMATIC_CALIBRATION_MIN_HISTORY_HOURS: float = 24.0
DEFAULT_AUTOMATIC_UFH_MIN_SELECTED_SEGMENTS: int = 2
DEFAULT_AUTOMATIC_UFH_BOUND_TOLERANCE_RATIO: float = 1e-6
DEFAULT_AUTOMATIC_UFH_MAX_R_RO_MISMATCH_RATIO: float = 4.0
DEFAULT_AUTOMATIC_UFH_MIN_PARAMETER_RATIO: float = 0.5
DEFAULT_AUTOMATIC_UFH_MAX_PARAMETER_RATIO: float = 2.0
DEFAULT_AUTOMATIC_UFH_REGULARIZATION_WEIGHT: float = 1.0
DEFAULT_AUTOMATIC_DHW_ACTIVE_MIN_SELECTED_SEGMENTS: int = 2
DEFAULT_AUTOMATIC_DHW_STANDBY_BOUND_TOLERANCE_RATIO: float = 1e-6
DEFAULT_AUTOMATIC_DHW_STANDBY_FIT_AMBIENT_TEMPERATURE_BIAS: bool = False
DEFAULT_AUTOMATIC_DHW_ACTIVE_BOUND_TOLERANCE_RATIO: float = 1e-6
DEFAULT_AUTOMATIC_COP_MIN_DHW_SAMPLE_COUNT: int = 1


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
    max_gti_w_per_m2: float = DEFAULT_MAX_GTI_W_PER_M2
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
    fit_ambient_temperature_bias: bool = DEFAULT_FIT_DHW_AMBIENT_TEMPERATURE_BIAS
    initial_ambient_temperature_bias_c: float = DEFAULT_INITIAL_DHW_AMBIENT_TEMPERATURE_BIAS_C
    min_ambient_temperature_bias_c: float = DEFAULT_MIN_DHW_AMBIENT_TEMPERATURE_BIAS_C
    max_ambient_temperature_bias_c: float = DEFAULT_MAX_DHW_AMBIENT_TEMPERATURE_BIAS_C

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
        if self.min_ambient_temperature_bias_c >= self.max_ambient_temperature_bias_c:
            raise ValueError("min_ambient_temperature_bias_c must be < max_ambient_temperature_bias_c.")
        if not (
            self.min_ambient_temperature_bias_c
            <= self.initial_ambient_temperature_bias_c
            <= self.max_ambient_temperature_bias_c
        ):
            raise ValueError("initial_ambient_temperature_bias_c must lie within its bounds.")

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
    fit_ambient_temperature_bias: bool
    fitted_ambient_temperature_bias_c: float
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
class COPCalibrationSample:
    """One filtered heat-pump operating sample for offline COP calibration.

    Attributes:
        bucket_start_utc: Start timestamp of the telemetry sample/window [UTC].
        bucket_end_utc: End timestamp of the telemetry sample/window [UTC].
        dt_hours: Sample/window duration Δt [h].
        mode_name: Heat-pump mode label (typically ``ufh`` or ``dhw``) [-].
        outdoor_temperature_mean_c: Δt-weighted mean outdoor temperature over the sample/window [°C].
        supply_target_temperature_mean_c: Mean commanded/target supply temperature [°C].
        supply_temperature_mean_c: Mean measured hydraulic supply temperature [°C].
        thermal_energy_kwh: Thermal energy delivered during the sample/window [kWh].
        electric_energy_kwh: Electrical energy consumed during the sample/window [kWh].
        source_bucket_count: Number of persisted telemetry buckets merged into this calibration sample [-].
    """

    bucket_start_utc: datetime
    bucket_end_utc: datetime
    dt_hours: float
    mode_name: str
    outdoor_temperature_mean_c: float
    supply_target_temperature_mean_c: float
    supply_temperature_mean_c: float
    thermal_energy_kwh: float
    electric_energy_kwh: float
    source_bucket_count: int = 1

    def __post_init__(self) -> None:
        if self.dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")
        if not self.mode_name.strip():
            raise ValueError("mode_name must not be blank.")
        if self.thermal_energy_kwh <= 0.0:
            raise ValueError("thermal_energy_kwh must be strictly positive.")
        if self.electric_energy_kwh <= 0.0:
            raise ValueError("electric_energy_kwh must be strictly positive.")
        if self.source_bucket_count <= 0:
            raise ValueError("source_bucket_count must be strictly positive.")

    @property
    def actual_cop(self) -> float:
        """Measured bucket COP = thermal energy / electrical energy [-]."""
        return self.thermal_energy_kwh / self.electric_energy_kwh


@dataclass(frozen=True, slots=True)
class COPCalibrationSegmentQuality:
    """Quality diagnostics for one raw contiguous COP calibration segment."""

    raw_segment_index: int
    mode_name: str
    selected: bool
    sample_count: int
    duration_hours: float
    thermal_energy_kwh: float
    electric_energy_kwh: float
    outdoor_temperature_span_c: float
    supply_target_temperature_span_c: float
    actual_cop_span: float
    supply_tracking_rmse_c: float
    score: float

    def __post_init__(self) -> None:
        if self.raw_segment_index < 0:
            raise ValueError("raw_segment_index must be non-negative.")
        if not self.mode_name.strip():
            raise ValueError("mode_name must not be blank.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")
        if self.duration_hours <= 0.0:
            raise ValueError("duration_hours must be strictly positive.")
        for name in (
            "thermal_energy_kwh",
            "electric_energy_kwh",
            "outdoor_temperature_span_c",
            "supply_target_temperature_span_c",
            "actual_cop_span",
            "supply_tracking_rmse_c",
            "score",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True, slots=True)
class COPCalibrationDataset:
    """Collection of filtered operating buckets used for offline COP calibration."""

    samples: tuple[COPCalibrationSample, ...]
    segment_qualities: tuple[COPCalibrationSegmentQuality, ...] = ()

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("COPCalibrationDataset requires at least one sample.")

    @property
    def sample_count(self) -> int:
        """Number of operating buckets in the COP calibration dataset [-]."""
        return len(self.samples)

    @property
    def ufh_sample_count(self) -> int:
        """Number of UFH-mode buckets in the dataset [-]."""
        return sum(sample.mode_name == DEFAULT_UFH_MODE_NAME for sample in self.samples)

    @property
    def dhw_sample_count(self) -> int:
        """Number of DHW-mode buckets in the dataset [-]."""
        return sum(sample.mode_name == DEFAULT_DHW_MODE_NAME for sample in self.samples)

    @property
    def start_utc(self) -> datetime:
        """Earliest timestamp in the COP calibration dataset [UTC]."""
        return self.samples[0].bucket_start_utc

    @property
    def end_utc(self) -> datetime:
        """Latest timestamp in the COP calibration dataset [UTC]."""
        return self.samples[-1].bucket_end_utc

    @property
    def raw_segment_count(self) -> int:
        """Number of raw contiguous COP segments evaluated before selection [-]."""
        return len(self.segment_qualities) if self.segment_qualities else 1

    @property
    def dropped_segment_count(self) -> int:
        """Number of raw COP segments rejected by the quality-selection policy [-]."""
        if not self.segment_qualities:
            return 0
        return sum(not quality.selected for quality in self.segment_qualities)

    @property
    def selected_segment_count(self) -> int:
        """Number of retained contiguous COP segments after quality selection [-]."""
        if not self.segment_qualities:
            return 1
        return sum(quality.selected for quality in self.segment_qualities)

    @property
    def selected_ufh_segment_count(self) -> int:
        """Number of retained UFH-mode COP segments after selection [-]."""
        return sum(
            quality.selected and quality.mode_name == DEFAULT_UFH_MODE_NAME
            for quality in self.segment_qualities
        )

    @property
    def selected_dhw_segment_count(self) -> int:
        """Number of retained DHW-mode COP segments after selection [-]."""
        return sum(
            quality.selected and quality.mode_name == DEFAULT_DHW_MODE_NAME
            for quality in self.segment_qualities
        )


@dataclass(frozen=True, slots=True)
class COPCalibrationDiagnostics:
    """Diagnostics for COP bucket filtering and segment-quality selection.

    Attributes:
        raw_row_count: Total telemetry rows evaluated before any COP filtering [-].
        mode_accepted_count: Rows surviving the UFH/DHW mode filter [-].
        defrost_accepted_count: Rows surviving the defrost-fraction filter [-].
        booster_accepted_count: Rows surviving the booster-fraction filter [-].
        dt_accepted_count: Rows with strictly positive bucket duration Δt [-].
        thermal_energy_accepted_count: Rows surviving the thermal-energy threshold [-].
        electric_energy_accepted_count: Rows whose raw electric delta individually meets the configured bucket threshold [-].
        finite_supply_accepted_count: Rows with finite target and measured supply temperatures [-].
        cop_accepted_count: Rows whose raw bucket COP individually lies within ``1 < COP <= cop_max`` [-].
        raw_segment_count: Number of contiguous same-mode raw segments formed before scoring [-].
        selected_segment_count: Number of segments retained after thresholding and optional top-N caps [-].
        selected_sample_count: Number of bucket samples retained after final segment selection [-].
        selected_ufh_sample_count: Number of retained UFH samples after final segment selection [-].
        selected_dhw_sample_count: Number of retained DHW samples after final segment selection [-].
        bucket_rejection_counts: Sorted ``(reason, count)`` pairs for bucket-level rejections [-].
        segment_failure_counts: Sorted ``(reason, count)`` pairs for segment-level rejections [-].
        segment_qualities: Final segment-quality objects after optional top-N capping [-].
    """

    raw_row_count: int
    mode_accepted_count: int
    defrost_accepted_count: int
    booster_accepted_count: int
    dt_accepted_count: int
    thermal_energy_accepted_count: int
    electric_energy_accepted_count: int
    finite_supply_accepted_count: int
    cop_accepted_count: int
    raw_segment_count: int
    selected_segment_count: int
    selected_sample_count: int
    selected_ufh_sample_count: int
    selected_dhw_sample_count: int
    bucket_rejection_counts: tuple[tuple[str, int], ...]
    segment_failure_counts: tuple[tuple[str, int], ...]
    segment_qualities: tuple[COPCalibrationSegmentQuality, ...]

    def __post_init__(self) -> None:
        for name in (
            "raw_row_count",
            "mode_accepted_count",
            "defrost_accepted_count",
            "booster_accepted_count",
            "dt_accepted_count",
            "thermal_energy_accepted_count",
            "electric_energy_accepted_count",
            "finite_supply_accepted_count",
            "cop_accepted_count",
            "raw_segment_count",
            "selected_segment_count",
            "selected_sample_count",
            "selected_ufh_sample_count",
            "selected_dhw_sample_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.selected_segment_count > self.raw_segment_count:
            raise ValueError("selected_segment_count may not exceed raw_segment_count.")
        if self.selected_ufh_sample_count + self.selected_dhw_sample_count != self.selected_sample_count:
            raise ValueError(
                "selected_ufh_sample_count + selected_dhw_sample_count must equal selected_sample_count."
            )
        for collection_name in ("bucket_rejection_counts", "segment_failure_counts"):
            for reason, count in getattr(self, collection_name):
                if not reason.strip():
                    raise ValueError(f"{collection_name} reasons must not be blank.")
                if count < 0:
                    raise ValueError(f"{collection_name} counts must be non-negative.")


@dataclass(frozen=True, slots=True)
class COPCalibrationSettings:
    """Validated settings for offline Carnot COP calibration.

    The current offline stage learns the UFH heating-curve shape plus separate
    UFH/DHW Carnot efficiency factors from historical buckets:

    * ``T_supply_min`` [°C] — UFH heating-curve intercept
    * ``T_ref_outdoor`` [°C] — UFH heating-curve balance-point / breakpoint
    * ``heating_curve_slope`` [K/K] — UFH heating-curve slope
    * ``eta_carnot_ufh`` [-] — UFH Carnot efficiency factor
    * ``eta_carnot_dhw`` [-] — DHW Carnot efficiency factor

    The approach temperatures and COP clamps stay fixed for identifiability and to
    preserve the convex MPC pre-calculation assumptions from §14.1:

    * ``delta_T_cond``
    * ``delta_T_evap``
    * ``cop_min``
    * ``cop_max``
    """

    ufh_mode_name: str = DEFAULT_UFH_MODE_NAME
    dhw_mode_name: str = DEFAULT_DHW_MODE_NAME
    max_defrost_active_fraction: float = DEFAULT_MAX_DEFROST_ACTIVE_FRACTION
    max_booster_active_fraction: float = DEFAULT_MAX_BOOSTER_ACTIVE_FRACTION
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT
    min_ufh_curve_sample_count: int = DEFAULT_MIN_UFH_CURVE_SAMPLE_COUNT
    min_thermal_energy_kwh: float = DEFAULT_MIN_THERMAL_ENERGY_KWH
    min_electric_energy_kwh: float = DEFAULT_MIN_ELECTRIC_ENERGY_KWH
    min_segment_samples: int = DEFAULT_MIN_COP_SEGMENT_SAMPLES
    min_segment_thermal_energy_kwh: float = DEFAULT_MIN_COP_SEGMENT_THERMAL_ENERGY_KWH
    min_segment_actual_cop_span: float = DEFAULT_MIN_COP_SEGMENT_ACTUAL_COP_SPAN
    max_segment_supply_tracking_rmse_c: float = DEFAULT_MAX_COP_SEGMENT_SUPPLY_TRACKING_RMSE_C
    min_ufh_segment_outdoor_temperature_span_c: float = DEFAULT_MIN_UFH_COP_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C
    min_ufh_segment_supply_target_span_c: float = DEFAULT_MIN_UFH_COP_SEGMENT_SUPPLY_TARGET_SPAN_C
    min_segment_score: float = DEFAULT_MIN_COP_SEGMENT_SCORE
    max_selected_ufh_segments: int | None = None
    max_selected_dhw_segments: int | None = None
    segment_score_weight_sample_count: float = DEFAULT_COP_SEGMENT_SCORE_WEIGHT_SAMPLE_COUNT
    segment_score_weight_thermal_energy: float = DEFAULT_COP_SEGMENT_SCORE_WEIGHT_THERMAL_ENERGY
    segment_score_weight_actual_cop_span: float = DEFAULT_COP_SEGMENT_SCORE_WEIGHT_ACTUAL_COP_SPAN
    segment_score_weight_outdoor_temperature_span: float = DEFAULT_COP_SEGMENT_SCORE_WEIGHT_OUTDOOR_TEMPERATURE_SPAN
    segment_score_weight_supply_target_span: float = DEFAULT_COP_SEGMENT_SCORE_WEIGHT_SUPPLY_TARGET_SPAN
    segment_score_weight_supply_tracking: float = DEFAULT_COP_SEGMENT_SCORE_WEIGHT_SUPPLY_TRACKING
    reaggregate_min_electric_energy_kwh: float = DEFAULT_COP_REAGGREGATE_MIN_ELECTRIC_ENERGY_KWH
    reaggregate_min_bucket_count: int = DEFAULT_COP_REAGGREGATE_MIN_BUCKET_COUNT
    max_segment_boundary_gap_ratio: float = DEFAULT_COP_MAX_SEGMENT_BOUNDARY_GAP_RATIO
    initial_eta_carnot: float = DEFAULT_INITIAL_ETA_CARNOT
    min_eta_carnot: float = DEFAULT_MIN_ETA_CARNOT
    max_eta_carnot: float = DEFAULT_MAX_ETA_CARNOT
    initial_t_supply_min_c: float = DEFAULT_INITIAL_T_SUPPLY_MIN_C
    min_t_supply_min_c: float = DEFAULT_MIN_T_SUPPLY_MIN_C
    max_t_supply_min_c: float = DEFAULT_MAX_T_SUPPLY_MIN_C
    initial_heating_curve_slope: float = DEFAULT_INITIAL_HEATING_CURVE_SLOPE
    min_heating_curve_slope: float = DEFAULT_MIN_HEATING_CURVE_SLOPE
    max_heating_curve_slope: float = DEFAULT_MAX_HEATING_CURVE_SLOPE
    fit_t_ref_outdoor: bool = DEFAULT_FIT_T_REF_OUTDOOR
    t_ref_outdoor_c: float = DEFAULT_T_REF_OUTDOOR_C
    min_t_ref_outdoor_c: float = DEFAULT_MIN_T_REF_OUTDOOR_C
    max_t_ref_outdoor_c: float = DEFAULT_MAX_T_REF_OUTDOOR_C
    min_ufh_t_ref_side_sample_count: int = DEFAULT_MIN_UFH_T_REF_SIDE_SAMPLE_COUNT
    delta_t_cond_k: float = DEFAULT_DELTA_T_COND_K
    delta_t_evap_k: float = DEFAULT_DELTA_T_EVAP_K
    cop_min: float = DEFAULT_COP_MIN
    cop_max: float = DEFAULT_COP_MAX
    heating_curve_loss_name: str = DEFAULT_COP_HEATING_CURVE_LOSS_NAME
    eta_loss_name: str = DEFAULT_COP_ETA_LOSS_NAME
    heating_curve_loss_scale_c: float = DEFAULT_COP_HEATING_CURVE_LOSS_SCALE_C
    eta_loss_scale_kwh: float = DEFAULT_COP_ETA_LOSS_SCALE_KWH

    def __post_init__(self) -> None:
        if not self.ufh_mode_name.strip():
            raise ValueError("ufh_mode_name must not be blank.")
        if not self.dhw_mode_name.strip():
            raise ValueError("dhw_mode_name must not be blank.")
        for name in (
            "min_thermal_energy_kwh",
            "min_electric_energy_kwh",
            "min_segment_thermal_energy_kwh",
            "min_segment_actual_cop_span",
            "max_segment_supply_tracking_rmse_c",
            "min_ufh_segment_outdoor_temperature_span_c",
            "min_ufh_segment_supply_target_span_c",
            "segment_score_weight_sample_count",
            "segment_score_weight_thermal_energy",
            "segment_score_weight_actual_cop_span",
            "segment_score_weight_outdoor_temperature_span",
            "segment_score_weight_supply_target_span",
            "segment_score_weight_supply_tracking",
            "max_segment_boundary_gap_ratio",
            "initial_eta_carnot",
            "min_eta_carnot",
            "max_eta_carnot",
            "initial_t_supply_min_c",
            "min_t_supply_min_c",
            "max_t_supply_min_c",
            "initial_heating_curve_slope",
            "t_ref_outdoor_c",
            "min_t_ref_outdoor_c",
            "max_t_ref_outdoor_c",
            "max_heating_curve_slope",
            "cop_min",
            "cop_max",
            "heating_curve_loss_scale_c",
            "eta_loss_scale_kwh",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be strictly positive.")
        if self.max_defrost_active_fraction < 0.0 or self.max_defrost_active_fraction > 1.0:
            raise ValueError("max_defrost_active_fraction must be in [0, 1].")
        if self.max_booster_active_fraction < 0.0 or self.max_booster_active_fraction > 1.0:
            raise ValueError("max_booster_active_fraction must be in [0, 1].")
        if self.min_sample_count < 2:
            raise ValueError("min_sample_count must be at least 2.")
        if self.min_ufh_curve_sample_count < 2:
            raise ValueError("min_ufh_curve_sample_count must be at least 2.")
        if self.min_segment_samples < 2:
            raise ValueError("min_segment_samples must be at least 2.")
        if self.min_segment_samples > self.min_sample_count:
            raise ValueError("min_segment_samples must be <= min_sample_count.")
        if self.reaggregate_min_bucket_count <= 0:
            raise ValueError("reaggregate_min_bucket_count must be strictly positive.")
        if self.min_segment_score < 0.0:
            raise ValueError("min_segment_score must be non-negative.")
        if self.max_selected_ufh_segments is not None and self.max_selected_ufh_segments <= 0:
            raise ValueError("max_selected_ufh_segments must be strictly positive when provided.")
        if self.max_selected_dhw_segments is not None and self.max_selected_dhw_segments <= 0:
            raise ValueError("max_selected_dhw_segments must be strictly positive when provided.")
        if not (self.min_eta_carnot <= self.initial_eta_carnot <= self.max_eta_carnot):
            raise ValueError("initial_eta_carnot must lie within its bounds.")
        if self.min_eta_carnot >= self.max_eta_carnot:
            raise ValueError("min_eta_carnot must be < max_eta_carnot.")
        if not (self.min_t_supply_min_c <= self.initial_t_supply_min_c <= self.max_t_supply_min_c):
            raise ValueError("initial_t_supply_min_c must lie within its bounds.")
        if self.min_t_supply_min_c >= self.max_t_supply_min_c:
            raise ValueError("min_t_supply_min_c must be < max_t_supply_min_c.")
        if self.min_heating_curve_slope < 0.0:
            raise ValueError("min_heating_curve_slope must be non-negative.")
        if not (
            self.min_heating_curve_slope
            <= self.initial_heating_curve_slope
            <= self.max_heating_curve_slope
        ):
            raise ValueError("initial_heating_curve_slope must lie within its bounds.")
        if self.min_heating_curve_slope >= self.max_heating_curve_slope:
            raise ValueError("min_heating_curve_slope must be < max_heating_curve_slope.")
        if not (self.min_t_ref_outdoor_c <= self.t_ref_outdoor_c <= self.max_t_ref_outdoor_c):
            raise ValueError("t_ref_outdoor_c must lie within its bounds.")
        if self.min_t_ref_outdoor_c >= self.max_t_ref_outdoor_c:
            raise ValueError("min_t_ref_outdoor_c must be < max_t_ref_outdoor_c.")
        if self.min_ufh_t_ref_side_sample_count <= 0:
            raise ValueError("min_ufh_t_ref_side_sample_count must be strictly positive.")
        if self.delta_t_cond_k < 0.0:
            raise ValueError("delta_t_cond_k must be non-negative.")
        if self.delta_t_evap_k < 0.0:
            raise ValueError("delta_t_evap_k must be non-negative.")
        if self.cop_min <= 1.0:
            raise ValueError("cop_min must be > 1.")
        if self.cop_max <= self.cop_min:
            raise ValueError("cop_max must be strictly greater than cop_min.")
        if self.heating_curve_loss_name not in COP_LEAST_SQUARES_LOSS_CHOICES:
            raise ValueError(
                "heating_curve_loss_name must be one of "
                f"{COP_LEAST_SQUARES_LOSS_CHOICES!r}."
            )
        if self.eta_loss_name not in COP_LEAST_SQUARES_LOSS_CHOICES:
            raise ValueError(f"eta_loss_name must be one of {COP_LEAST_SQUARES_LOSS_CHOICES!r}.")


@dataclass(frozen=True, slots=True)
class AutomaticCalibrationSettings:
    """Scheduler/runtime settings for automatic in-addon calibration.

    The addon executes the offline calibrators against persisted telemetry only
    after a minimum history window is available. Once enabled, the automatic
    calibration cycle always attempts all supported stages (UFH, DHW standby,
    DHW active, and COP) and retains the previous successful snapshot whenever a
    particular stage fails.

    Attributes:
        min_history_hours: Minimum persisted telemetry history before any
            automatic stage runs [h].
        ufh_active_fit_eta: Whether the automatic active-UFH stage is allowed to
            fit glazing transmittance ``eta`` [-]. Disabled by default because the
            real replay database currently offers too little excitation to separate
            envelope dynamics from solar-gain scaling robustly.
        ufh_active_fit_internal_gains_heat_fraction: Whether the automatic
            active-UFH stage is allowed to fit the baseload→heat mapping fraction
            ``internal_gains_heat_fraction`` [-]. Disabled by default because this
            nuisance parameter is weakly identifiable on short histories.
        ufh_active_min_selected_segments: Minimum number of selected active-UFH
            replay segments required before the fitted RC tuple is trusted [-].
            One short segment is often not structurally informative enough to
            separate solar/internal gains from the RC dynamics.
        ufh_active_bound_tolerance_ratio: Relative tolerance used when deciding
            whether a fitted UFH parameter effectively sits on one of its
            optimizer bounds [-]. Values close to the bound are treated as bound
            hits because such solutions are typically under-identified.
        ufh_active_max_r_ro_mismatch_ratio: Maximum allowed multiplicative
            mismatch between the active-UFH fitted ``R_ro`` and the passive
            off-mode envelope-derived ``R_ro`` [-]. Larger disagreement means the
            active replay fit is not physically self-consistent across stages.
        ufh_active_min_parameter_ratio: Lower bound for automatic active-UFH
            parameters relative to the reference tuple [-]. Tighter than the
            generic manual-calibration bounds to reduce under-identified drift.
        ufh_active_max_parameter_ratio: Upper bound for automatic active-UFH
            parameters relative to the reference tuple [-].
        ufh_active_regularization_weight: Tikhonov weight anchoring the automatic
            active-UFH fit to the reference tuple [-].
        dhw_active_min_selected_segments: Minimum number of selected active-DHW
            replay segments required before the fitted ``R_strat`` is trusted [-].
            One contiguous DHW charging run is often too weak to distinguish
            stratification dynamics from measurement noise and residual no-draw
            model mismatch.
        dhw_standby_bound_tolerance_ratio: Relative tolerance used when deciding
            whether the fitted DHW standby ``tau_standby`` / derived ``R_loss`` is
            effectively sitting on its optimizer bounds [-].
        dhw_standby_fit_ambient_temperature_bias: Whether the automatic standby
            stage may fit a boiler-ambient sensor bias [°C]. Disabled by default
            because the current production telemetry makes this bias weakly
            identifiable and prone to bound-hits without materially improving the fit.
        dhw_active_fit_total_capacity: Whether the automatic active-DHW stage is
            allowed to fit the total DHW heat capacity ``C_top + C_bot`` relative
            to the reference tuple [-]. Enabled by default because the runtime
            should not rely on a fixed tank-capacity assumption when charging data
            is available.
        dhw_active_fit_capacity_split: Whether the automatic active-DHW stage is
            allowed to fit the top/bottom heat-capacity split [-]. Disabled by
            default because the current production database drives this parameter
            to its box constraints, indicating under-identification.
        dhw_active_fit_temperature_biases: Whether the automatic active-DHW stage
            is allowed to fit DHW top/bottom sensor biases [°C]. Disabled by
            default because the current production database does not support stable
            identification of both sensor biases together with ``R_strat``.
        dhw_active_bound_tolerance_ratio: Relative tolerance used when deciding
            whether the fitted active-DHW ``R_strat`` is effectively sitting on its
            optimizer bounds [-].
        cop_min_dhw_sample_count: Minimum number of retained DHW COP samples
            required before the automatic COP stage may publish a DHW-specific
            Carnot efficiency factor [-]. This avoids silently reusing the UFH
            fit when no informative DHW COP data survived dataset selection.
    """

    min_history_hours: float = DEFAULT_AUTOMATIC_CALIBRATION_MIN_HISTORY_HOURS
    ufh_active_fit_eta: bool = False
    ufh_active_fit_internal_gains_heat_fraction: bool = False
    ufh_active_min_selected_segments: int = DEFAULT_AUTOMATIC_UFH_MIN_SELECTED_SEGMENTS
    ufh_active_bound_tolerance_ratio: float = DEFAULT_AUTOMATIC_UFH_BOUND_TOLERANCE_RATIO
    ufh_active_max_r_ro_mismatch_ratio: float = DEFAULT_AUTOMATIC_UFH_MAX_R_RO_MISMATCH_RATIO
    ufh_active_min_parameter_ratio: float = DEFAULT_AUTOMATIC_UFH_MIN_PARAMETER_RATIO
    ufh_active_max_parameter_ratio: float = DEFAULT_AUTOMATIC_UFH_MAX_PARAMETER_RATIO
    ufh_active_regularization_weight: float = DEFAULT_AUTOMATIC_UFH_REGULARIZATION_WEIGHT
    dhw_active_min_selected_segments: int = DEFAULT_AUTOMATIC_DHW_ACTIVE_MIN_SELECTED_SEGMENTS
    dhw_standby_bound_tolerance_ratio: float = DEFAULT_AUTOMATIC_DHW_STANDBY_BOUND_TOLERANCE_RATIO
    dhw_standby_fit_ambient_temperature_bias: bool = DEFAULT_AUTOMATIC_DHW_STANDBY_FIT_AMBIENT_TEMPERATURE_BIAS
    dhw_active_fit_total_capacity: bool = True
    dhw_active_fit_capacity_split: bool = False
    dhw_active_fit_temperature_biases: bool = False
    dhw_active_bound_tolerance_ratio: float = DEFAULT_AUTOMATIC_DHW_ACTIVE_BOUND_TOLERANCE_RATIO
    cop_min_dhw_sample_count: int = DEFAULT_AUTOMATIC_COP_MIN_DHW_SAMPLE_COUNT

    def __post_init__(self) -> None:
        if self.min_history_hours <= 0.0:
            raise ValueError("min_history_hours must be strictly positive.")
        if self.ufh_active_min_selected_segments <= 0:
            raise ValueError("ufh_active_min_selected_segments must be strictly positive.")
        if self.ufh_active_bound_tolerance_ratio < 0.0:
            raise ValueError("ufh_active_bound_tolerance_ratio must be non-negative.")
        if self.ufh_active_max_r_ro_mismatch_ratio < 1.0:
            raise ValueError("ufh_active_max_r_ro_mismatch_ratio must be >= 1.")
        if self.ufh_active_min_parameter_ratio <= 0.0:
            raise ValueError("ufh_active_min_parameter_ratio must be strictly positive.")
        if self.ufh_active_min_parameter_ratio >= self.ufh_active_max_parameter_ratio:
            raise ValueError("ufh_active_min_parameter_ratio must be < ufh_active_max_parameter_ratio.")
        if self.ufh_active_regularization_weight < 0.0:
            raise ValueError("ufh_active_regularization_weight must be non-negative.")
        if self.dhw_active_min_selected_segments <= 0:
            raise ValueError("dhw_active_min_selected_segments must be strictly positive.")
        if self.dhw_standby_bound_tolerance_ratio < 0.0:
            raise ValueError("dhw_standby_bound_tolerance_ratio must be non-negative.")
        if self.dhw_active_bound_tolerance_ratio < 0.0:
            raise ValueError("dhw_active_bound_tolerance_ratio must be non-negative.")
        if self.cop_min_dhw_sample_count <= 0:
            raise ValueError("cop_min_dhw_sample_count must be strictly positive.")


@dataclass(frozen=True, slots=True)
class COPCalibrationResult:
    """Result of fitting the offline Carnot COP model parameters."""

    fitted_parameters: HeatPumpCOPParameters
    t_ref_outdoor_was_fitted: bool
    rmse_supply_temperature_c: float
    rmse_electric_energy_kwh: float
    rmse_actual_cop: float
    ufh_rmse_electric_energy_kwh: float
    dhw_rmse_electric_energy_kwh: float | None
    ufh_rmse_actual_cop: float
    dhw_rmse_actual_cop: float | None
    ufh_bias_actual_cop: float
    dhw_bias_actual_cop: float | None
    diagnostic_eta_carnot_ufh: float
    diagnostic_eta_carnot_dhw: float | None
    sample_count: int
    ufh_sample_count: int
    dhw_sample_count: int
    dataset_start_utc: datetime
    dataset_end_utc: datetime
    heating_curve_optimizer_status: str
    eta_optimizer_status: str
    heating_curve_optimizer_cost: float
    eta_optimizer_cost: float

    def __post_init__(self) -> None:
        if self.rmse_supply_temperature_c < 0.0:
            raise ValueError("rmse_supply_temperature_c must be non-negative.")
        if self.rmse_electric_energy_kwh < 0.0:
            raise ValueError("rmse_electric_energy_kwh must be non-negative.")
        if self.rmse_actual_cop < 0.0:
            raise ValueError("rmse_actual_cop must be non-negative.")
        if self.ufh_rmse_electric_energy_kwh < 0.0:
            raise ValueError("ufh_rmse_electric_energy_kwh must be non-negative.")
        if self.dhw_rmse_electric_energy_kwh is not None and self.dhw_rmse_electric_energy_kwh < 0.0:
            raise ValueError("dhw_rmse_electric_energy_kwh must be non-negative when present.")
        if self.ufh_rmse_actual_cop < 0.0:
            raise ValueError("ufh_rmse_actual_cop must be non-negative.")
        if self.dhw_rmse_actual_cop is not None and self.dhw_rmse_actual_cop < 0.0:
            raise ValueError("dhw_rmse_actual_cop must be non-negative when present.")
        if self.diagnostic_eta_carnot_ufh <= 0.0:
            raise ValueError("diagnostic_eta_carnot_ufh must be strictly positive.")
        if self.diagnostic_eta_carnot_dhw is not None and self.diagnostic_eta_carnot_dhw <= 0.0:
            raise ValueError("diagnostic_eta_carnot_dhw must be strictly positive when present.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be strictly positive.")
        if self.ufh_sample_count <= 0:
            raise ValueError("ufh_sample_count must be strictly positive.")
        if self.dhw_sample_count < 0:
            raise ValueError("dhw_sample_count must be non-negative.")


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

    This stage fits ``R_strat`` and can optionally fit the total DHW heat
    capacity and/or the top/bottom capacity split from contiguous charging windows
    that are filtered to look like no-draw events. ``R_loss`` and
    ``lambda_water`` remain fixed from the injected reference parameter object.

    The fitter replays the 2-state DHW model with ``V_tap = 0`` and minimises the
    one-step residuals on both ``T_top`` and ``T_bot``.

    The active-DHW spread thresholds are intentionally treated as telemetry
    excitation heuristics, not as first-principles thermodynamic constants. Real
    boilers can partially remix while charging; therefore the defaults only reject
    near-isothermal top/bottom traces that are too close to the sensor-noise floor
    to identify ``R_strat`` robustly.

    ``R_strat`` itself is a grey-box parameter, so the optimisation box is carried
    explicitly in physical units [K/kW] instead of reusing the UFH relative-box
    heuristic. The lower bound must remain strictly positive for numerical
    well-posedness, but it cannot assume a strongly stratified residential tank:
    during charging the *effective* remixing resistance can approach perfect
    mixing. Therefore the default lower bound is near zero rather than the
    previously too-restrictive residential-typical range.
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
    min_r_strat_k_per_kw: float = DEFAULT_MIN_DHW_R_STRAT_K_PER_KW
    max_r_strat_k_per_kw: float = DEFAULT_MAX_DHW_R_STRAT_K_PER_KW
    fit_total_capacity: bool = DEFAULT_FIT_DHW_TOTAL_CAPACITY
    initial_c_total_scale: float = 1.0
    min_c_total_scale: float = DEFAULT_MIN_DHW_C_TOTAL_SCALE
    max_c_total_scale: float = DEFAULT_MAX_DHW_C_TOTAL_SCALE
    fit_capacity_split: bool = DEFAULT_FIT_DHW_CAPACITY_SPLIT
    initial_c_top_fraction: float | None = None
    min_c_top_fraction: float = DEFAULT_MIN_DHW_C_TOP_FRACTION
    max_c_top_fraction: float = DEFAULT_MAX_DHW_C_TOP_FRACTION
    ambient_temperature_bias_c: float = DEFAULT_INITIAL_DHW_AMBIENT_TEMPERATURE_BIAS_C
    fit_temperature_biases: bool = DEFAULT_FIT_DHW_TEMPERATURE_BIASES
    initial_t_top_bias_c: float = DEFAULT_INITIAL_DHW_TOP_TEMPERATURE_BIAS_C
    initial_t_bot_bias_c: float = DEFAULT_INITIAL_DHW_BOTTOM_TEMPERATURE_BIAS_C
    min_temperature_bias_c: float = DEFAULT_MIN_DHW_TEMPERATURE_BIAS_C
    max_temperature_bias_c: float = DEFAULT_MAX_DHW_TEMPERATURE_BIAS_C
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
            "min_r_strat_k_per_kw",
            "max_r_strat_k_per_kw",
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
        if self.min_r_strat_k_per_kw >= self.max_r_strat_k_per_kw:
            raise ValueError("min_r_strat_k_per_kw must be < max_r_strat_k_per_kw.")
        if not (self.min_r_strat_k_per_kw <= self.reference_parameters.R_strat <= self.max_r_strat_k_per_kw):
            raise ValueError(
                "reference_parameters.R_strat must lie within "
                "[min_r_strat_k_per_kw, max_r_strat_k_per_kw]."
            )
        if self.regularization_weight < 0.0:
            raise ValueError("regularization_weight must be non-negative.")
        if self.min_c_total_scale <= 0.0:
            raise ValueError("min_c_total_scale must be strictly positive.")
        if self.min_c_total_scale >= self.max_c_total_scale:
            raise ValueError("min_c_total_scale must be < max_c_total_scale.")
        if not (self.min_c_total_scale <= self.initial_c_total_scale <= self.max_c_total_scale):
            raise ValueError("initial_c_total_scale must lie within [min_c_total_scale, max_c_total_scale].")
        if self.min_c_top_fraction <= 0.0 or self.max_c_top_fraction >= 1.0:
            raise ValueError("Capacity split bounds must satisfy 0 < min_c_top_fraction < max_c_top_fraction < 1.")
        if self.min_c_top_fraction >= self.max_c_top_fraction:
            raise ValueError("min_c_top_fraction must be < max_c_top_fraction.")
        reference_c_top_fraction = self.reference_parameters.C_top / (
            self.reference_parameters.C_top + self.reference_parameters.C_bot
        )
        if not (self.min_c_top_fraction <= reference_c_top_fraction <= self.max_c_top_fraction):
            raise ValueError("reference C_top/C_total fraction must lie within the configured split bounds.")
        if self.initial_c_top_fraction is None:
            object.__setattr__(self, "initial_c_top_fraction", reference_c_top_fraction)
        if self.initial_c_top_fraction is None or not (
            self.min_c_top_fraction <= self.initial_c_top_fraction <= self.max_c_top_fraction
        ):
            raise ValueError("initial_c_top_fraction must lie within its bounds.")
        if self.min_temperature_bias_c >= self.max_temperature_bias_c:
            raise ValueError("min_temperature_bias_c must be < max_temperature_bias_c.")
        if not (
            DEFAULT_MIN_DHW_AMBIENT_TEMPERATURE_BIAS_C
            <= self.ambient_temperature_bias_c
            <= DEFAULT_MAX_DHW_AMBIENT_TEMPERATURE_BIAS_C
        ):
            raise ValueError(
                "ambient_temperature_bias_c must lie within the configured DHW ambient-bias bounds."
            )
        for name in ("initial_t_top_bias_c", "initial_t_bot_bias_c"):
            if not (self.min_temperature_bias_c <= getattr(self, name) <= self.max_temperature_bias_c):
                raise ValueError(f"{name} must lie within [min_temperature_bias_c, max_temperature_bias_c].")

    @property
    def reference_c_total_kwh_per_k(self) -> float:
        """Injected total DHW heat capacity ``C_top + C_bot`` [kWh/K]."""
        return self.reference_parameters.C_top + self.reference_parameters.C_bot


@dataclass(frozen=True, slots=True)
class DHWActiveCalibrationResult:
    """Result of fitting active DHW stratification parameters from no-draw charging runs."""

    fitted_parameters: DHWParameters
    fit_total_capacity: bool
    fitted_c_total_scale: float
    fit_capacity_split: bool
    fitted_c_top_fraction: float
    fit_temperature_biases: bool
    fitted_t_top_bias_c: float
    fitted_t_bot_bias_c: float
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
    :class:`home_optimizer.domain.estimation.kalman.UFHKalmanFilter` and learns the physical
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
    reference_internal_gains_kw: float = 0.3
    reference_internal_gains_heat_fraction: float = DEFAULT_INITIAL_INTERNAL_GAINS_HEAT_FRACTION
    fit_c_r: bool = False
    fit_eta: bool = DEFAULT_FIT_ETA
    min_eta: float = DEFAULT_MIN_ETA
    max_eta: float = DEFAULT_MAX_ETA
    fit_internal_gains_heat_fraction: bool = DEFAULT_FIT_INTERNAL_GAINS_HEAT_FRACTION
    min_internal_gains_heat_fraction: float = DEFAULT_MIN_INTERNAL_GAINS_HEAT_FRACTION
    max_internal_gains_heat_fraction: float = DEFAULT_MAX_INTERNAL_GAINS_HEAT_FRACTION
    fit_room_temperature_bias: bool = DEFAULT_FIT_ROOM_TEMPERATURE_BIAS
    initial_room_temperature_bias_c: float = DEFAULT_INITIAL_ROOM_TEMPERATURE_BIAS_C
    min_room_temperature_bias_c: float = DEFAULT_MIN_ROOM_TEMPERATURE_BIAS_C
    max_room_temperature_bias_c: float = DEFAULT_MAX_ROOM_TEMPERATURE_BIAS_C
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
            "reference_internal_gains_kw",
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
        if not 0.0 <= self.reference_internal_gains_heat_fraction <= 1.0:
            raise ValueError("reference_internal_gains_heat_fraction must be in [0, 1].")
        if not 0.0 <= self.min_eta <= self.max_eta <= 1.0:
            raise ValueError("Eta bounds must satisfy 0 <= min_eta <= max_eta <= 1.")
        if not (self.min_eta <= self.reference_parameters.eta <= self.max_eta):
            raise ValueError("reference_parameters.eta must lie within [min_eta, max_eta].")
        if not 0.0 <= self.min_internal_gains_heat_fraction <= self.max_internal_gains_heat_fraction <= 1.0:
            raise ValueError(
                "Internal-gains heat-fraction bounds must satisfy 0 <= min <= max <= 1."
            )
        if not (
            self.min_internal_gains_heat_fraction
            <= self.reference_internal_gains_heat_fraction
            <= self.max_internal_gains_heat_fraction
        ):
            raise ValueError(
                "reference_internal_gains_heat_fraction must lie within its configured bounds."
            )
        if self.regularization_weight < 0.0:
            raise ValueError("regularization_weight must be non-negative.")
        if self.min_room_temperature_bias_c >= self.max_room_temperature_bias_c:
            raise ValueError("min_room_temperature_bias_c must be < max_room_temperature_bias_c.")
        if not (
            self.min_room_temperature_bias_c
            <= self.initial_room_temperature_bias_c
            <= self.max_room_temperature_bias_c
        ):
            raise ValueError("initial_room_temperature_bias_c must lie within its bounds.")
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
    fit_eta: bool
    fitted_internal_gains_heat_fraction: float
    fit_internal_gains_heat_fraction: bool
    fit_room_temperature_bias: bool
    fitted_room_temperature_bias_c: float
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
