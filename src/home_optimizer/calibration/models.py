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
DEFAULT_MAX_PAIR_DT_HOURS: float = 0.25
DEFAULT_MAX_GTI_W_PER_M2: float = 25.0
DEFAULT_MAX_DEFROST_ACTIVE_FRACTION: float = 0.0
DEFAULT_MAX_BOOSTER_ACTIVE_FRACTION: float = 0.0
DEFAULT_MIN_SAMPLE_COUNT: int = 12
DEFAULT_INITIAL_TAU_HOURS: float = 24.0
DEFAULT_MIN_TAU_HOURS: float = 0.5
DEFAULT_MAX_TAU_HOURS: float = 500.0
DEFAULT_FORECAST_ALIGNMENT_TOLERANCE_HOURS: float = 1.0


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

