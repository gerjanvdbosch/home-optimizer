"""Shared builders for offline calibration stage settings.

These helpers keep the CLI and the automatic scheduler on one source of truth
for stage-specific defaults and thresholds. The goal is architectural DRYness:
automatic calibration must apply the same mathematical filters as the manual
CLI path unless the caller explicitly overrides a setting.
"""

from __future__ import annotations

from .models import (
    COPCalibrationSettings,
    DEFAULT_ACTIVE_MAX_GTI_W_PER_M2,
    DEFAULT_INITIAL_FLOOR_TEMPERATURE_OFFSET_C,
    DEFAULT_MIN_DHW_ACTIVE_SAMPLE_COUNT,
    DEFAULT_MAX_DHW_R_STRAT_K_PER_KW,
    DEFAULT_MAX_DHW_IMPLIED_TAP_M3_PER_H,
    DEFAULT_MAX_DHW_LAYER_TEMPERATURE_SPREAD_C,
    DEFAULT_MIN_DHW_LAYER_TEMPERATURE_SPREAD_C,
    DEFAULT_MIN_DHW_POWER_KW,
    DEFAULT_MIN_DHW_R_STRAT_K_PER_KW,
    DEFAULT_MIN_DHW_SEGMENT_BOTTOM_TEMPERATURE_RISE_C,
    DEFAULT_MIN_DHW_SEGMENT_DELIVERED_ENERGY_KWH,
    DEFAULT_MIN_DHW_SEGMENT_LAYER_SPREAD_SPAN_C,
    DEFAULT_MIN_DHW_SEGMENT_MEAN_LAYER_SPREAD_C,
    DEFAULT_MIN_DHW_SEGMENT_SAMPLES,
    DEFAULT_MIN_DHW_SEGMENT_SCORE,
    DEFAULT_MIN_DHW_SEGMENT_TOP_TEMPERATURE_RISE_C,
    DEFAULT_MIN_SAMPLE_COUNT,
    DEFAULT_MIN_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C,
    DEFAULT_MIN_SEGMENT_ROOM_TEMPERATURE_SPAN_C,
    DEFAULT_MIN_SEGMENT_SAMPLES,
    DEFAULT_MIN_SEGMENT_SCORE,
    DEFAULT_MIN_SEGMENT_UFH_POWER_SPAN_KW,
    DEFAULT_MIN_UFH_POWER_KW,
    DHWActiveCalibrationSettings,
    DHWStandbyCalibrationSettings,
    UFHActiveCalibrationSettings,
)
from ..types import DHWParameters, ThermalParameters


def build_ufh_active_calibration_settings(
    reference_parameters: ThermalParameters,
    *,
    max_gti_w_per_m2: float | None = None,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
    min_segment_samples: int = DEFAULT_MIN_SEGMENT_SAMPLES,
    min_segment_ufh_power_span_kw: float = DEFAULT_MIN_SEGMENT_UFH_POWER_SPAN_KW,
    min_segment_room_temperature_span_c: float = DEFAULT_MIN_SEGMENT_ROOM_TEMPERATURE_SPAN_C,
    min_segment_outdoor_temperature_span_c: float = DEFAULT_MIN_SEGMENT_OUTDOOR_TEMPERATURE_SPAN_C,
    min_segment_score: float = DEFAULT_MIN_SEGMENT_SCORE,
    max_selected_segments: int | None = None,
    min_ufh_power_kw: float = DEFAULT_MIN_UFH_POWER_KW,
    fit_c_r: bool = False,
    fit_initial_floor_temperature_offset: bool = False,
    initial_floor_temperature_offset_c: float = DEFAULT_INITIAL_FLOOR_TEMPERATURE_OFFSET_C,
) -> UFHActiveCalibrationSettings:
    """Build active-UFH calibration settings with CLI-equivalent defaults.

    Args:
        reference_parameters: Fixed UFH reference parameters, including replay
            timestep ``dt_hours`` [h].
        max_gti_w_per_m2: Optional GTI acceptance threshold [W/m²]. When not
            provided, use the wide active-UFH CLI default.
        min_sample_count: Minimum retained replay samples [-].
        min_segment_samples: Minimum samples per contiguous UFH segment [-].
        min_segment_ufh_power_span_kw: Minimum power excitation per segment [kW].
        min_segment_room_temperature_span_c: Minimum room-temperature span [°C].
        min_segment_outdoor_temperature_span_c: Minimum outdoor-temperature span [°C].
        min_segment_score: Minimum dimensionless segment-quality score [-].
        max_selected_segments: Optional cap on retained segments [-].
        min_ufh_power_kw: Minimum mean UFH power for a valid replay bucket [kW].
        fit_c_r: Whether ``C_r`` is fitted instead of held fixed [-].
        fit_initial_floor_temperature_offset: Whether the nuisance floor-offset
            state is fitted [-].
        initial_floor_temperature_offset_c: Initial/fixed floor offset [°C].

    Returns:
        Validated active-UFH calibration settings.
    """
    effective_max_gti_w_per_m2 = (
        DEFAULT_ACTIVE_MAX_GTI_W_PER_M2 if max_gti_w_per_m2 is None else max_gti_w_per_m2
    )
    return UFHActiveCalibrationSettings(
        reference_parameters=reference_parameters,
        max_gti_w_per_m2=effective_max_gti_w_per_m2,
        min_sample_count=min_sample_count,
        min_segment_samples=min_segment_samples,
        min_segment_ufh_power_span_kw=min_segment_ufh_power_span_kw,
        min_segment_room_temperature_span_c=min_segment_room_temperature_span_c,
        min_segment_outdoor_temperature_span_c=min_segment_outdoor_temperature_span_c,
        min_segment_score=min_segment_score,
        max_selected_segments=max_selected_segments,
        min_ufh_power_kw=min_ufh_power_kw,
        fit_c_r=fit_c_r,
        fit_initial_floor_temperature_offset=fit_initial_floor_temperature_offset,
        initial_floor_temperature_offset_c=initial_floor_temperature_offset_c,
    )


def build_dhw_standby_calibration_settings(
    *,
    dt_hours: float,
    reference_c_top_kwh_per_k: float,
    reference_c_bot_kwh_per_k: float,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
    max_layer_temperature_spread_c: float = DEFAULT_MAX_DHW_LAYER_TEMPERATURE_SPREAD_C,
) -> DHWStandbyCalibrationSettings:
    """Build standby-DHW calibration settings with CLI-equivalent defaults.

    Args:
        dt_hours: Reference replay timestep Δt [h].
        reference_c_top_kwh_per_k: Fixed top-layer capacity ``C_top`` [kWh/K].
        reference_c_bot_kwh_per_k: Fixed bottom-layer capacity ``C_bot`` [kWh/K].
        min_sample_count: Minimum retained standby samples [-].
        max_layer_temperature_spread_c: Maximum quasi-mixed layer spread [°C].

    Returns:
        Validated standby-DHW calibration settings.
    """
    return DHWStandbyCalibrationSettings(
        dt_hours=dt_hours,
        reference_c_top_kwh_per_k=reference_c_top_kwh_per_k,
        reference_c_bot_kwh_per_k=reference_c_bot_kwh_per_k,
        min_sample_count=min_sample_count,
        max_layer_temperature_spread_c=max_layer_temperature_spread_c,
    )


def build_dhw_active_calibration_settings(
    reference_parameters: DHWParameters,
    *,
    min_sample_count: int = DEFAULT_MIN_DHW_ACTIVE_SAMPLE_COUNT,
    min_segment_samples: int = DEFAULT_MIN_DHW_SEGMENT_SAMPLES,
    min_dhw_power_kw: float = DEFAULT_MIN_DHW_POWER_KW,
    min_layer_temperature_spread_c: float = DEFAULT_MIN_DHW_LAYER_TEMPERATURE_SPREAD_C,
    max_implied_tap_m3_per_h: float = DEFAULT_MAX_DHW_IMPLIED_TAP_M3_PER_H,
    min_segment_delivered_energy_kwh: float = DEFAULT_MIN_DHW_SEGMENT_DELIVERED_ENERGY_KWH,
    min_segment_mean_layer_spread_c: float = DEFAULT_MIN_DHW_SEGMENT_MEAN_LAYER_SPREAD_C,
    min_segment_layer_spread_span_c: float = DEFAULT_MIN_DHW_SEGMENT_LAYER_SPREAD_SPAN_C,
    min_segment_bottom_temperature_rise_c: float = DEFAULT_MIN_DHW_SEGMENT_BOTTOM_TEMPERATURE_RISE_C,
    min_segment_top_temperature_rise_c: float = DEFAULT_MIN_DHW_SEGMENT_TOP_TEMPERATURE_RISE_C,
    min_segment_score: float = DEFAULT_MIN_DHW_SEGMENT_SCORE,
    min_r_strat_k_per_kw: float = DEFAULT_MIN_DHW_R_STRAT_K_PER_KW,
    max_r_strat_k_per_kw: float = DEFAULT_MAX_DHW_R_STRAT_K_PER_KW,
    max_selected_segments: int | None = None,
) -> DHWActiveCalibrationSettings:
    """Build active-DHW calibration settings with CLI-equivalent defaults.

    Args:
        reference_parameters: Fixed 2-node DHW model parameters, including replay
            timestep ``dt_hours`` [h].
        min_sample_count: Minimum retained no-draw replay samples [-].
        min_segment_samples: Minimum samples per contiguous DHW segment [-].
        min_dhw_power_kw: Minimum mean DHW charging power [kW].
        min_layer_temperature_spread_c: Minimum required layer spread [°C].
        max_implied_tap_m3_per_h: Maximum allowed implied no-draw tap flow [m³/h].
        min_segment_delivered_energy_kwh: Minimum delivered DHW energy [kWh].
        min_segment_mean_layer_spread_c: Minimum mean layer spread [°C].
        min_segment_layer_spread_span_c: Minimum layer-spread span [°C].
        min_segment_bottom_temperature_rise_c: Minimum bottom rise [°C].
        min_segment_top_temperature_rise_c: Minimum top rise [°C].
        min_segment_score: Minimum dimensionless segment-quality score [-].
        min_r_strat_k_per_kw: Explicit lower optimiser bound for ``R_strat`` [K/kW].
        max_r_strat_k_per_kw: Explicit upper optimiser bound for ``R_strat`` [K/kW].
        max_selected_segments: Optional cap on retained segments [-].

    Returns:
        Validated active-DHW calibration settings.
    """
    return DHWActiveCalibrationSettings(
        reference_parameters=reference_parameters,
        min_sample_count=min_sample_count,
        min_segment_samples=min_segment_samples,
        min_dhw_power_kw=min_dhw_power_kw,
        min_layer_temperature_spread_c=min_layer_temperature_spread_c,
        max_implied_tap_m3_per_h=max_implied_tap_m3_per_h,
        min_segment_delivered_energy_kwh=min_segment_delivered_energy_kwh,
        min_segment_mean_layer_spread_c=min_segment_mean_layer_spread_c,
        min_segment_layer_spread_span_c=min_segment_layer_spread_span_c,
        min_segment_bottom_temperature_rise_c=min_segment_bottom_temperature_rise_c,
        min_segment_top_temperature_rise_c=min_segment_top_temperature_rise_c,
        min_segment_score=min_segment_score,
        min_r_strat_k_per_kw=min_r_strat_k_per_kw,
        max_r_strat_k_per_kw=max_r_strat_k_per_kw,
        max_selected_segments=max_selected_segments,
    )


def build_cop_calibration_settings(**kwargs: float | int | str | None) -> COPCalibrationSettings:
    """Build COP calibration settings.

    Args:
        **kwargs: Optional COP-setting overrides. Unspecified parameters fall back
            to the CLI/default offline-COP configuration.

    Returns:
        Validated COP calibration settings.
    """
    return COPCalibrationSettings(**kwargs)
