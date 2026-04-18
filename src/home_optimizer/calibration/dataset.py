"""Dataset builders for offline thermal-parameter calibration.

The first calibration stage uses only low-solar ``off`` windows to estimate an
*effective* building envelope model.  This is intentionally conservative:
without active UFH excitation the full 2-state UFH model is not structurally
identifiable, but the aggregate envelope loss and effective thermal capacity are.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime
from typing import Sequence

import numpy as np

from .models import (
    COPCalibrationDataset,
    COPCalibrationSegmentQuality,
    COPCalibrationSample,
    COPCalibrationSettings,
    DHWActiveCalibrationDataset,
    DHWActiveCalibrationSegmentQuality,
    DHWActiveCalibrationSample,
    DHWActiveCalibrationSettings,
    DHWStandbyCalibrationDataset,
    DHWStandbyCalibrationSample,
    DHWStandbyCalibrationSettings,
    UFHActiveCalibrationDataset,
    UFHActiveCalibrationSegmentQuality,
    UFHActiveCalibrationSample,
    UFHActiveCalibrationSettings,
    UFHCalibrationDataset,
    UFHCalibrationSample,
    UFHOffCalibrationSettings,
)
from ..telemetry.models import ForecastSnapshot, TelemetryAggregate

_SECONDS_PER_HOUR: float = 3600.0


def _dhw_total_energy_kwh(
    *,
    t_top_c: float,
    t_bot_c: float,
    c_top_kwh_per_k: float,
    c_bot_kwh_per_k: float,
) -> float:
    """Return total tank thermal energy proxy ``C_top·T_top + C_bot·T_bot`` [kWh]."""
    return c_top_kwh_per_k * t_top_c + c_bot_kwh_per_k * t_bot_c


def _implied_v_tap_m3_per_h(
    *,
    t_top_start_c: float,
    t_bot_start_c: float,
    t_top_end_c: float,
    t_bot_end_c: float,
    dt_hours: float,
    p_dhw_mean_kw: float,
    t_mains_c: float,
    t_amb_c: float,
    settings: DHWActiveCalibrationSettings,
) -> float:
    """Infer tap flow from the full-tank energy balance and clamp it to ``≥ 0``.

    Implements the rearranged total-energy balance from DHW §9.5:

        ΔE/Δt = P_dhw - λ·V_tap·(T_top - T_mains) - Q_loss

    so that

        V_tap = (P_dhw - Q_loss - ΔE/Δt) / (λ·(T_top - T_mains))

    The result is projected onto the physically feasible set ``V_tap ≥ 0`` before
    use in dataset filtering. Negative implied draw would be non-physical and is
    interpreted here as measurement/model noise rather than a real tap event.
    """
    p = settings.reference_parameters
    start_energy_kwh = _dhw_total_energy_kwh(
        t_top_c=t_top_start_c,
        t_bot_c=t_bot_start_c,
        c_top_kwh_per_k=p.C_top,
        c_bot_kwh_per_k=p.C_bot,
    )
    end_energy_kwh = _dhw_total_energy_kwh(
        t_top_c=t_top_end_c,
        t_bot_c=t_bot_end_c,
        c_top_kwh_per_k=p.C_top,
        c_bot_kwh_per_k=p.C_bot,
    )
    delta_energy_rate_kw = (end_energy_kwh - start_energy_kwh) / dt_hours
    q_loss_kw = (t_top_start_c - t_amb_c) / p.R_loss + (t_bot_start_c - t_amb_c) / p.R_loss
    tap_denominator = p.lambda_water * (t_top_start_c - t_mains_c)
    if tap_denominator <= 0.0:
        return float("inf")
    implied_v_tap = (p_dhw_mean_kw - q_loss_kw - delta_energy_rate_kw) / tap_denominator
    return float(max(0.0, implied_v_tap))


class _ForecastLookup:
    """Nearest-neighbour lookup from UTC timestamps to persisted forecast steps."""

    def __init__(self, forecast_rows: Sequence[ForecastSnapshot]) -> None:
        rows = sorted(forecast_rows, key=lambda row: row.valid_at_utc)
        self._rows = rows
        self._times = [row.valid_at_utc.timestamp() for row in rows]

    def nearest(
        self,
        target_utc: datetime,
        *,
        tolerance_hours: float,
    ) -> ForecastSnapshot | None:
        """Return the closest forecast step within the requested tolerance window."""
        if not self._rows:
            return None
        target_seconds = target_utc.timestamp()
        index = bisect_left(self._times, target_seconds)
        candidates: list[ForecastSnapshot] = []
        if index < len(self._rows):
            candidates.append(self._rows[index])
        if index > 0:
            candidates.append(self._rows[index - 1])
        if not candidates:
            return None
        closest = min(candidates, key=lambda row: abs(row.valid_at_utc.timestamp() - target_seconds))
        max_delta_seconds = tolerance_hours * _SECONDS_PER_HOUR
        if abs(closest.valid_at_utc.timestamp() - target_seconds) > max_delta_seconds:
            return None
        return closest


def build_ufh_off_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    forecast_rows: Sequence[ForecastSnapshot],
    settings: UFHOffCalibrationSettings,
) -> UFHCalibrationDataset:
    """Build low-solar off-mode transition samples for first-stage UFH calibration.

    The fitted transition sample uses the last room temperature of one bucket as
    ``T[k]`` and the last room temperature of the following bucket as ``T[k+1]``.
    Mean disturbances from the *second* bucket approximate the forcing over the
    interval ``[k, k+1]``.

    Args:
        aggregates: Historical telemetry buckets ordered arbitrarily.
        forecast_rows: Persisted forecast snapshots used to filter out solar-active windows.
        settings: Validated calibration thresholds and parameter bounds.

    Returns:
        Dataset containing validated off-mode transition samples.

    Raises:
        ValueError: If fewer than ``settings.min_sample_count`` usable samples remain.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    forecast_lookup = _ForecastLookup(forecast_rows)
    samples: list[UFHCalibrationSample] = []

    for previous_row, next_row in zip(rows, rows[1:]):
        if previous_row.hp_mode_last != settings.off_mode_name:
            continue
        if next_row.hp_mode_last != settings.off_mode_name:
            continue
        if previous_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if next_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if previous_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue
        if next_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue

        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            continue

        forecast = forecast_lookup.nearest(
            next_row.bucket_end_utc,
            tolerance_hours=settings.forecast_alignment_tolerance_hours,
        )
        gti_w_per_m2 = float(forecast.gti_w_per_m2) if forecast is not None else 0.0
        if gti_w_per_m2 > settings.max_gti_w_per_m2:
            continue

        samples.append(
            UFHCalibrationSample(
                interval_start_utc=previous_row.bucket_end_utc,
                interval_end_utc=next_row.bucket_end_utc,
                dt_hours=dt_hours,
                room_temperature_start_c=float(previous_row.room_temperature_last_c),
                room_temperature_end_c=float(next_row.room_temperature_last_c),
                outdoor_temperature_mean_c=float(next_row.outdoor_temperature_mean_c),
                gti_w_per_m2=gti_w_per_m2,
                household_elec_power_mean_kw=float(next_row.household_elec_power_mean_kw),
            )
        )

    if len(samples) < settings.min_sample_count:
        raise ValueError(
            "Not enough low-solar off-mode samples for UFH calibration: "
            f"required >= {settings.min_sample_count}, found {len(samples)}."
        )
    return UFHCalibrationDataset(samples=tuple(samples))


def _cop_segment_score(
    samples: Sequence[COPCalibrationSample],
    settings: COPCalibrationSettings,
) -> float:
    """Return a dimensionless quality score for one contiguous COP segment."""
    sample_count = float(len(samples))
    thermal_energy_kwh = float(sum(sample.thermal_energy_kwh for sample in samples))
    actual_cops = np.array([sample.actual_cop for sample in samples], dtype=float)
    outdoor_temperatures_c = np.array([sample.outdoor_temperature_mean_c for sample in samples], dtype=float)
    supply_targets_c = np.array([sample.supply_target_temperature_mean_c for sample in samples], dtype=float)
    supply_measured_c = np.array([sample.supply_temperature_mean_c for sample in samples], dtype=float)
    actual_cop_span = float(np.ptp(actual_cops))
    outdoor_span_c = float(np.ptp(outdoor_temperatures_c))
    supply_target_span_c = float(np.ptp(supply_targets_c))
    supply_tracking_rmse_c = float(np.sqrt(np.mean(np.square(supply_measured_c - supply_targets_c))))
    tracking_margin = max(
        0.0,
        1.0 - supply_tracking_rmse_c / settings.max_segment_supply_tracking_rmse_c,
    )

    score = (
        settings.segment_score_weight_sample_count * sample_count / float(settings.min_segment_samples)
        + settings.segment_score_weight_thermal_energy
        * thermal_energy_kwh
        / settings.min_segment_thermal_energy_kwh
        + settings.segment_score_weight_actual_cop_span
        * actual_cop_span
        / settings.min_segment_actual_cop_span
        + settings.segment_score_weight_supply_tracking * tracking_margin
    )
    if samples[0].mode_name == settings.ufh_mode_name:
        score += (
            settings.segment_score_weight_outdoor_temperature_span
            * outdoor_span_c
            / settings.min_ufh_segment_outdoor_temperature_span_c
            + settings.segment_score_weight_supply_target_span
            * supply_target_span_c
            / settings.min_ufh_segment_supply_target_span_c
        )
    return float(score)


def _cop_segment_quality(
    samples: Sequence[COPCalibrationSample],
    *,
    raw_segment_index: int,
    settings: COPCalibrationSettings,
) -> COPCalibrationSegmentQuality:
    """Summarise one raw contiguous COP segment before top-N selection."""
    actual_cops = np.array([sample.actual_cop for sample in samples], dtype=float)
    outdoor_temperatures_c = np.array([sample.outdoor_temperature_mean_c for sample in samples], dtype=float)
    supply_targets_c = np.array([sample.supply_target_temperature_mean_c for sample in samples], dtype=float)
    supply_measured_c = np.array([sample.supply_temperature_mean_c for sample in samples], dtype=float)
    quality = COPCalibrationSegmentQuality(
        raw_segment_index=raw_segment_index,
        mode_name=samples[0].mode_name,
        selected=False,
        sample_count=len(samples),
        duration_hours=float(sum(sample.dt_hours for sample in samples)),
        thermal_energy_kwh=float(sum(sample.thermal_energy_kwh for sample in samples)),
        electric_energy_kwh=float(sum(sample.electric_energy_kwh for sample in samples)),
        outdoor_temperature_span_c=float(np.ptp(outdoor_temperatures_c)),
        supply_target_temperature_span_c=float(np.ptp(supply_targets_c)),
        actual_cop_span=float(np.ptp(actual_cops)),
        supply_tracking_rmse_c=float(np.sqrt(np.mean(np.square(supply_measured_c - supply_targets_c)))),
        score=_cop_segment_score(samples, settings),
    )
    hard_selected = (
        quality.sample_count >= settings.min_segment_samples
        and quality.thermal_energy_kwh >= settings.min_segment_thermal_energy_kwh
        and quality.actual_cop_span >= settings.min_segment_actual_cop_span
        and quality.supply_tracking_rmse_c <= settings.max_segment_supply_tracking_rmse_c
        and quality.score >= settings.min_segment_score
    )
    if quality.mode_name == settings.ufh_mode_name:
        hard_selected = hard_selected and (
            quality.outdoor_temperature_span_c >= settings.min_ufh_segment_outdoor_temperature_span_c
            and quality.supply_target_temperature_span_c >= settings.min_ufh_segment_supply_target_span_c
        )
    return COPCalibrationSegmentQuality(
        raw_segment_index=quality.raw_segment_index,
        mode_name=quality.mode_name,
        selected=hard_selected,
        sample_count=quality.sample_count,
        duration_hours=quality.duration_hours,
        thermal_energy_kwh=quality.thermal_energy_kwh,
        electric_energy_kwh=quality.electric_energy_kwh,
        outdoor_temperature_span_c=quality.outdoor_temperature_span_c,
        supply_target_temperature_span_c=quality.supply_target_temperature_span_c,
        actual_cop_span=quality.actual_cop_span,
        supply_tracking_rmse_c=quality.supply_tracking_rmse_c,
        score=quality.score,
    )


def _apply_cop_segment_caps(
    qualities: list[COPCalibrationSegmentQuality],
    settings: COPCalibrationSettings,
) -> list[COPCalibrationSegmentQuality]:
    """Apply optional top-N per-mode caps after hard-threshold screening."""
    selected_by_mode: dict[str, list[COPCalibrationSegmentQuality]] = {}
    for quality in qualities:
        if not quality.selected:
            continue
        selected_by_mode.setdefault(quality.mode_name, []).append(quality)

    selected_raw_indices: set[int] = set()
    for mode_name, mode_qualities in selected_by_mode.items():
        max_selected_segments = (
            settings.max_selected_ufh_segments
            if mode_name == settings.ufh_mode_name
            else settings.max_selected_dhw_segments
        )
        ranked = sorted(
            mode_qualities,
            key=lambda quality: (-quality.score, quality.supply_tracking_rmse_c, quality.raw_segment_index),
        )
        kept = ranked if max_selected_segments is None else ranked[:max_selected_segments]
        selected_raw_indices.update(quality.raw_segment_index for quality in kept)

    return [
        COPCalibrationSegmentQuality(
            raw_segment_index=quality.raw_segment_index,
            mode_name=quality.mode_name,
            selected=quality.selected and quality.raw_segment_index in selected_raw_indices,
            sample_count=quality.sample_count,
            duration_hours=quality.duration_hours,
            thermal_energy_kwh=quality.thermal_energy_kwh,
            electric_energy_kwh=quality.electric_energy_kwh,
            outdoor_temperature_span_c=quality.outdoor_temperature_span_c,
            supply_target_temperature_span_c=quality.supply_target_temperature_span_c,
            actual_cop_span=quality.actual_cop_span,
            supply_tracking_rmse_c=quality.supply_tracking_rmse_c,
            score=quality.score,
        )
        for quality in qualities
    ]


def build_cop_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    settings: COPCalibrationSettings,
) -> COPCalibrationDataset:
    """Build a filtered operating-bucket dataset for offline COP calibration.

    The builder keeps only UFH and DHW buckets that are physically meaningful for
    COP fitting: positive thermal/electrical energy, no defrost/booster overlap,
    and measured bucket COP within the physically meaningful interval
    ``(1, cop_max]``. Surviving buckets are then grouped into contiguous same-mode
    segments, scored for excitation/supply tracking quality, and optionally capped
    per mode before the final dataset is emitted.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    raw_segments: list[list[COPCalibrationSample]] = []
    current_segment: list[COPCalibrationSample] = []
    accepted_modes = {settings.ufh_mode_name, settings.dhw_mode_name}

    def flush_segment() -> None:
        nonlocal current_segment
        if current_segment:
            raw_segments.append(current_segment)
            current_segment = []

    for row in rows:
        mode_name = str(row.hp_mode_last)
        if mode_name not in accepted_modes:
            flush_segment()
            continue
        if float(row.defrost_active_fraction) > settings.max_defrost_active_fraction:
            flush_segment()
            continue
        if float(row.booster_heater_active_fraction) > settings.max_booster_active_fraction:
            flush_segment()
            continue

        dt_hours = (row.bucket_end_utc - row.bucket_start_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0:
            flush_segment()
            continue

        thermal_energy_kwh = float(row.hp_thermal_power_mean_kw) * dt_hours
        electric_energy_kwh = float(row.hp_electric_energy_delta_kwh)
        if thermal_energy_kwh < settings.min_thermal_energy_kwh:
            flush_segment()
            continue
        if electric_energy_kwh < settings.min_electric_energy_kwh:
            flush_segment()
            continue

        supply_target_temperature_mean_c = float(row.hp_supply_target_temperature_mean_c)
        supply_temperature_mean_c = float(row.hp_supply_temperature_mean_c)
        if not np.isfinite(supply_target_temperature_mean_c):
            flush_segment()
            continue
        if not np.isfinite(supply_temperature_mean_c):
            flush_segment()
            continue

        sample = COPCalibrationSample(
            bucket_start_utc=row.bucket_start_utc,
            bucket_end_utc=row.bucket_end_utc,
            dt_hours=dt_hours,
            mode_name=mode_name,
            outdoor_temperature_mean_c=float(row.outdoor_temperature_mean_c),
            supply_target_temperature_mean_c=supply_target_temperature_mean_c,
            supply_temperature_mean_c=supply_temperature_mean_c,
            thermal_energy_kwh=thermal_energy_kwh,
            electric_energy_kwh=electric_energy_kwh,
        )
        if sample.actual_cop <= 1.0:
            flush_segment()
            continue
        if sample.actual_cop > settings.cop_max:
            flush_segment()
            continue
        if current_segment and (
            current_segment[-1].mode_name != sample.mode_name
            or current_segment[-1].bucket_end_utc != sample.bucket_start_utc
        ):
            flush_segment()
        current_segment.append(sample)

    flush_segment()

    segment_qualities = _apply_cop_segment_caps(
        [
            _cop_segment_quality(segment_samples, raw_segment_index=index, settings=settings)
            for index, segment_samples in enumerate(raw_segments)
        ],
        settings,
    )
    selected_segment_indices = {
        quality.raw_segment_index for quality in segment_qualities if quality.selected
    }
    samples = [
        sample
        for segment_index, segment_samples in enumerate(raw_segments)
        if segment_index in selected_segment_indices
        for sample in segment_samples
    ]

    if len(samples) < settings.min_sample_count:
        raise ValueError(
            "Not enough valid operating buckets for COP calibration: "
            f"required >= {settings.min_sample_count}, found {len(samples)}."
        )
    dataset = COPCalibrationDataset(samples=tuple(samples), segment_qualities=tuple(segment_qualities))
    if dataset.ufh_sample_count < settings.min_ufh_curve_sample_count:
        raise ValueError(
            "Not enough UFH buckets for heating-curve calibration: "
            f"required >= {settings.min_ufh_curve_sample_count}, found {dataset.ufh_sample_count}."
        )
    return dataset


def build_dhw_standby_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    settings: DHWStandbyCalibrationSettings,
) -> DHWStandbyCalibrationDataset:
    """Build conservative quasi-mixed standby DHW samples for loss calibration.

    This stage intentionally keeps only intervals where the tank is not in DHW
    mode and both layer temperatures remain close together.  Under those
    conditions the full 2-node DHW energy balance reduces to a one-state standby
    envelope in the weighted mean tank temperature.

    Args:
        aggregates: Historical telemetry buckets ordered arbitrarily.
        settings: Validated standby-calibration thresholds and reference capacities.

    Returns:
        Dataset containing validated standby DHW transition samples.

    Raises:
        ValueError: If fewer than ``settings.min_sample_count`` usable samples remain.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    samples: list[DHWStandbyCalibrationSample] = []

    for previous_row, next_row in zip(rows, rows[1:]):
        if previous_row.hp_mode_last == settings.dhw_mode_name:
            continue
        if next_row.hp_mode_last == settings.dhw_mode_name:
            continue
        if previous_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if next_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if previous_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue
        if next_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue

        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            continue
        if abs(dt_hours - settings.dt_hours) > settings.dt_compatibility_tolerance_hours:
            continue

        layer_spread_start_c = abs(
            float(previous_row.dhw_top_temperature_last_c) - float(previous_row.dhw_bottom_temperature_last_c)
        )
        layer_spread_end_c = abs(
            float(next_row.dhw_top_temperature_last_c) - float(next_row.dhw_bottom_temperature_last_c)
        )
        if layer_spread_start_c > settings.max_layer_temperature_spread_c:
            continue
        if layer_spread_end_c > settings.max_layer_temperature_spread_c:
            continue

        samples.append(
            DHWStandbyCalibrationSample(
                interval_start_utc=previous_row.bucket_end_utc,
                interval_end_utc=next_row.bucket_end_utc,
                dt_hours=dt_hours,
                t_top_start_c=float(previous_row.dhw_top_temperature_last_c),
                t_top_end_c=float(next_row.dhw_top_temperature_last_c),
                t_bot_start_c=float(previous_row.dhw_bottom_temperature_last_c),
                t_bot_end_c=float(next_row.dhw_bottom_temperature_last_c),
                boiler_ambient_mean_c=float(next_row.boiler_ambient_temp_mean_c),
            )
        )

    if len(samples) < settings.min_sample_count:
        raise ValueError(
            "Not enough quasi-mixed standby DHW samples for calibration: "
            f"required >= {settings.min_sample_count}, found {len(samples)}."
        )
    return DHWStandbyCalibrationDataset(samples=tuple(samples))


def build_dhw_active_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    settings: DHWActiveCalibrationSettings,
) -> DHWActiveCalibrationDataset:
    """Build active DHW no-draw replay samples for ``R_strat`` calibration.

    Each sample replays one transition ``x[k] -> x[k+1]`` during DHW charging,
    using the measured top/bottom state at the start of the interval and the mean
    power/ambient/mains disturbances from the second bucket of the pair.

    To stay physically consistent without a tap-flow meter, this stage keeps only
    pairs whose total-energy-balance-implied draw remains below the configured
    threshold.  Those windows are then replayed with ``V_tap = 0``.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    raw_segments: list[list[DHWActiveCalibrationSample]] = []
    current_segment_samples: list[DHWActiveCalibrationSample] = []
    dt_reference_hours = settings.reference_parameters.dt_hours

    def flush_current_segment() -> None:
        nonlocal current_segment_samples
        if len(current_segment_samples) >= settings.min_segment_samples:
            raw_segments.append(current_segment_samples)
        current_segment_samples = []

    def meets_threshold(value: float, lower_bound: float) -> bool:
        """Return whether a metric satisfies a configured lower bound."""
        return value > lower_bound or bool(np.isclose(value, lower_bound))

    def compute_segment_quality(
        raw_segment_index: int,
        segment_samples: list[DHWActiveCalibrationSample],
    ) -> DHWActiveCalibrationSegmentQuality:
        top_temperatures = np.array(
            [segment_samples[0].t_top_start_c] + [sample.t_top_end_c for sample in segment_samples],
            dtype=float,
        )
        bot_temperatures = np.array(
            [segment_samples[0].t_bot_start_c] + [sample.t_bot_end_c for sample in segment_samples],
            dtype=float,
        )
        layer_spreads = np.abs(top_temperatures - bot_temperatures)
        delivered_energy_kwh = float(
            sum(sample.p_dhw_mean_kw * sample.dt_hours for sample in segment_samples)
        )
        implied_v_taps = np.array(
            [sample.implied_v_tap_m3_per_h for sample in segment_samples],
            dtype=float,
        )

        sample_count = len(segment_samples)
        duration_hours = float(sum(sample.dt_hours for sample in segment_samples))
        mean_layer_spread_c = float(np.mean(layer_spreads))
        layer_spread_span_c = float(np.max(layer_spreads) - np.min(layer_spreads))
        bottom_temperature_rise_c = float(bot_temperatures[-1] - bot_temperatures[0])
        top_temperature_rise_c = float(top_temperatures[-1] - top_temperatures[0])
        p95_implied_v_tap_m3_per_h = float(np.percentile(implied_v_taps, 95))
        max_implied_v_tap_m3_per_h = float(np.max(implied_v_taps))
        tap_margin = max(
            0.0,
            1.0 - p95_implied_v_tap_m3_per_h / settings.max_implied_tap_m3_per_h,
        )

        score = (
            settings.segment_score_weight_sample_count * (sample_count / settings.min_segment_samples)
            + settings.segment_score_weight_delivered_energy
            * (delivered_energy_kwh / settings.min_segment_delivered_energy_kwh)
            + settings.segment_score_weight_mean_layer_spread
            * (mean_layer_spread_c / settings.min_segment_mean_layer_spread_c)
            + settings.segment_score_weight_layer_spread_span
            * (layer_spread_span_c / settings.min_segment_layer_spread_span_c)
            + settings.segment_score_weight_bottom_temperature_rise
            * (bottom_temperature_rise_c / settings.min_segment_bottom_temperature_rise_c)
            + settings.segment_score_weight_top_temperature_rise
            * (top_temperature_rise_c / settings.min_segment_top_temperature_rise_c)
            + settings.segment_score_weight_tap_margin * tap_margin
        )
        passes_hard_thresholds = (
            sample_count >= settings.min_segment_samples
            and meets_threshold(delivered_energy_kwh, settings.min_segment_delivered_energy_kwh)
            and meets_threshold(mean_layer_spread_c, settings.min_segment_mean_layer_spread_c)
            and meets_threshold(layer_spread_span_c, settings.min_segment_layer_spread_span_c)
            and meets_threshold(bottom_temperature_rise_c, settings.min_segment_bottom_temperature_rise_c)
            and meets_threshold(top_temperature_rise_c, settings.min_segment_top_temperature_rise_c)
            and p95_implied_v_tap_m3_per_h <= settings.max_implied_tap_m3_per_h
            and meets_threshold(score, settings.min_segment_score)
        )
        return DHWActiveCalibrationSegmentQuality(
            raw_segment_index=raw_segment_index,
            selected_segment_index=raw_segment_index if passes_hard_thresholds else None,
            selected=passes_hard_thresholds,
            sample_count=sample_count,
            duration_hours=duration_hours,
            delivered_energy_kwh=delivered_energy_kwh,
            mean_layer_spread_c=mean_layer_spread_c,
            layer_spread_span_c=layer_spread_span_c,
            bottom_temperature_rise_c=bottom_temperature_rise_c,
            top_temperature_rise_c=top_temperature_rise_c,
            p95_implied_v_tap_m3_per_h=p95_implied_v_tap_m3_per_h,
            max_implied_v_tap_m3_per_h=max_implied_v_tap_m3_per_h,
            score=float(score),
        )

    for previous_row, next_row in zip(rows, rows[1:]):
        pair_is_valid = True
        if previous_row.hp_mode_last != settings.active_mode_name:
            pair_is_valid = False
        if next_row.hp_mode_last != settings.active_mode_name:
            pair_is_valid = False
        if previous_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            pair_is_valid = False
        if next_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            pair_is_valid = False
        if previous_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            pair_is_valid = False
        if next_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            pair_is_valid = False

        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            pair_is_valid = False
        if abs(dt_hours - dt_reference_hours) > settings.dt_compatibility_tolerance_hours:
            pair_is_valid = False

        p_dhw_mean_kw = float(next_row.hp_thermal_power_mean_kw)
        if p_dhw_mean_kw < settings.min_dhw_power_kw:
            pair_is_valid = False

        layer_spread_start_c = abs(
            float(previous_row.dhw_top_temperature_last_c) - float(previous_row.dhw_bottom_temperature_last_c)
        )
        layer_spread_end_c = abs(
            float(next_row.dhw_top_temperature_last_c) - float(next_row.dhw_bottom_temperature_last_c)
        )
        if max(layer_spread_start_c, layer_spread_end_c) < settings.min_layer_temperature_spread_c:
            pair_is_valid = False

        t_mains_c = float(next_row.t_mains_estimated_mean_c)
        t_amb_c = float(next_row.boiler_ambient_temp_mean_c)
        implied_v_tap_m3_per_h = _implied_v_tap_m3_per_h(
            t_top_start_c=float(previous_row.dhw_top_temperature_last_c),
            t_bot_start_c=float(previous_row.dhw_bottom_temperature_last_c),
            t_top_end_c=float(next_row.dhw_top_temperature_last_c),
            t_bot_end_c=float(next_row.dhw_bottom_temperature_last_c),
            dt_hours=dt_hours,
            p_dhw_mean_kw=p_dhw_mean_kw,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
            settings=settings,
        )
        if implied_v_tap_m3_per_h > settings.max_implied_tap_m3_per_h:
            pair_is_valid = False

        if not pair_is_valid:
            flush_current_segment()
            continue

        current_segment_samples.append(
            DHWActiveCalibrationSample(
                interval_start_utc=previous_row.bucket_end_utc,
                interval_end_utc=next_row.bucket_end_utc,
                dt_hours=dt_hours,
                t_top_start_c=float(previous_row.dhw_top_temperature_last_c),
                t_top_end_c=float(next_row.dhw_top_temperature_last_c),
                t_bot_start_c=float(previous_row.dhw_bottom_temperature_last_c),
                t_bot_end_c=float(next_row.dhw_bottom_temperature_last_c),
                p_dhw_mean_kw=p_dhw_mean_kw,
                t_mains_c=t_mains_c,
                t_amb_c=t_amb_c,
                implied_v_tap_m3_per_h=implied_v_tap_m3_per_h,
                segment_index=len(raw_segments),
            )
        )

    flush_current_segment()
    raw_segment_qualities = tuple(
        compute_segment_quality(index, segment_samples) for index, segment_samples in enumerate(raw_segments)
    )
    selected_raw_indices = [quality.raw_segment_index for quality in raw_segment_qualities if quality.selected]
    if settings.max_selected_segments is not None and len(selected_raw_indices) > settings.max_selected_segments:
        selected_raw_indices = [
            quality.raw_segment_index
            for quality in sorted(
                (quality for quality in raw_segment_qualities if quality.selected),
                key=lambda quality: (
                    -quality.score,
                    quality.p95_implied_v_tap_m3_per_h,
                    quality.raw_segment_index,
                ),
            )[: settings.max_selected_segments]
        ]
        selected_raw_indices.sort()

    raw_to_selected_index = {raw_index: selected_index for selected_index, raw_index in enumerate(selected_raw_indices)}
    selected_samples: list[DHWActiveCalibrationSample] = []
    for raw_index in selected_raw_indices:
        for sample in raw_segments[raw_index]:
            selected_samples.append(
                DHWActiveCalibrationSample(
                    interval_start_utc=sample.interval_start_utc,
                    interval_end_utc=sample.interval_end_utc,
                    dt_hours=sample.dt_hours,
                    t_top_start_c=sample.t_top_start_c,
                    t_top_end_c=sample.t_top_end_c,
                    t_bot_start_c=sample.t_bot_start_c,
                    t_bot_end_c=sample.t_bot_end_c,
                    p_dhw_mean_kw=sample.p_dhw_mean_kw,
                    t_mains_c=sample.t_mains_c,
                    t_amb_c=sample.t_amb_c,
                    implied_v_tap_m3_per_h=sample.implied_v_tap_m3_per_h,
                    segment_index=raw_to_selected_index[raw_index],
                )
            )

    segment_qualities = tuple(
        DHWActiveCalibrationSegmentQuality(
            raw_segment_index=quality.raw_segment_index,
            selected_segment_index=raw_to_selected_index.get(quality.raw_segment_index),
            selected=quality.raw_segment_index in raw_to_selected_index,
            sample_count=quality.sample_count,
            duration_hours=quality.duration_hours,
            delivered_energy_kwh=quality.delivered_energy_kwh,
            mean_layer_spread_c=quality.mean_layer_spread_c,
            layer_spread_span_c=quality.layer_spread_span_c,
            bottom_temperature_rise_c=quality.bottom_temperature_rise_c,
            top_temperature_rise_c=quality.top_temperature_rise_c,
            p95_implied_v_tap_m3_per_h=quality.p95_implied_v_tap_m3_per_h,
            max_implied_v_tap_m3_per_h=quality.max_implied_v_tap_m3_per_h,
            score=quality.score,
        )
        for quality in raw_segment_qualities
    )

    if len(selected_samples) < settings.min_sample_count:
        raise ValueError(
            "Not enough active no-draw DHW samples for stratification calibration: "
            f"required >= {settings.min_sample_count}, found {len(selected_samples)}."
        )
    return DHWActiveCalibrationDataset(
        samples=tuple(selected_samples),
        segment_qualities=segment_qualities,
    )


def build_ufh_active_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    forecast_rows: Sequence[ForecastSnapshot],
    settings: UFHActiveCalibrationSettings,
) -> UFHActiveCalibrationDataset:
    """Build replay samples for active UFH RC calibration.

    Each sample replays one transition ``x[k] -> x[k+1]`` using the mean UFH power
    and disturbances from the second bucket of a consecutive pair, while the room
    temperature at the pair endpoints supplies the measured scalar output.

    Args:
        aggregates: Historical telemetry buckets ordered arbitrarily.
        forecast_rows: Persisted forecast rows used to reconstruct GTI [W/m²].
        settings: Validated active-calibration configuration and reference parameters.

    Returns:
        Dataset containing validated replay samples for the active UFH fitter.

    Raises:
        ValueError: If too few usable active samples remain after filtering.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    forecast_lookup = _ForecastLookup(forecast_rows)
    raw_segments: list[list[UFHActiveCalibrationSample]] = []
    current_segment_samples: list[UFHActiveCalibrationSample] = []
    dt_reference_hours = settings.reference_parameters.dt_hours

    def flush_current_segment() -> None:
        nonlocal current_segment_samples
        if len(current_segment_samples) >= settings.min_segment_samples:
            raw_segments.append(current_segment_samples)
        current_segment_samples = []

    def compute_segment_quality(
        raw_segment_index: int,
        segment_samples: list[UFHActiveCalibrationSample],
    ) -> UFHActiveCalibrationSegmentQuality:
        def meets_threshold(value: float, lower_bound: float) -> bool:
            """Return whether a metric satisfies a configured lower bound.

            The comparison is inclusive up to floating-point round-off because
            telemetry aggregates often land exactly on the configured threshold.
            """

            return value > lower_bound or bool(np.isclose(value, lower_bound))

        room_temperatures = np.array(
            [segment_samples[0].room_temperature_start_c]
            + [sample.room_temperature_end_c for sample in segment_samples],
            dtype=float,
        )
        outdoor_temperatures = np.array(
            [sample.outdoor_temperature_mean_c for sample in segment_samples], dtype=float
        )
        ufh_powers = np.array([sample.ufh_power_mean_kw for sample in segment_samples], dtype=float)
        gti_values = np.array([sample.gti_w_per_m2 for sample in segment_samples], dtype=float)

        sample_count = len(segment_samples)
        duration_hours = float(sum(sample.dt_hours for sample in segment_samples))
        ufh_power_span_kw = float(np.max(ufh_powers) - np.min(ufh_powers))
        room_temperature_span_c = float(np.max(room_temperatures) - np.min(room_temperatures))
        outdoor_temperature_span_c = float(np.max(outdoor_temperatures) - np.min(outdoor_temperatures))
        mean_gti_w_per_m2 = float(np.mean(gti_values))

        score = (
            settings.segment_score_weight_sample_count * (sample_count / settings.min_segment_samples)
            + settings.segment_score_weight_ufh_power_span
            * (ufh_power_span_kw / settings.min_segment_ufh_power_span_kw)
            + settings.segment_score_weight_room_temperature_span
            * (room_temperature_span_c / settings.min_segment_room_temperature_span_c)
            + settings.segment_score_weight_outdoor_temperature_span
            * (outdoor_temperature_span_c / settings.min_segment_outdoor_temperature_span_c)
        )
        passes_hard_thresholds = (
            sample_count >= settings.min_segment_samples
            and meets_threshold(ufh_power_span_kw, settings.min_segment_ufh_power_span_kw)
            and meets_threshold(room_temperature_span_c, settings.min_segment_room_temperature_span_c)
            and meets_threshold(outdoor_temperature_span_c, settings.min_segment_outdoor_temperature_span_c)
            and meets_threshold(score, settings.min_segment_score)
        )
        return UFHActiveCalibrationSegmentQuality(
            raw_segment_index=raw_segment_index,
            selected_segment_index=raw_segment_index if passes_hard_thresholds else None,
            selected=passes_hard_thresholds,
            sample_count=sample_count,
            duration_hours=duration_hours,
            ufh_power_span_kw=ufh_power_span_kw,
            room_temperature_span_c=room_temperature_span_c,
            outdoor_temperature_span_c=outdoor_temperature_span_c,
            mean_gti_w_per_m2=mean_gti_w_per_m2,
            score=float(score),
        )

    for previous_row, next_row in zip(rows, rows[1:]):
        pair_is_valid = True
        if previous_row.hp_mode_last != settings.active_mode_name:
            pair_is_valid = False
        if next_row.hp_mode_last != settings.active_mode_name:
            pair_is_valid = False
        if previous_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            pair_is_valid = False
        if next_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            pair_is_valid = False
        if previous_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            pair_is_valid = False
        if next_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            pair_is_valid = False

        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            pair_is_valid = False
        if abs(dt_hours - dt_reference_hours) > settings.dt_compatibility_tolerance_hours:
            pair_is_valid = False
        if float(next_row.hp_thermal_power_mean_kw) < settings.min_ufh_power_kw:
            pair_is_valid = False

        forecast = forecast_lookup.nearest(
            next_row.bucket_end_utc,
            tolerance_hours=settings.forecast_alignment_tolerance_hours,
        )
        if forecast is None:
            pair_is_valid = False
        elif float(forecast.gti_w_per_m2) > settings.max_gti_w_per_m2:
            pair_is_valid = False

        if not pair_is_valid:
            flush_current_segment()
            continue

        current_segment_samples.append(
            UFHActiveCalibrationSample(
                interval_start_utc=previous_row.bucket_end_utc,
                interval_end_utc=next_row.bucket_end_utc,
                dt_hours=dt_hours,
                room_temperature_start_c=float(previous_row.room_temperature_last_c),
                room_temperature_end_c=float(next_row.room_temperature_last_c),
                outdoor_temperature_mean_c=float(next_row.outdoor_temperature_mean_c),
                gti_w_per_m2=float(forecast.gti_w_per_m2),
                internal_gain_proxy_kw=float(next_row.household_elec_power_mean_kw),
                ufh_power_mean_kw=float(next_row.hp_thermal_power_mean_kw),
                segment_index=len(raw_segments),
            )
        )

    flush_current_segment()

    raw_segment_qualities = tuple(
        compute_segment_quality(index, segment_samples) for index, segment_samples in enumerate(raw_segments)
    )
    selected_raw_indices = [
        quality.raw_segment_index for quality in raw_segment_qualities if quality.selected
    ]
    if settings.max_selected_segments is not None and len(selected_raw_indices) > settings.max_selected_segments:
        selected_raw_indices = [
            quality.raw_segment_index
            for quality in sorted(
                (quality for quality in raw_segment_qualities if quality.selected),
                key=lambda quality: (-quality.score, quality.raw_segment_index),
            )[: settings.max_selected_segments]
        ]
        selected_raw_indices.sort()

    raw_to_selected_index = {raw_index: selected_index for selected_index, raw_index in enumerate(selected_raw_indices)}
    selected_samples: list[UFHActiveCalibrationSample] = []
    for raw_index in selected_raw_indices:
        for sample in raw_segments[raw_index]:
            selected_samples.append(
                UFHActiveCalibrationSample(
                    interval_start_utc=sample.interval_start_utc,
                    interval_end_utc=sample.interval_end_utc,
                    dt_hours=sample.dt_hours,
                    room_temperature_start_c=sample.room_temperature_start_c,
                    room_temperature_end_c=sample.room_temperature_end_c,
                    outdoor_temperature_mean_c=sample.outdoor_temperature_mean_c,
                    gti_w_per_m2=sample.gti_w_per_m2,
                    internal_gain_proxy_kw=sample.internal_gain_proxy_kw,
                    ufh_power_mean_kw=sample.ufh_power_mean_kw,
                    segment_index=raw_to_selected_index[raw_index],
                )
            )

    segment_qualities = tuple(
        UFHActiveCalibrationSegmentQuality(
            raw_segment_index=quality.raw_segment_index,
            selected_segment_index=raw_to_selected_index.get(quality.raw_segment_index),
            selected=quality.raw_segment_index in raw_to_selected_index,
            sample_count=quality.sample_count,
            duration_hours=quality.duration_hours,
            ufh_power_span_kw=quality.ufh_power_span_kw,
            room_temperature_span_c=quality.room_temperature_span_c,
            outdoor_temperature_span_c=quality.outdoor_temperature_span_c,
            mean_gti_w_per_m2=quality.mean_gti_w_per_m2,
            score=quality.score,
        )
        for quality in raw_segment_qualities
    )

    if len(selected_samples) < settings.min_sample_count:
        raise ValueError(
            "Not enough active UFH samples for RC calibration: "
            f"required >= {settings.min_sample_count}, found {len(selected_samples)}."
        )
    return UFHActiveCalibrationDataset(
        samples=tuple(selected_samples),
        segment_qualities=segment_qualities,
    )


