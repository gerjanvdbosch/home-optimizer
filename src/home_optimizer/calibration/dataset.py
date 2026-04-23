"""Dataset builders for offline thermal-parameter calibration.

The first calibration stage uses only low-solar ``off`` windows to estimate an
*effective* building envelope model.  This is intentionally conservative:
without active UFH excitation the full 2-state UFH model is not structurally
identifiable, but the aggregate envelope loss and effective thermal capacity are.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from .models import (
    COPCalibrationDiagnostics,
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


@dataclass(frozen=True, slots=True)
class _COPRawBucket:
    """One admissible raw telemetry bucket prior to COP-only re-aggregation."""

    bucket_start_utc: datetime
    bucket_end_utc: datetime
    dt_hours: float
    mode_name: str
    outdoor_temperature_mean_c: float
    supply_target_temperature_mean_c: float
    supply_temperature_mean_c: float
    thermal_energy_kwh: float
    electric_energy_kwh: float

    @property
    def has_valid_bucket_cop(self) -> bool:
        """Return whether the raw bucket exposes a finite physical COP."""
        return self.electric_energy_kwh > 0.0

    @property
    def actual_cop(self) -> float:
        """Return the raw bucket COP when electric energy is strictly positive."""
        if self.electric_energy_kwh <= 0.0:
            raise ValueError("actual_cop is undefined when electric_energy_kwh <= 0.")
        return self.thermal_energy_kwh / self.electric_energy_kwh


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

    The standby-loss term and outlet temperature are evaluated on interval means
    instead of start-of-interval values. This reduces systematic false positives
    from measurement noise and from the fact that persisted telemetry buckets
    represent finite windows rather than infinitesimal samples.

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
    mean_t_top_c = 0.5 * (t_top_start_c + t_top_end_c)
    mean_t_bot_c = 0.5 * (t_bot_start_c + t_bot_end_c)
    q_loss_kw = (mean_t_top_c - t_amb_c) / p.R_loss + (mean_t_bot_c - t_amb_c) / p.R_loss
    tap_denominator = p.lambda_water * (mean_t_top_c - t_mains_c)
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


def _cop_supply_tracking_rmse_c(raw_buckets: Sequence[_COPRawBucket]) -> float:
    """Return the Δt-weighted supply-tracking RMSE over raw telemetry buckets [°C]."""
    dt_hours = np.array([bucket.dt_hours for bucket in raw_buckets], dtype=float)
    squared_errors = np.square(
        np.array([bucket.supply_temperature_mean_c for bucket in raw_buckets], dtype=float)
        - np.array([bucket.supply_target_temperature_mean_c for bucket in raw_buckets], dtype=float)
    )
    return float(np.sqrt(np.average(squared_errors, weights=dt_hours)))


def _cop_segment_tracking_margin(
    raw_buckets: Sequence[_COPRawBucket],
    settings: COPCalibrationSettings,
) -> float:
    """Return the tracking-quality margin used in COP segment scoring [-].

    UFH segments are scored against hydraulic supply-target tracking because the
    retained telemetry directly exercises the heating-curve path. For DHW
    segments that same target signal is not a reliable excitation-quality proxy
    in persisted runtime data, so DHW segments receive a neutral tracking score
    instead of being penalised by an unrelated UFH-oriented metric.
    """
    if raw_buckets[0].mode_name != settings.ufh_mode_name:
        return 1.0
    supply_tracking_rmse_c = _cop_supply_tracking_rmse_c(raw_buckets)
    return max(
        0.0,
        1.0 - supply_tracking_rmse_c / settings.max_segment_supply_tracking_rmse_c,
    )


def _cop_segment_passes_tracking_requirement(
    quality: COPCalibrationSegmentQuality,
    settings: COPCalibrationSettings,
) -> bool:
    """Return whether the segment passes the mode-specific tracking gate."""
    if quality.mode_name != settings.ufh_mode_name:
        return True
    return quality.supply_tracking_rmse_c <= settings.max_segment_supply_tracking_rmse_c


def _cop_segment_score(
    raw_buckets: Sequence[_COPRawBucket],
    calibration_samples: Sequence[COPCalibrationSample],
    settings: COPCalibrationSettings,
) -> float:
    """Return a dimensionless quality score for one contiguous COP segment."""
    sample_count = float(len(raw_buckets))
    thermal_energy_kwh = float(sum(bucket.thermal_energy_kwh for bucket in raw_buckets))
    actual_cops = np.array([sample.actual_cop for sample in calibration_samples], dtype=float)
    outdoor_temperatures_c = np.array(
        [bucket.outdoor_temperature_mean_c for bucket in raw_buckets],
        dtype=float,
    )
    supply_targets_c = np.array(
        [bucket.supply_target_temperature_mean_c for bucket in raw_buckets],
        dtype=float,
    )
    actual_cop_span = 0.0 if actual_cops.size == 0 else float(np.ptp(actual_cops))
    outdoor_span_c = float(np.ptp(outdoor_temperatures_c))
    supply_target_span_c = float(np.ptp(supply_targets_c))
    tracking_margin = _cop_segment_tracking_margin(raw_buckets, settings)

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
    if raw_buckets[0].mode_name == settings.ufh_mode_name:
        score += (
            settings.segment_score_weight_outdoor_temperature_span
            * outdoor_span_c
            / settings.min_ufh_segment_outdoor_temperature_span_c
            + settings.segment_score_weight_supply_target_span
            * supply_target_span_c
            / settings.min_ufh_segment_supply_target_span_c
        )
    return float(score)


def _aggregate_cop_window(raw_buckets: Sequence[_COPRawBucket]) -> COPCalibrationSample | None:
    """Aggregate contiguous raw COP buckets into one calibration sample using Δt-weighted means."""
    if not raw_buckets:
        raise ValueError("raw_buckets must not be empty.")
    electric_energy_kwh = float(sum(bucket.electric_energy_kwh for bucket in raw_buckets))
    if electric_energy_kwh <= 0.0:
        return None
    dt_hours = float(sum(bucket.dt_hours for bucket in raw_buckets))
    dt_weights = np.array([bucket.dt_hours for bucket in raw_buckets], dtype=float)
    return COPCalibrationSample(
        bucket_start_utc=raw_buckets[0].bucket_start_utc,
        bucket_end_utc=raw_buckets[-1].bucket_end_utc,
        dt_hours=dt_hours,
        mode_name=raw_buckets[0].mode_name,
        outdoor_temperature_mean_c=float(
            np.average(
                np.array([bucket.outdoor_temperature_mean_c for bucket in raw_buckets], dtype=float),
                weights=dt_weights,
            )
        ),
        supply_target_temperature_mean_c=float(
            np.average(
                np.array(
                    [bucket.supply_target_temperature_mean_c for bucket in raw_buckets],
                    dtype=float,
                ),
                weights=dt_weights,
            )
        ),
        supply_temperature_mean_c=float(
            np.average(
                np.array([bucket.supply_temperature_mean_c for bucket in raw_buckets], dtype=float),
                weights=dt_weights,
            )
        ),
        thermal_energy_kwh=float(sum(bucket.thermal_energy_kwh for bucket in raw_buckets)),
        electric_energy_kwh=electric_energy_kwh,
        source_bucket_count=len(raw_buckets),
    )


def _effective_cop_sample_min_electric_energy_kwh(settings: COPCalibrationSettings) -> float:
    """Return the effective post-reaggregation electric-energy threshold [kWh]."""
    return max(settings.min_electric_energy_kwh, settings.reaggregate_min_electric_energy_kwh)


def _reaggregate_cop_segment(
    raw_buckets: Sequence[_COPRawBucket],
    settings: COPCalibrationSettings,
) -> list[COPCalibrationSample]:
    """Greedily merge contiguous raw buckets into larger COP calibration windows."""
    if not raw_buckets:
        return []
    target_electric_energy_kwh = settings.reaggregate_min_electric_energy_kwh
    min_bucket_count = settings.reaggregate_min_bucket_count
    if target_electric_energy_kwh <= 0.0 and min_bucket_count <= 1:
        return [
            sample
            for sample in (_aggregate_cop_window((bucket,)) for bucket in raw_buckets)
            if sample is not None
        ]

    grouped_windows: list[list[_COPRawBucket]] = []
    current_window: list[_COPRawBucket] = []
    current_electric_energy_kwh = 0.0
    for bucket in raw_buckets:
        current_window.append(bucket)
        current_electric_energy_kwh += bucket.electric_energy_kwh
        if (
            current_electric_energy_kwh >= target_electric_energy_kwh
            and len(current_window) >= min_bucket_count
        ):
            grouped_windows.append(list(current_window))
            current_window = []
            current_electric_energy_kwh = 0.0

    if current_window:
        if grouped_windows and (
            current_electric_energy_kwh < target_electric_energy_kwh
            or len(current_window) < min_bucket_count
        ):
            grouped_windows[-1].extend(current_window)
        else:
            grouped_windows.append(list(current_window))

    effective_min_electric_energy_kwh = _effective_cop_sample_min_electric_energy_kwh(settings)
    samples: list[COPCalibrationSample] = []
    for grouped_window in grouped_windows:
        sample = _aggregate_cop_window(grouped_window)
        if sample is None:
            continue
        if sample.source_bucket_count < settings.reaggregate_min_bucket_count:
            continue
        if sample.electric_energy_kwh < effective_min_electric_energy_kwh:
            continue
        if sample.thermal_energy_kwh < settings.min_thermal_energy_kwh:
            continue
        if sample.actual_cop <= 1.0 or sample.actual_cop > settings.cop_max:
            continue
        samples.append(sample)
    return samples


def _cop_segment_quality(
    raw_buckets: Sequence[_COPRawBucket],
    calibration_samples: Sequence[COPCalibrationSample],
    *,
    raw_segment_index: int,
    settings: COPCalibrationSettings,
) -> COPCalibrationSegmentQuality:
    """Summarise one raw contiguous COP segment before top-N selection."""
    actual_cops = np.array([sample.actual_cop for sample in calibration_samples], dtype=float)
    outdoor_temperatures_c = np.array(
        [bucket.outdoor_temperature_mean_c for bucket in raw_buckets],
        dtype=float,
    )
    supply_targets_c = np.array(
        [bucket.supply_target_temperature_mean_c for bucket in raw_buckets],
        dtype=float,
    )
    quality = COPCalibrationSegmentQuality(
        raw_segment_index=raw_segment_index,
        mode_name=raw_buckets[0].mode_name,
        selected=False,
        sample_count=len(raw_buckets),
        duration_hours=float(sum(bucket.dt_hours for bucket in raw_buckets)),
        thermal_energy_kwh=float(sum(bucket.thermal_energy_kwh for bucket in raw_buckets)),
        electric_energy_kwh=float(sum(bucket.electric_energy_kwh for bucket in raw_buckets)),
        outdoor_temperature_span_c=float(np.ptp(outdoor_temperatures_c)),
        supply_target_temperature_span_c=float(np.ptp(supply_targets_c)),
        actual_cop_span=0.0 if actual_cops.size == 0 else float(np.ptp(actual_cops)),
        supply_tracking_rmse_c=_cop_supply_tracking_rmse_c(raw_buckets),
        score=_cop_segment_score(raw_buckets, calibration_samples, settings),
    )
    hard_selected = (
        quality.sample_count >= settings.min_segment_samples
        and quality.thermal_energy_kwh >= settings.min_segment_thermal_energy_kwh
        and quality.actual_cop_span >= settings.min_segment_actual_cop_span
        and _cop_segment_passes_tracking_requirement(quality, settings)
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


def _cop_samples_are_contiguous(
    previous_bucket: _COPRawBucket,
    current_bucket: _COPRawBucket,
    settings: COPCalibrationSettings,
) -> bool:
    """Return whether two same-mode COP buckets belong to one continuous operating segment."""
    if previous_bucket.mode_name != current_bucket.mode_name:
        return False
    gap_hours = (
        current_bucket.bucket_start_utc - previous_bucket.bucket_end_utc
    ).total_seconds() / _SECONDS_PER_HOUR
    if gap_hours < 0.0:
        return False
    max_allowed_gap_hours = (
        min(previous_bucket.dt_hours, current_bucket.dt_hours)
        * settings.max_segment_boundary_gap_ratio
    )
    return gap_hours <= max_allowed_gap_hours


def _collect_cop_segments(
    aggregates: Sequence[TelemetryAggregate],
    settings: COPCalibrationSettings,
) -> tuple[list[list[_COPRawBucket]], Counter[str], Counter[str]]:
    """Collect raw contiguous COP segments and bucket-level rejection counters."""
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    raw_segments: list[list[_COPRawBucket]] = []
    current_segment: list[_COPRawBucket] = []
    accepted_modes = {settings.ufh_mode_name, settings.dhw_mode_name}
    stage_counts: Counter[str] = Counter(raw_row_count=len(rows))
    rejection_counts: Counter[str] = Counter()

    def flush_segment() -> None:
        nonlocal current_segment
        if current_segment:
            raw_segments.append(current_segment)
            current_segment = []

    for row in rows:
        mode_name = str(row.hp_mode_last)
        if mode_name not in accepted_modes:
            rejection_counts["mode_not_ufh_or_dhw"] += 1
            flush_segment()
            continue
        stage_counts["mode_accepted_count"] += 1

        if float(row.defrost_active_fraction) > settings.max_defrost_active_fraction:
            rejection_counts["defrost_fraction"] += 1
            flush_segment()
            continue
        stage_counts["defrost_accepted_count"] += 1

        if float(row.booster_heater_active_fraction) > settings.max_booster_active_fraction:
            rejection_counts["booster_fraction"] += 1
            flush_segment()
            continue
        stage_counts["booster_accepted_count"] += 1

        dt_hours = (row.bucket_end_utc - row.bucket_start_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0:
            rejection_counts["non_positive_dt"] += 1
            flush_segment()
            continue
        stage_counts["dt_accepted_count"] += 1

        thermal_energy_kwh = float(row.hp_thermal_power_mean_kw) * dt_hours
        electric_energy_kwh = float(row.hp_electric_energy_delta_kwh)
        if thermal_energy_kwh < settings.min_thermal_energy_kwh:
            rejection_counts["thermal_energy_below_min"] += 1
            flush_segment()
            continue
        stage_counts["thermal_energy_accepted_count"] += 1

        supply_target_temperature_mean_c = float(row.hp_supply_target_temperature_mean_c)
        supply_temperature_mean_c = float(row.hp_supply_temperature_mean_c)
        if not np.isfinite(supply_target_temperature_mean_c):
            rejection_counts["target_supply_not_finite"] += 1
            flush_segment()
            continue
        if not np.isfinite(supply_temperature_mean_c):
            rejection_counts["measured_supply_not_finite"] += 1
            flush_segment()
            continue
        stage_counts["finite_supply_accepted_count"] += 1

        raw_bucket = _COPRawBucket(
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
        if electric_energy_kwh >= settings.min_electric_energy_kwh:
            stage_counts["electric_energy_accepted_count"] += 1
        else:
            rejection_counts["electric_energy_below_min"] += 1
        if raw_bucket.has_valid_bucket_cop:
            if 1.0 < raw_bucket.actual_cop <= settings.cop_max:
                stage_counts["cop_accepted_count"] += 1
            elif raw_bucket.actual_cop <= 1.0:
                rejection_counts["actual_cop_leq_1"] += 1
            else:
                rejection_counts["actual_cop_above_cop_max"] += 1

        if current_segment and not _cop_samples_are_contiguous(
            current_segment[-1],
            raw_bucket,
            settings,
        ):
            rejection_counts["segment_break_mode_or_gap"] += 1
            flush_segment()
        current_segment.append(raw_bucket)

    flush_segment()
    return raw_segments, stage_counts, rejection_counts


def _cop_segment_failure_counts(
    *,
    uncapped_qualities: Sequence[COPCalibrationSegmentQuality],
    capped_qualities: Sequence[COPCalibrationSegmentQuality],
    settings: COPCalibrationSettings,
) -> Counter[str]:
    """Return counters for the exact segment-selection rules that rejected each segment."""
    failure_counts: Counter[str] = Counter()
    for uncapped_quality, capped_quality in zip(uncapped_qualities, capped_qualities, strict=True):
        if capped_quality.selected:
            continue
        if uncapped_quality.selected and not capped_quality.selected:
            failure_counts["top_n_cap"] += 1
            continue
        if uncapped_quality.sample_count < settings.min_segment_samples:
            failure_counts["sample_count"] += 1
        if uncapped_quality.thermal_energy_kwh < settings.min_segment_thermal_energy_kwh:
            failure_counts["thermal_energy"] += 1
        if uncapped_quality.actual_cop_span < settings.min_segment_actual_cop_span:
            failure_counts["actual_cop_span"] += 1
        if not _cop_segment_passes_tracking_requirement(uncapped_quality, settings):
            failure_counts["supply_tracking_rmse"] += 1
        if uncapped_quality.mode_name == settings.ufh_mode_name:
            if uncapped_quality.outdoor_temperature_span_c < settings.min_ufh_segment_outdoor_temperature_span_c:
                failure_counts["ufh_outdoor_span"] += 1
            if (
                uncapped_quality.supply_target_temperature_span_c
                < settings.min_ufh_segment_supply_target_span_c
            ):
                failure_counts["ufh_supply_target_span"] += 1
        if uncapped_quality.score < settings.min_segment_score:
            failure_counts["score"] += 1
    return failure_counts


def diagnose_cop_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    settings: COPCalibrationSettings,
) -> COPCalibrationDiagnostics:
    """Diagnose COP bucket filtering and segment selection without requiring a fit-ready dataset."""
    raw_segments, stage_counts, rejection_counts = _collect_cop_segments(aggregates, settings)
    reaggregated_segments = [
        _reaggregate_cop_segment(raw_segment, settings) for raw_segment in raw_segments
    ]
    uncapped_qualities = [
        _cop_segment_quality(
            raw_segment,
            reaggregated_segment,
            raw_segment_index=index,
            settings=settings,
        )
        for index, (raw_segment, reaggregated_segment) in enumerate(
            zip(raw_segments, reaggregated_segments, strict=True)
        )
    ]
    capped_qualities = _apply_cop_segment_caps(list(uncapped_qualities), settings)
    selected_segment_indices = {
        quality.raw_segment_index for quality in capped_qualities if quality.selected
    }
    selected_samples = [
        sample
        for segment_index, segment_samples in enumerate(reaggregated_segments)
        if segment_index in selected_segment_indices
        for sample in segment_samples
    ]
    segment_failure_counts = _cop_segment_failure_counts(
        uncapped_qualities=uncapped_qualities,
        capped_qualities=capped_qualities,
        settings=settings,
    )
    return COPCalibrationDiagnostics(
        raw_row_count=stage_counts["raw_row_count"],
        mode_accepted_count=stage_counts["mode_accepted_count"],
        defrost_accepted_count=stage_counts["defrost_accepted_count"],
        booster_accepted_count=stage_counts["booster_accepted_count"],
        dt_accepted_count=stage_counts["dt_accepted_count"],
        thermal_energy_accepted_count=stage_counts["thermal_energy_accepted_count"],
        electric_energy_accepted_count=stage_counts["electric_energy_accepted_count"],
        finite_supply_accepted_count=stage_counts["finite_supply_accepted_count"],
        cop_accepted_count=stage_counts["cop_accepted_count"],
        raw_segment_count=len(raw_segments),
        selected_segment_count=sum(quality.selected for quality in capped_qualities),
        selected_sample_count=len(selected_samples),
        selected_ufh_sample_count=sum(
            sample.mode_name == settings.ufh_mode_name for sample in selected_samples
        ),
        selected_dhw_sample_count=sum(
            sample.mode_name == settings.dhw_mode_name for sample in selected_samples
        ),
        bucket_rejection_counts=tuple(sorted(rejection_counts.items())),
        segment_failure_counts=tuple(sorted(segment_failure_counts.items())),
        segment_qualities=tuple(capped_qualities),
    )


def build_cop_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    settings: COPCalibrationSettings,
) -> COPCalibrationDataset:
    """Build a filtered operating-bucket dataset for offline COP calibration.

    The builder first filters out non-operating and physically invalid raw buckets,
    then groups contiguous same-mode runs. Within each run it optionally re-aggregates
    multiple persisted telemetry buckets into larger COP calibration windows so coarse
    electrical counters remain usable. Segment scoring still evaluates the full raw
    excitation, while the fitter consumes the re-aggregated windows.
    """
    raw_segments, _, _ = _collect_cop_segments(aggregates, settings)
    reaggregated_segments = [
        _reaggregate_cop_segment(raw_segment, settings) for raw_segment in raw_segments
    ]
    segment_qualities = _apply_cop_segment_caps(
        [
            _cop_segment_quality(
                raw_segment,
                reaggregated_segment,
                raw_segment_index=index,
                settings=settings,
            )
            for index, (raw_segment, reaggregated_segment) in enumerate(
                zip(raw_segments, reaggregated_segments, strict=True)
            )
        ],
        settings,
    )
    selected_segment_indices = {
        quality.raw_segment_index for quality in segment_qualities if quality.selected
    }
    samples = [
        sample
        for segment_index, segment_samples in enumerate(reaggregated_segments)
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
    threshold. Those windows are then replayed with ``V_tap = 0``.

    The pair-level layer-spread gate is intentionally only a *minimum excitation
    floor*: charging can remix a real tank, so low-but-nonzero spreads may still
    contain usable ``R_strat`` information when the *segment as a whole* shows a
    measurable rise and spread evolution.

    Existing calibrated DHW sensor-bias corrections are applied during the
    no-draw selection logic only. The dataset itself continues to store the raw
    measured temperatures so the active fitter can either keep those injected
    biases fixed or refine them further when ``fit_temperature_biases=True``.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    raw_segments: list[list[DHWActiveCalibrationSample]] = []
    current_segment_samples: list[DHWActiveCalibrationSample] = []
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
        # Active-DHW replay uses the exact sample-specific ``dt_hours`` stored in
        # each sample, so small persisted-bucket jitter is physically acceptable.
        # Only non-positive or excessively large gaps are rejected here.

        p_dhw_mean_kw = float(next_row.hp_thermal_power_mean_kw)
        if p_dhw_mean_kw < settings.min_dhw_power_kw:
            pair_is_valid = False

        layer_spread_start_c = abs(
            (float(previous_row.dhw_top_temperature_last_c) + settings.initial_t_top_bias_c)
            - (float(previous_row.dhw_bottom_temperature_last_c) + settings.initial_t_bot_bias_c)
        )
        layer_spread_end_c = abs(
            (float(next_row.dhw_top_temperature_last_c) + settings.initial_t_top_bias_c)
            - (float(next_row.dhw_bottom_temperature_last_c) + settings.initial_t_bot_bias_c)
        )
        if max(layer_spread_start_c, layer_spread_end_c) < settings.min_layer_temperature_spread_c:
            pair_is_valid = False

        t_mains_c = float(next_row.t_mains_estimated_mean_c)
        t_amb_c = float(next_row.boiler_ambient_temp_mean_c) + settings.ambient_temperature_bias_c
        implied_v_tap_m3_per_h = _implied_v_tap_m3_per_h(
            t_top_start_c=float(previous_row.dhw_top_temperature_last_c) + settings.initial_t_top_bias_c,
            t_bot_start_c=float(previous_row.dhw_bottom_temperature_last_c) + settings.initial_t_bot_bias_c,
            t_top_end_c=float(next_row.dhw_top_temperature_last_c) + settings.initial_t_top_bias_c,
            t_bot_end_c=float(next_row.dhw_bottom_temperature_last_c) + settings.initial_t_bot_bias_c,
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

