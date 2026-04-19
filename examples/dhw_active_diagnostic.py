"""Diagnose why active-DHW calibration does or does not learn ``R_strat``.

This helper inspects persisted telemetry and reports which active-DHW dataset
filters reject candidate pairs. It also probes a small threshold grid so real
installations can see whether sample starvation is caused by power, layer-spread,
or implied-tap assumptions.
"""

from __future__ import annotations

from collections import Counter
from itertools import product

from home_optimizer.calibration.dataset import _implied_v_tap_m3_per_h
from home_optimizer.calibration.service import (
    _infer_calibration_replay_dt_hours,
    _load_calibration_aggregates,
    build_dhw_active_dataset_from_repository,
    calibrate_dhw_standby_from_repository,
)
from home_optimizer.calibration.settings_factory import build_dhw_active_calibration_settings
from home_optimizer.calibration.settings_factory import build_dhw_standby_calibration_settings
from home_optimizer.telemetry.repository import TelemetryRepository
from home_optimizer.types import DHWParameters


def main() -> None:
    repository = TelemetryRepository(database_url="sqlite:///database.sqlite3")
    rows = _load_calibration_aggregates(repository)
    dt_ref_hours = _infer_calibration_replay_dt_hours(repository)
    layer_capacity_kwh_per_k = 200 * 1.1628e-3 / 2.0
    standby_result = calibrate_dhw_standby_from_repository(
        repository,
        build_dhw_standby_calibration_settings(
            dt_hours=dt_ref_hours,
            reference_c_top_kwh_per_k=layer_capacity_kwh_per_k,
            reference_c_bot_kwh_per_k=layer_capacity_kwh_per_k,
        ),
    )
    reference = DHWParameters(
        dt_hours=dt_ref_hours,
        C_top=layer_capacity_kwh_per_k,
        C_bot=layer_capacity_kwh_per_k,
        R_strat=10.0,
        R_loss=standby_result.suggested_r_loss_k_per_kw,
    )
    settings = build_dhw_active_calibration_settings(reference_parameters=reference)

    pair_rejection_counts: Counter[str] = Counter()
    dhw_pair_count = 0
    for previous_row, next_row in zip(rows, rows[1:]):
        if previous_row.hp_mode_last != settings.active_mode_name or next_row.hp_mode_last != settings.active_mode_name:
            continue
        dhw_pair_count += 1
        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / 3600.0
        implied_v_tap_m3_per_h = _implied_v_tap_m3_per_h(
            t_top_start_c=float(previous_row.dhw_top_temperature_last_c),
            t_bot_start_c=float(previous_row.dhw_bottom_temperature_last_c),
            t_top_end_c=float(next_row.dhw_top_temperature_last_c),
            t_bot_end_c=float(next_row.dhw_bottom_temperature_last_c),
            dt_hours=dt_hours,
            p_dhw_mean_kw=float(next_row.hp_thermal_power_mean_kw),
            t_mains_c=float(next_row.t_mains_estimated_mean_c),
            t_amb_c=float(next_row.boiler_ambient_temp_mean_c),
            settings=settings,
        )
        layer_spread_start_c = abs(
            float(previous_row.dhw_top_temperature_last_c) - float(previous_row.dhw_bottom_temperature_last_c)
        )
        layer_spread_end_c = abs(
            float(next_row.dhw_top_temperature_last_c) - float(next_row.dhw_bottom_temperature_last_c)
        )
        if float(next_row.hp_thermal_power_mean_kw) < settings.min_dhw_power_kw:
            pair_rejection_counts["power_below_min"] += 1
        if max(layer_spread_start_c, layer_spread_end_c) < settings.min_layer_temperature_spread_c:
            pair_rejection_counts["layer_spread_below_min"] += 1
        if implied_v_tap_m3_per_h > settings.max_implied_tap_m3_per_h:
            pair_rejection_counts["implied_tap_above_max"] += 1
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            pair_rejection_counts["dt_out_of_range"] += 1

    print(f"dt_ref_hours={dt_ref_hours:.6f}")
    print(f"standby_r_loss_k_per_kw={standby_result.suggested_r_loss_k_per_kw:.6f}")
    print(f"dhw_pair_count={dhw_pair_count}")
    print(f"default_settings={{'max_implied_tap_m3_per_h': {settings.max_implied_tap_m3_per_h}, 'min_layer_temperature_spread_c': {settings.min_layer_temperature_spread_c}}}")
    print("pair_rejection_counts=", dict(pair_rejection_counts))

    print("\nthreshold sweep:")
    for min_layer_temperature_spread_c, max_implied_tap_m3_per_h in product((3.0, 2.0, 1.0, 0.5), (0.01, 0.02, 0.05, 0.1)):
        try:
            dataset = build_dhw_active_dataset_from_repository(
                repository,
                build_dhw_active_calibration_settings(
                    reference_parameters=reference,
                    min_layer_temperature_spread_c=min_layer_temperature_spread_c,
                    max_implied_tap_m3_per_h=max_implied_tap_m3_per_h,
                ),
            )
            print(
                "OK",
                {
                    "min_layer_temperature_spread_c": min_layer_temperature_spread_c,
                    "max_implied_tap_m3_per_h": max_implied_tap_m3_per_h,
                    "sample_count": dataset.sample_count,
                    "segment_count": dataset.segment_count,
                    "raw_segment_count": dataset.raw_segment_count,
                    "dropped_segment_count": dataset.dropped_segment_count,
                },
            )
        except Exception as exc:  # noqa: BLE001
            print(
                "FAIL",
                {
                    "min_layer_temperature_spread_c": min_layer_temperature_spread_c,
                    "max_implied_tap_m3_per_h": max_implied_tap_m3_per_h,
                    "error": str(exc),
                },
            )

    print("\nfully relaxed probe:")
    try:
        relaxed_dataset = build_dhw_active_dataset_from_repository(
            repository,
            build_dhw_active_calibration_settings(
                reference_parameters=reference,
                min_sample_count=2,
                min_segment_samples=2,
                min_layer_temperature_spread_c=0.5,
                max_implied_tap_m3_per_h=0.1,
                min_segment_delivered_energy_kwh=0.05,
                min_segment_mean_layer_spread_c=0.5,
                min_segment_layer_spread_span_c=0.001,
                min_segment_bottom_temperature_rise_c=0.001,
                min_segment_top_temperature_rise_c=0.001,
            ),
        )
        print(
            "OK",
            {
                "sample_count": relaxed_dataset.sample_count,
                "segment_count": relaxed_dataset.segment_count,
                "raw_segment_count": relaxed_dataset.raw_segment_count,
                "dropped_segment_count": relaxed_dataset.dropped_segment_count,
            },
        )
        for quality in relaxed_dataset.segment_qualities:
            print(quality)
    except Exception as exc:  # noqa: BLE001
        print("FAIL", {"error": str(exc)})


if __name__ == "__main__":
    main()

