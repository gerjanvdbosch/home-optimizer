"""Tests for offline thermal-parameter calibration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import cos, sin
from typing import cast

import numpy as np

from home_optimizer.calibration import (
    AutomaticCalibrationSettings,
    COPCalibrationDiagnostics,
    COPCalibrationDataset,
    COPCalibrationSample,
    COPCalibrationSettings,
    DHWActiveCalibrationDataset,
    DHWActiveCalibrationSample,
    DHWActiveCalibrationSettings,
    DHWStandbyCalibrationDataset,
    DHWStandbyCalibrationSample,
    DHWStandbyCalibrationSettings,
    build_cop_calibration_dataset,
    build_automatic_calibration_snapshot,
    build_dhw_active_calibration_dataset,
    diagnose_cop_calibration_dataset,
    UFHActiveCalibrationDataset,
    UFHActiveCalibrationSample,
    UFHActiveCalibrationSettings,
    UFHCalibrationDataset,
    UFHCalibrationSample,
    UFHOffCalibrationSettings,
    build_dhw_standby_calibration_dataset,
    build_ufh_active_calibration_dataset,
    calibrate_ufh_active_rc,
    calibrate_cop_model,
    calibrate_dhw_active_stratification,
    calibrate_dhw_standby_loss,
    build_ufh_off_calibration_dataset,
    calibrate_ufh_off_envelope,
)
from home_optimizer.calibration.models import (
    DEFAULT_ACTIVE_MAX_GTI_W_PER_M2,
    DEFAULT_MAX_GTI_W_PER_M2,
    DEFAULT_MIN_SAMPLE_COUNT,
)
from home_optimizer.application.optimizer import RunRequest
from home_optimizer.domain.dhw.model import DHWModel
from home_optimizer.domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters
from home_optimizer.domain.ufh.model import ThermalModel, solar_gain_kw
from home_optimizer.telemetry import TelemetryRepository
from home_optimizer.types import CalibrationParameterOverrides, CalibrationSnapshotPayload, DHWParameters, ThermalParameters


class SimpleNamespace:
    """Minimal local namespace helper for test fixtures.

    This avoids depending on a top-level ``types`` import while still offering the
    attribute-style test data used throughout this module.
    """

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def test_build_ufh_off_calibration_dataset_filters_to_low_solar_off_windows() -> None:
    """Only consecutive low-solar off-mode windows may enter the first-stage dataset."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.0,
            outdoor_temperature_mean_c=10.0,
            household_elec_power_mean_kw=0.2,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=19.9,
            outdoor_temperature_mean_c=10.0,
            household_elec_power_mean_kw=0.2,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=19.8,
            outdoor_temperature_mean_c=10.0,
            household_elec_power_mean_kw=0.2,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=19.7,
            outdoor_temperature_mean_c=10.0,
            household_elec_power_mean_kw=0.2,
        ),
    ]
    forecasts = [
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=5), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=10), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=15), gti_w_per_m2=100.0),
    ]

    dataset = build_ufh_off_calibration_dataset(
        aggregates=cast(list, aggregates),
        forecast_rows=cast(list, forecasts),
        settings=UFHOffCalibrationSettings(min_sample_count=2, max_gti_w_per_m2=25.0),
    )

    assert dataset.sample_count == 2
    sample = dataset.samples[0]
    assert sample.room_temperature_start_c == 20.0
    assert sample.room_temperature_end_c == 19.9
    assert sample.gti_w_per_m2 == 0.0


def test_ufh_off_calibration_settings_use_passive_defaults() -> None:
    """Passive UFH calibration defaults must stay aligned with the low-solar off-stage design."""
    settings = UFHOffCalibrationSettings()

    assert settings.max_gti_w_per_m2 == DEFAULT_MAX_GTI_W_PER_M2
    assert settings.min_sample_count == DEFAULT_MIN_SAMPLE_COUNT


def test_calibrate_ufh_off_envelope_recovers_synthetic_parameters() -> None:
    """Least-squares off-mode calibration must recover synthetic envelope parameters."""
    tau_true_hours = 91.0
    dt_hours = 5.0 / 60.0
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    t_room = 21.0
    t_out_profile = np.linspace(6.0, 10.0, 48)
    samples: list[UFHCalibrationSample] = []
    for step_k, t_out_c in enumerate(t_out_profile):
        t_next = t_room + dt_hours / tau_true_hours * (-(t_room - t_out_c))
        interval_start = start + timedelta(hours=step_k * dt_hours)
        interval_end = interval_start + timedelta(hours=dt_hours)
        samples.append(
            UFHCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=dt_hours,
                room_temperature_start_c=t_room,
                room_temperature_end_c=t_next,
                outdoor_temperature_mean_c=float(t_out_c),
                gti_w_per_m2=0.0,
                household_elec_power_mean_kw=0.2,
            )
        )
        t_room = t_next

    dataset = UFHCalibrationDataset(samples=tuple(samples))
    settings = UFHOffCalibrationSettings(
        min_sample_count=10,
        initial_tau_hours=24.0,
        reference_c_eff_kwh_per_k=14.0,
    )

    result = calibrate_ufh_off_envelope(dataset, settings)

    np.testing.assert_allclose(result.tau_house_hours, tau_true_hours, rtol=1e-2)
    np.testing.assert_allclose(result.suggested_r_ro_k_per_kw, tau_true_hours / 14.0, rtol=1e-2)
    assert result.rmse_room_temperature_c < 1e-6


def test_build_cop_calibration_dataset_filters_to_valid_operating_buckets() -> None:
    """Only physically meaningful UFH/DHW operating buckets may enter the COP dataset."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_start_utc=start,
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=35.0,
            hp_supply_temperature_mean_c=34.0,
            hp_thermal_power_mean_kw=3.6,
            hp_electric_energy_delta_kwh=0.25,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=5),
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.0,
            hp_supply_target_temperature_mean_c=37.0,
            hp_supply_temperature_mean_c=36.5,
            hp_thermal_power_mean_kw=3.8,
            hp_electric_energy_delta_kwh=0.28,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.0,
            hp_thermal_power_mean_kw=2.4,
            hp_electric_energy_delta_kwh=0.18,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=15),
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=7.0,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.2,
            hp_thermal_power_mean_kw=2.6,
            hp_electric_energy_delta_kwh=0.19,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=20),
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=25.0,
            hp_supply_temperature_mean_c=25.0,
            hp_thermal_power_mean_kw=0.0,
            hp_electric_energy_delta_kwh=0.0,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=25),
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="ufh",
            defrost_active_fraction=1.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=36.0,
            hp_supply_temperature_mean_c=35.0,
            hp_thermal_power_mean_kw=3.0,
            hp_electric_energy_delta_kwh=0.25,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=30),
            bucket_end_utc=start + timedelta(minutes=35),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=36.0,
            hp_supply_temperature_mean_c=35.0,
            hp_thermal_power_mean_kw=1.3,
            hp_electric_energy_delta_kwh=0.06,
        ),
    ]

    dataset = build_cop_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=COPCalibrationSettings(
            min_sample_count=2,
            min_ufh_curve_sample_count=2,
            min_thermal_energy_kwh=0.1,
            min_electric_energy_kwh=0.05,
            reaggregate_min_electric_energy_kwh=0.0,
            reaggregate_min_bucket_count=1,
            min_segment_samples=2,
            min_segment_thermal_energy_kwh=0.2,
            min_segment_actual_cop_span=0.02,
            min_ufh_segment_outdoor_temperature_span_c=0.5,
            min_ufh_segment_supply_target_span_c=0.5,
        ),
    )

    assert dataset.sample_count == 4
    assert dataset.ufh_sample_count == 2
    assert dataset.dhw_sample_count == 2
    assert dataset.raw_segment_count == 3
    assert dataset.dropped_segment_count == 1
    assert all(sample.actual_cop > 1.0 for sample in dataset.samples)


def test_build_cop_calibration_dataset_drops_weak_ufh_segments() -> None:
    """Weak UFH COP segments with poor excitation must be rejected before fitting."""
    start = datetime(2026, 4, 17, 1, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_start_utc=start,
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=35.0,
            hp_supply_temperature_mean_c=34.9,
            hp_thermal_power_mean_kw=2.2,
            hp_electric_energy_delta_kwh=0.15,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=5),
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.1,
            hp_supply_target_temperature_mean_c=35.1,
            hp_supply_temperature_mean_c=35.0,
            hp_thermal_power_mean_kw=2.3,
            hp_electric_energy_delta_kwh=0.16,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=7.5,
            hp_supply_target_temperature_mean_c=25.0,
            hp_supply_temperature_mean_c=25.0,
            hp_thermal_power_mean_kw=0.0,
            hp_electric_energy_delta_kwh=0.0,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=15),
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.0,
            hp_supply_target_temperature_mean_c=38.0,
            hp_supply_temperature_mean_c=37.3,
            hp_thermal_power_mean_kw=3.0,
            hp_electric_energy_delta_kwh=0.21,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=20),
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=2.5,
            hp_supply_target_temperature_mean_c=41.0,
            hp_supply_temperature_mean_c=40.1,
            hp_thermal_power_mean_kw=3.3,
            hp_electric_energy_delta_kwh=0.25,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=25),
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.0,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.0,
            hp_thermal_power_mean_kw=2.5,
            hp_electric_energy_delta_kwh=0.18,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=30),
            bucket_end_utc=start + timedelta(minutes=35),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=4.5,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.2,
            hp_thermal_power_mean_kw=2.7,
            hp_electric_energy_delta_kwh=0.17,
        ),
    ]

    dataset = build_cop_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=COPCalibrationSettings(
            min_sample_count=4,
            min_ufh_curve_sample_count=2,
            min_thermal_energy_kwh=0.1,
            min_electric_energy_kwh=0.05,
            reaggregate_min_electric_energy_kwh=0.0,
            reaggregate_min_bucket_count=1,
            min_segment_samples=2,
            min_segment_thermal_energy_kwh=0.2,
            min_segment_actual_cop_span=0.02,
            min_ufh_segment_outdoor_temperature_span_c=1.0,
            min_ufh_segment_supply_target_span_c=1.0,
        ),
    )

    assert dataset.sample_count == 4
    assert dataset.ufh_sample_count == 2
    assert dataset.dhw_sample_count == 2
    assert dataset.raw_segment_count == 3
    assert dataset.dropped_segment_count == 1
    selected_ufh = [sample for sample in dataset.samples if sample.mode_name == "ufh"]
    assert selected_ufh[0].outdoor_temperature_mean_c == 5.0


def test_build_cop_calibration_dataset_keeps_best_ufh_segments_when_capped() -> None:
    """Top-N COP segment selection must keep the best-excited UFH run."""
    start = datetime(2026, 4, 17, 2, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_start_utc=start,
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=35.0,
            hp_supply_temperature_mean_c=34.8,
            hp_thermal_power_mean_kw=2.1,
            hp_electric_energy_delta_kwh=0.15,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=5),
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.5,
            hp_supply_target_temperature_mean_c=36.5,
            hp_supply_temperature_mean_c=36.1,
            hp_thermal_power_mean_kw=2.2,
            hp_electric_energy_delta_kwh=0.16,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.5,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.1,
            hp_thermal_power_mean_kw=2.5,
            hp_electric_energy_delta_kwh=0.18,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=15),
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=4.5,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.0,
            hp_thermal_power_mean_kw=2.6,
            hp_electric_energy_delta_kwh=0.18,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=20),
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=4.0,
            hp_supply_target_temperature_mean_c=25.0,
            hp_supply_temperature_mean_c=25.0,
            hp_thermal_power_mean_kw=0.0,
            hp_electric_energy_delta_kwh=0.0,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=25),
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.0,
            hp_supply_target_temperature_mean_c=38.0,
            hp_supply_temperature_mean_c=37.4,
            hp_thermal_power_mean_kw=3.0,
            hp_electric_energy_delta_kwh=0.21,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=30),
            bucket_end_utc=start + timedelta(minutes=35),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=1.5,
            hp_supply_target_temperature_mean_c=42.0,
            hp_supply_temperature_mean_c=41.1,
            hp_thermal_power_mean_kw=3.5,
            hp_electric_energy_delta_kwh=0.27,
        ),
    ]

    dataset = build_cop_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=COPCalibrationSettings(
            min_sample_count=4,
            min_ufh_curve_sample_count=2,
            min_thermal_energy_kwh=0.1,
            min_electric_energy_kwh=0.05,
            reaggregate_min_electric_energy_kwh=0.0,
            reaggregate_min_bucket_count=1,
            min_segment_samples=2,
            min_segment_thermal_energy_kwh=0.2,
            min_segment_actual_cop_span=0.02,
            min_ufh_segment_outdoor_temperature_span_c=1.0,
            min_ufh_segment_supply_target_span_c=1.0,
            max_selected_ufh_segments=1,
        ),
    )

    assert dataset.sample_count == 4
    assert dataset.ufh_sample_count == 2
    assert dataset.dhw_sample_count == 2
    assert dataset.raw_segment_count == 3
    assert dataset.dropped_segment_count == 1
    selected_scores = [quality.score for quality in dataset.segment_qualities if quality.selected and quality.mode_name == "ufh"]
    dropped_scores = [quality.score for quality in dataset.segment_qualities if not quality.selected and quality.mode_name == "ufh"]
    assert selected_scores[0] > dropped_scores[0]
    assert dataset.samples[0].mode_name == "dhw"
    assert dataset.samples[-1].outdoor_temperature_mean_c == 1.5


def test_build_cop_calibration_dataset_keeps_dhw_segments_with_high_supply_tracking_rmse() -> None:
    """DHW COP selection must not reuse the UFH supply-tracking gate as a hard reject."""
    start = datetime(2026, 4, 17, 2, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_start_utc=start,
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=7.5,
            hp_supply_target_temperature_mean_c=35.0,
            hp_supply_temperature_mean_c=34.6,
            hp_thermal_power_mean_kw=2.8,
            hp_electric_energy_delta_kwh=0.20,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=5),
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.0,
            hp_supply_target_temperature_mean_c=38.0,
            hp_supply_temperature_mean_c=37.2,
            hp_thermal_power_mean_kw=3.2,
            hp_electric_energy_delta_kwh=0.22,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.5,
            hp_supply_target_temperature_mean_c=30.0,
            hp_supply_temperature_mean_c=55.0,
            hp_thermal_power_mean_kw=2.5,
            hp_electric_energy_delta_kwh=0.20,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=15),
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.0,
            hp_supply_target_temperature_mean_c=30.0,
            hp_supply_temperature_mean_c=56.0,
            hp_thermal_power_mean_kw=2.9,
            hp_electric_energy_delta_kwh=0.18,
        ),
    ]

    dataset = build_cop_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=COPCalibrationSettings(
            min_sample_count=4,
            min_ufh_curve_sample_count=2,
            min_thermal_energy_kwh=0.1,
            min_electric_energy_kwh=0.05,
            reaggregate_min_electric_energy_kwh=0.0,
            reaggregate_min_bucket_count=1,
            min_segment_samples=2,
            min_segment_thermal_energy_kwh=0.2,
            min_segment_actual_cop_span=0.02,
            max_segment_supply_tracking_rmse_c=1.0,
            min_ufh_segment_outdoor_temperature_span_c=1.0,
            min_ufh_segment_supply_target_span_c=1.0,
        ),
    )

    assert dataset.ufh_sample_count == 2
    assert dataset.dhw_sample_count == 2
    assert dataset.selected_dhw_segment_count == 1
    dhw_quality = [quality for quality in dataset.segment_qualities if quality.mode_name == "dhw"][0]
    assert dhw_quality.supply_tracking_rmse_c > 20.0
    assert dhw_quality.selected is True


def test_build_cop_calibration_dataset_keeps_same_mode_buckets_with_small_boundary_gap() -> None:
    """Small telemetry sampling gaps between same-mode COP buckets must not split a segment."""
    start = datetime(2026, 4, 17, 2, 40, tzinfo=timezone.utc)
    gap_seconds = 10.0
    bucket_duration = timedelta(minutes=5) - timedelta(seconds=gap_seconds)
    aggregates = [
        SimpleNamespace(
            bucket_start_utc=start,
            bucket_end_utc=start + bucket_duration,
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=35.0,
            hp_supply_temperature_mean_c=34.7,
            hp_thermal_power_mean_kw=3.2,
            hp_electric_energy_delta_kwh=0.20,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=5),
            bucket_end_utc=start + timedelta(minutes=5) + bucket_duration,
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.0,
            hp_supply_target_temperature_mean_c=37.0,
            hp_supply_temperature_mean_c=36.5,
            hp_thermal_power_mean_kw=3.3,
            hp_electric_energy_delta_kwh=0.20,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=10) + bucket_duration,
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=4.0,
            hp_supply_target_temperature_mean_c=39.5,
            hp_supply_temperature_mean_c=38.9,
            hp_thermal_power_mean_kw=3.5,
            hp_electric_energy_delta_kwh=0.20,
        ),
    ]

    dataset = build_cop_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=COPCalibrationSettings(
            min_sample_count=3,
            min_ufh_curve_sample_count=3,
            reaggregate_min_electric_energy_kwh=0.0,
            reaggregate_min_bucket_count=1,
            min_segment_samples=3,
            min_segment_thermal_energy_kwh=0.2,
            min_segment_actual_cop_span=0.02,
            min_ufh_segment_outdoor_temperature_span_c=1.0,
            min_ufh_segment_supply_target_span_c=1.0,
            max_segment_boundary_gap_ratio=0.1,
        ),
    )

    assert dataset.sample_count == 3
    assert dataset.raw_segment_count == 1
    assert dataset.selected_segment_count == 1
    assert dataset.dropped_segment_count == 0


def test_build_cop_calibration_dataset_reaggregates_zero_delta_buckets_into_one_window() -> None:
    """COP re-aggregation must merge coarse-counter UFH buckets into one fit-worthy window."""
    start = datetime(2026, 4, 17, 4, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_start_utc=start,
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=8.0,
            hp_supply_target_temperature_mean_c=35.0,
            hp_supply_temperature_mean_c=34.8,
            hp_thermal_power_mean_kw=3.0,
            hp_electric_energy_delta_kwh=0.0,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=5),
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.5,
            hp_supply_target_temperature_mean_c=36.5,
            hp_supply_temperature_mean_c=36.0,
            hp_thermal_power_mean_kw=3.1,
            hp_electric_energy_delta_kwh=0.1,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.0,
            hp_supply_target_temperature_mean_c=38.0,
            hp_supply_temperature_mean_c=37.2,
            hp_thermal_power_mean_kw=3.3,
            hp_electric_energy_delta_kwh=0.0,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=15),
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=3.0,
            hp_supply_target_temperature_mean_c=40.0,
            hp_supply_temperature_mean_c=39.1,
            hp_thermal_power_mean_kw=3.5,
            hp_electric_energy_delta_kwh=0.1,
        ),
    ]

    dataset = build_cop_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=COPCalibrationSettings(
            min_sample_count=2,
            min_ufh_curve_sample_count=2,
            min_thermal_energy_kwh=0.1,
            min_electric_energy_kwh=0.05,
            reaggregate_min_electric_energy_kwh=0.1,
            reaggregate_min_bucket_count=2,
            min_segment_samples=2,
            min_segment_thermal_energy_kwh=0.2,
            min_segment_actual_cop_span=0.02,
            min_ufh_segment_outdoor_temperature_span_c=1.0,
            min_ufh_segment_supply_target_span_c=1.0,
        ),
    )

    assert dataset.sample_count == 2
    assert [sample.source_bucket_count for sample in dataset.samples] == [2, 2]
    np.testing.assert_allclose(
        [sample.electric_energy_kwh for sample in dataset.samples],
        [0.1, 0.1],
    )
    assert dataset.raw_segment_count == 1
    assert dataset.selected_segment_count == 1


def test_diagnose_cop_calibration_dataset_reports_bucket_and_segment_dropoffs() -> None:
    """COP diagnostics must explain where rows and segments are rejected before fitting."""
    start = datetime(2026, 4, 17, 3, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_start_utc=start,
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.0,
            hp_supply_target_temperature_mean_c=37.0,
            hp_supply_temperature_mean_c=36.5,
            hp_thermal_power_mean_kw=3.2,
            hp_electric_energy_delta_kwh=0.20,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=5),
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=4.0,
            hp_supply_target_temperature_mean_c=40.0,
            hp_supply_temperature_mean_c=39.3,
            hp_thermal_power_mean_kw=3.4,
            hp_electric_energy_delta_kwh=0.22,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.0,
            hp_supply_target_temperature_mean_c=25.0,
            hp_supply_temperature_mean_c=25.0,
            hp_thermal_power_mean_kw=0.0,
            hp_electric_energy_delta_kwh=0.0,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=15),
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=6.0,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.0,
            hp_thermal_power_mean_kw=2.6,
            hp_electric_energy_delta_kwh=0.18,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=20),
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=5.8,
            hp_supply_target_temperature_mean_c=55.0,
            hp_supply_temperature_mean_c=54.1,
            hp_thermal_power_mean_kw=2.7,
            hp_electric_energy_delta_kwh=0.18,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=25),
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="ufh",
            defrost_active_fraction=1.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=3.5,
            hp_supply_target_temperature_mean_c=41.0,
            hp_supply_temperature_mean_c=40.2,
            hp_thermal_power_mean_kw=3.4,
            hp_electric_energy_delta_kwh=0.22,
        ),
        SimpleNamespace(
            bucket_start_utc=start + timedelta(minutes=30),
            bucket_end_utc=start + timedelta(minutes=35),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            outdoor_temperature_mean_c=3.4,
            hp_supply_target_temperature_mean_c=41.2,
            hp_supply_temperature_mean_c=40.6,
            hp_thermal_power_mean_kw=2.0,
            hp_electric_energy_delta_kwh=0.15,
        ),
    ]
    settings = COPCalibrationSettings(
        min_sample_count=2,
        min_ufh_curve_sample_count=2,
        min_thermal_energy_kwh=0.1,
        min_electric_energy_kwh=0.05,
        reaggregate_min_electric_energy_kwh=0.0,
        reaggregate_min_bucket_count=1,
        min_segment_samples=2,
        min_segment_thermal_energy_kwh=0.2,
        min_segment_actual_cop_span=0.02,
        min_ufh_segment_outdoor_temperature_span_c=0.5,
        min_ufh_segment_supply_target_span_c=0.5,
    )

    diagnostics = diagnose_cop_calibration_dataset(cast(list, aggregates), settings)

    assert isinstance(diagnostics, COPCalibrationDiagnostics)
    assert diagnostics.raw_row_count == 7
    assert diagnostics.mode_accepted_count == 6
    assert diagnostics.cop_accepted_count == 5
    assert diagnostics.raw_segment_count == 3
    assert diagnostics.selected_segment_count == 2
    assert diagnostics.selected_sample_count == 4
    assert diagnostics.selected_ufh_sample_count == 2
    assert diagnostics.selected_dhw_sample_count == 2
    assert ("mode_not_ufh_or_dhw", 1) in diagnostics.bucket_rejection_counts
    assert ("defrost_fraction", 1) in diagnostics.bucket_rejection_counts
    assert ("sample_count", 1) in diagnostics.segment_failure_counts
    assert ("actual_cop_span", 1) in diagnostics.segment_failure_counts


def test_calibrate_cop_model_recovers_synthetic_parameters() -> None:
    """Offline COP calibration must recover synthetic heating-curve and eta parameters."""
    true_parameters = HeatPumpCOPParameters(
        eta_carnot_ufh=0.47,
        eta_carnot_dhw=0.49,
        delta_T_cond=5.0,
        delta_T_evap=5.0,
        T_supply_min=27.0,
        T_ref_outdoor=18.0,
        heating_curve_slope=0.9,
        cop_min=1.5,
        cop_max=7.0,
    )
    model = HeatPumpCOPModel(true_parameters)
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    dt_hours = 5.0 / 60.0
    samples: list[COPCalibrationSample] = []

    for step_k in range(48):
        bucket_start = start + timedelta(hours=step_k * dt_hours)
        bucket_end = bucket_start + timedelta(hours=dt_hours)
        t_out = 8.0 - 0.25 * step_k
        t_supply_target = float(model.heating_curve(np.array([t_out], dtype=float))[0])
        thermal_energy_kwh = 0.26 + 0.03 * np.sin(step_k / 5.0)
        predicted_cop = float(model.cop_ufh(np.array([t_out], dtype=float))[0])
        electric_energy_kwh = thermal_energy_kwh / predicted_cop
        samples.append(
            COPCalibrationSample(
                bucket_start_utc=bucket_start,
                bucket_end_utc=bucket_end,
                dt_hours=dt_hours,
                mode_name="ufh",
                outdoor_temperature_mean_c=t_out,
                supply_target_temperature_mean_c=t_supply_target,
                supply_temperature_mean_c=t_supply_target - 0.5,
                thermal_energy_kwh=thermal_energy_kwh,
                electric_energy_kwh=electric_energy_kwh,
            )
        )

    for step_k in range(32):
        bucket_start = start + timedelta(hours=(48 + step_k) * dt_hours)
        bucket_end = bucket_start + timedelta(hours=dt_hours)
        t_out = 4.0 + 0.15 * step_k
        t_supply_target = 55.0
        thermal_energy_kwh = 0.22 + 0.02 * np.cos(step_k / 4.0)
        predicted_cop = float(model.cop_dhw(np.array([t_out], dtype=float), t_dhw_supply=t_supply_target)[0])
        electric_energy_kwh = thermal_energy_kwh / predicted_cop
        samples.append(
            COPCalibrationSample(
                bucket_start_utc=bucket_start,
                bucket_end_utc=bucket_end,
                dt_hours=dt_hours,
                mode_name="dhw",
                outdoor_temperature_mean_c=t_out,
                supply_target_temperature_mean_c=t_supply_target,
                supply_temperature_mean_c=t_supply_target - 0.7,
                thermal_energy_kwh=thermal_energy_kwh,
                electric_energy_kwh=electric_energy_kwh,
            )
        )

    dataset = COPCalibrationDataset(samples=tuple(samples))
    settings = COPCalibrationSettings(
        min_sample_count=16,
        min_ufh_curve_sample_count=12,
        initial_eta_carnot=0.4,
        initial_t_supply_min_c=24.0,
        initial_heating_curve_slope=1.2,
        t_ref_outdoor_c=true_parameters.T_ref_outdoor,
        delta_t_cond_k=true_parameters.delta_T_cond,
        delta_t_evap_k=true_parameters.delta_T_evap,
        cop_min=true_parameters.cop_min,
        cop_max=true_parameters.cop_max,
    )

    result = calibrate_cop_model(dataset, settings)

    np.testing.assert_allclose(result.fitted_parameters.eta_carnot_ufh, true_parameters.eta_carnot_ufh, rtol=2e-2)
    np.testing.assert_allclose(result.fitted_parameters.eta_carnot_dhw, true_parameters.eta_carnot_dhw, rtol=2e-2)
    np.testing.assert_allclose(result.fitted_parameters.T_supply_min, true_parameters.T_supply_min, rtol=2e-2)
    np.testing.assert_allclose(result.fitted_parameters.T_ref_outdoor, true_parameters.T_ref_outdoor, rtol=1e-9)
    np.testing.assert_allclose(
        result.fitted_parameters.heating_curve_slope,
        true_parameters.heating_curve_slope,
        rtol=2e-2,
    )
    assert result.t_ref_outdoor_was_fitted is False
    assert result.ufh_sample_count == 48
    assert result.dhw_sample_count == 32
    assert result.rmse_supply_temperature_c < 1e-6
    assert result.rmse_electric_energy_kwh < 1e-6
    assert result.rmse_actual_cop < 1e-6
    assert result.ufh_rmse_electric_energy_kwh < 1e-6
    assert result.dhw_rmse_electric_energy_kwh is not None
    assert result.dhw_rmse_electric_energy_kwh < 1e-6
    assert result.ufh_rmse_actual_cop < 1e-6
    assert result.dhw_rmse_actual_cop is not None
    assert result.dhw_rmse_actual_cop < 1e-6
    np.testing.assert_allclose(result.diagnostic_eta_carnot_ufh, true_parameters.eta_carnot_ufh, rtol=2e-2)
    assert result.diagnostic_eta_carnot_dhw is not None
    np.testing.assert_allclose(result.diagnostic_eta_carnot_dhw, true_parameters.eta_carnot_dhw, rtol=2e-2)


def test_calibrate_cop_model_recovers_t_ref_outdoor_when_ufh_data_spans_breakpoint() -> None:
    """COP calibration must fit T_ref_outdoor when UFH data excite both curve branches."""
    true_parameters = HeatPumpCOPParameters(
        eta_carnot_ufh=0.44,
        eta_carnot_dhw=0.46,
        delta_T_cond=5.0,
        delta_T_evap=5.0,
        T_supply_min=26.5,
        T_ref_outdoor=17.5,
        heating_curve_slope=0.85,
        cop_min=1.5,
        cop_max=7.0,
    )
    model = HeatPumpCOPModel(true_parameters)
    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    dt_hours = 5.0 / 60.0
    samples: list[COPCalibrationSample] = []

    ufh_outdoor_profile = np.linspace(22.0, -4.0, 60)
    for step_k, t_out in enumerate(ufh_outdoor_profile):
        bucket_start = start + timedelta(hours=step_k * dt_hours)
        bucket_end = bucket_start + timedelta(hours=dt_hours)
        t_supply_target = float(model.heating_curve(np.array([t_out], dtype=float))[0])
        thermal_energy_kwh = 0.24 + 0.04 * np.sin(step_k / 6.0)
        predicted_cop = float(model.cop_ufh(np.array([t_out], dtype=float))[0])
        electric_energy_kwh = thermal_energy_kwh / predicted_cop
        samples.append(
            COPCalibrationSample(
                bucket_start_utc=bucket_start,
                bucket_end_utc=bucket_end,
                dt_hours=dt_hours,
                mode_name="ufh",
                outdoor_temperature_mean_c=float(t_out),
                supply_target_temperature_mean_c=t_supply_target,
                supply_temperature_mean_c=t_supply_target - 0.3,
                thermal_energy_kwh=thermal_energy_kwh,
                electric_energy_kwh=electric_energy_kwh,
            )
        )

    for step_k, t_out in enumerate(np.linspace(8.0, -1.0, 24), start=len(samples)):
        bucket_start = start + timedelta(hours=step_k * dt_hours)
        bucket_end = bucket_start + timedelta(hours=dt_hours)
        t_supply_target = 55.0
        thermal_energy_kwh = 0.20 + 0.02 * np.cos(step_k / 4.0)
        predicted_cop = float(model.cop_dhw(np.array([t_out], dtype=float), t_dhw_supply=t_supply_target)[0])
        electric_energy_kwh = thermal_energy_kwh / predicted_cop
        samples.append(
            COPCalibrationSample(
                bucket_start_utc=bucket_start,
                bucket_end_utc=bucket_end,
                dt_hours=dt_hours,
                mode_name="dhw",
                outdoor_temperature_mean_c=float(t_out),
                supply_target_temperature_mean_c=t_supply_target,
                supply_temperature_mean_c=t_supply_target - 0.5,
                thermal_energy_kwh=thermal_energy_kwh,
                electric_energy_kwh=electric_energy_kwh,
            )
        )

    result = calibrate_cop_model(
        COPCalibrationDataset(samples=tuple(samples)),
        COPCalibrationSettings(
            min_sample_count=24,
            min_ufh_curve_sample_count=24,
            initial_eta_carnot=0.40,
            initial_t_supply_min_c=24.0,
            initial_heating_curve_slope=1.0,
            t_ref_outdoor_c=18.0,
            min_t_ref_outdoor_c=10.0,
            max_t_ref_outdoor_c=22.0,
            delta_t_cond_k=true_parameters.delta_T_cond,
            delta_t_evap_k=true_parameters.delta_T_evap,
            cop_min=true_parameters.cop_min,
            cop_max=true_parameters.cop_max,
        ),
    )

    assert result.t_ref_outdoor_was_fitted is True
    np.testing.assert_allclose(result.fitted_parameters.T_ref_outdoor, true_parameters.T_ref_outdoor, atol=1e-3)
    np.testing.assert_allclose(result.fitted_parameters.T_supply_min, true_parameters.T_supply_min, atol=1e-3)
    np.testing.assert_allclose(result.fitted_parameters.heating_curve_slope, true_parameters.heating_curve_slope, atol=1e-3)
    np.testing.assert_allclose(result.fitted_parameters.eta_carnot_ufh, true_parameters.eta_carnot_ufh, atol=1e-3)
    np.testing.assert_allclose(result.fitted_parameters.eta_carnot_dhw, true_parameters.eta_carnot_dhw, atol=1e-3)
    assert result.rmse_supply_temperature_c < 1e-6
    assert result.rmse_actual_cop < 1e-6


def test_cop_calibration_settings_reject_invalid_loss_names() -> None:
    """COP settings must fail fast when an unsupported SciPy robust loss name is configured."""
    try:
        COPCalibrationSettings(heating_curve_loss_name="not_a_loss")
    except ValueError as exc:
        assert "heating_curve_loss_name" in str(exc)
    else:
        raise AssertionError("Expected invalid heating_curve_loss_name to raise ValueError.")

    try:
        COPCalibrationSettings(eta_loss_name="still_not_a_loss")
    except ValueError as exc:
        assert "eta_loss_name" in str(exc)
    else:
        raise AssertionError("Expected invalid eta_loss_name to raise ValueError.")


def test_calibrate_cop_model_uses_non_saturated_eta_initial_guess() -> None:
    """The eta fit must escape a fully clipped cop_max plateau caused by a high initial η.

    This regression covers the real-data failure mode where warm UFH samples and
    a conservative ``cop_max`` make the least-squares objective locally flat at
    the user-provided initial η. The fitter must derive a data-driven initial η
    below the full-saturation threshold so the optimisation remains identifiable.
    """
    true_parameters = HeatPumpCOPParameters(
        eta_carnot_ufh=0.27,
        eta_carnot_dhw=0.27,
        delta_T_cond=5.0,
        delta_T_evap=5.0,
        T_supply_min=25.0,
        T_ref_outdoor=18.0,
        heating_curve_slope=0.8,
        cop_min=1.5,
        cop_max=7.0,
    )
    model = HeatPumpCOPModel(true_parameters)
    start = datetime(2026, 4, 17, 10, 0, tzinfo=timezone.utc)
    dt_hours = 5.0 / 60.0
    samples: list[COPCalibrationSample] = []

    for step_k, t_out in enumerate(np.linspace(16.5, 20.0, 8)):
        bucket_start = start + timedelta(hours=step_k * dt_hours)
        bucket_end = bucket_start + timedelta(hours=dt_hours)
        t_supply_target = float(model.heating_curve(np.array([t_out], dtype=float))[0])
        thermal_energy_kwh = 0.38 + 0.015 * np.sin(step_k / 3.0)
        actual_cop = float(model.cop_ufh(np.array([t_out], dtype=float))[0])
        electric_energy_kwh = thermal_energy_kwh / actual_cop
        samples.append(
            COPCalibrationSample(
                bucket_start_utc=bucket_start,
                bucket_end_utc=bucket_end,
                dt_hours=dt_hours,
                mode_name="ufh",
                outdoor_temperature_mean_c=float(t_out),
                supply_target_temperature_mean_c=t_supply_target,
                supply_temperature_mean_c=t_supply_target - 0.2,
                thermal_energy_kwh=thermal_energy_kwh,
                electric_energy_kwh=electric_energy_kwh,
            )
        )

    dataset = COPCalibrationDataset(samples=tuple(samples))
    settings = COPCalibrationSettings(
        min_sample_count=8,
        min_ufh_curve_sample_count=8,
        initial_eta_carnot=0.45,
        initial_t_supply_min_c=24.0,
        initial_heating_curve_slope=1.0,
        t_ref_outdoor_c=true_parameters.T_ref_outdoor,
        delta_t_cond_k=true_parameters.delta_T_cond,
        delta_t_evap_k=true_parameters.delta_T_evap,
        cop_min=true_parameters.cop_min,
        cop_max=true_parameters.cop_max,
    )

    result = calibrate_cop_model(dataset, settings)

    assert result.fitted_parameters.eta_carnot_ufh < settings.initial_eta_carnot
    np.testing.assert_allclose(result.fitted_parameters.eta_carnot_ufh, true_parameters.eta_carnot_ufh, rtol=2e-2)
    np.testing.assert_allclose(result.diagnostic_eta_carnot_ufh, true_parameters.eta_carnot_ufh, rtol=2e-2)
    assert result.rmse_actual_cop < 1e-6


def test_calibrate_cop_model_soft_l1_is_more_robust_to_outliers_than_linear() -> None:
    """The configured robust COP losses must reduce parameter drift under bucket outliers."""
    true_parameters = HeatPumpCOPParameters(
        eta_carnot_ufh=0.46,
        eta_carnot_dhw=0.46,
        delta_T_cond=5.0,
        delta_T_evap=5.0,
        T_supply_min=28.0,
        T_ref_outdoor=18.0,
        heating_curve_slope=0.85,
        cop_min=1.5,
        cop_max=7.0,
    )
    model = HeatPumpCOPModel(true_parameters)
    start = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    dt_hours = 5.0 / 60.0
    samples: list[COPCalibrationSample] = []

    for step_k in range(40):
        bucket_start = start + timedelta(hours=step_k * dt_hours)
        bucket_end = bucket_start + timedelta(hours=dt_hours)
        t_out = 9.0 - 0.22 * step_k
        t_supply_target = float(model.heating_curve(np.array([t_out], dtype=float))[0])
        thermal_energy_kwh = 0.28 + 0.02 * np.sin(step_k / 4.0)
        predicted_cop = float(model.cop_ufh(np.array([t_out], dtype=float))[0])
        electric_energy_kwh = thermal_energy_kwh / predicted_cop

        if step_k in {9, 27}:
            t_supply_target += 12.0
            electric_energy_kwh += 0.12

        samples.append(
            COPCalibrationSample(
                bucket_start_utc=bucket_start,
                bucket_end_utc=bucket_end,
                dt_hours=dt_hours,
                mode_name="ufh",
                outdoor_temperature_mean_c=t_out,
                supply_target_temperature_mean_c=t_supply_target,
                supply_temperature_mean_c=t_supply_target - 0.4,
                thermal_energy_kwh=thermal_energy_kwh,
                electric_energy_kwh=electric_energy_kwh,
            )
        )

    dataset = COPCalibrationDataset(samples=tuple(samples))
    common_settings = dict(
        min_sample_count=16,
        min_ufh_curve_sample_count=16,
        initial_eta_carnot=0.4,
        initial_t_supply_min_c=24.0,
        initial_heating_curve_slope=1.1,
        t_ref_outdoor_c=true_parameters.T_ref_outdoor,
        delta_t_cond_k=true_parameters.delta_T_cond,
        delta_t_evap_k=true_parameters.delta_T_evap,
        cop_min=true_parameters.cop_min,
        cop_max=true_parameters.cop_max,
        heating_curve_loss_scale_c=1.0,
        eta_loss_scale_kwh=0.03,
    )
    linear_result = calibrate_cop_model(
        dataset,
        COPCalibrationSettings(
            heating_curve_loss_name="linear",
            eta_loss_name="linear",
            **common_settings,
        ),
    )
    robust_result = calibrate_cop_model(
        dataset,
        COPCalibrationSettings(
            heating_curve_loss_name="soft_l1",
            eta_loss_name="soft_l1",
            **common_settings,
        ),
    )

    linear_parameter_error = sum(
        (
            abs(linear_result.fitted_parameters.T_supply_min - true_parameters.T_supply_min),
            abs(linear_result.fitted_parameters.heating_curve_slope - true_parameters.heating_curve_slope),
            abs(linear_result.fitted_parameters.eta_carnot_ufh - true_parameters.eta_carnot_ufh),
        )
    )
    robust_parameter_error = sum(
        (
            abs(robust_result.fitted_parameters.T_supply_min - true_parameters.T_supply_min),
            abs(robust_result.fitted_parameters.heating_curve_slope - true_parameters.heating_curve_slope),
            abs(robust_result.fitted_parameters.eta_carnot_ufh - true_parameters.eta_carnot_ufh),
        )
    )

    assert robust_parameter_error < linear_parameter_error
    assert abs(robust_result.fitted_parameters.eta_carnot_ufh - true_parameters.eta_carnot_ufh) < abs(
        linear_result.fitted_parameters.eta_carnot_ufh - true_parameters.eta_carnot_ufh
    )


def test_build_automatic_calibration_snapshot_merges_previous_successful_overrides(monkeypatch) -> None:
    """Automatic calibration must retain prior successful parameters when a stage is disabled or absent."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)

    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (
            start,
            datetime(2026, 4, 18, 6, 0, tzinfo=timezone.utc),
        ),
        get_latest_calibration_snapshot=lambda: CalibrationSnapshotPayload(
            generated_at_utc=datetime(2026, 4, 18, 5, 0, tzinfo=timezone.utc),
            effective_parameters=CalibrationParameterOverrides(
                dhw_R_loss=52.0,
                eta_carnot_ufh=0.33,
                eta_carnot_dhw=0.31,
            ),
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: [
            SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index)) for index in range(4)
        ],
    )

    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=ThermalParameters(
                dt_hours=1.0,
                C_r=7.5,
                C_b=11.0,
                R_br=1.2,
                R_ro=8.8,
                alpha=0.25,
                eta=0.55,
                A_glass=7.5,
            ),
            fit_c_r=False,
            fit_initial_floor_temperature_offset=False,
            fitted_initial_floor_temperature_offset_c=1.0,
            sample_count=24,
            segment_count=3,
            dataset_start_utc=datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc),
            dataset_end_utc=datetime(2026, 4, 18, 6, 0, tzinfo=timezone.utc),
            optimizer_status="ok",
            rmse_room_temperature_c=0.18,
            max_abs_innovation_c=0.30,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=9.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=HeatPumpCOPParameters(
                eta_carnot_ufh=0.41,
                eta_carnot_dhw=0.39,
                delta_T_cond=5.0,
                delta_T_evap=5.0,
                T_supply_min=26.2,
                T_ref_outdoor=18.0,
                heating_curve_slope=0.9,
                cop_min=1.5,
                cop_max=7.0,
            ),
            t_ref_outdoor_was_fitted=True,
            rmse_supply_temperature_c=0.15,
            rmse_electric_energy_kwh=0.08,
            sample_count=14,
            ufh_sample_count=8,
            dhw_sample_count=6,
            dataset_start_utc=datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc),
            dataset_end_utc=datetime(2026, 4, 18, 6, 0, tzinfo=timezone.utc),
            ufh_rmse_electric_energy_kwh=0.07,
            dhw_rmse_electric_energy_kwh=0.09,
            eta_optimizer_status="ok",
            rmse_actual_cop=0.22,
            ufh_rmse_actual_cop=0.20,
            dhw_rmse_actual_cop=0.25,
            ufh_bias_actual_cop=-0.01,
            dhw_bias_actual_cop=0.02,
            diagnostic_eta_carnot_ufh=0.41,
            diagnostic_eta_carnot_dhw=0.39,
            heating_curve_optimizer_status="ok",
            heating_curve_optimizer_cost=0.05,
            eta_optimizer_cost=0.04,
        ),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.effective_parameters.C_r == 7.5
    assert snapshot.effective_parameters.R_ro == 8.8
    assert snapshot.effective_parameters.eta_carnot_ufh == 0.41
    assert snapshot.effective_parameters.eta_carnot_dhw == 0.39
    assert snapshot.effective_parameters.T_supply_min == 26.2
    assert snapshot.effective_parameters.T_ref_outdoor_curve == 18.0
    assert snapshot.effective_parameters.heating_curve_slope == 0.9
    assert snapshot.effective_parameters.dhw_R_loss_top == 52.0
    assert snapshot.effective_parameters.dhw_R_loss_bot == 52.0
    assert snapshot.ufh_active is not None and snapshot.ufh_active.succeeded is True
    assert snapshot.cop is not None and snapshot.cop.succeeded is True
    assert snapshot.cop.diagnostics["dhw_sample_count"] == 6


def test_build_automatic_calibration_snapshot_matches_cli_stage_settings(monkeypatch) -> None:
    """Automatic calibration must reuse replay/filter defaults while applying automatic-only fit overrides."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]
    captured_settings: dict[str, object] = {}

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, settings: (
            captured_settings.setdefault("ufh_active", settings),
            SimpleNamespace(
                fitted_parameters=ThermalParameters(
                    dt_hours=settings.reference_parameters.dt_hours,
                    C_r=settings.reference_parameters.C_r,
                    C_b=settings.reference_parameters.C_b,
                    R_br=settings.reference_parameters.R_br,
                    R_ro=settings.reference_parameters.R_ro,
                    alpha=settings.reference_parameters.alpha,
                    eta=settings.reference_parameters.eta,
                    A_glass=settings.reference_parameters.A_glass,
                ),
                fit_c_r=False,
                fit_initial_floor_temperature_offset=False,
                fitted_initial_floor_temperature_offset_c=1.0,
                sample_count=12,
                segment_count=2,
                dataset_start_utc=start,
                dataset_end_utc=start + timedelta(hours=1),
                optimizer_status="ok",
                rmse_room_temperature_c=0.1,
                max_abs_innovation_c=0.20,
            ),
        )[1],
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, settings: (
            captured_settings.setdefault("dhw_standby", settings),
            SimpleNamespace(
                tau_standby_hours=8.0,
                suggested_r_loss_k_per_kw=60.0,
                sample_count=12,
                dataset_start_utc=start,
                dataset_end_utc=start + timedelta(hours=1),
                optimizer_status="ok",
                rmse_mean_tank_temperature_c=0.10,
                max_abs_residual_c=0.20,
            ),
        )[1],
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, settings: (
            captured_settings.setdefault("dhw_active", settings),
            SimpleNamespace(
                fitted_parameters=DHWParameters(
                    dt_hours=settings.reference_parameters.dt_hours,
                    C_top=settings.reference_parameters.C_top,
                    C_bot=settings.reference_parameters.C_bot,
                    R_strat=settings.reference_parameters.R_strat,
                    R_loss=settings.reference_parameters.R_loss,
                ),
                sample_count=12,
                segment_count=2,
                dataset_start_utc=start,
                dataset_end_utc=start + timedelta(hours=1),
                optimizer_status="ok",
                rmse_t_top_c=0.1,
                rmse_t_bot_c=0.1,
                max_abs_residual_c=0.2,
            ),
        )[1],
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=HeatPumpCOPParameters(
                eta_carnot_ufh=0.41,
                eta_carnot_dhw=0.39,
                delta_T_cond=5.0,
                delta_T_evap=5.0,
                T_supply_min=26.0,
                T_ref_outdoor=18.0,
                heating_curve_slope=0.9,
                cop_min=1.5,
                cop_max=7.0,
            ),
            t_ref_outdoor_was_fitted=True,
            rmse_supply_temperature_c=0.10,
            rmse_electric_energy_kwh=0.04,
            sample_count=12,
            ufh_sample_count=8,
            dhw_sample_count=4,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=1),
            ufh_rmse_electric_energy_kwh=0.03,
            dhw_rmse_electric_energy_kwh=0.05,
            eta_optimizer_status="ok",
            rmse_actual_cop=0.1,
            ufh_rmse_actual_cop=0.09,
            dhw_rmse_actual_cop=0.12,
            ufh_bias_actual_cop=-0.01,
            dhw_bias_actual_cop=0.01,
            diagnostic_eta_carnot_ufh=0.41,
            diagnostic_eta_carnot_dhw=0.39,
            heating_curve_optimizer_status="ok",
            heating_curve_optimizer_cost=0.02,
            eta_optimizer_cost=0.03,
        ),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    ufh_settings = cast(UFHActiveCalibrationSettings, captured_settings["ufh_active"])
    standby_settings = cast(DHWStandbyCalibrationSettings, captured_settings["dhw_standby"])
    dhw_active_settings = cast(DHWActiveCalibrationSettings, captured_settings["dhw_active"])
    assert ufh_settings.reference_parameters.dt_hours == 5.0 / 60.0
    assert ufh_settings.max_gti_w_per_m2 == DEFAULT_ACTIVE_MAX_GTI_W_PER_M2
    assert ufh_settings.reference_parameters.dt_hours != RunRequest.model_validate({}).dt_hours
    assert ufh_settings.fit_eta is False
    assert ufh_settings.fit_internal_gains_heat_fraction is False
    assert ufh_settings.min_parameter_ratio == AutomaticCalibrationSettings().ufh_active_min_parameter_ratio
    assert ufh_settings.max_parameter_ratio == AutomaticCalibrationSettings().ufh_active_max_parameter_ratio
    assert ufh_settings.regularization_weight == AutomaticCalibrationSettings().ufh_active_regularization_weight
    assert standby_settings.dt_hours == 5.0 / 60.0
    assert standby_settings.fit_ambient_temperature_bias is False
    assert standby_settings.initial_ambient_temperature_bias_c == RunRequest.model_validate({}).dhw_boiler_ambient_bias_c
    assert dhw_active_settings.reference_parameters.dt_hours == 5.0 / 60.0
    assert snapshot.cop is not None and snapshot.cop.diagnostics["required_min_dhw_sample_count"] == 1


def test_build_automatic_calibration_snapshot_rejects_cop_fit_without_real_dhw_identification(monkeypatch) -> None:
    """Automatic calibration must reject COP snapshots that only reuse the UFH eta for DHW."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw standby")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=HeatPumpCOPParameters(
                eta_carnot_ufh=0.41,
                eta_carnot_dhw=0.41,
                delta_T_cond=5.0,
                delta_T_evap=5.0,
                T_supply_min=26.0,
                T_ref_outdoor=18.0,
                heating_curve_slope=0.9,
                cop_min=1.5,
                cop_max=7.0,
            ),
            t_ref_outdoor_was_fitted=True,
            rmse_supply_temperature_c=0.10,
            rmse_electric_energy_kwh=0.04,
            rmse_actual_cop=0.1,
            ufh_rmse_electric_energy_kwh=0.03,
            dhw_rmse_electric_energy_kwh=None,
            ufh_rmse_actual_cop=0.09,
            dhw_rmse_actual_cop=None,
            ufh_bias_actual_cop=-0.01,
            dhw_bias_actual_cop=None,
            diagnostic_eta_carnot_ufh=0.41,
            diagnostic_eta_carnot_dhw=None,
            sample_count=8,
            ufh_sample_count=8,
            dhw_sample_count=0,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=1),
            heating_curve_optimizer_status="ok",
            eta_optimizer_status="UFH: ok | DHW: No DHW samples retained; reusing eta_carnot_ufh.",
            heating_curve_optimizer_cost=0.02,
            eta_optimizer_cost=0.03,
        ),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.cop is not None
    assert snapshot.cop.succeeded is False
    assert "insufficient retained DHW COP samples" in snapshot.cop.message
    assert snapshot.cop.diagnostics["dhw_sample_count"] == 0


def test_build_automatic_calibration_snapshot_accepts_ufh_fit_with_exact_zoh_runtime(monkeypatch) -> None:
    """Automatic calibration may persist a UFH tuple when runtime exact-ZOH remains valid."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: CalibrationSnapshotPayload(
            generated_at_utc=start,
            effective_parameters=CalibrationParameterOverrides(dhw_R_loss=52.0),
        ),
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=ThermalParameters(
                dt_hours=5.0 / 60.0,
                C_r=6.0,
                C_b=10.5,
                R_br=1.0,
                R_ro=8.5,
                alpha=0.25,
                eta=0.55,
                A_glass=7.5,
            ),
            fit_c_r=False,
            fit_initial_floor_temperature_offset=False,
            fitted_initial_floor_temperature_offset_c=1.0,
            sample_count=25,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_room_temperature_c=0.04,
            max_abs_innovation_c=0.12,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=3.2),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=60.0,
            sample_count=12,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=1),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({"dt_hours": 1.0}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.ufh_active is not None
    assert snapshot.ufh_active.succeeded is True
    assert snapshot.effective_parameters.C_r == 6.0
    assert snapshot.effective_parameters.C_b == 10.5
    assert snapshot.effective_parameters.R_br == 1.0
    assert snapshot.effective_parameters.R_ro == 8.5
    assert snapshot.effective_parameters.dhw_R_loss_top == 60.0
    assert snapshot.effective_parameters.dhw_R_loss_bot == 60.0


def test_build_automatic_calibration_snapshot_rejects_ufh_fit_that_hits_bounds(monkeypatch) -> None:
    """Automatic calibration must reject active UFH fits that converge onto box constraints."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=ThermalParameters(
                dt_hours=5.0 / 60.0,
                C_r=6.0,
                C_b=2.5,
                R_br=1.2,
                R_ro=8.0,
                alpha=0.25,
                eta=0.55,
                A_glass=7.5,
            ),
            fit_c_r=False,
            fit_initial_floor_temperature_offset=False,
            fitted_initial_floor_temperature_offset_c=1.0,
            sample_count=20,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_room_temperature_c=0.05,
            max_abs_innovation_c=0.15,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=8.2),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw standby")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(
            min_history_hours=12.0,
            ufh_active_fit_eta=True,
            ufh_active_fit_internal_gains_heat_fraction=True,
        ),
    )

    assert snapshot is not None
    assert snapshot.ufh_active is not None
    assert snapshot.ufh_active.succeeded is False
    assert "parameter bounds" in snapshot.ufh_active.message


def test_build_automatic_calibration_snapshot_rejects_ufh_fit_with_eta_near_zero_bound(monkeypatch) -> None:
    """Automatic calibration must reject UFH fits when the newly added eta collapses to near-zero."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=ThermalParameters(
                dt_hours=5.0 / 60.0,
                C_r=6.0,
                C_b=11.0,
                R_br=1.1,
                R_ro=8.5,
                alpha=0.25,
                eta=1e-12,
                A_glass=7.5,
            ),
            fit_c_r=False,
            fit_eta=True,
            fit_internal_gains_heat_fraction=True,
            fitted_internal_gains_heat_fraction=0.7,
            fit_initial_floor_temperature_offset=False,
            fitted_initial_floor_temperature_offset_c=1.0,
            sample_count=20,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_room_temperature_c=0.05,
            max_abs_innovation_c=0.15,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=8.6),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw standby")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(
            min_history_hours=12.0,
            dhw_active_fit_capacity_split=True,
        ),
    )

    assert snapshot is not None
    assert snapshot.ufh_active is not None
    assert snapshot.ufh_active.succeeded is False
    assert "eta=" in snapshot.ufh_active.message
    assert snapshot.ufh_active.diagnostics["bound_violations"]
    assert snapshot.ufh_active.diagnostics["selected_segment_count"] == 2


def test_build_automatic_calibration_snapshot_rejects_ufh_fit_with_too_few_segments(monkeypatch) -> None:
    """Automatic calibration must reject active UFH fits built from only one short selected segment."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=ThermalParameters(
                dt_hours=5.0 / 60.0,
                C_r=6.0,
                C_b=11.0,
                R_br=1.1,
                R_ro=8.5,
                alpha=0.25,
                eta=0.55,
                A_glass=7.5,
            ),
            fit_c_r=False,
            fit_initial_floor_temperature_offset=False,
            fitted_initial_floor_temperature_offset_c=1.0,
            sample_count=12,
            segment_count=1,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=1),
            optimizer_status="ok",
            rmse_room_temperature_c=0.05,
            max_abs_innovation_c=0.15,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=8.6),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw standby")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(
            min_history_hours=12.0,
            dhw_active_fit_capacity_split=True,
            dhw_active_fit_temperature_biases=True,
        ),
    )

    assert snapshot is not None
    assert snapshot.ufh_active is not None
    assert snapshot.ufh_active.succeeded is False
    assert "insufficient active excitation" in snapshot.ufh_active.message


def test_build_automatic_calibration_snapshot_rejects_ufh_fit_inconsistent_with_passive_r_ro(monkeypatch) -> None:
    """Automatic calibration must reject active UFH fits that disagree strongly with the passive envelope stage."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=ThermalParameters(
                dt_hours=5.0 / 60.0,
                C_r=6.0,
                C_b=11.0,
                R_br=1.1,
                R_ro=6.0,
                alpha=0.25,
                eta=0.55,
                A_glass=7.5,
            ),
            fit_c_r=False,
            fit_initial_floor_temperature_offset=False,
            fitted_initial_floor_temperature_offset_c=1.0,
            sample_count=18,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_room_temperature_c=0.04,
            max_abs_innovation_c=0.12,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=30.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw standby")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(
            min_history_hours=12.0,
            dhw_standby_fit_ambient_temperature_bias=True,
        ),
    )

    assert snapshot is not None
    assert snapshot.ufh_active is not None
    assert snapshot.ufh_active.succeeded is False
    assert "active/passive R_ro mismatch" in snapshot.ufh_active.message


def test_build_automatic_calibration_snapshot_rejects_dhw_standby_fit_that_hits_bounds(monkeypatch) -> None:
    """Automatic calibration must reject DHW standby fits that collapse onto tau bounds."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=0.5,
            suggested_r_loss_k_per_kw=5.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.20,
            max_abs_residual_c=0.40,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.dhw_standby is not None
    assert snapshot.dhw_standby.succeeded is False
    assert "tau_standby converged to the lower bound" in snapshot.dhw_standby.message


def test_build_automatic_calibration_snapshot_rejects_dhw_standby_bias_at_bound(monkeypatch) -> None:
    """Automatic calibration must reject standby fits when the ambient bias lands on its bound."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=70.0,
            fit_ambient_temperature_bias=True,
            fitted_ambient_temperature_bias_c=5.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip dhw active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.dhw_standby is not None
    assert snapshot.dhw_standby.succeeded is False
    assert "ambient sensor bias converged to its bound" in snapshot.dhw_standby.message


def test_build_automatic_calibration_snapshot_rejects_dhw_active_fit_with_too_few_segments(monkeypatch) -> None:
    """Automatic calibration must reject active DHW fits built from only one selected segment."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=70.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=DHWParameters(
                dt_hours=5.0 / 60.0,
                C_top=0.5814,
                C_bot=0.5814,
                R_strat=12.0,
                R_loss=70.0,
            ),
            sample_count=12,
            segment_count=1,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_t_top_c=0.08,
            rmse_t_bot_c=0.09,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.dhw_active is not None
    assert snapshot.dhw_active.succeeded is False
    assert "insufficient active DHW excitation" in snapshot.dhw_active.message
    assert snapshot.dhw_active.diagnostics["selected_segment_count"] == 1
    assert snapshot.dhw_active.diagnostics["required_min_selected_segments"] == 2


def test_build_automatic_calibration_snapshot_accepts_dhw_active_fit_at_near_zero_lower_bound(monkeypatch) -> None:
    """Automatic calibration may accept a near-zero active-DHW lower-bound hit as strong mixing."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=70.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=DHWParameters(
                dt_hours=5.0 / 60.0,
                C_top=0.5814,
                C_bot=0.5814,
                R_strat=1e-3,
                R_loss=70.0,
            ),
            sample_count=18,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_t_top_c=0.08,
            rmse_t_bot_c=0.09,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.dhw_active is not None
    assert snapshot.dhw_active.succeeded is True
    assert snapshot.dhw_active.diagnostics["hits_lower_bound"] is True
    assert snapshot.dhw_active.diagnostics["near_perfect_mixing_regime"] is True
    np.testing.assert_allclose(snapshot.effective_parameters.dhw_R_strat, 1e-3)


def test_build_automatic_calibration_snapshot_publishes_total_dhw_capacity_when_fitted(monkeypatch) -> None:
    """Automatic DHW calibration must publish fitted C_top/C_bot when total capacity is identified."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=70.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=DHWParameters(
                dt_hours=5.0 / 60.0,
                C_top=0.09,
                C_bot=0.05,
                R_strat=12.0,
                R_loss=70.0,
            ),
            fit_total_capacity=True,
            fitted_c_total_scale=(0.09 + 0.05) / (0.11628 + 0.11628),
            fit_capacity_split=False,
            fitted_c_top_fraction=0.09 / (0.09 + 0.05),
            fit_temperature_biases=False,
            fitted_t_top_bias_c=0.0,
            fitted_t_bot_bias_c=0.0,
            sample_count=18,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_t_top_c=0.08,
            rmse_t_bot_c=0.09,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.effective_parameters.dhw_C_top == 0.09
    assert snapshot.effective_parameters.dhw_C_bot == 0.05
    assert snapshot.dhw_active is not None
    assert snapshot.dhw_active.succeeded is True


def test_build_automatic_calibration_snapshot_rejects_dhw_active_fit_that_hits_upper_bound(monkeypatch) -> None:
    """Automatic calibration must still reject active DHW fits that converge to the upper bound."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=70.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=DHWParameters(
                dt_hours=5.0 / 60.0,
                C_top=0.5814,
                C_bot=0.5814,
                R_strat=50.0,
                R_loss=70.0,
            ),
            sample_count=18,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_t_top_c=0.08,
            rmse_t_bot_c=0.09,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.dhw_active is not None
    assert snapshot.dhw_active.succeeded is False
    assert "R_strat converged to the upper bound" in snapshot.dhw_active.message


def test_build_automatic_calibration_snapshot_rejects_dhw_active_capacity_split_at_bound(monkeypatch) -> None:
    """Automatic calibration must reject active DHW fits when C_top_fraction collapses to its bound."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=70.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=DHWParameters(
                dt_hours=5.0 / 60.0,
                C_top=0.023256,
                C_bot=0.209304,
                R_strat=5.0,
                R_loss=70.0,
            ),
            fit_capacity_split=True,
            fitted_c_top_fraction=0.1,
            fit_temperature_biases=False,
            sample_count=18,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_t_top_c=0.08,
            rmse_t_bot_c=0.09,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.dhw_active is not None
    assert snapshot.dhw_active.succeeded is False
    assert "C_top_fraction converged to its bound" in snapshot.dhw_active.message


def test_build_automatic_calibration_snapshot_rejects_dhw_active_temperature_bias_at_bound(monkeypatch) -> None:
    """Automatic calibration must reject active DHW fits when a layer-sensor bias lands on its bound."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    repository = SimpleNamespace(
        get_aggregate_time_bounds=lambda: (start, start + timedelta(hours=30)),
        get_latest_calibration_snapshot=lambda: None,
    )
    telemetry_rows = [
        SimpleNamespace(bucket_end_utc=start + timedelta(minutes=5 * index))
        for index in range(6)
    ]

    monkeypatch.setattr(
        "home_optimizer.calibration.service._load_calibration_aggregates",
        lambda _repository: telemetry_rows,
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_active_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip ufh active")),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_ufh_off_from_repository",
        lambda _repository, _settings: SimpleNamespace(suggested_r_ro_k_per_kw=10.0),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_standby_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            tau_standby_hours=8.0,
            suggested_r_loss_k_per_kw=70.0,
            sample_count=20,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=3),
            optimizer_status="ok",
            rmse_mean_tank_temperature_c=0.10,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_dhw_active_from_repository",
        lambda _repository, _settings: SimpleNamespace(
            fitted_parameters=DHWParameters(
                dt_hours=5.0 / 60.0,
                C_top=0.12,
                C_bot=0.12,
                R_strat=5.0,
                R_loss=70.0,
            ),
            fit_capacity_split=True,
            fitted_c_top_fraction=0.5,
            fit_temperature_biases=True,
            fitted_t_top_bias_c=-5.0,
            fitted_t_bot_bias_c=0.0,
            sample_count=18,
            segment_count=2,
            dataset_start_utc=start,
            dataset_end_utc=start + timedelta(hours=2),
            optimizer_status="ok",
            rmse_t_top_c=0.08,
            rmse_t_bot_c=0.09,
            max_abs_residual_c=0.20,
        ),
    )
    monkeypatch.setattr(
        "home_optimizer.calibration.service.calibrate_cop_from_repository",
        lambda _repository, _settings: (_ for _ in ()).throw(ValueError("skip cop")),
    )

    snapshot = build_automatic_calibration_snapshot(
        repository=cast(TelemetryRepository, cast(object, repository)),
        base_request=RunRequest.model_validate({}),
        settings=AutomaticCalibrationSettings(min_history_hours=12.0),
    )

    assert snapshot is not None
    assert snapshot.dhw_active is not None
    assert snapshot.dhw_active.succeeded is False
    assert "T_top sensor bias converged to its bound" in snapshot.dhw_active.message


def test_build_dhw_standby_calibration_dataset_filters_to_quasi_mixed_non_dhw_windows() -> None:
    """Only non-DHW quasi-mixed windows may enter the first-stage standby dataset."""
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=50.2,
            dhw_bottom_temperature_last_c=49.6,
            boiler_ambient_temp_mean_c=20.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=49.9,
            dhw_bottom_temperature_last_c=49.4,
            boiler_ambient_temp_mean_c=20.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=49.7,
            dhw_bottom_temperature_last_c=49.3,
            boiler_ambient_temp_mean_c=20.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=54.0,
            dhw_bottom_temperature_last_c=46.0,
            boiler_ambient_temp_mean_c=20.0,
        ),
    ]

    dataset = build_dhw_standby_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=DHWStandbyCalibrationSettings(
            dt_hours=5.0 / 60.0,
            reference_c_top_kwh_per_k=0.058,
            reference_c_bot_kwh_per_k=0.058,
            min_sample_count=2,
            max_layer_temperature_spread_c=1.0,
        ),
    )

    assert dataset.sample_count == 2
    sample = dataset.samples[0]
    assert sample.t_top_start_c == 50.2
    assert sample.t_bot_end_c == 49.4
    assert sample.boiler_ambient_mean_c == 20.0


def test_calibrate_dhw_standby_loss_recovers_synthetic_r_loss() -> None:
    """Standby DHW calibration must recover the standby time constant and R_loss."""
    parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=10.0,
        R_loss=50.0,
    )
    model = DHWModel(parameters)
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    state = np.array([55.0, 55.0], dtype=float)
    ambient_c = 20.0
    samples: list[DHWStandbyCalibrationSample] = []
    for step_k in range(64):
        interval_start = start + timedelta(hours=step_k * parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=parameters.dt_hours)
        next_state = model.step(
            state=state,
            control_kw=0.0,
            v_tap_m3_per_h=0.0,
            t_mains_c=10.0,
            t_amb_c=ambient_c,
        )
        samples.append(
            DHWStandbyCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=parameters.dt_hours,
                t_top_start_c=float(state[0]),
                t_top_end_c=float(next_state[0]),
                t_bot_start_c=float(state[1]),
                t_bot_end_c=float(next_state[1]),
                boiler_ambient_mean_c=ambient_c,
            )
        )
        state = next_state

    dataset = DHWStandbyCalibrationDataset(samples=tuple(samples))
    settings = DHWStandbyCalibrationSettings(
        dt_hours=parameters.dt_hours,
        reference_c_top_kwh_per_k=parameters.C_top,
        reference_c_bot_kwh_per_k=parameters.C_bot,
        min_sample_count=16,
        initial_tau_hours=4.0,
    )

    result = calibrate_dhw_standby_loss(dataset, settings)

    expected_tau_hours = (parameters.C_top + parameters.C_bot) * parameters.R_loss / 2.0
    np.testing.assert_allclose(result.tau_standby_hours, expected_tau_hours, rtol=2e-2)
    np.testing.assert_allclose(result.suggested_r_loss_k_per_kw, parameters.R_loss, rtol=2e-2)
    assert result.rmse_mean_tank_temperature_c < 1e-6


def test_build_dhw_active_calibration_dataset_filters_to_no_draw_dhw_windows() -> None:
    """Only active DHW windows with low implied tap draw may enter the active dataset."""
    reference_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=12.0,
        R_loss=50.0,
    )
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.0,
            dhw_bottom_temperature_last_c=42.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.4,
            dhw_bottom_temperature_last_c=44.2,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.5,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.8,
            dhw_bottom_temperature_last_c=45.8,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.6,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=46.0,
            dhw_bottom_temperature_last_c=44.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.7,
        ),
    ]

    dataset = build_dhw_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=DHWActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=2,
            min_segment_samples=2,
            min_dhw_power_kw=0.5,
            min_layer_temperature_spread_c=2.0,
            max_implied_tap_m3_per_h=0.01,
            min_segment_delivered_energy_kwh=0.1,
            min_segment_mean_layer_spread_c=2.0,
            min_segment_layer_spread_span_c=0.1,
            min_segment_bottom_temperature_rise_c=0.1,
            min_segment_top_temperature_rise_c=0.1,
        ),
    )

    assert dataset.sample_count == 2
    assert dataset.segment_count == 1
    assert dataset.raw_segment_count == 1
    assert dataset.dropped_segment_count == 0
    assert all(sample.implied_v_tap_m3_per_h <= 0.01 for sample in dataset.samples)
    assert dataset.samples[0].t_top_start_c == 48.0
    assert dataset.samples[1].t_bot_end_c == 45.8


def test_build_dhw_active_calibration_dataset_splits_contiguous_runs() -> None:
    """A gap in valid DHW charging pairs must start a new active-DHW segment."""
    reference_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=12.0,
        R_loss=50.0,
    )
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=47.0,
            dhw_bottom_temperature_last_c=41.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=47.4,
            dhw_bottom_temperature_last_c=43.3,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.4,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=47.9,
            dhw_bottom_temperature_last_c=44.9,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.5,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=47.9,
            dhw_bottom_temperature_last_c=44.9,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.0,
            dhw_bottom_temperature_last_c=42.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.4,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.3,
            dhw_bottom_temperature_last_c=44.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.5,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.7,
            dhw_bottom_temperature_last_c=45.5,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.6,
        ),
    ]

    dataset = build_dhw_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=DHWActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=4,
            min_segment_samples=2,
            min_dhw_power_kw=0.5,
            min_layer_temperature_spread_c=2.0,
            max_implied_tap_m3_per_h=0.01,
            min_segment_delivered_energy_kwh=0.1,
            min_segment_mean_layer_spread_c=2.0,
            min_segment_layer_spread_span_c=0.1,
            min_segment_bottom_temperature_rise_c=0.1,
            min_segment_top_temperature_rise_c=0.1,
        ),
    )

    assert dataset.sample_count == 4
    assert dataset.segment_count == 2
    assert dataset.raw_segment_count == 2
    assert dataset.dropped_segment_count == 0
    assert [sample.segment_index for sample in dataset.samples] == [0, 0, 1, 1]


def test_build_dhw_active_calibration_dataset_keeps_small_bucket_jitter() -> None:
    """Active-DHW calibration must keep informative no-draw runs despite mild dt jitter."""
    reference_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=12.0,
        R_loss=50.0,
    )
    start = datetime(2026, 4, 18, 6, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=47.8,
            dhw_bottom_temperature_last_c=41.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5, seconds=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.1,
            dhw_bottom_temperature_last_c=42.2,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.4,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10, seconds=5),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.4,
            dhw_bottom_temperature_last_c=43.5,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.5,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15, seconds=25),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.8,
            dhw_bottom_temperature_last_c=44.9,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.6,
        ),
    ]

    dataset = build_dhw_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=DHWActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=3,
            min_segment_samples=3,
            min_dhw_power_kw=0.5,
            min_layer_temperature_spread_c=2.0,
            max_implied_tap_m3_per_h=0.02,
            min_segment_delivered_energy_kwh=0.2,
            min_segment_mean_layer_spread_c=2.0,
            min_segment_layer_spread_span_c=0.1,
            min_segment_bottom_temperature_rise_c=0.2,
            min_segment_top_temperature_rise_c=0.1,
        ),
    )

    assert dataset.sample_count == 3
    np.testing.assert_allclose(
        [sample.dt_hours for sample in dataset.samples],
        [
            (aggregates[1].bucket_end_utc - aggregates[0].bucket_end_utc).total_seconds() / 3600.0,
            (aggregates[2].bucket_end_utc - aggregates[1].bucket_end_utc).total_seconds() / 3600.0,
            (aggregates[3].bucket_end_utc - aggregates[2].bucket_end_utc).total_seconds() / 3600.0,
        ],
        rtol=1e-12,
    )


def test_build_dhw_active_calibration_dataset_keeps_mixed_charging_runs() -> None:
    """Default active-DHW spread thresholds must keep mixed but informative charging runs."""
    reference_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=12.0,
        R_loss=50.0,
    )
    start = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=47.0,
            dhw_bottom_temperature_last_c=46.4,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=47.4,
            dhw_bottom_temperature_last_c=46.8,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.0,
            dhw_bottom_temperature_last_c=47.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.1,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.7,
            dhw_bottom_temperature_last_c=47.2,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.2,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=49.5,
            dhw_bottom_temperature_last_c=47.5,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.3,
        ),
    ]

    dataset = build_dhw_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=DHWActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=4,
            min_segment_samples=4,
            max_implied_tap_m3_per_h=0.05,
        ),
    )

    assert dataset.sample_count == 4
    assert dataset.segment_count == 1
    assert dataset.raw_segment_count == 1
    assert dataset.dropped_segment_count == 0
    quality = dataset.segment_qualities[0]
    assert quality.mean_layer_spread_c >= 1.0
    assert quality.bottom_temperature_rise_c > 0.5
    assert quality.top_temperature_rise_c > 0.1


def test_build_dhw_active_calibration_dataset_drops_weak_segments() -> None:
    """Weak but no-draw DHW runs must be dropped by the richer segment-quality filter."""
    reference_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=12.0,
        R_loss=50.0,
    )
    start = datetime(2026, 4, 18, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.0,
            dhw_bottom_temperature_last_c=42.5,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.1,
            dhw_bottom_temperature_last_c=42.7,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.8,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.15,
            dhw_bottom_temperature_last_c=42.8,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.85,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.15,
            dhw_bottom_temperature_last_c=42.8,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.0,
            dhw_bottom_temperature_last_c=41.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.9,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.5,
            dhw_bottom_temperature_last_c=43.8,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.6,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=49.0,
            dhw_bottom_temperature_last_c=45.6,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.8,
        ),
    ]

    dataset = build_dhw_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=DHWActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=2,
            min_segment_samples=2,
            min_dhw_power_kw=0.5,
            min_layer_temperature_spread_c=2.0,
            max_implied_tap_m3_per_h=0.05,
            min_segment_delivered_energy_kwh=0.2,
            min_segment_mean_layer_spread_c=3.0,
            min_segment_layer_spread_span_c=0.3,
            min_segment_bottom_temperature_rise_c=0.5,
            min_segment_top_temperature_rise_c=0.2,
        ),
    )

    assert dataset.raw_segment_count == 2
    assert dataset.segment_count == 1
    assert dataset.dropped_segment_count == 1
    assert [sample.segment_index for sample in dataset.samples] == [0, 0]
    kept_quality = [quality for quality in dataset.segment_qualities if quality.selected][0]
    dropped_quality = [quality for quality in dataset.segment_qualities if not quality.selected][0]
    assert kept_quality.bottom_temperature_rise_c > dropped_quality.bottom_temperature_rise_c


def test_build_dhw_active_calibration_dataset_keeps_best_segments_when_capped() -> None:
    """Top-N active-DHW segment selection must keep the highest-scoring informative runs."""
    reference_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=12.0,
        R_loss=50.0,
    )
    start = datetime(2026, 4, 18, 1, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.0,
            dhw_bottom_temperature_last_c=42.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.2,
            dhw_bottom_temperature_last_c=43.0,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=1.2,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.35,
            dhw_bottom_temperature_last_c=43.7,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=1.3,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.35,
            dhw_bottom_temperature_last_c=43.7,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.0,
            dhw_bottom_temperature_last_c=40.5,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=1.6,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=48.5,
            dhw_bottom_temperature_last_c=43.8,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=2.8,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="dhw",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            dhw_top_temperature_last_c=49.1,
            dhw_bottom_temperature_last_c=45.9,
            boiler_ambient_temp_mean_c=20.0,
            t_mains_estimated_mean_c=10.0,
            hp_thermal_power_mean_kw=3.0,
        ),
    ]

    dataset = build_dhw_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        settings=DHWActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=2,
            min_segment_samples=2,
            min_dhw_power_kw=0.5,
            min_layer_temperature_spread_c=2.0,
            max_implied_tap_m3_per_h=0.05,
            min_segment_delivered_energy_kwh=0.1,
            min_segment_mean_layer_spread_c=2.0,
            min_segment_layer_spread_span_c=0.1,
            min_segment_bottom_temperature_rise_c=0.1,
            min_segment_top_temperature_rise_c=0.1,
            max_selected_segments=1,
        ),
    )

    assert dataset.raw_segment_count == 2
    assert dataset.segment_count == 1
    assert dataset.dropped_segment_count == 1
    selected_scores = [quality.score for quality in dataset.segment_qualities if quality.selected]
    dropped_scores = [quality.score for quality in dataset.segment_qualities if not quality.selected]
    assert selected_scores[0] > dropped_scores[0]
    assert dataset.samples[0].t_bot_start_c == 40.5


def test_calibrate_dhw_active_stratification_recovers_synthetic_r_strat() -> None:
    """Active DHW no-draw calibration must recover synthetic ``R_strat`` from exact data."""
    true_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=14.0,
        R_loss=50.0,
    )
    reference_parameters = DHWParameters(
        dt_hours=true_parameters.dt_hours,
        C_top=true_parameters.C_top,
        C_bot=true_parameters.C_bot,
        R_strat=10.0,
        R_loss=true_parameters.R_loss,
    )
    model = DHWModel(true_parameters)
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    state = np.array([47.0, 41.0], dtype=float)
    samples: list[DHWActiveCalibrationSample] = []
    for step_k in range(96):
        interval_start = start + timedelta(hours=step_k * true_parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
        control_kw = 2.1 + 0.4 * np.sin(step_k / 6.0) + 0.15 * np.cos(step_k / 9.0)
        t_mains_c = 10.0 + 0.3 * np.sin(step_k / 20.0)
        t_amb_c = 20.0 + 0.2 * np.cos(step_k / 18.0)
        next_state = model.step(
            state=state,
            control_kw=control_kw,
            v_tap_m3_per_h=0.0,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
        )
        samples.append(
            DHWActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=true_parameters.dt_hours,
                t_top_start_c=float(state[0]),
                t_top_end_c=float(next_state[0]),
                t_bot_start_c=float(state[1]),
                t_bot_end_c=float(next_state[1]),
                p_dhw_mean_kw=control_kw,
                t_mains_c=t_mains_c,
                t_amb_c=t_amb_c,
                implied_v_tap_m3_per_h=0.0,
                segment_index=0,
            )
        )
        state = next_state

    dataset = DHWActiveCalibrationDataset(samples=tuple(samples))
    settings = DHWActiveCalibrationSettings(
        reference_parameters=reference_parameters,
        min_sample_count=24,
        min_segment_samples=4,
    )

    result = calibrate_dhw_active_stratification(dataset, settings)

    np.testing.assert_allclose(result.fitted_parameters.R_strat, true_parameters.R_strat, rtol=5e-2)
    np.testing.assert_allclose(result.fitted_parameters.R_loss, true_parameters.R_loss, rtol=1e-12)
    assert result.segment_count == 1
    assert result.rmse_t_top_c < 1e-6
    assert result.rmse_t_bot_c < 1e-6


def test_calibrate_dhw_active_stratification_recovers_strong_charge_time_mixing() -> None:
    """Active-DHW calibration must allow effective ``R_strat`` values far below 5 K/kW.

    Real charging telemetry can behave nearly like a mixed tank because the fitted
    ``R_strat`` is an *effective* remixing resistance, not a material constant.
    This regression protects against reintroducing an overly aggressive lower box
    bound that would force the optimiser onto an artificial constraint.
    """
    true_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=0.03,
        R_loss=50.0,
    )
    reference_parameters = DHWParameters(
        dt_hours=true_parameters.dt_hours,
        C_top=true_parameters.C_top,
        C_bot=true_parameters.C_bot,
        R_strat=1.0,
        R_loss=true_parameters.R_loss,
    )
    model = DHWModel(true_parameters)
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    state = np.array([47.0, 41.0], dtype=float)
    samples: list[DHWActiveCalibrationSample] = []
    for step_k in range(96):
        interval_start = start + timedelta(hours=step_k * true_parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
        control_kw = 2.0 + 0.35 * np.sin(step_k / 7.0) + 0.1 * np.cos(step_k / 13.0)
        t_mains_c = 10.0 + 0.25 * np.sin(step_k / 19.0)
        t_amb_c = 20.0 + 0.15 * np.cos(step_k / 17.0)
        next_state = model.step(
            state=state,
            control_kw=control_kw,
            v_tap_m3_per_h=0.0,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
        )
        samples.append(
            DHWActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=true_parameters.dt_hours,
                t_top_start_c=float(state[0]),
                t_top_end_c=float(next_state[0]),
                t_bot_start_c=float(state[1]),
                t_bot_end_c=float(next_state[1]),
                p_dhw_mean_kw=control_kw,
                t_mains_c=t_mains_c,
                t_amb_c=t_amb_c,
                implied_v_tap_m3_per_h=0.0,
                segment_index=0,
            )
        )
        state = next_state

    dataset = DHWActiveCalibrationDataset(samples=tuple(samples))
    settings = DHWActiveCalibrationSettings(
        reference_parameters=reference_parameters,
        min_sample_count=24,
        min_segment_samples=4,
    )

    result = calibrate_dhw_active_stratification(dataset, settings)

    np.testing.assert_allclose(result.fitted_parameters.R_strat, true_parameters.R_strat, rtol=5e-2)
    assert result.fitted_parameters.R_strat < 5.0
    assert result.rmse_t_top_c < 1e-6
    assert result.rmse_t_bot_c < 1e-6


def test_calibrate_dhw_active_stratification_recovers_synthetic_r_strat_with_variable_dt() -> None:
    """Active-DHW calibration must recover ``R_strat`` when persisted bucket dt jitters slightly."""
    dt_hours_sequence = (5.0 / 60.0, 4.0 / 60.0, 6.0 / 60.0, 5.5 / 60.0) * 24
    true_parameters = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=0.058,
        C_bot=0.058,
        R_strat=14.0,
        R_loss=50.0,
    )
    reference_parameters = DHWParameters(
        dt_hours=true_parameters.dt_hours,
        C_top=true_parameters.C_top,
        C_bot=true_parameters.C_bot,
        R_strat=10.0,
        R_loss=true_parameters.R_loss,
    )
    start = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    state = np.array([47.0, 41.0], dtype=float)
    samples: list[DHWActiveCalibrationSample] = []
    current_time = start
    for step_k, dt_hours in enumerate(dt_hours_sequence):
        interval_start = current_time
        interval_end = interval_start + timedelta(hours=dt_hours)
        control_kw = 2.0 + 0.35 * np.sin(step_k / 8.0) + 0.12 * np.cos(step_k / 11.0)
        t_mains_c = 10.0 + 0.3 * np.sin(step_k / 20.0)
        t_amb_c = 20.0 + 0.2 * np.cos(step_k / 18.0)
        next_state = DHWModel(
            DHWParameters(
                dt_hours=dt_hours,
                C_top=true_parameters.C_top,
                C_bot=true_parameters.C_bot,
                R_strat=true_parameters.R_strat,
                R_loss=true_parameters.R_loss,
            )
        ).step(
            state=state,
            control_kw=control_kw,
            v_tap_m3_per_h=0.0,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
        )
        samples.append(
            DHWActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=dt_hours,
                t_top_start_c=float(state[0]),
                t_top_end_c=float(next_state[0]),
                t_bot_start_c=float(state[1]),
                t_bot_end_c=float(next_state[1]),
                p_dhw_mean_kw=control_kw,
                t_mains_c=t_mains_c,
                t_amb_c=t_amb_c,
                implied_v_tap_m3_per_h=0.0,
                segment_index=0,
            )
        )
        state = next_state
        current_time = interval_end

    dataset = DHWActiveCalibrationDataset(samples=tuple(samples))
    settings = DHWActiveCalibrationSettings(
        reference_parameters=reference_parameters,
        min_sample_count=24,
        min_segment_samples=4,
    )

    result = calibrate_dhw_active_stratification(dataset, settings)

    np.testing.assert_allclose(result.fitted_parameters.R_strat, true_parameters.R_strat, rtol=5e-2)
    assert result.rmse_t_top_c < 1e-6
    assert result.rmse_t_bot_c < 1e-6


def test_build_ufh_active_calibration_dataset_filters_to_excited_ufh_windows() -> None:
    """Only consecutive, sufficiently excited UFH windows may enter the active dataset."""
    reference_parameters = ThermalParameters(
        dt_hours=5.0 / 60.0,
        C_r=3.0,
        C_b=18.0,
        R_br=2.5,
        R_ro=4.0,
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.0,
            outdoor_temperature_mean_c=10.0,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.35,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.1,
            outdoor_temperature_mean_c=10.2,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.5,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.2,
            outdoor_temperature_mean_c=10.4,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.45,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.2,
            outdoor_temperature_mean_c=10.4,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.0,
        ),
    ]
    forecasts = [
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=5), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=10), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=15), gti_w_per_m2=0.0),
    ]

    dataset = build_ufh_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        forecast_rows=cast(list, forecasts),
        settings=UFHActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=2,
            min_segment_samples=2,
            min_segment_ufh_power_span_kw=0.01,
            min_segment_room_temperature_span_c=0.01,
            min_segment_outdoor_temperature_span_c=0.01,
            min_segment_score=0.0,
            min_ufh_power_kw=0.1,
        ),
    )

    assert dataset.sample_count == 2
    assert dataset.segment_count == 1
    assert dataset.raw_segment_count == 1
    assert dataset.dropped_segment_count == 0
    sample = dataset.samples[0]
    assert sample.room_temperature_start_c == 20.0
    assert sample.room_temperature_end_c == 20.1
    assert sample.ufh_power_mean_kw == 0.5
    assert sample.segment_index == 0


def test_build_ufh_active_calibration_dataset_splits_contiguous_runs() -> None:
    """A gap in valid UFH replay pairs must start a new calibration segment."""
    reference_parameters = ThermalParameters(
        dt_hours=5.0 / 60.0,
        C_r=3.0,
        C_b=18.0,
        R_br=2.5,
        R_ro=4.0,
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.0,
            outdoor_temperature_mean_c=9.5,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.4,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.1,
            outdoor_temperature_mean_c=9.6,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.45,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.18,
            outdoor_temperature_mean_c=9.7,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.5,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.1,
            outdoor_temperature_mean_c=9.7,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.2,
            outdoor_temperature_mean_c=9.8,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.55,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.3,
            outdoor_temperature_mean_c=9.9,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.6,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.45,
            outdoor_temperature_mean_c=10.0,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.65,
        ),
    ]
    forecasts = [
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=5), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=10), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=15), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=20), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=25), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=30), gti_w_per_m2=0.0),
    ]

    dataset = build_ufh_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        forecast_rows=cast(list, forecasts),
        settings=UFHActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=4,
            min_segment_samples=2,
            min_segment_ufh_power_span_kw=0.01,
            min_segment_room_temperature_span_c=0.01,
            min_segment_outdoor_temperature_span_c=0.01,
            min_segment_score=0.0,
            min_ufh_power_kw=0.1,
        ),
    )

    assert dataset.sample_count == 4
    assert dataset.segment_count == 2
    assert dataset.raw_segment_count == 2
    assert dataset.dropped_segment_count == 0
    assert [sample.segment_index for sample in dataset.samples] == [0, 0, 1, 1]


def test_build_ufh_active_calibration_dataset_keeps_only_best_segments_when_capped() -> None:
    """Top-N segment selection must keep the highest-scoring informative UFH runs."""
    reference_parameters = ThermalParameters(
        dt_hours=5.0 / 60.0,
        C_r=3.0,
        C_b=18.0,
        R_br=2.5,
        R_ro=4.0,
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )
    start = datetime(2026, 4, 18, 0, 0, tzinfo=timezone.utc)
    aggregates = [
        SimpleNamespace(
            bucket_end_utc=start,
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.0,
            outdoor_temperature_mean_c=6.0,
            household_elec_power_mean_kw=0.2,
            hp_thermal_power_mean_kw=0.3,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=5),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.04,
            outdoor_temperature_mean_c=6.05,
            household_elec_power_mean_kw=0.2,
            hp_thermal_power_mean_kw=0.35,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=10),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.08,
            outdoor_temperature_mean_c=6.1,
            household_elec_power_mean_kw=0.2,
            hp_thermal_power_mean_kw=0.4,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=15),
            hp_mode_last="off",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.08,
            outdoor_temperature_mean_c=6.1,
            household_elec_power_mean_kw=0.2,
            hp_thermal_power_mean_kw=0.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=20),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.2,
            outdoor_temperature_mean_c=4.5,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=0.5,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=25),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.5,
            outdoor_temperature_mean_c=5.1,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=1.0,
        ),
        SimpleNamespace(
            bucket_end_utc=start + timedelta(minutes=30),
            hp_mode_last="ufh",
            defrost_active_fraction=0.0,
            booster_heater_active_fraction=0.0,
            room_temperature_last_c=20.85,
            outdoor_temperature_mean_c=5.8,
            household_elec_power_mean_kw=0.25,
            hp_thermal_power_mean_kw=1.5,
        ),
    ]
    forecasts = [
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=5), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=10), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=15), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=20), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=25), gti_w_per_m2=0.0),
        SimpleNamespace(valid_at_utc=start + timedelta(minutes=30), gti_w_per_m2=0.0),
    ]

    dataset = build_ufh_active_calibration_dataset(
        aggregates=cast(list, aggregates),
        forecast_rows=cast(list, forecasts),
        settings=UFHActiveCalibrationSettings(
            reference_parameters=reference_parameters,
            min_sample_count=2,
            min_segment_samples=2,
            max_selected_segments=1,
            min_segment_room_temperature_span_c=0.01,
            min_segment_outdoor_temperature_span_c=0.01,
            min_segment_ufh_power_span_kw=0.01,
            min_ufh_power_kw=0.1,
        ),
    )

    assert dataset.raw_segment_count == 2
    assert dataset.segment_count == 1
    assert dataset.dropped_segment_count == 1
    assert [sample.segment_index for sample in dataset.samples] == [0, 0]
    assert dataset.samples[0].room_temperature_start_c == 20.2
    selected_scores = [quality.score for quality in dataset.segment_qualities if quality.selected]
    dropped_scores = [quality.score for quality in dataset.segment_qualities if not quality.selected]
    assert selected_scores[0] > dropped_scores[0]


def test_calibrate_ufh_active_rc_recovers_synthetic_parameters() -> None:
    """Active UFH Kalman replay must recover synthetic RC parameters from exact data."""
    true_parameters = ThermalParameters(
        dt_hours=5.0 / 60.0,
        C_r=3.4,
        C_b=20.0,
        R_br=1.9,
        R_ro=6.2,
        alpha=0.3,
        eta=0.58,
        A_glass=11.0,
    )
    reference_parameters = ThermalParameters(
        dt_hours=true_parameters.dt_hours,
        C_r=true_parameters.C_r,
        C_b=17.0,
        R_br=2.4,
        R_ro=5.4,
        alpha=true_parameters.alpha,
        eta=true_parameters.eta,
        A_glass=true_parameters.A_glass,
    )
    model = ThermalModel(true_parameters)

    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    state = np.array([20.0, 22.2], dtype=float)
    samples: list[UFHActiveCalibrationSample] = []
    for step_k in range(192):
        interval_start = start + timedelta(hours=step_k * true_parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
        control_kw = 1.2 + 0.55 * np.sin(step_k / 4.0) + 0.2 * np.cos(step_k / 11.0)
        outdoor_temperature_c = 2.5 + 4.0 * np.cos(step_k / 17.0)
        gti_w_per_m2 = 0.0
        solar_gain_kw_value = float(
            solar_gain_kw(
                gti_w_per_m2,
                glass_area_m2=true_parameters.A_glass,
                transmittance=true_parameters.eta,
            )
        )
        internal_gain_kw = 0.18 + 0.12 * np.cos(step_k / 6.0)
        next_state = model.step(
            state=state,
            control_kw=control_kw,
            outdoor_temperature_c=outdoor_temperature_c,
            solar_gain_kw_value=solar_gain_kw_value,
            internal_gain_kw=internal_gain_kw,
        )
        samples.append(
            UFHActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=true_parameters.dt_hours,
                room_temperature_start_c=float(state[0]),
                room_temperature_end_c=float(next_state[0]),
                outdoor_temperature_mean_c=outdoor_temperature_c,
                gti_w_per_m2=gti_w_per_m2,
                internal_gain_proxy_kw=internal_gain_kw,
                ufh_power_mean_kw=control_kw,
                segment_index=0,
            )
        )
        state = next_state

    dataset = UFHActiveCalibrationDataset(samples=tuple(samples))
    settings = UFHActiveCalibrationSettings(
        reference_parameters=reference_parameters,
        min_sample_count=32,
        fit_c_r=False,
        initial_floor_temperature_offset_c=2.2,
        initial_room_covariance_k2=1e-4,
        initial_floor_covariance_k2=1e-2,
        process_noise_room_k2=1e-8,
        process_noise_floor_k2=1e-8,
        measurement_variance_k2=1e-8,
    )

    result = calibrate_ufh_active_rc(dataset, settings)

    np.testing.assert_allclose(result.fitted_parameters.C_b, true_parameters.C_b, rtol=6e-2)
    np.testing.assert_allclose(result.fitted_parameters.R_br, true_parameters.R_br, rtol=6e-2)
    np.testing.assert_allclose(result.fitted_parameters.R_ro, true_parameters.R_ro, rtol=6e-2)
    np.testing.assert_allclose(result.fitted_parameters.C_r, true_parameters.C_r, rtol=1e-12)
    assert result.segment_count == 1
    assert result.rmse_room_temperature_c < 1e-5


def test_calibrate_ufh_active_rc_fits_initial_floor_offset_across_segments() -> None:
    """The optional nuisance floor-offset parameter must be recovered across replay resets."""
    true_parameters = ThermalParameters(
        dt_hours=5.0 / 60.0,
        C_r=3.2,
        C_b=19.0,
        R_br=2.1,
        R_ro=5.8,
        alpha=0.3,
        eta=0.58,
        A_glass=10.5,
    )
    true_floor_offset_c = 3.6
    model = ThermalModel(true_parameters)
    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    segment_lengths = (72, 60)
    samples: list[UFHActiveCalibrationSample] = []

    elapsed_hours = 0.0
    for segment_index, segment_length in enumerate(segment_lengths):
        room_temperature_c = 20.0 + 0.2 * segment_index
        state = np.array([room_temperature_c, room_temperature_c + true_floor_offset_c], dtype=float)
        for local_step in range(segment_length):
            interval_start = start + timedelta(hours=elapsed_hours)
            interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
            global_step = sum(segment_lengths[:segment_index]) + local_step
            control_kw = 1.1 + 0.45 * np.sin(global_step / 5.0)
            outdoor_temperature_c = 3.0 + 2.5 * np.cos(global_step / 13.0)
            gti_w_per_m2 = 0.0
            internal_gain_kw = 0.22 + 0.05 * np.cos(global_step / 7.0)
            next_state = model.step(
                state=state,
                control_kw=control_kw,
                outdoor_temperature_c=outdoor_temperature_c,
                solar_gain_kw_value=0.0,
                internal_gain_kw=internal_gain_kw,
            )
            samples.append(
                UFHActiveCalibrationSample(
                    interval_start_utc=interval_start,
                    interval_end_utc=interval_end,
                    dt_hours=true_parameters.dt_hours,
                    room_temperature_start_c=float(state[0]),
                    room_temperature_end_c=float(next_state[0]),
                    outdoor_temperature_mean_c=outdoor_temperature_c,
                    gti_w_per_m2=gti_w_per_m2,
                    internal_gain_proxy_kw=internal_gain_kw,
                    ufh_power_mean_kw=control_kw,
                    segment_index=segment_index,
                )
            )
            state = next_state
            elapsed_hours += true_parameters.dt_hours
        elapsed_hours += true_parameters.dt_hours

    dataset = UFHActiveCalibrationDataset(samples=tuple(samples))
    settings = UFHActiveCalibrationSettings(
        reference_parameters=true_parameters,
        min_sample_count=32,
        min_segment_samples=16,
        fit_initial_floor_temperature_offset=True,
        initial_floor_temperature_offset_c=1.0,
        min_initial_floor_temperature_offset_c=-1.0,
        max_initial_floor_temperature_offset_c=6.0,
        min_parameter_ratio=0.999,
        max_parameter_ratio=1.001,
        initial_room_covariance_k2=1e-4,
        initial_floor_covariance_k2=1e-2,
        process_noise_room_k2=1e-8,
        process_noise_floor_k2=1e-8,
        measurement_variance_k2=1e-8,
    )

    result = calibrate_ufh_active_rc(dataset, settings)

    np.testing.assert_allclose(
        result.fitted_initial_floor_temperature_offset_c,
        true_floor_offset_c,
        rtol=5e-2,
    )
    assert result.segment_count == 2
    assert result.rmse_room_temperature_c < 1e-5


def test_calibrate_ufh_active_rc_recovers_gain_parameters() -> None:
    """Active UFH calibration must recover solar gain and internal-gain mapping."""
    true_parameters = ThermalParameters(
        dt_hours=0.25,
        C_r=5.5,
        C_b=11.0,
        R_br=1.1,
        R_ro=8.5,
        alpha=0.25,
        eta=0.42,
        A_glass=8.0,
    )
    true_internal_gains_kw = 0.25
    true_internal_gains_heat_fraction = 0.58
    model = ThermalModel(true_parameters)
    state = np.array([20.0, 22.0], dtype=float)
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    samples: list[UFHActiveCalibrationSample] = []

    for index in range(48):
        interval_start = start + timedelta(hours=index * true_parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
        outdoor_temperature_c = 2.0 + 4.0 * sin(index / 7.0)
        gti_w_per_m2 = 250.0 + 180.0 * max(sin(index / 6.0), 0.0)
        ufh_power_mean_kw = 1.0 + 0.6 * max(cos(index / 5.0), 0.0)
        internal_gain_proxy_kw = 0.4 + 0.9 * max(sin(index / 4.0), 0.0)
        solar_gain_kw_value = float(
            solar_gain_kw(
                gti_w_per_m2,
                glass_area_m2=true_parameters.A_glass,
                transmittance=true_parameters.eta,
            )
        )
        internal_gain_kw = float(
            max(true_internal_gains_kw, true_internal_gains_heat_fraction * internal_gain_proxy_kw)
        )
        next_state = model.step(
            state=state,
            control_kw=ufh_power_mean_kw,
            outdoor_temperature_c=outdoor_temperature_c,
            solar_gain_kw_value=solar_gain_kw_value,
            internal_gain_kw=internal_gain_kw,
        )
        samples.append(
            UFHActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=true_parameters.dt_hours,
                room_temperature_start_c=float(state[0]),
                room_temperature_end_c=float(next_state[0]),
                outdoor_temperature_mean_c=outdoor_temperature_c,
                gti_w_per_m2=gti_w_per_m2,
                internal_gain_proxy_kw=internal_gain_proxy_kw,
                ufh_power_mean_kw=ufh_power_mean_kw,
                segment_index=0,
            )
        )
        state = next_state

    dataset = UFHActiveCalibrationDataset(samples=tuple(samples))
    settings = UFHActiveCalibrationSettings(
        reference_parameters=ThermalParameters(
            dt_hours=true_parameters.dt_hours,
            C_r=true_parameters.C_r,
            C_b=true_parameters.C_b,
            R_br=true_parameters.R_br,
            R_ro=true_parameters.R_ro,
            alpha=true_parameters.alpha,
            eta=0.55,
            A_glass=true_parameters.A_glass,
        ),
        reference_internal_gains_kw=true_internal_gains_kw,
        reference_internal_gains_heat_fraction=0.75,
        min_sample_count=24,
        min_segment_samples=12,
        fit_eta=True,
        fit_internal_gains_heat_fraction=True,
        min_parameter_ratio=0.999,
        max_parameter_ratio=1.001,
        min_eta=0.2,
        max_eta=0.8,
        min_internal_gains_heat_fraction=0.0,
        max_internal_gains_heat_fraction=1.0,
        initial_room_covariance_k2=1e-4,
        initial_floor_covariance_k2=1e-2,
        process_noise_room_k2=1e-8,
        process_noise_floor_k2=1e-8,
        measurement_variance_k2=1e-8,
    )

    result = calibrate_ufh_active_rc(dataset, settings)

    np.testing.assert_allclose(result.fitted_parameters.eta, true_parameters.eta, atol=5e-2)
    np.testing.assert_allclose(
        result.fitted_internal_gains_heat_fraction,
        true_internal_gains_heat_fraction,
        atol=7e-2,
    )


def test_calibrate_ufh_active_rc_supports_room_temperature_bias_parameterization() -> None:
    """Active UFH calibration must safely support a room-temperature bias parameter."""
    true_parameters = ThermalParameters(
        dt_hours=0.25,
        C_r=5.5,
        C_b=11.0,
        R_br=1.1,
        R_ro=8.5,
        alpha=0.25,
        eta=0.42,
        A_glass=8.0,
    )
    true_room_bias_c = 0.45
    model = ThermalModel(true_parameters)
    state = np.array([20.0, 22.0], dtype=float)
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    samples: list[UFHActiveCalibrationSample] = []

    for index in range(48):
        interval_start = start + timedelta(hours=index * true_parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
        outdoor_temperature_c = 1.0 + 5.0 * sin(index / 9.0)
        ufh_power_mean_kw = 0.9 + 0.7 * max(cos(index / 6.0), 0.0)
        internal_gain_proxy_kw = 0.2
        next_state = model.step(
            state=state,
            control_kw=ufh_power_mean_kw,
            outdoor_temperature_c=outdoor_temperature_c,
            solar_gain_kw_value=0.0,
            internal_gain_kw=0.2,
        )
        samples.append(
            UFHActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=true_parameters.dt_hours,
                room_temperature_start_c=float(state[0] - true_room_bias_c),
                room_temperature_end_c=float(next_state[0] - true_room_bias_c),
                outdoor_temperature_mean_c=outdoor_temperature_c,
                gti_w_per_m2=0.0,
                internal_gain_proxy_kw=internal_gain_proxy_kw,
                ufh_power_mean_kw=ufh_power_mean_kw,
                segment_index=0,
            )
        )
        state = next_state

    dataset = UFHActiveCalibrationDataset(samples=tuple(samples))
    settings = UFHActiveCalibrationSettings(
        reference_parameters=true_parameters,
        reference_internal_gains_kw=0.2,
        reference_internal_gains_heat_fraction=0.0,
        min_sample_count=24,
        min_segment_samples=12,
        fit_room_temperature_bias=True,
        min_parameter_ratio=0.999,
        max_parameter_ratio=1.001,
        min_room_temperature_bias_c=-1.0,
        max_room_temperature_bias_c=1.0,
        initial_room_covariance_k2=1e-4,
        initial_floor_covariance_k2=1e-2,
        process_noise_room_k2=1e-8,
        process_noise_floor_k2=1e-8,
        measurement_variance_k2=1e-8,
    )

    result = calibrate_ufh_active_rc(dataset, settings)

    assert result.fit_room_temperature_bias is True
    assert settings.min_room_temperature_bias_c <= result.fitted_room_temperature_bias_c <= settings.max_room_temperature_bias_c
    assert np.isfinite(result.fitted_room_temperature_bias_c)
    assert result.rmse_room_temperature_c >= 0.0


def test_calibrate_dhw_standby_loss_recovers_ambient_sensor_bias() -> None:
    """Standby DHW calibration must recover an ambient-temperature sensor bias."""
    c_top = 0.06
    c_bot = 0.06
    c_total = c_top + c_bot
    true_r_loss = 40.0
    tau_true_hours = c_total * true_r_loss / 2.0
    true_ambient_bias_c = 0.9
    dt_hours = 0.1
    mean_tank_c = 55.0
    true_ambient_c = 18.0
    raw_ambient_c = true_ambient_c - true_ambient_bias_c
    start = datetime(2026, 2, 2, tzinfo=timezone.utc)
    samples: list[DHWStandbyCalibrationSample] = []

    for index in range(30):
        next_mean_tank_c = mean_tank_c + dt_hours / tau_true_hours * (-(mean_tank_c - true_ambient_c))
        interval_start = start + timedelta(hours=index * dt_hours)
        interval_end = interval_start + timedelta(hours=dt_hours)
        samples.append(
            DHWStandbyCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=dt_hours,
                t_top_start_c=mean_tank_c,
                t_top_end_c=next_mean_tank_c,
                t_bot_start_c=mean_tank_c,
                t_bot_end_c=next_mean_tank_c,
                boiler_ambient_mean_c=raw_ambient_c,
            )
        )
        mean_tank_c = next_mean_tank_c

    result = calibrate_dhw_standby_loss(
        DHWStandbyCalibrationDataset(samples=tuple(samples)),
        DHWStandbyCalibrationSettings(
            dt_hours=dt_hours,
            reference_c_top_kwh_per_k=c_top,
            reference_c_bot_kwh_per_k=c_bot,
            min_sample_count=12,
            fit_ambient_temperature_bias=True,
            min_tau_hours=0.5,
            max_tau_hours=8.0,
            initial_tau_hours=3.0,
            min_ambient_temperature_bias_c=-2.0,
            max_ambient_temperature_bias_c=2.0,
        ),
    )

    np.testing.assert_allclose(result.suggested_r_loss_k_per_kw, true_r_loss, atol=5e-2)
    np.testing.assert_allclose(result.fitted_ambient_temperature_bias_c, true_ambient_bias_c, atol=5e-2)


def test_calibrate_dhw_active_stratification_recovers_capacity_split_and_sensor_biases() -> None:
    """Active DHW calibration must recover the C_top/C_bot split and layer-temperature biases."""
    true_parameters = DHWParameters(
        dt_hours=0.1,
        C_top=0.078,
        C_bot=0.042,
        R_strat=8.5,
        R_loss=45.0,
    )
    true_top_bias_c = 0.6
    true_bot_bias_c = -0.35
    model = DHWModel(true_parameters)
    state = np.array([54.0, 44.0], dtype=float)
    start = datetime(2026, 2, 3, tzinfo=timezone.utc)
    samples: list[DHWActiveCalibrationSample] = []

    for index in range(36):
        interval_start = start + timedelta(hours=index * true_parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
        p_dhw_mean_kw = 1.4 + 0.8 * max(sin(index / 4.0), 0.0)
        t_mains_c = 10.0 + 0.5 * sin(index / 9.0)
        t_amb_c = 19.0
        next_state = model.step(
            state=state,
            control_kw=p_dhw_mean_kw,
            v_tap_m3_per_h=0.0,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
        )
        samples.append(
            DHWActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=true_parameters.dt_hours,
                t_top_start_c=float(state[0] - true_top_bias_c),
                t_top_end_c=float(next_state[0] - true_top_bias_c),
                t_bot_start_c=float(state[1] - true_bot_bias_c),
                t_bot_end_c=float(next_state[1] - true_bot_bias_c),
                p_dhw_mean_kw=p_dhw_mean_kw,
                t_mains_c=t_mains_c,
                t_amb_c=t_amb_c,
                implied_v_tap_m3_per_h=0.0,
                segment_index=0,
            )
        )
        state = next_state

    result = calibrate_dhw_active_stratification(
        DHWActiveCalibrationDataset(samples=tuple(samples)),
        DHWActiveCalibrationSettings(
            reference_parameters=DHWParameters(
                dt_hours=true_parameters.dt_hours,
                C_top=0.06,
                C_bot=0.06,
                R_strat=10.0,
                R_loss=true_parameters.R_loss,
            ),
            min_sample_count=12,
            min_segment_samples=6,
            fit_capacity_split=True,
            fit_temperature_biases=True,
            initial_c_top_fraction=0.5,
            min_c_top_fraction=0.2,
            max_c_top_fraction=0.8,
            min_temperature_bias_c=-1.0,
            max_temperature_bias_c=1.0,
            min_r_strat_k_per_kw=2.0,
            max_r_strat_k_per_kw=15.0,
        ),
    )

    np.testing.assert_allclose(result.fitted_parameters.C_top, true_parameters.C_top, atol=5e-3)
    np.testing.assert_allclose(result.fitted_parameters.C_bot, true_parameters.C_bot, atol=5e-3)
    np.testing.assert_allclose(result.fitted_parameters.R_strat, true_parameters.R_strat, atol=2e-1)
    np.testing.assert_allclose(result.fitted_t_top_bias_c, true_top_bias_c, atol=5e-2)
    np.testing.assert_allclose(result.fitted_t_bot_bias_c, true_bot_bias_c, atol=5e-2)


def test_calibrate_dhw_active_stratification_recovers_total_capacity_scale() -> None:
    """Active DHW calibration must recover the total tank capacity instead of relying on a fixed assumption."""
    true_parameters = DHWParameters(
        dt_hours=0.1,
        C_top=0.078,
        C_bot=0.042,
        R_strat=8.5,
        R_loss=45.0,
    )
    model = DHWModel(true_parameters)
    state = np.array([54.0, 44.0], dtype=float)
    start = datetime(2026, 2, 3, tzinfo=timezone.utc)
    samples: list[DHWActiveCalibrationSample] = []

    for index in range(36):
        interval_start = start + timedelta(hours=index * true_parameters.dt_hours)
        interval_end = interval_start + timedelta(hours=true_parameters.dt_hours)
        p_dhw_mean_kw = 1.4 + 0.8 * max(sin(index / 4.0), 0.0)
        t_mains_c = 10.0 + 0.5 * sin(index / 9.0)
        t_amb_c = 19.0
        next_state = model.step(
            state=state,
            control_kw=p_dhw_mean_kw,
            v_tap_m3_per_h=0.0,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
        )
        samples.append(
            DHWActiveCalibrationSample(
                interval_start_utc=interval_start,
                interval_end_utc=interval_end,
                dt_hours=true_parameters.dt_hours,
                t_top_start_c=float(state[0]),
                t_top_end_c=float(next_state[0]),
                t_bot_start_c=float(state[1]),
                t_bot_end_c=float(next_state[1]),
                p_dhw_mean_kw=p_dhw_mean_kw,
                t_mains_c=t_mains_c,
                t_amb_c=t_amb_c,
                implied_v_tap_m3_per_h=0.0,
                segment_index=0,
            )
        )
        state = next_state

    result = calibrate_dhw_active_stratification(
        DHWActiveCalibrationDataset(samples=tuple(samples)),
        DHWActiveCalibrationSettings(
            reference_parameters=DHWParameters(
                dt_hours=true_parameters.dt_hours,
                C_top=0.06,
                C_bot=0.06,
                R_strat=10.0,
                R_loss=true_parameters.R_loss,
            ),
            min_sample_count=12,
            min_segment_samples=6,
            fit_total_capacity=True,
            fit_capacity_split=True,
            initial_c_total_scale=1.0,
            min_c_total_scale=0.5,
            max_c_total_scale=1.5,
            initial_c_top_fraction=0.5,
            min_c_top_fraction=0.2,
            max_c_top_fraction=0.8,
            min_r_strat_k_per_kw=2.0,
            max_r_strat_k_per_kw=15.0,
        ),
    )

    np.testing.assert_allclose(
        result.fitted_parameters.C_top + result.fitted_parameters.C_bot,
        true_parameters.C_top + true_parameters.C_bot,
        atol=5e-3,
    )
    np.testing.assert_allclose(result.fitted_parameters.C_top, true_parameters.C_top, atol=5e-3)
    np.testing.assert_allclose(result.fitted_parameters.C_bot, true_parameters.C_bot, atol=5e-3)
