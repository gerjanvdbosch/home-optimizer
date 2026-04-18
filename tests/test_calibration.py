"""Tests for offline thermal-parameter calibration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import numpy as np

from home_optimizer.calibration import (
    UFHActiveCalibrationDataset,
    UFHActiveCalibrationSample,
    UFHActiveCalibrationSettings,
    UFHCalibrationDataset,
    UFHCalibrationSample,
    UFHOffCalibrationSettings,
    build_ufh_active_calibration_dataset,
    calibrate_ufh_active_rc,
    build_ufh_off_calibration_dataset,
    calibrate_ufh_off_envelope,
)
from home_optimizer.thermal_model import ThermalModel, solar_gain_kw
from home_optimizer.types import ThermalParameters


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
            min_ufh_power_kw=0.1,
        ),
    )

    assert dataset.sample_count == 2
    sample = dataset.samples[0]
    assert sample.room_temperature_start_c == 20.0
    assert sample.room_temperature_end_c == 20.1
    assert sample.ufh_power_mean_kw == 0.5


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
    assert result.rmse_room_temperature_c < 1e-5


