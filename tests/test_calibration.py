"""Tests for offline thermal-parameter calibration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast

import numpy as np

from home_optimizer.calibration import (
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
    build_dhw_active_calibration_dataset,
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
from home_optimizer.cop_model import HeatPumpCOPModel, HeatPumpCOPParameters
from home_optimizer.dhw_model import DHWModel
from home_optimizer.thermal_model import ThermalModel, solar_gain_kw
from home_optimizer.types import DHWParameters
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
            bucket_start_utc=start + timedelta(minutes=10),
            bucket_end_utc=start + timedelta(minutes=15),
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
            bucket_start_utc=start + timedelta(minutes=15),
            bucket_end_utc=start + timedelta(minutes=20),
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
            bucket_start_utc=start + timedelta(minutes=20),
            bucket_end_utc=start + timedelta(minutes=25),
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
        ),
    )

    assert dataset.sample_count == 3
    assert dataset.ufh_sample_count == 2
    assert dataset.dhw_sample_count == 1
    assert all(sample.actual_cop > 1.0 for sample in dataset.samples)


def test_calibrate_cop_model_recovers_synthetic_parameters() -> None:
    """Offline COP calibration must recover synthetic heating-curve and eta parameters."""
    true_parameters = HeatPumpCOPParameters(
        eta_carnot=0.47,
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

    np.testing.assert_allclose(result.fitted_parameters.eta_carnot, true_parameters.eta_carnot, rtol=2e-2)
    np.testing.assert_allclose(result.fitted_parameters.T_supply_min, true_parameters.T_supply_min, rtol=2e-2)
    np.testing.assert_allclose(
        result.fitted_parameters.heating_curve_slope,
        true_parameters.heating_curve_slope,
        rtol=2e-2,
    )
    assert result.ufh_sample_count == 48
    assert result.dhw_sample_count == 32
    assert result.rmse_supply_temperature_c < 1e-6
    assert result.rmse_electric_energy_kwh < 1e-6
    assert result.rmse_actual_cop < 1e-6


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
    np.testing.assert_allclose(result.tau_standby_hours, expected_tau_hours, rtol=1e-3)
    np.testing.assert_allclose(result.suggested_r_loss_k_per_kw, parameters.R_loss, rtol=1e-3)
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


