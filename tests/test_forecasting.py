"""Tests for the ML-based runtime forecasting layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from home_optimizer.forecasting import DHWTapForecaster, ForecastService, ShutterForecaster
from home_optimizer.application.optimizer import Optimizer, RunRequest
from home_optimizer.sensors import LiveReadings
from home_optimizer.telemetry import TelemetryRepository, aggregate_readings
from home_optimizer.types.calibration import CalibrationParameterOverrides, CalibrationSnapshotPayload
from home_optimizer.types.constants import LAMBDA_WATER_KWH_PER_M3_K


def _reading(
    timestamp: datetime,
    *,
    shutter_living_room_pct: float,
    outdoor_temperature_c: float,
    household_elec_power_kw: float = 0.0,
    hp_mode: str = "ufh",
    hp_supply_temperature_c: float = 31.0,
    hp_supply_target_temperature_c: float = 33.0,
    hp_return_temperature_c: float = 27.0,
    hp_flow_lpm: float = 9.0,
    hp_electric_power_kw: float = 2.0,
    dhw_top_temperature_c: float = 52.0,
    dhw_bottom_temperature_c: float = 45.0,
) -> LiveReadings:
    """Create one fully populated telemetry sample for forecasting tests."""
    pv_output_kw = 0.6
    return LiveReadings(
        room_temperature_c=20.5,
        outdoor_temperature_c=outdoor_temperature_c,
        hp_supply_temperature_c=hp_supply_temperature_c,
        hp_supply_target_temperature_c=hp_supply_target_temperature_c,
        hp_return_temperature_c=hp_return_temperature_c,
        hp_flow_lpm=hp_flow_lpm,
        hp_electric_power_kw=hp_electric_power_kw,
        hp_mode=hp_mode,
        p1_net_power_kw=household_elec_power_kw + hp_electric_power_kw - pv_output_kw,
        pv_output_kw=pv_output_kw,
        thermostat_setpoint_c=20.5,
        dhw_top_temperature_c=dhw_top_temperature_c,
        dhw_bottom_temperature_c=dhw_bottom_temperature_c,
        shutter_living_room_pct=shutter_living_room_pct,
        defrost_active=False,
        booster_heater_active=False,
        boiler_ambient_temp_c=18.0,
        refrigerant_condensation_temp_c=38.0,
        refrigerant_liquid_line_temp_c=28.0,
        discharge_temp_c=65.0,
        t_mains_estimated_c=10.5,
        timestamp=timestamp,
        pv_total_kwh=1000.0,
        hp_electric_total_kwh=500.0,
        p1_import_total_kwh=800.0,
        p1_export_total_kwh=200.0,
    )


def _synthetic_gti_w_per_m2(valid_at_utc: datetime) -> float:
    """Deterministic irradiance profile used to train a predictable shutter pattern."""
    hour = valid_at_utc.hour
    if 11 <= hour <= 14:
        return 650.0
    if 8 <= hour <= 10 or 15 <= hour <= 17:
        return 180.0
    return 0.0


def _synthetic_shutter_pct(gti_w_per_m2: float) -> float:
    """Behavior rule used to generate training data for the ML model."""
    if gti_w_per_m2 >= 500.0:
        return 20.0
    if gti_w_per_m2 > 0.0:
        return 65.0
    return 100.0


def _populate_shutter_history(repository: TelemetryRepository, *, start_utc: datetime, hours: int) -> None:
    """Populate telemetry history plus aligned weather rows for shutter-model training."""
    for step in range(hours):
        valid_at_utc = start_utc + timedelta(hours=step)
        gti_w_per_m2 = _synthetic_gti_w_per_m2(valid_at_utc)
        shutter_pct = _synthetic_shutter_pct(gti_w_per_m2)
        outdoor_temperature_c = 5.0 + 0.25 * valid_at_utc.hour
        aggregate = aggregate_readings(
            [
                _reading(
                    valid_at_utc - timedelta(minutes=5),
                    shutter_living_room_pct=shutter_pct,
                    outdoor_temperature_c=outdoor_temperature_c,
                    household_elec_power_kw=0.4,
                ),
                _reading(
                    valid_at_utc,
                    shutter_living_room_pct=shutter_pct,
                    outdoor_temperature_c=outdoor_temperature_c,
                    household_elec_power_kw=0.4,
                ),
            ]
        )
        aggregate.update(
            {
                "electricity_price_mean_eur_per_kwh": 0.25,
                "electricity_price_last_eur_per_kwh": 0.25,
                "feed_in_price_eur_per_kwh": 0.0,
            }
        )
        repository.add_aggregate(aggregate)
        repository.bulk_add_forecast_snapshots(
            [
                {
                    "fetched_at_utc": valid_at_utc,
                    "valid_at_utc": valid_at_utc,
                    "step_k": 0,
                    "dt_hours": 1.0,
                    "t_out_c": outdoor_temperature_c,
                    "gti_w_per_m2": gti_w_per_m2,
                    "gti_pv_w_per_m2": 0.0,
                }
            ]
        )


def _add_future_forecast_batch(
    repository: TelemetryRepository,
    *,
    fetched_at_utc: datetime,
    gti_profile_w_per_m2: list[float],
) -> None:
    """Persist one future hourly forecast batch for inference-time testing."""
    repository.bulk_add_forecast_snapshots(
        [
            {
                "fetched_at_utc": fetched_at_utc,
                "valid_at_utc": fetched_at_utc + timedelta(hours=step_k),
                "step_k": step_k,
                "dt_hours": 1.0,
                "t_out_c": 6.0 + 0.5 * step_k,
                "gti_w_per_m2": gti_w_per_m2,
                "gti_pv_w_per_m2": 0.0,
            }
            for step_k, gti_w_per_m2 in enumerate(gti_profile_w_per_m2)
        ]
    )


def _synthetic_baseload_kw(valid_at_utc: datetime) -> float:
    """Deterministic daily electrical baseload profile [kW]."""
    hour = valid_at_utc.hour
    if 6 <= hour <= 8:
        return 0.85
    if 18 <= hour <= 22:
        return 1.25
    if 0 <= hour <= 5:
        return 0.30
    return 0.55


def _populate_baseload_history(repository: TelemetryRepository, *, start_utc: datetime, hours: int) -> None:
    """Populate telemetry history plus aligned weather rows for baseload-model training."""
    for step in range(hours):
        valid_at_utc = start_utc + timedelta(hours=step)
        gti_w_per_m2 = _synthetic_gti_w_per_m2(valid_at_utc)
        baseload_kw = _synthetic_baseload_kw(valid_at_utc)
        outdoor_temperature_c = 5.0 + 0.25 * valid_at_utc.hour
        aggregate = aggregate_readings(
            [
                _reading(
                    valid_at_utc - timedelta(minutes=5),
                    shutter_living_room_pct=100.0,
                    outdoor_temperature_c=outdoor_temperature_c,
                    household_elec_power_kw=baseload_kw,
                ),
                _reading(
                    valid_at_utc,
                    shutter_living_room_pct=100.0,
                    outdoor_temperature_c=outdoor_temperature_c,
                    household_elec_power_kw=baseload_kw,
                ),
            ]
        )
        aggregate.update(
            {
                "electricity_price_mean_eur_per_kwh": 0.25,
                "electricity_price_last_eur_per_kwh": 0.25,
                "feed_in_price_eur_per_kwh": 0.0,
            }
        )
        repository.add_aggregate(aggregate)
        repository.bulk_add_forecast_snapshots(
            [
                {
                    "fetched_at_utc": valid_at_utc,
                    "valid_at_utc": valid_at_utc,
                    "step_k": 0,
                    "dt_hours": 1.0,
                    "t_out_c": outdoor_temperature_c,
                    "gti_w_per_m2": gti_w_per_m2,
                    "gti_pv_w_per_m2": 0.0,
                }
            ]
        )


def _synthetic_dhw_tap_temperatures(valid_at_utc: datetime) -> tuple[float, float]:
    """Return a recurring DHW temperature profile with evening draw around 22:00–23:00 UTC."""
    hour = valid_at_utc.hour
    if hour == 22:
        return 48.0, 41.0
    if hour == 23:
        return 44.0, 38.0
    if hour == 0:
        return 43.5, 38.0
    return 54.0, 46.0


def _populate_dhw_tap_history(repository: TelemetryRepository, *, start_utc: datetime, hours: int) -> None:
    """Populate telemetry history with a recurring evening DHW draw signature."""
    for step in range(hours):
        valid_at_utc = start_utc + timedelta(hours=step)
        dhw_top_temperature_c, dhw_bottom_temperature_c = _synthetic_dhw_tap_temperatures(valid_at_utc)
        aggregate = aggregate_readings(
            [
                _reading(
                    valid_at_utc - timedelta(minutes=5),
                    shutter_living_room_pct=100.0,
                    outdoor_temperature_c=8.0,
                    household_elec_power_kw=0.4,
                    hp_mode="off",
                    hp_supply_temperature_c=20.0,
                    hp_supply_target_temperature_c=20.0,
                    hp_return_temperature_c=20.0,
                    hp_flow_lpm=0.0,
                    hp_electric_power_kw=0.0,
                    dhw_top_temperature_c=dhw_top_temperature_c,
                    dhw_bottom_temperature_c=dhw_bottom_temperature_c,
                ),
                _reading(
                    valid_at_utc,
                    shutter_living_room_pct=100.0,
                    outdoor_temperature_c=8.0,
                    household_elec_power_kw=0.4,
                    hp_mode="off",
                    hp_supply_temperature_c=20.0,
                    hp_supply_target_temperature_c=20.0,
                    hp_return_temperature_c=20.0,
                    hp_flow_lpm=0.0,
                    hp_electric_power_kw=0.0,
                    dhw_top_temperature_c=dhw_top_temperature_c,
                    dhw_bottom_temperature_c=dhw_bottom_temperature_c,
                ),
            ]
        )
        aggregate.update(
            {
                "electricity_price_mean_eur_per_kwh": 0.25,
                "electricity_price_last_eur_per_kwh": 0.25,
                "feed_in_price_eur_per_kwh": 0.0,
            }
        )
        repository.add_aggregate(aggregate)
        repository.bulk_add_forecast_snapshots(
            [
                {
                    "fetched_at_utc": valid_at_utc,
                    "valid_at_utc": valid_at_utc,
                    "step_k": 0,
                    "dt_hours": 1.0,
                    "t_out_c": 8.0,
                    "gti_w_per_m2": 0.0,
                    "gti_pv_w_per_m2": 0.0,
                }
            ]
        )


def _train_persisted_shutter_model(repository: TelemetryRepository) -> object:
    """Train and persist one shutter artifact for runtime inference tests."""
    metadata = ShutterForecaster().train_and_persist_from_repository(repository=repository)
    assert metadata is not None
    return metadata


def _train_persisted_baseload_model(repository: TelemetryRepository) -> object:
    """Train and persist one baseload artifact for runtime inference tests."""
    from home_optimizer.forecasting import BaseloadForecaster

    metadata = BaseloadForecaster().train_and_persist_from_repository(repository=repository)
    assert metadata is not None
    return metadata


def _add_dhw_calibration_snapshot(
    repository: TelemetryRepository,
    *,
    generated_at_utc: datetime,
    c_top_kwh_per_k: float,
    c_bot_kwh_per_k: float,
    r_loss_k_per_kw: float,
) -> object:
    """Persist one minimal calibration snapshot with the DHW parameters needed for tap-profile training."""

    payload = CalibrationSnapshotPayload(
        generated_at_utc=generated_at_utc,
        effective_parameters=CalibrationParameterOverrides(
            dhw_C_top=c_top_kwh_per_k,
            dhw_C_bot=c_bot_kwh_per_k,
            dhw_R_loss=r_loss_k_per_kw,
        ),
    )
    return repository.add_calibration_snapshot(payload)


def _train_persisted_dhw_tap_profile(
    repository: TelemetryRepository,
    *,
    c_top_kwh_per_k: float,
    c_bot_kwh_per_k: float,
    r_loss_k_per_kw: float,
) -> object:
    """Train and persist one recurring DHW tap-profile artifact for inference tests."""

    metadata = DHWTapForecaster().train_and_persist_from_repository(
        repository=repository,
        c_top_kwh_per_k=c_top_kwh_per_k,
        c_bot_kwh_per_k=c_bot_kwh_per_k,
        r_loss_k_per_kw=r_loss_k_per_kw,
        lambda_water_kwh_per_m3_k=LAMBDA_WATER_KWH_PER_M3_K,
    )
    assert metadata is not None
    return metadata


def test_shutter_forecaster_learns_daylight_closure_pattern(tmp_path) -> None:
    """Sunny hours should yield a lower predicted shutter opening than dark hours."""
    database_url = f"sqlite:///{tmp_path / 'shutter-forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    _populate_shutter_history(repository, start_utc=history_start_utc, hours=72)
    future_fetched_at_utc = history_start_utc + timedelta(hours=72)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 180.0, 650.0, 650.0, 180.0, 0.0],
    )
    _train_persisted_shutter_model(repository)

    rows = repository.get_latest_forecast_batch()
    prediction = ShutterForecaster().predict_from_repository(
        repository=repository,
        weather_rows=rows,
        horizon_steps=6,
        initial_shutter_pct=100.0,
    )

    assert prediction is not None
    assert prediction.shape == (6,)
    assert np.all(prediction >= 0.0)
    assert np.all(prediction <= 100.0)
    assert prediction[2] < prediction[0]
    assert prediction[3] < prediction[5]
    assert float(np.mean(prediction[2:4])) < float(np.mean(prediction[[0, 5]]))


def test_forecast_service_returns_no_shutter_override_without_enough_history(tmp_path) -> None:
    """The ML service must fail safe and keep scalar fallback behavior on sparse data."""
    database_url = f"sqlite:///{tmp_path / 'shutter-forecast-sparse.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    _populate_shutter_history(repository, start_utc=history_start_utc, hours=8)
    future_fetched_at_utc = history_start_utc + timedelta(hours=8)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 180.0, 650.0, 0.0],
    )

    rows = repository.get_latest_forecast_batch()
    overrides = ForecastService().build_missing_overrides(
        request_data={
            "horizon_hours": 4,
            "shutter_living_room_pct": 100.0,
            "shutter_forecast": None,
        },
        repository=repository,
        weather_rows=rows,
        current_overrides={},
    )

    assert overrides == {}


def test_forecast_service_builds_dhw_tap_forecast_from_history(tmp_path) -> None:
    """The runtime forecast service must learn recurring DHW draw hours from telemetry history."""
    database_url = f"sqlite:///{tmp_path / 'dhw-tap-forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc)
    _populate_dhw_tap_history(repository, start_utc=history_start_utc, hours=72)
    future_fetched_at_utc = history_start_utc + timedelta(hours=92)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 0.0, 0.0, 0.0],
    )

    rows = repository.get_latest_forecast_batch()
    overrides = ForecastService().build_missing_overrides(
        request_data=RunRequest.model_validate(
            {
                "horizon_hours": 4,
                "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
                "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
                "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
                "dhw_enabled": True,
                "dhw_C_top": 0.11628,
                "dhw_C_bot": 0.11628,
                "dhw_R_loss": 462.0,
            }
        ).model_dump(mode="python"),
        repository=repository,
        weather_rows=rows[:4],
    )

    assert "dhw_v_tap_forecast" in overrides
    tap_forecast = np.asarray(overrides["dhw_v_tap_forecast"], dtype=float)
    assert tap_forecast.shape == (4,)
    assert np.all(tap_forecast >= 0.0)
    assert [row.valid_at_utc.hour for row in rows[:4]] == [20, 21, 22, 23]
    assert float(np.max(tap_forecast[2:4])) > float(np.max(tap_forecast[:2]))


def test_dhw_tap_forecaster_reuses_persisted_profile_without_recomputing_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trained DHW tap artifact must be reused at runtime without re-inferring history."""

    database_url = f"sqlite:///{tmp_path / 'dhw-tap-persisted.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc)
    c_top_kwh_per_k = 0.11628
    c_bot_kwh_per_k = 0.11628
    r_loss_k_per_kw = 462.0
    _populate_dhw_tap_history(repository, start_utc=history_start_utc, hours=72)
    future_fetched_at_utc = history_start_utc + timedelta(hours=92)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 0.0, 0.0, 0.0],
    )
    DHWTapForecaster.clear_model_cache()
    _train_persisted_dhw_tap_profile(
        repository,
        c_top_kwh_per_k=c_top_kwh_per_k,
        c_bot_kwh_per_k=c_bot_kwh_per_k,
        r_loss_k_per_kw=r_loss_k_per_kw,
    )

    forecaster = DHWTapForecaster()

    def fail_if_history_is_used(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("predict_from_repository should have reused the persisted DHW profile artifact.")

    monkeypatch.setattr(forecaster, "_predict_from_history", fail_if_history_is_used)
    rows = repository.get_latest_forecast_batch()
    prediction = forecaster.predict_from_repository(
        repository=repository,
        horizon_valid_at_utc=[row.valid_at_utc for row in rows[:4]],
        c_top_kwh_per_k=c_top_kwh_per_k,
        c_bot_kwh_per_k=c_bot_kwh_per_k,
        r_loss_k_per_kw=r_loss_k_per_kw,
        lambda_water_kwh_per_m3_k=LAMBDA_WATER_KWH_PER_M3_K,
    )

    assert prediction is not None
    assert prediction.shape == (4,)
    assert [row.valid_at_utc.hour for row in rows[:4]] == [20, 21, 22, 23]
    assert float(np.max(prediction[2:4])) > float(np.max(prediction[:2]))


def test_forecast_service_trains_persisted_dhw_tap_profile_from_calibration_snapshot(tmp_path) -> None:
    """Nightly forecast training must persist a DHW tap artifact when calibrated DHW physics are available."""

    database_url = f"sqlite:///{tmp_path / 'dhw-tap-training.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc)
    c_top_kwh_per_k = 0.11628
    c_bot_kwh_per_k = 0.11628
    r_loss_k_per_kw = 462.0
    _populate_dhw_tap_history(repository, start_utc=history_start_utc, hours=72)
    _add_dhw_calibration_snapshot(
        repository,
        generated_at_utc=history_start_utc + timedelta(hours=72),
        c_top_kwh_per_k=c_top_kwh_per_k,
        c_bot_kwh_per_k=c_bot_kwh_per_k,
        r_loss_k_per_kw=r_loss_k_per_kw,
    )

    training_results = ForecastService().train_and_persist_models(repository=repository)

    dhw_training_result = training_results["dhw_v_tap_forecast"]
    assert dhw_training_result is not None
    assert int(getattr(dhw_training_result, "sample_count", -1)) >= 24
    assert repository.forecast_artifact_path("dhw_tap_profile").exists()


def test_optimizer_scheduled_input_injects_ml_shutter_forecast(tmp_path) -> None:
    """Scheduled optimizer input must add ML shutter forecasts when none are supplied."""
    database_url = f"sqlite:///{tmp_path / 'scheduled-shutter-forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    _populate_shutter_history(repository, start_utc=history_start_utc, hours=72)
    future_fetched_at_utc = history_start_utc + timedelta(hours=72)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 180.0, 650.0, 650.0, 180.0, 0.0],
    )
    _train_persisted_shutter_model(repository)

    base_input = RunRequest.model_validate({"horizon_hours": 6, "shutter_living_room_pct": 100.0})
    scheduled_input = Optimizer._build_scheduled_input(
        base_input=base_input,
        backend=None,
        repository=repository,
    )

    assert scheduled_input.shutter_forecast is not None
    assert len(scheduled_input.shutter_forecast) == 6
    assert scheduled_input.shutter_forecast[2] < scheduled_input.shutter_forecast[0]
    assert scheduled_input.shutter_forecast[3] < scheduled_input.shutter_forecast[5]


def test_forecast_service_injects_baseload_forecast_only(tmp_path) -> None:
    """The ML service must provide a baseload forecast without precomputing thermal gains."""
    database_url = f"sqlite:///{tmp_path / 'baseload-forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    _populate_baseload_history(repository, start_utc=history_start_utc, hours=72)
    future_fetched_at_utc = history_start_utc + timedelta(hours=72)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 0.0, 120.0, 120.0, 0.0, 0.0],
    )
    _train_persisted_baseload_model(repository)

    rows = repository.get_latest_forecast_batch()
    overrides = ForecastService().build_missing_overrides(
        request_data={
            "horizon_hours": 6,
            "shutter_living_room_pct": 100.0,
            "shutter_forecast": None,
            "baseload_forecast": None,
        },
        repository=repository,
        weather_rows=rows,
        current_overrides={},
    )

    assert "baseload_forecast" in overrides
    baseload_forecast = np.asarray(overrides["baseload_forecast"], dtype=float)
    assert baseload_forecast.shape == (6,)
    assert np.all(baseload_forecast >= 0.0)


def test_optimizer_maps_baseload_forecast_with_derived_reference() -> None:
    """UFH forecast construction must derive the baseload floor from the thermal baseline."""
    from home_optimizer.domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters

    run_request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
            "baseload_forecast": [0.35, 0.45, 0.95, 1.10],
            "internal_gains_kw": 0.20,
            "internal_gains_heat_fraction": 0.50,
            "pv_enabled": False,
        }
    )
    cop_model = HeatPumpCOPModel(
        HeatPumpCOPParameters(
            eta_carnot=run_request.eta_carnot,
            delta_T_cond=run_request.delta_T_cond,
            delta_T_evap=run_request.delta_T_evap,
            T_supply_min=run_request.T_supply_min,
            T_ref_outdoor=run_request.T_ref_outdoor_curve,
            heating_curve_slope=run_request.heating_curve_slope,
            cop_min=run_request.cop_min,
            cop_max=run_request.cop_max,
        )
    )

    forecast = Optimizer._build_ufh_forecast(run_request, start_hour=0, cop_model=cop_model)

    np.testing.assert_allclose(forecast.internal_gains_kw, np.array([0.20, 0.225, 0.475, 0.55]))


def test_optimizer_does_not_add_heat_when_baseload_heat_stays_below_baseline() -> None:
    """Baseload must not create extra Q_int when its useful heat stays below the baseline level."""
    from home_optimizer.domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters

    run_request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
            "baseload_forecast": [0.05, 0.10, 0.20, 0.25],
            "internal_gains_kw": 0.20,
            "internal_gains_heat_fraction": 0.50,
            "pv_enabled": False,
        }
    )
    cop_model = HeatPumpCOPModel(
        HeatPumpCOPParameters(
            eta_carnot=run_request.eta_carnot,
            delta_T_cond=run_request.delta_T_cond,
            delta_T_evap=run_request.delta_T_evap,
            T_supply_min=run_request.T_supply_min,
            T_ref_outdoor=run_request.T_ref_outdoor_curve,
            heating_curve_slope=run_request.heating_curve_slope,
            cop_min=run_request.cop_min,
            cop_max=run_request.cop_max,
        )
    )

    forecast = Optimizer._build_ufh_forecast(run_request, start_hour=0, cop_model=cop_model)

    np.testing.assert_allclose(forecast.internal_gains_kw, np.full(4, 0.20))


def test_optimizer_keeps_baseline_internal_gains_when_heat_fraction_is_zero() -> None:
    """Zero baseload heat fraction must collapse to the scalar baseline without division tricks."""
    from home_optimizer.domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters

    run_request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
            "baseload_forecast": [0.50, 1.00, 1.50, 2.00],
            "internal_gains_kw": 0.20,
            "internal_gains_heat_fraction": 0.0,
            "pv_enabled": False,
        }
    )
    cop_model = HeatPumpCOPModel(
        HeatPumpCOPParameters(
            eta_carnot=run_request.eta_carnot,
            delta_T_cond=run_request.delta_T_cond,
            delta_T_evap=run_request.delta_T_evap,
            T_supply_min=run_request.T_supply_min,
            T_ref_outdoor=run_request.T_ref_outdoor_curve,
            heating_curve_slope=run_request.heating_curve_slope,
            cop_min=run_request.cop_min,
            cop_max=run_request.cop_max,
        )
    )

    forecast = Optimizer._build_ufh_forecast(run_request, start_hour=0, cop_model=cop_model)

    np.testing.assert_allclose(forecast.internal_gains_kw, np.full(4, 0.20))


def test_optimizer_prefers_explicit_internal_gains_forecast_over_baseload_mapping() -> None:
    """An explicit internal-gains forecast must override the derived baseload heat mapping."""
    from home_optimizer.domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters

    run_request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
            "baseload_forecast": [0.35, 0.45, 0.95, 1.10],
            "internal_gains_forecast": [0.10, 0.20, 0.30, 0.40],
            "internal_gains_kw": 0.20,
            "internal_gains_heat_fraction": 0.50,
            "pv_enabled": False,
        }
    )
    cop_model = HeatPumpCOPModel(
        HeatPumpCOPParameters(
            eta_carnot=run_request.eta_carnot,
            delta_T_cond=run_request.delta_T_cond,
            delta_T_evap=run_request.delta_T_evap,
            T_supply_min=run_request.T_supply_min,
            T_ref_outdoor=run_request.T_ref_outdoor_curve,
            heating_curve_slope=run_request.heating_curve_slope,
            cop_min=run_request.cop_min,
            cop_max=run_request.cop_max,
        )
    )

    forecast = Optimizer._build_ufh_forecast(run_request, start_hour=0, cop_model=cop_model)

    np.testing.assert_allclose(forecast.internal_gains_kw, np.array([0.10, 0.20, 0.30, 0.40]))


def test_optimizer_rejects_negative_baseload_forecast_for_internal_gains_mapping() -> None:
    """Negative electrical baseload is physically impossible and must fail fast."""
    from home_optimizer.domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters

    run_request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
            "baseload_forecast": [0.35, -0.10, 0.95, 1.10],
            "internal_gains_kw": 0.20,
            "internal_gains_heat_fraction": 0.50,
            "pv_enabled": False,
        }
    )
    cop_model = HeatPumpCOPModel(
        HeatPumpCOPParameters(
            eta_carnot=run_request.eta_carnot,
            delta_T_cond=run_request.delta_T_cond,
            delta_T_evap=run_request.delta_T_evap,
            T_supply_min=run_request.T_supply_min,
            T_ref_outdoor=run_request.T_ref_outdoor_curve,
            heating_curve_slope=run_request.heating_curve_slope,
            cop_min=run_request.cop_min,
            cop_max=run_request.cop_max,
        )
    )

    with pytest.raises(ValueError, match="baseload_forecast must remain non-negative"):
        Optimizer._build_ufh_forecast(run_request, start_hour=0, cop_model=cop_model)


def test_shutter_forecaster_reuses_cached_model_for_unchanged_history(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated predictions on an unchanged artifact must not hit disk deserialisation twice."""
    database_url = f"sqlite:///{tmp_path / 'cached-shutter-forecast.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    _populate_shutter_history(repository, start_utc=history_start_utc, hours=72)
    future_fetched_at_utc = history_start_utc + timedelta(hours=72)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 180.0, 650.0, 650.0, 180.0, 0.0],
    )
    ShutterForecaster.clear_model_cache()
    _train_persisted_shutter_model(repository)
    ShutterForecaster.clear_model_cache()
    rows = repository.get_latest_forecast_batch()

    load_calls = 0
    original_load = ShutterForecaster._load_artifact_from_disk

    def counting_load(self: ShutterForecaster, artifact_path):  # noqa: ANN001
        nonlocal load_calls
        load_calls += 1
        return original_load(self, artifact_path)

    monkeypatch.setattr(ShutterForecaster, "_load_artifact_from_disk", counting_load)

    first_forecaster = ShutterForecaster()
    second_forecaster = ShutterForecaster()
    first_prediction = first_forecaster.predict_from_repository(
        repository=repository,
        weather_rows=rows,
        horizon_steps=6,
        initial_shutter_pct=100.0,
    )
    second_prediction = second_forecaster.predict_from_repository(
        repository=repository,
        weather_rows=rows,
        horizon_steps=6,
        initial_shutter_pct=100.0,
    )

    assert load_calls == 1
    assert first_prediction is not None and second_prediction is not None
    np.testing.assert_allclose(first_prediction, second_prediction)


def test_shutter_forecaster_refreshes_cached_model_after_retraining(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A same-process nightly retrain must refresh the cached runtime model without extra disk loads."""
    database_url = f"sqlite:///{tmp_path / 'cache-invalidation-shutter.sqlite3'}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()

    history_start_utc = datetime(2026, 4, 14, 0, 0, tzinfo=timezone.utc)
    _populate_shutter_history(repository, start_utc=history_start_utc, hours=72)
    future_fetched_at_utc = history_start_utc + timedelta(hours=72)
    _add_future_forecast_batch(
        repository,
        fetched_at_utc=future_fetched_at_utc,
        gti_profile_w_per_m2=[0.0, 180.0, 650.0, 650.0, 180.0, 0.0],
    )
    ShutterForecaster.clear_model_cache()
    _train_persisted_shutter_model(repository)
    ShutterForecaster.clear_model_cache()
    rows = repository.get_latest_forecast_batch()

    load_calls = 0
    original_load = ShutterForecaster._load_artifact_from_disk

    def counting_load(self: ShutterForecaster, artifact_path):  # noqa: ANN001
        nonlocal load_calls
        load_calls += 1
        return original_load(self, artifact_path)

    monkeypatch.setattr(ShutterForecaster, "_load_artifact_from_disk", counting_load)

    forecaster = ShutterForecaster()
    first_prediction = forecaster.predict_from_repository(
        repository=repository,
        weather_rows=rows,
        horizon_steps=6,
        initial_shutter_pct=100.0,
    )
    first_cached_sample_count = forecaster._load_cached_artifact(repository).sample_count  # noqa: SLF001

    _populate_shutter_history(repository, start_utc=history_start_utc + timedelta(hours=72), hours=1)
    _train_persisted_shutter_model(repository)

    second_prediction = forecaster.predict_from_repository(
        repository=repository,
        weather_rows=rows,
        horizon_steps=6,
        initial_shutter_pct=100.0,
    )
    refreshed_cached_sample_count = forecaster._load_cached_artifact(repository).sample_count  # noqa: SLF001

    assert load_calls == 1
    assert first_prediction is not None and second_prediction is not None
    assert refreshed_cached_sample_count > first_cached_sample_count

