"""Tests for the ML-based runtime forecasting layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from home_optimizer.forecasting import ForecastService, ShutterForecaster
from home_optimizer.optimizer import Optimizer, RunRequest
from home_optimizer.sensors import LiveReadings
from home_optimizer.telemetry import TelemetryRepository, aggregate_readings


def _reading(
    timestamp: datetime,
    *,
    shutter_living_room_pct: float,
    outdoor_temperature_c: float,
    household_elec_power_kw: float = 0.0,
) -> LiveReadings:
    """Create one fully populated telemetry sample for forecasting tests."""
    hp_electric_power_kw = 2.0
    pv_output_kw = 0.6
    return LiveReadings(
        room_temperature_c=20.5,
        outdoor_temperature_c=outdoor_temperature_c,
        hp_supply_temperature_c=31.0,
        hp_supply_target_temperature_c=33.0,
        hp_return_temperature_c=27.0,
        hp_flow_lpm=9.0,
        hp_electric_power_kw=hp_electric_power_kw,
        hp_mode="ufh",
        p1_net_power_kw=household_elec_power_kw + hp_electric_power_kw - pv_output_kw,
        pv_output_kw=pv_output_kw,
        thermostat_setpoint_c=20.5,
        dhw_top_temperature_c=52.0,
        dhw_bottom_temperature_c=45.0,
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


def test_forecast_service_injects_baseload_and_internal_gains_proxy(tmp_path) -> None:
    """The ML service must provide a baseload forecast and reuse it as internal-gains proxy."""
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
            "internal_gains_forecast": None,
        },
        repository=repository,
        weather_rows=rows,
        current_overrides={},
    )

    assert "baseload_forecast" in overrides
    assert "internal_gains_forecast" in overrides
    baseload_forecast = np.asarray(overrides["baseload_forecast"], dtype=float)
    internal_gains_forecast = np.asarray(overrides["internal_gains_forecast"], dtype=float)
    assert baseload_forecast.shape == (6,)
    np.testing.assert_allclose(baseload_forecast, internal_gains_forecast)
    assert np.all(baseload_forecast >= 0.0)


def test_optimizer_uses_baseload_forecast_as_internal_gains_proxy() -> None:
    """UFH forecast construction must use baseload_forecast when no explicit internal-gains forecast exists."""
    from home_optimizer.cop_model import HeatPumpCOPModel, HeatPumpCOPParameters

    run_request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "gti_pv_forecast": [0.0, 0.0, 0.0, 0.0],
            "baseload_forecast": [0.35, 0.45, 0.95, 1.10],
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

    np.testing.assert_allclose(forecast.internal_gains_kw, np.array([0.35, 0.45, 0.95, 1.10]))


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

