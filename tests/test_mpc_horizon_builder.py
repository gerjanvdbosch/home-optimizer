from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.features.mpc import (
    DEFAULT_OUTDOOR_TEMPERATURE_FORECAST_NAME,
    DEFAULT_SOLAR_GAIN_FORECAST_NAME,
    MpcHorizonBuilder,
    MpcHorizonBuildRequest,
)


def test_mpc_horizon_builder_uses_latest_forecast_targets_and_prices() -> None:
    start_time = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    created_old = start_time - timedelta(hours=2)
    created_new = start_time - timedelta(hours=1)
    forecast_entries = [
        ForecastEntry(
            created_at_utc=created_old,
            forecast_time_utc=start_time,
            name=DEFAULT_OUTDOOR_TEMPERATURE_FORECAST_NAME,
            value=3.0,
            unit="C",
            source="test",
        ),
        ForecastEntry(
            created_at_utc=created_new,
            forecast_time_utc=start_time,
            name=DEFAULT_OUTDOOR_TEMPERATURE_FORECAST_NAME,
            value=4.0,
            unit="C",
            source="test",
        ),
        ForecastEntry(
            created_at_utc=created_new,
            forecast_time_utc=start_time,
            name=DEFAULT_SOLAR_GAIN_FORECAST_NAME,
            value=200.0,
            unit="W/m2",
            source="test",
        ),
    ]
    price_intervals = [
        PriceInterval(
            start_time_utc=start_time,
            end_time_utc=start_time + timedelta(minutes=20),
            source="test",
            value=0.30,
        )
    ]
    target_schedule = [
        TemperatureTargetWindow(time=time(0, 0), target=19.0, low_margin=0.5, high_margin=1.0),
        TemperatureTargetWindow(time=time(7, 0), target=20.0, low_margin=0.5, high_margin=1.0),
    ]

    horizon = MpcHorizonBuilder().build(
        MpcHorizonBuildRequest(
            start_time_utc=start_time,
            horizon_steps=3,
            interval_minutes=10,
            target_schedule=target_schedule,
            forecast_entries=forecast_entries,
            price_intervals=price_intervals,
            default_effective_heating_kw=2.5,
            solar_gain_input_scale=0.01,
            default_occupied=1.0,
        )
    )

    assert len(horizon) == 3
    assert horizon[0].outdoor_temp_c == 4.0
    assert horizon[0].solar_gain_kw == 2.0
    assert horizon[0].price_eur_kwh == 0.30
    assert horizon[0].temp_min_c == 19.5
    assert horizon[0].temp_max_c == 21.0
    assert horizon[1].price_eur_kwh == 0.30
    assert horizon[2].price_eur_kwh == 0.0
    assert horizon[0].effective_heating_kw_forecast == 2.5
    assert horizon[0].hp_electric_power_forecast_kw == 2.5
    assert horizon[0].pv_available_power_forecast_kw == 0.0
    assert horizon[0].base_load_power_forecast_kw == 0.0
    assert horizon[0].import_price_eur_kwh == 0.30
    assert horizon[0].export_price_eur_kwh == 0.0
    assert horizon[0].occupied == 1.0
