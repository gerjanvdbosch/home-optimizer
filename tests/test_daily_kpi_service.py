from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    COMPRESSOR_FREQUENCY,
    DHW_TOP_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_ELECTRIC_POWER,
    OUTDOOR_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    NumericPoint,
    NumericSeries,
    normalize_utc_timestamp,
)
from home_optimizer.domain.time import current_local_timezone, parse_datetime
from home_optimizer.features.kpi.service import DailyKpiService


class FakeKpiDataReader:
    def __init__(self, series: list[NumericSeries]) -> None:
        self._series = series

    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return [
            shift_series_to_day_boundary(series, start_time)
            for series in self._series
            if series.name in names
        ]

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return [
            shift_series_to_day_boundary(series, start_time)
            for series in self._series
            if series.name in names
        ]

    def read_electricity_price_series(
        self,
        start_time,
        end_time,
        *,
        source,
        interval_minutes=15,
    ) -> NumericSeries:
        return NumericSeries(name="electricity_price", unit="EUR/kWh", points=[])


def build_half_hourly_series(
    *,
    name: str,
    unit: str,
    start_time: datetime,
    values: list[float],
) -> NumericSeries:
    return NumericSeries(
        name=name,
        unit=unit,
        points=[
            NumericPoint(
                timestamp=normalize_utc_timestamp(start_time + timedelta(minutes=30 * index)),
                value=value,
            )
            for index, value in enumerate(values)
        ],
    )


def shift_series_to_day_boundary(series: NumericSeries, start_time: datetime) -> NumericSeries:
    if not series.points:
        return series

    delta = start_time - parse_datetime(series.points[0].timestamp)
    return NumericSeries(
        name=series.name,
        unit=series.unit,
        points=[
            NumericPoint(
                timestamp=normalize_utc_timestamp(parse_datetime(point.timestamp) + delta),
                value=point.value,
            )
            for point in series.points
        ],
    )


def build_settings() -> AppSettings:
    return AppSettings.from_options(
        {
            "database_path": "/tmp/home-optimizer-test.db",
            "electricity_pricing": {
                "mode": "fixed",
                "peak_price": 0.25,
                "off_peak_price": 0.25,
                "feed_in_tariff": 0.10,
            },
            "room_target": [
                {"time": "00:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
            ],
            "dhw_target": [
                {"time": "00:00", "target": 50.0, "low_margin": 2.0, "high_margin": 5.0},
            ],
        }
    )


def test_daily_kpi_service_computes_baseline_metrics() -> None:
    local_timezone = current_local_timezone()
    day = date(2026, 4, 25)
    start_time = datetime.combine(day, time.min, tzinfo=local_timezone)
    hp_values = [2.0] * 24 + [0.0] * 24
    net_values = [3.0] * 24 + [-0.5] * 24
    pv_values = [0.0] * 24 + [2.0] * 12 + [0.0] * 12
    compressor_values = [0.0] * 4 + [35.0] * 8 + [0.0] * 8 + [42.0] * 28
    room_values = [18.0] * 48
    outdoor_values = [8.0] * 24 + [10.0] * 24
    solar_values = [100.0] * 24 + [300.0] * 24
    shutter_values = [50.0] * 48
    dhw_values = [45.0] * 48
    setpoint_values = [19.0] * 16 + [20.0] * 24 + [18.0] * 8

    service = DailyKpiService(
        FakeKpiDataReader(
            [
                build_half_hourly_series(
                    name=HP_ELECTRIC_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=hp_values,
                ),
                build_half_hourly_series(
                    name=P1_NET_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=net_values,
                ),
                build_half_hourly_series(
                    name=PV_OUTPUT_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=pv_values,
                ),
                build_half_hourly_series(
                    name=COMPRESSOR_FREQUENCY,
                    unit="Hz",
                    start_time=start_time,
                    values=compressor_values,
                ),
                build_half_hourly_series(
                    name=ROOM_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=room_values,
                ),
                build_half_hourly_series(
                    name=OUTDOOR_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=outdoor_values,
                ),
                build_half_hourly_series(
                    name=GTI_LIVING_ROOM_WINDOWS,
                    unit="W/m2",
                    start_time=start_time,
                    values=solar_values,
                ),
                build_half_hourly_series(
                    name=SHUTTER_LIVING_ROOM,
                    unit="percent",
                    start_time=start_time,
                    values=shutter_values,
                ),
                build_half_hourly_series(
                    name=DHW_TOP_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=dhw_values,
                ),
                build_half_hourly_series(
                    name=THERMOSTAT_SETPOINT,
                    unit="°C",
                    start_time=start_time,
                    values=setpoint_values,
                ),
            ]
        ),
        build_settings(),
    )

    kpis = service.get_day_kpis(day)

    assert kpis.is_valid_for_control_evaluation is True
    assert kpis.validity_reasons == []
    assert kpis.data_coverage_pct == 100.0
    assert kpis.largest_data_gap_minutes == 30.0
    assert kpis.hp_electric_kwh == 24.0
    assert kpis.total_import_kwh == 36.0
    assert kpis.total_export_kwh == 6.0
    assert kpis.pv_generation_kwh == 12.0
    assert kpis.solar_irradiance_mean_w_m2 == 200.0
    assert kpis.shutter_open_pct_mean == 50.0
    assert kpis.outdoor_temperature_mean_c == 9.0
    assert kpis.self_consumption_ratio == 0.5
    assert kpis.electricity_cost_eur == pytest.approx(8.4)
    assert kpis.room_temperature_mae_c == 1.0
    assert kpis.room_comfort_undershoot_degree_hours == 12.0
    assert kpis.comfort_overshoot_while_heating_degree_hours == 0.0
    assert kpis.comfort_overshoot_passive_degree_hours == 0.0
    assert kpis.dhw_comfort_undershoot_minutes == 1440.0
    assert kpis.thermostat_setpoint_changes == 2
    assert kpis.compressor_starts == 2


def test_daily_kpi_service_marks_day_invalid_when_required_series_are_missing() -> None:
    service = DailyKpiService(
        FakeKpiDataReader([]),
        build_settings(),
    )

    kpis = service.get_day_kpis(date(2026, 4, 25))

    assert kpis.is_valid_for_control_evaluation is False
    assert kpis.validity_reasons == [
        "missing_room_temperature",
        "missing_thermostat_setpoint",
        "missing_compressor_frequency",
        "missing_heatpump_electricity",
        "missing_grid_energy",
        "missing_pv_generation",
        "missing_outdoor_temperature",
        "missing_dhw_temperature",
    ]


def test_daily_kpi_service_marks_day_invalid_when_gap_exceeds_thirty_minutes() -> None:
    local_timezone = current_local_timezone()
    day = date(2026, 4, 25)
    start_time = datetime.combine(day, time.min, tzinfo=local_timezone)

    service = DailyKpiService(
        FakeKpiDataReader(
            [
                NumericSeries(
                    name=ROOM_TEMPERATURE,
                    unit="°C",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=19.0),
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(start_time + timedelta(hours=1)),
                            value=19.0,
                        ),
                    ],
                ),
                build_half_hourly_series(
                    name=THERMOSTAT_SETPOINT,
                    unit="°C",
                    start_time=start_time,
                    values=[19.0] * 48,
                ),
                build_half_hourly_series(
                    name=COMPRESSOR_FREQUENCY,
                    unit="Hz",
                    start_time=start_time,
                    values=[0.0] * 48,
                ),
                build_half_hourly_series(
                    name=HP_ELECTRIC_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[1.0] * 48,
                ),
                build_half_hourly_series(
                    name=P1_NET_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[1.0] * 48,
                ),
                build_half_hourly_series(
                    name=PV_OUTPUT_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[0.0] * 48,
                ),
                build_half_hourly_series(
                    name=DHW_TOP_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[45.0] * 48,
                ),
                build_half_hourly_series(
                    name=OUTDOOR_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[10.0] * 48,
                ),
                build_half_hourly_series(
                    name=GTI_LIVING_ROOM_WINDOWS,
                    unit="W/m2",
                    start_time=start_time,
                    values=[200.0] * 48,
                ),
                build_half_hourly_series(
                    name=SHUTTER_LIVING_ROOM,
                    unit="percent",
                    start_time=start_time,
                    values=[60.0] * 48,
                ),
            ]
        ),
        build_settings(),
    )

    kpis = service.get_day_kpis(day)

    assert kpis.is_valid_for_control_evaluation is False
    assert "room_temperature_gap_too_large" in kpis.validity_reasons


def test_daily_kpi_service_builds_baseline_summary_from_valid_days() -> None:
    local_timezone = current_local_timezone()
    start_time = datetime.combine(date(2026, 4, 25), time.min, tzinfo=local_timezone)
    valid_values = [19.0] * 48

    service = DailyKpiService(
        FakeKpiDataReader(
            [
                build_half_hourly_series(
                    name=ROOM_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=valid_values,
                ),
                build_half_hourly_series(
                    name=THERMOSTAT_SETPOINT,
                    unit="°C",
                    start_time=start_time,
                    values=[19.0] * 48,
                ),
                build_half_hourly_series(
                    name=COMPRESSOR_FREQUENCY,
                    unit="Hz",
                    start_time=start_time,
                    values=[0.0] * 48,
                ),
                build_half_hourly_series(
                    name=HP_ELECTRIC_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[2.0] * 48,
                ),
                build_half_hourly_series(
                    name=P1_NET_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[3.0] * 48,
                ),
                build_half_hourly_series(
                    name=PV_OUTPUT_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[1.0] * 48,
                ),
                build_half_hourly_series(
                    name=OUTDOOR_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[10.0] * 48,
                ),
                build_half_hourly_series(
                    name=GTI_LIVING_ROOM_WINDOWS,
                    unit="W/m2",
                    start_time=start_time,
                    values=[200.0] * 48,
                ),
                build_half_hourly_series(
                    name=SHUTTER_LIVING_ROOM,
                    unit="percent",
                    start_time=start_time,
                    values=[60.0] * 48,
                ),
                build_half_hourly_series(
                    name=DHW_TOP_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[45.0] * 48,
                ),
            ]
        ),
        build_settings(),
    )

    summary = service.get_baseline_summary(date(2026, 4, 25), date(2026, 4, 27))

    assert summary.number_of_days == 3
    assert summary.number_of_valid_days == 3
    assert summary.mean_hp_electric_kwh_per_day == 48.0
    assert summary.mean_electricity_cost_eur_per_day == pytest.approx(18.0)
    assert summary.mean_room_temperature_mae_c == 0.0
    assert summary.mean_solar_irradiance_w_m2 == 200.0
    assert summary.mean_shutter_open_pct == 60.0
    assert summary.total_comfort_undershoot_degree_hours == 0.0
    assert summary.total_comfort_overshoot_while_heating_degree_hours == 0.0
    assert summary.total_comfort_overshoot_passive_degree_hours == 0.0
    assert summary.total_dhw_undershoot_minutes == 4320.0
    assert summary.mean_compressor_starts_per_day == 0.0
    assert summary.mean_self_consumption_ratio == 1.0
