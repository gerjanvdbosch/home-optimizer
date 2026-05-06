from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    COMPRESSOR_FREQUENCY,
    DHW_TOP_TEMPERATURE,
    HP_ELECTRIC_POWER,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    THERMOSTAT_SETPOINT,
    NumericPoint,
    NumericSeries,
    normalize_utc_timestamp,
)
from home_optimizer.domain.time import current_local_timezone
from home_optimizer.features.kpi.service import DailyKpiService


class FakeKpiDataReader:
    def __init__(self, series: list[NumericSeries]) -> None:
        self._series = series

    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return [series for series in self._series if series.name in names]

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
    assert kpis.hp_electric_kwh == 24.0
    assert kpis.total_import_kwh == 36.0
    assert kpis.total_export_kwh == 6.0
    assert kpis.pv_generation_kwh == 12.0
    assert kpis.self_consumption_ratio == 0.5
    assert kpis.electricity_cost_eur == pytest.approx(8.4)
    assert kpis.room_temperature_mae_c == 1.0
    assert kpis.room_comfort_violation_degree_hours == 12.0
    assert kpis.dhw_comfort_violation_minutes == 1440.0
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
            ]
        ),
        build_settings(),
    )

    kpis = service.get_day_kpis(day)

    assert kpis.is_valid_for_control_evaluation is False
    assert "room_temperature_gap_too_large" in kpis.validity_reasons
