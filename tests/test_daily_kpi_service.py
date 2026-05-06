from __future__ import annotations

from datetime import date, datetime, time, timedelta

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
    midday = start_time + timedelta(hours=12)
    evening = start_time + timedelta(hours=18)

    service = DailyKpiService(
        FakeKpiDataReader(
            [
                NumericSeries(
                    name=HP_ELECTRIC_POWER,
                    unit="kW",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=2.0),
                        NumericPoint(timestamp=normalize_utc_timestamp(midday), value=0.0),
                    ],
                ),
                NumericSeries(
                    name=P1_NET_POWER,
                    unit="kW",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=3.0),
                        NumericPoint(timestamp=normalize_utc_timestamp(midday), value=-0.5),
                    ],
                ),
                NumericSeries(
                    name=PV_OUTPUT_POWER,
                    unit="kW",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=0.0),
                        NumericPoint(timestamp=normalize_utc_timestamp(midday), value=2.0),
                        NumericPoint(timestamp=normalize_utc_timestamp(evening), value=0.0),
                    ],
                ),
                NumericSeries(
                    name=COMPRESSOR_FREQUENCY,
                    unit="Hz",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=0.0),
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(start_time + timedelta(hours=2)),
                            value=35.0,
                        ),
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(start_time + timedelta(hours=6)),
                            value=0.0,
                        ),
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(start_time + timedelta(hours=10)),
                            value=42.0,
                        ),
                    ],
                ),
                NumericSeries(
                    name=ROOM_TEMPERATURE,
                    unit="°C",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=18.0),
                    ],
                ),
                NumericSeries(
                    name=DHW_TOP_TEMPERATURE,
                    unit="°C",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=45.0),
                    ],
                ),
                NumericSeries(
                    name=THERMOSTAT_SETPOINT,
                    unit="°C",
                    points=[
                        NumericPoint(timestamp=normalize_utc_timestamp(start_time), value=19.0),
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(start_time + timedelta(hours=8)),
                            value=20.0,
                        ),
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(start_time + timedelta(hours=16)),
                            value=20.0,
                        ),
                        NumericPoint(
                            timestamp=normalize_utc_timestamp(start_time + timedelta(hours=20)),
                            value=18.0,
                        ),
                    ],
                ),
            ]
        ),
        build_settings(),
    )

    kpis = service.get_day_kpis(day)

    assert kpis.hp_electric_kwh == 24.0
    assert kpis.total_import_kwh == 36.0
    assert kpis.total_export_kwh == 6.0
    assert kpis.pv_generation_kwh == 12.0
    assert kpis.self_consumption_ratio == 0.5
    assert kpis.electricity_cost_eur == 8.4
    assert kpis.room_temperature_mae_c == 1.0
    assert kpis.room_comfort_violation_degree_hours == 12.0
    assert kpis.dhw_comfort_violation_minutes == 1440.0
    assert kpis.thermostat_setpoint_changes == 2
    assert kpis.compressor_starts == 2
