from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    DHW_BOTTOM_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_ELECTRIC_POWER,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    OUTDOOR_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    NumericPoint,
    NumericSeries,
    TextPoint,
    TextSeries,
    normalize_utc_timestamp,
)
from home_optimizer.features.identification.service import IdentificationDatasetService


class FakeIdentificationDataReader:
    def __init__(
        self,
        *,
        numeric_series: list[NumericSeries],
        text_series: list[TextSeries],
        forecast_series: list[NumericSeries],
        price_series: NumericSeries,
    ) -> None:
        self.numeric_series = numeric_series
        self.text_series = text_series
        self.forecast_series = forecast_series
        self.price_series = price_series

    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return [series for series in self.numeric_series if series.name in names]

    def read_text_series(self, names, start_time, end_time) -> list[TextSeries]:
        return [series for series in self.text_series if series.name in names]

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return [series for series in self.forecast_series if series.name in names]

    def read_electricity_price_series(
        self,
        start_time,
        end_time,
        *,
        source,
        interval_minutes=15,
    ) -> NumericSeries:
        return self.price_series


def build_numeric_series(
    *,
    name: str,
    unit: str,
    start_time: datetime,
    values: list[float],
    interval_minutes: int = 15,
) -> NumericSeries:
    return NumericSeries(
        name=name,
        unit=unit,
        points=[
            NumericPoint(
                timestamp=normalize_utc_timestamp(
                    start_time + timedelta(minutes=interval_minutes * index)
                ),
                value=value,
            )
            for index, value in enumerate(values)
        ],
    )


def build_text_series(
    *,
    name: str,
    start_time: datetime,
    values: list[str],
    interval_minutes: int = 15,
) -> TextSeries:
    return TextSeries(
        name=name,
        points=[
            TextPoint(
                timestamp=normalize_utc_timestamp(
                    start_time + timedelta(minutes=interval_minutes * index)
                ),
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
                "peak_price": 0.30,
                "off_peak_price": 0.20,
                "feed_in_tariff": 0.08,
            },
            "room_target": [
                {"time": "00:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.0},
                {"time": "07:00", "target": 20.5, "low_margin": 0.5, "high_margin": 1.0},
            ],
        }
    )


def test_identification_dataset_service_builds_fifteen_minute_rows_with_validation_flags() -> None:
    start_time = datetime(2026, 4, 25, 7, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(minutes=45)
    service = IdentificationDatasetService(
        FakeIdentificationDataReader(
            numeric_series=[
                build_numeric_series(
                    name=ROOM_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[20.1, 20.2, 20.3],
                ),
                build_numeric_series(
                    name=OUTDOOR_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[8.0, 8.0, 8.1],
                ),
                build_numeric_series(
                    name=DHW_TOP_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[49.0, 48.0, 48.1],
                ),
                build_numeric_series(
                    name=DHW_BOTTOM_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[45.0, 44.8, 44.9],
                ),
                build_numeric_series(
                    name=HP_ELECTRIC_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[1.5, 1.6, 1.4],
                ),
                build_numeric_series(
                    name=DEFROST_ACTIVE,
                    unit="bool",
                    start_time=start_time,
                    values=[0.0, 1.0, 0.0],
                ),
                build_numeric_series(
                    name=BOOSTER_HEATER_ACTIVE,
                    unit="bool",
                    start_time=start_time,
                    values=[0.0, 0.0, 1.0],
                ),
                build_numeric_series(
                    name=PV_OUTPUT_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[0.2, 0.3, 0.5],
                ),
                build_numeric_series(
                    name=P1_NET_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[1.8, 1.6, 1.2],
                ),
                build_numeric_series(
                    name=SHUTTER_LIVING_ROOM,
                    unit="%",
                    start_time=start_time,
                    values=[50.0, 25.0, 0.0],
                ),
                build_numeric_series(
                    name=THERMOSTAT_SETPOINT,
                    unit="°C",
                    start_time=start_time,
                    values=[20.5, 20.5, 20.5],
                ),
                build_numeric_series(
                    name=HP_SUPPLY_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[31.0, 32.0, 31.5],
                ),
                build_numeric_series(
                    name=HP_RETURN_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[27.0, 28.0, 28.0],
                ),
                build_numeric_series(
                    name=HP_FLOW,
                    unit="L/min",
                    start_time=start_time,
                    values=[10.0, 10.0, 10.0],
                ),
            ],
            text_series=[
                build_text_series(
                    name=HP_MODE,
                    start_time=start_time,
                    values=["space_heating", "dhw", "off"],
                )
            ],
            forecast_series=[
                build_numeric_series(
                    name=GTI_LIVING_ROOM_WINDOWS,
                    unit="W/m2",
                    start_time=start_time,
                    values=[300.0, 400.0, 500.0],
                )
            ],
            price_series=build_numeric_series(
                name="electricity_price",
                unit="EUR/kWh",
                start_time=start_time,
                values=[0.25, 0.25, 0.25],
            ),
        ),
        build_settings(),
    )

    dataset = service.build_dataset(
        start_time=start_time,
        end_time=end_time,
    )

    assert dataset.interval_minutes == 15
    assert len(dataset.rows) == 3

    first_row = dataset.rows[0]
    assert first_row.mode_space == 1
    assert first_row.mode_dhw == 0
    assert first_row.mode_off == 0
    assert first_row.hp_delta_t_c == 4.0
    assert first_row.thermal_output_estimate_kw == 2.7906666666666666
    assert first_row.cop_estimate == 2.7906666666666666 / 1.5
    assert first_row.solar_gain_proxy_w_m2 == 150.0
    assert first_row.price_import_eur_kwh == 0.25
    assert first_row.price_export_eur_kwh == 0.08
    assert first_row.occupied_flag == 1
    assert first_row.dhw_draw_detected == 0
    assert first_row.defrost_active == 0
    assert first_row.booster_heater_active == 0
    assert first_row.is_valid_for_room_identification is True
    assert first_row.is_valid_for_dhw_identification is True
    assert first_row.is_valid_for_cop_identification is True
    assert first_row.exclusion_reasons == []

    second_row = dataset.rows[1]
    assert second_row.mode_space == 0
    assert second_row.mode_dhw == 1
    assert second_row.mode_off == 0
    assert second_row.defrost_active == 1
    assert second_row.dhw_draw_detected == 1
    assert second_row.solar_gain_proxy_w_m2 == 100.0
    assert second_row.is_valid_for_room_identification is False
    assert second_row.is_valid_for_dhw_identification is False
    assert second_row.is_valid_for_cop_identification is False
    assert "defrost_active" in second_row.exclusion_reasons

    third_row = dataset.rows[2]
    assert third_row.mode_space == 0
    assert third_row.mode_dhw == 0
    assert third_row.mode_off == 1
    assert third_row.booster_heater_active == 1
    assert third_row.is_valid_for_room_identification is False
    assert third_row.is_valid_for_dhw_identification is False
    assert third_row.is_valid_for_cop_identification is False
    assert "booster_heater_active" in third_row.exclusion_reasons
