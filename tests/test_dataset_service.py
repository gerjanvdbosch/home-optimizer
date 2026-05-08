from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    DHW_TOP_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_ELECTRIC_POWER,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    OUTDOOR_TEMPERATURE,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    NumericPoint,
    NumericSeries,
    TextPoint,
    TextSeries,
    normalize_utc_timestamp,
)
from home_optimizer.features.dataset import MpcDatasetService


class FakeDatasetDataReader:
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


def test_mpc_dataset_service_builds_generic_dataset_rows() -> None:
    start_time = datetime(2026, 4, 25, 7, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(minutes=30)
    service = MpcDatasetService(
        FakeDatasetDataReader(
            numeric_series=[
                build_numeric_series(
                    name=ROOM_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[20.1, 20.2],
                ),
                build_numeric_series(
                    name=OUTDOOR_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[8.0, 8.1],
                ),
                build_numeric_series(
                    name=DHW_TOP_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[49.0, 48.8],
                ),
                build_numeric_series(
                    name=HP_ELECTRIC_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[1.5, 1.6],
                ),
                build_numeric_series(
                    name=PV_OUTPUT_POWER,
                    unit="kW",
                    start_time=start_time,
                    values=[0.2, 0.3],
                ),
                build_numeric_series(
                    name=SHUTTER_LIVING_ROOM,
                    unit="%",
                    start_time=start_time,
                    values=[50.0, 0.0],
                ),
                build_numeric_series(
                    name=HP_SUPPLY_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[31.0, 32.0],
                ),
                build_numeric_series(
                    name=HP_RETURN_TEMPERATURE,
                    unit="°C",
                    start_time=start_time,
                    values=[27.0, 28.0],
                ),
                build_numeric_series(
                    name=HP_FLOW,
                    unit="L/min",
                    start_time=start_time,
                    values=[10.0, 10.0],
                ),
            ],
            text_series=[
                build_text_series(
                    name=HP_MODE,
                    start_time=start_time,
                    values=["space_heating", "off"],
                )
            ],
            forecast_series=[
                build_numeric_series(
                    name=GTI_LIVING_ROOM_WINDOWS,
                    unit="W/m2",
                    start_time=start_time,
                    values=[300.0, 400.0],
                )
            ],
            price_series=build_numeric_series(
                name="electricity_price",
                unit="EUR/kWh",
                start_time=start_time,
                values=[0.25, 0.25],
            ),
        ),
        build_settings(),
    )

    dataset = service.build_dataset(
        start_time=start_time,
        end_time=end_time,
    )
    summary = service.summarize_dataset(dataset)

    assert dataset.interval_minutes == 15
    assert len(dataset.rows) == 2
    assert dataset.rows[0].mode_space == 1
    assert dataset.rows[0].thermal_output_estimate_kw == 2.7906666666666666
    assert dataset.rows[0].solar_gain_proxy_w_m2 == 150.0
    assert dataset.rows[1].mode_off == 1
    assert dataset.rows[1].solar_gain_proxy_w_m2 == 0.0
    assert summary.total_rows == 2
    assert summary.mode_space_rows == 1
    assert summary.mode_off_rows == 1
