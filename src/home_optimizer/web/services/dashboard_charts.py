from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from home_optimizer.web.mappers import series_response, text_series_response
from home_optimizer.web.ports import DashboardDataReader
from home_optimizer.web.schemas import DashboardChartsResponse


class DashboardChartsService:
    def __init__(self, reader: DashboardDataReader) -> None:
        self.reader = reader

    def get_day_charts(
        self,
        chart_date: date,
        timezone_name: str = "UTC",
    ) -> DashboardChartsResponse:
        local_timezone = ZoneInfo(timezone_name)
        start_time = datetime.combine(chart_date, time.min, tzinfo=local_timezone)
        end_time = start_time + timedelta(days=1)
        series = self.reader.read_series(
            names=[
                "room_temperature",
                "dhw_top_temperature",
                "dhw_bottom_temperature",
                "hp_electric_power",
                "defrost_active",
                "booster_heater_active",
            ],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = self.reader.read_text_series(
            names=["hp_mode"],
            start_time=start_time,
            end_time=end_time,
        )
        forecast_series = self.reader.read_forecast_series(
            names=["temperature", "gti_pv", "gti_living_room_windows"],
            start_time=start_time,
            end_time=end_time,
        )
        series_by_name = {item.name: item for item in series}
        text_series_by_name = {item.name: item for item in text_series}
        forecast_series_by_name = {item.name: item for item in forecast_series}

        return DashboardChartsResponse(
            date=chart_date.isoformat(),
            room_temperature=series_response(series_by_name["room_temperature"]),
            dhw_temperatures=[
                series_response(series_by_name["dhw_top_temperature"]),
                series_response(series_by_name["dhw_bottom_temperature"]),
            ],
            heatpump_power=series_response(series_by_name["hp_electric_power"]),
            heatpump_mode=text_series_response(text_series_by_name["hp_mode"]),
            heatpump_statuses=[
                series_response(series_by_name["defrost_active"]),
                series_response(series_by_name["booster_heater_active"]),
            ],
            forecast_temperature=series_response(forecast_series_by_name["temperature"]),
            forecast_gti=[
                series_response(forecast_series_by_name["gti_pv"]),
                series_response(forecast_series_by_name["gti_living_room_windows"]),
            ],
        )
