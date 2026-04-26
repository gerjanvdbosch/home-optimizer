from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from home_optimizer.web.mappers import series_response, text_series_response
from home_optimizer.web.ports import DashboardDataReader
from home_optimizer.web.schemas import DashboardChartsResponse


class DashboardChartsService:
    def __init__(self, reader: DashboardDataReader) -> None:
        self.reader = reader

    def get_day_charts(self, chart_date: date) -> DashboardChartsResponse:
        start_time = datetime.combine(chart_date, time.min, tzinfo=timezone.utc)
        end_time = start_time + timedelta(days=1)
        series = self.reader.read_series(
            names=[
                "room_temperature",
                "dhw_top_temperature",
                "dhw_bottom_temperature",
                "hp_electric_power",
            ],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = self.reader.read_text_series(
            names=["hp_mode"],
            start_time=start_time,
            end_time=end_time,
        )
        series_by_name = {item.name: item for item in series}
        text_series_by_name = {item.name: item for item in text_series}

        return DashboardChartsResponse(
            date=chart_date.isoformat(),
            room_temperature=series_response(series_by_name["room_temperature"]),
            dhw_temperatures=[
                series_response(series_by_name["dhw_top_temperature"]),
                series_response(series_by_name["dhw_bottom_temperature"]),
            ],
            heatpump_power=series_response(series_by_name["hp_electric_power"]),
            heatpump_mode=text_series_response(text_series_by_name["hp_mode"]),
        )
