from __future__ import annotations

from home_optimizer.domain.charts import ChartPoint, ChartSeries
from home_optimizer.web.services.dashboard_charts import adjusted_gti_with_shutter


def test_adjusted_gti_with_shutter_uses_latest_known_open_percentage() -> None:
    window_gti = ChartSeries(
        name="gti_living_room_windows",
        unit="Wm2",
        points=[
            ChartPoint(timestamp="2026-04-25T09:00:00+00:00", value=100.0),
            ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=200.0),
            ChartPoint(timestamp="2026-04-25T15:00:00+00:00", value=300.0),
        ],
    )
    shutter_position = ChartSeries(
        name="shutter_living_room",
        unit="percent",
        points=[
            ChartPoint(timestamp="2026-04-25T08:30:00+00:00", value=25.0),
            ChartPoint(timestamp="2026-04-25T13:30:00+00:00", value=80.0),
        ],
    )

    adjusted = adjusted_gti_with_shutter(window_gti, shutter_position)

    assert adjusted.name == "gti_living_room_windows_adjusted"
    assert adjusted.unit == "Wm2"
    assert adjusted.points == [
        ChartPoint(timestamp="2026-04-25T09:00:00+00:00", value=25.0),
        ChartPoint(timestamp="2026-04-25T12:00:00+00:00", value=50.0),
        ChartPoint(timestamp="2026-04-25T15:00:00+00:00", value=240.0),
    ]


def test_adjusted_gti_with_shutter_defaults_to_fully_open_when_no_position_is_known() -> None:
    window_gti = ChartSeries(
        name="gti_living_room_windows",
        unit="Wm2",
        points=[ChartPoint(timestamp="2026-04-25T09:00:00+00:00", value=150.0)],
    )
    shutter_position = ChartSeries(
        name="shutter_living_room",
        unit="percent",
        points=[],
    )

    adjusted = adjusted_gti_with_shutter(window_gti, shutter_position)

    assert adjusted.points == [ChartPoint(timestamp="2026-04-25T09:00:00+00:00", value=150.0)]

