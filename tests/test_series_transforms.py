from __future__ import annotations

from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    NumericPoint,
    NumericSeries,
    TextPoint,
    TextSeries,
    build_space_heating_thermal_output_series,
)


def test_build_space_heating_thermal_output_series_filters_dhw_defrost_and_booster() -> None:
    thermal_output = build_space_heating_thermal_output_series(
        NumericSeries(
            name=HP_FLOW,
            unit="Lmin",
            points=[
                NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=10.0),
                NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=10.0),
                NumericPoint(timestamp="2026-04-25T00:30:00+00:00", value=10.0),
                NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=10.0),
            ],
        ),
        NumericSeries(
            name=HP_SUPPLY_TEMPERATURE,
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=35.0),
                NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=35.0),
                NumericPoint(timestamp="2026-04-25T00:30:00+00:00", value=35.0),
                NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=35.0),
            ],
        ),
        NumericSeries(
            name=HP_RETURN_TEMPERATURE,
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=30.0),
                NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=30.0),
                NumericPoint(timestamp="2026-04-25T00:30:00+00:00", value=30.0),
                NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=30.0),
            ],
        ),
        defrost_active=NumericSeries(
            name=DEFROST_ACTIVE,
            unit="bool",
            points=[NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=1.0)],
        ),
        booster_heater_active=NumericSeries(
            name=BOOSTER_HEATER_ACTIVE,
            unit="bool",
            points=[NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=1.0)],
        ),
        hp_mode=TextSeries(
            name=HP_MODE,
            points=[
                TextPoint(timestamp="2026-04-25T00:00:00+00:00", value="heat"),
                TextPoint(timestamp="2026-04-25T00:30:00+00:00", value="dhw"),
            ],
        ),
    )

    assert [point.timestamp for point in thermal_output.points] == [
        "2026-04-25T00:00:00+00:00",
    ]
