from __future__ import annotations

from datetime import timedelta

from home_optimizer.features.system_identification.building_model import (
    RoomTemperatureModelIdentifier,
    RoomTemperatureModelInputs,
)
from home_optimizer.features.system_identification.dataset import SeriesLookup, timed_values
from home_optimizer.features.system_identification.schemas import (
    NumericPoint,
    NumericSeries,
    RoomTemperatureModelResult,
    TextSeries,
)


class SystemIdentificationError(ValueError):
    pass


class SystemIdentificationService:
    def __init__(
        self,
        *,
        sample_interval_minutes: int = 15,
        train_fraction: float = 0.7,
    ) -> None:
        self.sample_interval_minutes = sample_interval_minutes
        self.train_fraction = train_fraction

    def identify_room_temperature_model(
        self,
        *,
        numeric_series: list[NumericSeries],
        text_series: list[TextSeries] | None = None,
    ) -> RoomTemperatureModelResult:
        numeric_by_name = {series.name: series for series in numeric_series}
        text_by_name = {series.name: series for series in text_series or []}

        try:
            inputs = RoomTemperatureModelInputs(
                room_temperature=numeric_by_name["room_temperature"],
                outdoor_temperature=numeric_by_name["outdoor_temperature"],
                thermal_output=self._thermal_output_series(numeric_by_name),
                solar_gain=self._optional_series(
                    numeric_by_name,
                    "gti_living_room_windows_adjusted",
                    "solar_gain",
                ),
                defrost_active=numeric_by_name.get("defrost_active"),
                booster_heater_active=numeric_by_name.get("booster_heater_active"),
                hp_mode=text_by_name.get("hp_mode"),
            )
            return RoomTemperatureModelIdentifier(
                sample_interval_minutes=self.sample_interval_minutes,
                train_fraction=self.train_fraction,
            ).identify(inputs)
        except KeyError as error:
            raise SystemIdentificationError(f"missing required series: {error.args[0]}") from error
        except ValueError as error:
            raise SystemIdentificationError(str(error)) from error

    def _thermal_output_series(self, series_by_name: dict[str, NumericSeries]) -> NumericSeries:
        if "thermal_output" in series_by_name:
            return series_by_name["thermal_output"]

        required_names = [
            "hp_flow",
            "hp_supply_temperature",
            "hp_return_temperature",
        ]
        if all(name in series_by_name for name in required_names):
            return self._build_thermal_output_series(
                flow=series_by_name["hp_flow"],
                supply=series_by_name["hp_supply_temperature"],
                return_temperature=series_by_name["hp_return_temperature"],
            )

        raise KeyError("thermal_output")

    def _build_thermal_output_series(
        self,
        *,
        flow: NumericSeries,
        supply: NumericSeries,
        return_temperature: NumericSeries,
    ) -> NumericSeries:
        supply_lookup = SeriesLookup(timed_values(supply))
        return_lookup = SeriesLookup(timed_values(return_temperature))
        max_age = timedelta(minutes=20)
        points: list[NumericPoint] = []

        for flow_point in timed_values(flow):
            supply_value = supply_lookup.latest_at(flow_point.timestamp, max_age)
            return_value = return_lookup.latest_at(flow_point.timestamp, max_age)
            if supply_value is None or return_value is None:
                continue

            delta_t = supply_value - return_value
            thermal_kw = max(0.0, flow_point.value * delta_t * 4186.0 / 60000.0)
            points.append(
                NumericPoint(
                    timestamp=flow_point.timestamp.isoformat(),
                    value=thermal_kw,
                )
            )

        return NumericSeries(
            name="thermal_output",
            unit="kW",
            points=points,
        )

    def _optional_series(
        self,
        series_by_name: dict[str, NumericSeries],
        *names: str,
    ) -> NumericSeries | None:
        for name in names:
            if name in series_by_name:
                return series_by_name[name]
        return None
