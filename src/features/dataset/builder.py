from datetime import datetime

import pandas as pd

from domain.models.config import (
    Aggregation,
    DatasetSpec,
    DataSpec,
    FillMethod,
    ForecastSpec,
    SensorReference,
    TimeSeriesSpec,
)


class DatasetLoader:
    def __init__(self, loaders: list):
        self.loaders = loaders

    def load(self, dataset: DatasetSpec, start: datetime, end: datetime) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for spec in dataset.specs:
            loader = self._find(spec)

            frame = loader.load(
                spec=spec,
                start=start,
                end=end,
            )

            frames.append(frame)

        return self._merge(frames)

    def _find(self, spec: DataSpec):
        for loader in self.loaders:
            if loader.supports(spec):
                return loader

        raise ValueError(f"No loader found for sensor: {spec.name}")

    def _merge(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()

        result = frames[0]

        for frame in frames[1:]:
            result = result.merge(
                frame,
                on="time",
                how="outer",
            )

        return result.sort_values("time").reset_index(drop=True)


class DatasetBuilder:
    def __init__(self):
        self._specs: list[DataSpec] = []

    def timeseries(
        self,
        name: str,
        sensor: SensorReference,
        aggregation: Aggregation | None = None,
        interval: str = "1min",
        fill: FillMethod = "none",
    ) -> "DatasetBuilder":
        self._specs.append(
            TimeSeriesSpec(
                name=name,
                sensor=sensor,
                aggregation=aggregation,
                interval=interval,
                fill=fill,
            )
        )

        return self

    def forecast(self, name: str, sensor: SensorReference) -> "DatasetBuilder":
        self._specs.append(
            ForecastSpec(
                name=name,
                sensor=sensor,
            )
        )

        return self

    def build(self) -> DatasetSpec:
        return DatasetSpec(specs=self._specs)


# class HomeDatasetBuilder:
#
#     def __init__(
#         self,
#         config: HomeConfig,
#     ):
#         self.config = config
#
#     def training(self):
#
#         return (
#             DatasetBuilder()
#
#             .timeseries(
#                 "pv_production",
#                 self.config.solar.production,
#                 aggregation="mean",
#                 interval="30min",
#             )
#
#             .forecast(
#                 "solar_p10",
#                 self.config.solar.forecast.p10,
#             )
#
#             .forecast(
#                 "solar_p50",
#                 self.config.solar.forecast.p50,
#             )
#
#             .forecast(
#                 "solar_p90",
#                 self.config.solar.forecast.p90,
#             )
#
#             .timeseries(
#                 "heat_pump_mode",
#                 self.config.heat_pump.mode,
#                 aggregation="last",
#                 fill="ffill",
#             )
#
#             .timeseries(
#                 "boiler_top",
#                 self.config.heat_pump.boiler_temperature.top,
#                 aggregation="mean",
#                 fill="interpolate",
#             )
#
#             .timeseries(
#                 "boiler_bottom",
#                 self.config.heat_pump.boiler_temperature.bottom,
#                 aggregation="mean",
#                 fill="interpolate",
#             )
#
#             .build()
#         )
