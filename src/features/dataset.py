from datetime import datetime

import pandas as pd


from datetime import datetime

import pandas as pd

from domain.models import DatasetSpec, SensorSpec


class DatasetLoader:
    def __init__(
        self,
        loaders: list,
    ):
        self.loaders = loaders

    def load(
        self,
        dataset: DatasetSpec,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:

        frames: list[pd.DataFrame] = []

        for sensor in dataset.sensors:
            loader = self._find_loader(sensor)

            frame = loader.load(
                sensor=sensor,
                start=start,
                end=end,
            )

            frames.append(frame)

        return self._merge(frames)

    def _find_loader(
        self,
        sensor: SensorSpec,
    ):
        for loader in self.loaders:
            if loader.supports(sensor):
                return loader

        raise ValueError(f"No loader found for sensor: {sensor.name}")

    def _merge(
        self,
        frames: list[pd.DataFrame],
    ) -> pd.DataFrame:

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
        self._requests: list[TimeSeriesSpec | ForecastSpec] = []

    def timeseries(
        self,
        name: str,
        sensor: SensorReference,
        aggregation: Aggregation = "mean",
        interval: str = "15min",
        fill: FillMethod = "none",
    ) -> "DatasetBuilder":

        self._requests.append(
            TimeSeriesSpec(
                name=name,
                sensor=sensor,
                aggregation=aggregation,
                interval=interval,
                fill=fill,
            )
        )

        return self

    def forecast(
        self,
        name: str,
        sensor: SensorReference,
    ) -> "DatasetBuilder":

        self._requests.append(
            ForecastSpec(
                name=name,
                sensor=sensor,
            )
        )

        return self

    def build(self) -> DatasetSpec:

        return DatasetSpec(
            sensors=self._requests,
        )


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


from datetime import datetime

import pandas as pd

from domain.models import (
    TimeSeriesSpec,
    Resample,
)


class TimeSeriesLoader:

    def __init__(
        self,
        influx,
        resolver,
    ):
        self.influx = influx
        self.resolver = resolver

    def supports(
        self,
        request,
    ):

        return isinstance(
            request,
            TimeSeriesSpec,
        )

    def load(
        self,
        request: TimeSeriesSpec,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:

        sensor = self.resolver.resolve(
            request.sensor,
        )

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
            resample=Resample(
                aggregation=request.aggregation,
                interval=request.interval,
            ),
        )

        rows = [
            {
                "time": parse_datetime(point["time"]),
                request.name: point["value"],
            }
            for point in points
            if point["value"] is not None
        ]

        df = pd.DataFrame(rows)

        if request.fill == "ffill":
            df[request.name] = df[request.name].ffill()

        elif request.fill == "interpolate":
            df[request.name] = df[request.name].astype(float).interpolate()

        return df


class ForecastLoader:

    def __init__(
        self,
        influx,
        resolver,
    ):
        self.influx = influx
        self.resolver = resolver

    def supports(
        self,
        request,
    ):

        return isinstance(
            request,
            ForecastSpec,
        )

    def load(
        self,
        request,
        start,
        end,
    ) -> pd.DataFrame:

        sensor = self.resolver.resolve(
            request.sensor,
        )

        points = self.influx.find_series(
            measurement=sensor.measurement,
            entity_id=sensor.entity_id,
            field=sensor.field,
            start=start,
            end=end,
        )

        rows = []

        for point in points:

            values = ast.literal_eval(point["value"])

            for target_time, watts in values.items():

                rows.append(
                    {
                        "time": parse_datetime(target_time),
                        request.name: float(watts),
                    }
                )

        return pd.DataFrame(rows)
