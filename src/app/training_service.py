import ast
from datetime import datetime, timedelta, timezone

import pandas as pd

from domain.models import Resample, TrainRequest
from domain.time import parse_datetime
from features.generator import SolarForecastFeatureGenerator
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver


class TrainingService:
    def __init__(
        self,
        influx: InfluxDatabase,
        resolver: InfluxSensorResolver,
        generator: SolarForecastFeatureGenerator,
    ):
        self.influx = influx
        self.resolver = resolver
        self.generator = generator

    def train(
        self,
        request: TrainRequest,
    ) -> None:
        end = datetime.now(timezone.utc)
        start = end - timedelta(
            days=request.days,
        )

        df = self._load(
            request=request,
            start=start,
            end=end,
        )

        df = self.generator.transform(df)

        print(df)

        # fit(df)

    def _load(
        self,
        request: TrainRequest,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        forecast = self._load_solar_forecast(
            request=request,
            start=start,
            end=end,
        )

        production = self._load_pv_production(
            request=request,
            start=start,
            end=end,
        )

        return (
            forecast.merge(
                production,
                on="target_time",
                how="inner",
            )
            .sort_values(
                ["target_time", "issue_time"],
            )
            .reset_index(drop=True)
        )

    def _load_solar_forecast(
        self,
        request: TrainRequest,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        rows = []

        for name, sensor in request.solar_forecast.items():
            influx_sensor = self.resolver.resolve(sensor)

            points = self.influx.find_series(
                measurement=influx_sensor.measurement,
                entity_id=influx_sensor.entity_id,
                field=influx_sensor.field,
                start=start,
                end=end,
            )

            for point in points:
                issue_time = parse_datetime(point["time"])

                value = ast.literal_eval(point["value"])

                for target_time, watts in value.items():
                    rows.append(
                        {
                            "issue_time": issue_time,
                            "target_time": parse_datetime(target_time),
                            "forecast": name,
                            "watts": float(watts),
                        }
                    )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        return (
            df.pivot_table(
                index=[
                    "issue_time",
                    "target_time",
                ],
                columns="forecast",
                values="watts",
                aggfunc="first",
            )
            .reset_index()
            .sort_values(
                [
                    "target_time",
                    "issue_time",
                ]
            )
            .reset_index(drop=True)
        )

    def _load_pv_production(
        self,
        request: TrainRequest,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:

        influx_sensor = self.resolver.resolve(
            request.pv_production,
        )

        points = self.influx.find_series(
            measurement=influx_sensor.measurement,
            entity_id=influx_sensor.entity_id,
            field=influx_sensor.field,
            start=start,
            end=end,
            resample=Resample(
                interval="5m",
                aggregation="mean",
            ),
        )

        rows = [
            {
                "target_time": parse_datetime(point["time"]),
                "pv_production": float(point["value"]),
            }
            for point in points
            if point["value"] is not None
        ]

        return pd.DataFrame(rows)
