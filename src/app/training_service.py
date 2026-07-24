import ast
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pandas as pd

from domain.models import Resample, SensorReferenceRequest, Storage, TrainRequest
from domain.time import parse_datetime
from features.forecaster import DhwForecaster, SolarForecaster
from features.generator import SolarForecastFeatureGenerator
from infrastructure.influx import InfluxDatabase, InfluxSensorResolver


class TrainingService:
    def __init__(
        self,
        influx: InfluxDatabase,
        resolver: InfluxSensorResolver,
        generator: SolarForecastFeatureGenerator,
        forecaster: SolarForecaster,
        storage: Storage,
    ):
        self.influx = influx
        self.resolver = resolver
        self.generator = generator
        self.forecaster = forecaster
        self.storage = storage

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

        records = cast(
            dict[str, Any],
            cast(object, df.tail(100).to_dict(orient="records")),
        )

        self.storage.save(records)

        df = self._load_boiler_features(
            start=start,
            end=end,
        )

        dhw = DhwForecaster(lags=96)

        dhw.fit(
            temperature=df["temp"],
            exog=df[
                [
                    "mode",
                ]
            ],
        )

        dhw.tune(
            temperature=df["temp"],
            exog=df[
                [
                    "mode",
                ]
            ],
        )

        print(dhw.best_params)
        print(dhw.best_score)

        metrics, predictions = dhw.backtest(
            temperature=df["temp"],
            exog=df[
                [
                    "mode",
                ]
            ],
        )

        print(metrics)

        future_index = pd.date_range(
            start=df.index[-1] + pd.Timedelta("15min"),
            periods=24,
            freq="15min",
        )

        future_exog = pd.DataFrame(
            {
                "mode": ["Uit"] * 24,
            },
            index=future_index,
        )

        future = dhw.predict(
            steps=24,
            exog=future_exog,
        )

        print(future)

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

        print(forecast[["time", "target_time"]].head(10))
        print(production.head(10))

        return (
            forecast.merge(
                production,
                on="target_time",
                how="inner",
            )
            .sort_values(
                ["target_time", "time"],
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
                time = parse_datetime(point["time"])

                value = ast.literal_eval(point["value"])

                for target_time, watts in value.items():
                    target_time = parse_datetime(target_time)

                    if target_time < time:
                        continue

                    if target_time - time > timedelta(hours=48):
                        continue

                    rows.append(
                        {
                            "time": time.replace(second=0, microsecond=0),
                            "target_time": target_time.replace(second=0, microsecond=0),
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
                    "time",
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
                    "time",
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
                interval="30m",
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

    def _load_boiler_features(self, start, end):

        temp_sensor = self.resolver.resolve(
            SensorReferenceRequest(
                entity_id="sensor.ecodan_heatpump_ca09ec_sww_2e_temp_sensor",
                attribute="value",
            )
        )

        mode_sensor = self.resolver.resolve(
            SensorReferenceRequest(
                entity_id="sensor.ecodan_heatpump_ca09ec_status_bedrijf",
                attribute="state",
            )
        )

        temp_points = self.influx.find_series(
            measurement=temp_sensor.measurement,
            entity_id=temp_sensor.entity_id,
            field=temp_sensor.field,
            start=start,
            end=end,
            resample=Resample(
                interval="15m",
                aggregation="mean",
            ),
        )

        mode_points = self.influx.find_series(
            measurement=mode_sensor.measurement,
            entity_id=mode_sensor.entity_id,
            field=mode_sensor.field,
            start=start,
            end=end,
        )

        temp = pd.DataFrame(
            {
                "time": [parse_datetime(p["time"]) for p in temp_points if p["value"] is not None],
                "temp": [float(p["value"]) for p in temp_points if p["value"] is not None],
            }
        )

        mode = pd.DataFrame(
            {
                "time": [parse_datetime(p["time"]) for p in mode_points if p["value"] is not None],
                "mode": [str(p["value"]) for p in mode_points if p["value"] is not None],
            }
        )

        df = pd.merge_asof(
            temp.sort_values("time"),
            mode.sort_values("time"),
            on="time",
            direction="backward",
        )

        return (
            df.set_index("time")
            .sort_index()
            .resample("15min")
            .agg(
                {
                    "temp": "mean",
                    "mode": "last",
                }
            )
            .assign(
                mode=lambda x: x["mode"].ffill().fillna("Uit"),
                temp=lambda x: x["temp"].interpolate(),
            )
        )
