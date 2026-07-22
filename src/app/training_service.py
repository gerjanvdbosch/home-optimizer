import ast
from datetime import datetime, timedelta, timezone

import pandas as pd

from domain.models import Resample, Storage, TrainRequest
from domain.time import parse_datetime
from features.forecaster import SolarForecaster
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

        self.storage.save(df.tail(100).to_dict(orient="records"))

        feature_columns = self._feature_columns(df)

        split = int(len(df) * 0.8)

        train = df.iloc[:split]
        test = df.iloc[split:]

        X_train = train[feature_columns]
        y_train = train["pv_production"]

        X_test = test[feature_columns]
        y_test = test["pv_production"]

        self.forecaster.fit(
            y=y_train,
            exog=X_train,
        )

        prediction = self.forecaster.predict(
            steps=len(X_test),
            exog=X_test,
        )

        metrics = self.forecaster.evaluate(
            y_true=y_test,
            y_pred=prediction,
        )

        print(metrics)

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

    def _feature_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:

        excluded = {
            "time",
            "target_time",
            "pv_production",
        }

        return [column for column in df.columns if column not in excluded]
