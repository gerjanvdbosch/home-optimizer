import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from optuna import Study, Trial, create_study
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

from domain.models.config import Config, ForecasterType
from domain.models.dataset import DatasetDefinition
from domain.models.state import BacktestPoint, BacktestResult
from domain.time import to_local_time
from features.dataset import DatasetBuilder
from features.forecaster import SklearnForecaster


class SolarForecaster(SklearnForecaster):
    @property
    def name(self) -> ForecasterType:
        return "solar"

    @property
    def label(self) -> str:
        return "Power"

    @property
    def unit(self) -> str:
        return "W"

    @property
    def target_column(self) -> str:
        return "P_solar"

    @property
    def exog_columns(self) -> list[str]:
        return [
            "p10",
            # "p50",
            "p90",
            "lead_time_hours",
            "lead_time_hours_sq",
            "spread",
            "spread_log",
            "hour_sin",
            "hour_cos",
            # "day_of_year_sin",
            # "day_of_year_cos",
            "spread_x_lead",
            "p50_x_lead",
        ]

    def create(self, **overrides: Any) -> HistGradientBoostingRegressor:
        params: dict[str, Any] = dict(
            # loss="quantile",
            # quantile=0.5,
            max_iter=100,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            l2_regularization=1.0,
            max_depth=None,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        params.update(overrides)

        return HistGradientBoostingRegressor(**params)

    def predict_arguments(self, df: pd.DataFrame, steps: int = 24):
        now = datetime.now(UTC)

        last_window = df[
            (df["target_time"] <= now) & df["P_solar"].notna()
        ].sort_values("target_time")

        future = (
            df[(df["target_time"] > now) & (df["time"] <= now)]
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .sort_values("target_time")
        )

        print(last_window)
        print(future)

        return {
            "X": future[self.exog_columns].iloc[:steps],
        }

    def arguments(self, df: pd.DataFrame) -> dict[str, Any]:
        df = df.dropna(subset=[self.target_column, *self.exog_columns]).copy()

        df["error"] = df[self.target_column] - df["p50"]

        return {
            "X": df[self.exog_columns],
            "y": df["error"],
        }

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["target_time"] > df["time"]].copy()

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600
        df["lead_time_hours_sq"] = df["lead_time_hours"] ** 2

        df["spread"] = df["p90"] - df["p10"]
        df["spread_log"] = np.log1p(df["spread"])

        df["spread_x_lead"] = df["spread"] * df["lead_time_hours"]
        df["p50_x_lead"] = df["p50"] * df["lead_time_hours"]

        hour = df["target_time"].dt.hour + df["target_time"].dt.minute / 60

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        day = df["target_time"].dt.dayofyear

        df["day_of_year_sin"] = np.sin(2 * np.pi * day / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * day / 365.25)

        # print(
        #     df[
        #         [
        #             "time",
        #             "target_time",
        #             "lead_time_hours",
        #             "P_solar",
        #             "p10",
        #             "p50",
        #             "p90",
        #             "spread",
        #             "spread_relative",
        #         ]
        #     ].to_string()
        # )

        df = df.sort_values(["time", "target_time"])

        return df

    def predict_result(self, prediction: np.ndarray, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            df["p50"].to_numpy() + prediction,
            index=df["target_time"],
            name="pred",
        )

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult:
        df = self.prepare(df)

        def make_points(df: pd.DataFrame, value_col: str) -> list[dict]:
            records = df[["target_time", value_col]].to_dict("records")

            return [
                {
                    "time": pd.to_datetime(record["target_time"]).isoformat(),
                    "value": float(record[value_col]),
                }
                for record in records
            ]

        actual = (
            df[["target_time", "P_solar"]]
            .dropna(subset=["P_solar"])
            .drop_duplicates("target_time")
            .sort_values("target_time")
        )

        backtest_points = [
            BacktestPoint(
                label="Actual",
                group="Actual",
                color="white",
                points=make_points(actual, "P_solar"),
            )
        ]

        baseline_errors: list[float] = []
        ml_errors: list[float] = []

        groups = df.sort_values("time").groupby("time")

        progress = tqdm(
            groups,
            total=groups.ngroups,
            desc="Solar backtest",
        )

        retrain_every = pd.Timedelta(hours=2)
        max_train_days = 60

        last_model = None
        last_trained: pd.Timestamp | None = None

        for update_time, update_df in progress:
            forecast = (
                update_df[update_df["target_time"] > update_time]
                .sort_values("target_time")
                .drop_duplicates("target_time", keep="last")
            )

            test = forecast[forecast["P_solar"].notna()].iloc[:steps].copy()

            if test.empty:
                continue

            need_retrain = (
                last_model is None
                or last_trained is None
                or (update_time - last_trained) >= retrain_every
            )

            if need_retrain:
                window_start = update_time - pd.Timedelta(days=max_train_days)

                train = df[
                    (df["time"] >= window_start)
                    & (df["time"] < update_time)
                    & (df["target_time"] < update_time)
                    & (df["target_time"] > df["time"])
                    & df["P_solar"].notna()
                ].dropna(subset=[self.target_column, *self.exog_columns])

                if not train.empty:
                    error = train["P_solar"] - train["p50"]
                    last_model = self.create()
                    last_model.fit(train[self.exog_columns], error)
                    last_trained = update_time

            if last_model is None:
                continue

            error_prediction = last_model.predict(test[self.exog_columns])
            test["pred"] = test["p50"] + error_prediction

            baseline_errors.extend((test["P_solar"] - test["p50"]).abs().tolist())

            ml_errors.extend((test["P_solar"] - test["pred"]).abs().tolist())

            ts = pd.to_datetime(str(update_time))
            label = to_local_time(ts.to_pydatetime()).strftime("%d-%m %H:%M")

            backtest_points.append(
                BacktestPoint(
                    label=f"ML {label}",
                    group="ML",
                    points=make_points(test, "pred"),
                )
            )

            backtest_points.append(
                BacktestPoint(
                    label=f"Update {label}",
                    group="Solcast",
                    points=make_points(
                        update_df.sort_values("target_time"),
                        "p50",
                    ),
                )
            )

        baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else 0.0

        ml_mae = float(np.mean(ml_errors)) if ml_errors else 0.0

        improvement = (
            100 * (baseline_mae - ml_mae) / baseline_mae if baseline_mae else 0.0
        )

        logging.info(
            "Solar MAE: baseline=%.2f W, ML=%.2f W, improvement=%.1f%%",
            baseline_mae,
            ml_mae,
            improvement,
        )

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=ml_mae,
            points=backtest_points,
        )

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True,
            ),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes",
                15,
                127,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                10,
                50,
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization",
                0.1,
                10.0,
                log=True,
            ),
            "max_iter": trial.suggest_int(
                "max_iter",
                100,
                400,
            ),
            # "max_depth": trial.suggest_int("max_depth", 3, 12),
        }

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 30,
        study_storage: str | Path | None = None,
        retrain_every: pd.Timedelta = pd.Timedelta(hours=6),
        max_train_days: int = 30,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df)

        def objective(trial: Trial) -> float:
            params = self.search_space(trial)

            ml_errors: list[float] = []

            last_model = None
            last_trained: pd.Timestamp | None = None

            groups = df.sort_values("time").groupby("time")

            for update_time, update_df in groups:
                forecast = (
                    update_df[update_df["target_time"] > update_time]
                    .sort_values("target_time")
                    .drop_duplicates("target_time", keep="last")
                )

                test = forecast[forecast["P_solar"].notna()].iloc[:steps].copy()

                if test.empty:
                    continue

                need_retrain = (
                    last_model is None
                    or last_trained is None
                    or (update_time - last_trained) >= retrain_every
                )

                if need_retrain:
                    window_start = update_time - pd.Timedelta(days=max_train_days)

                    train = df[
                        (df["time"] >= window_start)
                        & (df["time"] < update_time)
                        & (df["target_time"] < update_time)
                        & (df["target_time"] > df["time"])
                        & df["P_solar"].notna()
                    ].dropna(subset=[self.target_column, *self.exog_columns])

                    if not train.empty:
                        error = train["P_solar"] - train["p50"]

                        model = self.create(**params)
                        model.fit(train[self.exog_columns], error)

                        last_model = model
                        last_trained = update_time

                if last_model is None:
                    continue

                error_pred = last_model.predict(test[self.exog_columns])
                pred = (test["p50"] + error_pred).clip(lower=0)

                ml_errors.extend((test["P_solar"] - pred).abs().tolist())

            if not ml_errors:
                return 9999.0

            return float(np.mean(ml_errors))

        storage = str(study_storage) if study_storage else None

        study = create_study(
            study_name=self.name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            storage=storage,
            load_if_exists=True,
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params

        self.forecaster = self.create(**best_params)

        trials_df = study.trials_dataframe()

        logging.info(
            "Tune finished - best MAE: %.2f | params: %s",
            study.best_value,
            best_params,
        )

        return trials_df, study

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "P_solar",
                config.solar.production,
                interval="30m",
                aggregation="mean",
                fill=0,
            )
            .attribute_timeseries(
                "p10",
                config.solar.forecast.p10,
                interval="30m",
                aggregation="last",
            )
            .attribute_timeseries(
                "p50",
                config.solar.forecast.p50,
                interval="30m",
                aggregation="last",
            )
            .attribute_timeseries(
                "p90",
                config.solar.forecast.p90,
                interval="30m",
                aggregation="last",
            )
            .join("p50", "p10", on=("time", "target_time"), how="outer")
            .join("p50", "p90", on=("time", "target_time"), how="outer")
            .join(
                "p50",
                "P_solar",
                left_on=("target_time",),
                right_on=("time",),
                how="outer",
            )
            .build()
        )
