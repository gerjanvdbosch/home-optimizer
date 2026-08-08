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
            "solar_lag1",
            "solar_lag2",
            "solar_lag3",
            "solar_lag4",
            # "error_lag1",
            # "error_lag2",
            # "error_lag3",
            # "error_lag4",
            # "day_of_year_sin",
            # "day_of_year_cos",
            # "spread_upper",
            # "spread_lower",
            # "spread_upper_log",
            # "spread_lower_log",
            "spread_x_lead",
            "p50_x_lead",
        ]

    def features_from_params(self, params: dict[str, Any]) -> list[str]:
        features = ["lead_time_hours"]
        if params.get("use_lags"):
            features.extend(["solar_lag1", "solar_lag2", "solar_lag3", "solar_lag4"])
        if params.get("use_error_lags"):
            features.extend(["error_lag1", "error_lag2", "error_lag3", "error_lag4"])
        if params.get("use_p50"):
            features.append("p50")
        if params.get("use_time_features"):
            features.extend(
                ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
            )
        if params.get("use_lead_sq"):
            features.append("lead_time_hours_sq")
        if params.get("use_spread"):
            features.extend(["spread", "spread_log", "spread_x_lead"])
        if params.get("use_asym_spread"):
            features.extend(
                ["spread_upper", "spread_lower", "spread_upper_log", "spread_lower_log"]
            )
        if params.get("use_quantiles"):
            features.extend(["p10", "p90", "p50_x_lead"])
        return features

    def model_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in params.items() if not k.startswith("use_")}

    @property
    def features(self) -> list[str]:
        return self._tuned_features or self.exog_columns

    def create(self, **overrides: Any) -> HistGradientBoostingRegressor:
        params: dict[str, Any] = dict(
            max_iter=100,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=31,
            l2_regularization=1.0,
            max_depth=None,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
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
            "X": future[self.features].iloc[:steps],
        }

    def arguments(self, df: pd.DataFrame) -> dict[str, Any]:
        df = df.dropna(subset=[self.target_column, *self.features]).copy()

        df["error"] = df[self.target_column] - df["p50"]

        return {
            "X": df[self.features],
            "y": df["error"],
        }

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["baseline_error"] = df["P_solar"] - df["p50"]

        actual_series = (
            df[["target_time", "time", "P_solar", "baseline_error"]]
            .dropna(subset=["target_time", "P_solar", "baseline_error"])
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .rename(
                columns={
                    "target_time": "obs_time",
                    "P_solar": "actual_value",
                    "baseline_error": "error_value",
                }
            )
        )

        for i in range(4):
            actual_series[f"solar_lag{i + 1}"] = actual_series["actual_value"].shift(i)
            actual_series[f"error_lag{i + 1}"] = actual_series["error_value"].shift(i)

        lag_cols = [f"solar_lag{i}" for i in range(1, 5)] + [
            f"error_lag{i}" for i in range(1, 5)
        ]

        df = df.dropna(subset=["time"]).sort_values("time")

        df = pd.merge_asof(
            df,
            actual_series[["obs_time", *lag_cols]],
            left_on="time",
            right_on="obs_time",
            direction="backward",
        ).drop(columns=["obs_time"])

        df = df[df["target_time"] > df["time"]].copy()

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600
        df["lead_time_hours_sq"] = df["lead_time_hours"] ** 2

        df["spread"] = df["p90"] - df["p10"]
        df["spread_log"] = np.log1p(df["spread"])

        df["spread_upper"] = df["p90"] - df["p50"]
        df["spread_lower"] = df["p50"] - df["p10"]
        df["spread_upper_log"] = np.log1p(df["spread_upper"])
        df["spread_lower_log"] = np.log1p(df["spread_lower"])

        df["spread_x_lead"] = df["spread"] * df["lead_time_hours"]
        df["p50_x_lead"] = df["p50"] * df["lead_time_hours"]

        hour = df["target_time"].dt.hour + df["target_time"].dt.minute / 60

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        day = df["target_time"].dt.dayofyear

        df["day_of_year_sin"] = np.sin(2 * np.pi * day / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * day / 365.25)

        df = df.sort_values(["time", "target_time"])
        #
        # print(
        #     df[
        #         [
        #             "time",
        #             "target_time",
        #             "P_solar",
        #             "p50",
        #             "solar_lag1",
        #             "solar_lag2",
        #             "solar_lag3",
        #             "solar_lag4",
        #             "error_lag1",
        #             "error_lag2",
        #             "error_lag3",
        #             "error_lag4",
        #         ]
        #     ].to_string()
        # )

        return df

    def predict_result(self, prediction: np.ndarray, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            df["p50"].to_numpy() + prediction,
            index=df["target_time"],
            name="pred",
        )

    def generate_walk_forward_folds(
        self,
        df: pd.DataFrame,
        steps: int,
        retrain_every_hours: int = 6,
        max_train_days: int = 30,
    ) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame, bool]]:

        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

        # DatetimeIndex i.p.v. numpy-array: behoudt tz-info correct bij vergelijken
        time_index = pd.DatetimeIndex(df_sorted["time"])
        update_times = df_sorted["time"].unique()

        starts = time_index.searchsorted(update_times, side="left")
        ends = time_index.searchsorted(update_times, side="right")

        retrain_every = pd.Timedelta(hours=retrain_every_hours)
        max_train_window = pd.Timedelta(days=max_train_days)

        last_trained_time = None
        train_df = pd.DataFrame()

        for update_time, start, end in zip(update_times, starts, ends):
            update_time = pd.Timestamp(update_time)
            group = df_sorted.iloc[start:end]

            forecast = group[group["target_time"] > update_time].drop_duplicates(
                "target_time", keep="last"
            )
            test_df = forecast[forecast[self.target_column].notna()].iloc[:steps].copy()
            if test_df.empty:
                continue

            need_retrain = (
                last_trained_time is None
                or (update_time - last_trained_time) >= retrain_every
            )

            if need_retrain:
                window_start = update_time - max_train_window

                # Geen .to_datetime64() meer nodig — pandas regelt tz-vergelijking zelf
                train_start_idx = time_index.searchsorted(window_start, side="left")

                train_slice = df_sorted.iloc[train_start_idx:start]
                train_mask = (
                    (train_slice["target_time"] < update_time)
                    & (train_slice["target_time"] > train_slice["time"])
                    & train_slice[self.target_column].notna()
                )
                candidate_train = train_slice[train_mask]
                if not candidate_train.empty:
                    train_df = candidate_train
                    last_trained_time = update_time

            yield update_time, train_df, test_df, need_retrain

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult:
        df = self.prepare(df).dropna(subset=[self.target_column, "p50"])
        df["error_target"] = df[self.target_column] - df["p50"]
        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

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
            df_sorted[["target_time", "P_solar"]]
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
        model = None

        folds = self.generate_walk_forward_folds(
            df_sorted,
            steps=steps,
        )

        total_updates = df_sorted["time"].nunique()

        progress = tqdm(folds, total=total_updates, desc="Solar backtest")

        for update_time, train_df, test_df, need_retrain in progress:
            if need_retrain and not train_df.empty:
                train_clean = train_df.dropna(subset=self.features)
                if not train_clean.empty:
                    model = self.create(**self._tuned_params)
                    model.fit(train_clean[self.features], train_clean["error_target"])

            if model is None:
                continue

            test_clean = test_df.dropna(subset=self.features)
            if test_clean.empty:
                continue

            error_pred = model.predict(test_clean[self.features])
            test_clean["pred"] = (test_clean["p50"] + error_pred).clip(lower=0)

            baseline_errors.extend(
                (test_clean["P_solar"] - test_clean["p50"]).abs().tolist()
            )
            ml_errors.extend(
                (test_clean["P_solar"] - test_clean["pred"]).abs().tolist()
            )

            ts = pd.to_datetime(str(update_time))
            label = to_local_time(ts.to_pydatetime()).strftime("%d-%m %H:%M")

            backtest_points.append(
                BacktestPoint(
                    label=f"ML {label}",
                    group="ML",
                    points=make_points(test_clean, "pred"),
                )
            )
            backtest_points.append(
                BacktestPoint(
                    label=f"Update {label}",
                    group="Solcast",
                    points=make_points(test_clean.sort_values("target_time"), "p50"),
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
        params = {
            "loss": trial.suggest_categorical(
                "loss", ["squared_error", "absolute_error"]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.1, 10.0, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 10, 200),
        }

        features = ["lead_time_hours"]

        if trial.suggest_categorical("use_lags", [True, False]):
            features.extend(["solar_lag1", "solar_lag2", "solar_lag3", "solar_lag4"])

        if trial.suggest_categorical("use_error_lags", [True, False]):
            features.extend(["error_lag1", "error_lag2", "error_lag3", "error_lag4"])

        if trial.suggest_categorical("use_p50", [True, False]):
            features.append("p50")

        if trial.suggest_categorical("use_time_features", [True, False]):
            features.extend(
                ["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]
            )

        if trial.suggest_categorical("use_lead_sq", [True, False]):
            features.append("lead_time_hours_sq")

        if trial.suggest_categorical("use_spread", [True, False]):
            features.extend(["spread", "spread_log", "spread_x_lead"])

        if trial.suggest_categorical("use_asym_spread", [True, False]):
            features.extend(
                ["spread_upper", "spread_lower", "spread_upper_log", "spread_lower_log"]
            )

        if trial.suggest_categorical("use_quantiles", [True, False]):
            features.extend(["p10", "p90", "p50_x_lead"])

        params["_features"] = features
        return params

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 12,
        n_trials: int = 30,
        study_storage: str | Path | None = None,
        tune_retrain_every_hours: int = 6 * 7,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df).dropna(subset=[self.target_column, "p50"])
        df["error_target"] = df[self.target_column] - df["p50"]
        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

        # Eén keer materialiseren — hergebruikt over alle Optuna-trials
        folds = list(
            self.generate_walk_forward_folds(
                df_sorted,
                steps=steps,
                retrain_every_hours=tune_retrain_every_hours,
                max_train_days=30,
            )
        )

        logging.info(
            "Tune folds: %d totaal, %d retrain-momenten (retrain_every=%dh)",
            len(folds),
            sum(1 for _, _, _, need_retrain in folds if need_retrain),
            tune_retrain_every_hours,
        )

        def objective(trial: Trial) -> float:
            params = self.search_space(trial)
            features = params.pop("_features")

            ml_errors: list[float] = []
            model = None

            for update_time, train_df, test_df, need_retrain in folds:
                if need_retrain and not train_df.empty:
                    train_clean = train_df.dropna(subset=features)
                    if not train_clean.empty:
                        model = self.create(**params)
                        model.fit(train_clean[features], train_clean["error_target"])

                if model is None:
                    continue

                test_clean = test_df.dropna(subset=features)
                if test_clean.empty:
                    continue

                error_pred = model.predict(test_clean[features])
                pred = (test_clean["p50"] + error_pred).clip(lower=0)
                ml_errors.extend((test_clean["P_solar"] - pred).abs().tolist())

            return float(np.mean(ml_errors)) if ml_errors else 9999.0

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
        self._tuned_features = self.features_from_params(best_params)
        self._tuned_params = self.model_params(best_params)
        self.forecaster = self.create(**self._tuned_params)

        trials_df = study.trials_dataframe()

        logging.info(
            "Tune finished - best MAE: %.2f | params: %s | features: %s",
            study.best_value,
            self._tuned_params,
            self._tuned_features,
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
