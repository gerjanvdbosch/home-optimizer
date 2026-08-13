import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from optuna import Study, Trial, create_study
from optuna.samplers import TPESampler
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

from domain.models import (
    BacktestPoint,
    BacktestResult,
    Config,
    DatasetDefinition,
    ForecasterType,
)
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
            # "p10",
            "p50",
            # "p90",
            "gti",
            "gti_delta",
            "temperature",
            "lead_time_hours",
            # "lead_time_hours_sq",
            "spread",
            # "spread_log",
            # "is_day",
            "hour_sin",
            "hour_cos",
            # "day_of_year_sin",
            # "day_of_year_cos",
            "solar_lag1",
            # "solar_lag2",
            # "solar_lag3",
            # "solar_lag4",
            "error_lag1",
            # "error_lag2",
            # "error_lag3",
            # "error_lag4",
            # "spread_x_lead",
            # "p50_x_lead",
        ]

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "loss": trial.suggest_categorical(
                "loss", ["squared_error", "absolute_error"]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 150),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 3.0, 100.0, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 25, 100),
        }

    def create(self, **overrides: Any) -> HistGradientBoostingRegressor:
        params: dict[str, Any] = dict(
            loss="absolute_error",
            max_iter=48,
            learning_rate=0.042,
            max_leaf_nodes=13,
            min_samples_leaf=77,
            l2_regularization=86.4,
            max_depth=None,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
        )
        params.update(overrides)

        return HistGradientBoostingRegressor(**params)

    def predict_arguments(self, df: pd.DataFrame, steps: int = 24) -> dict[str, Any]:
        now = datetime.now(UTC)

        future = (
            df[(df["target_time"] > now) & (df["time"] <= now)]
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .sort_values("target_time")
        )

        return {
            "X": future[self.exog_columns].iloc[:steps],
        }

    def arguments(self, df: pd.DataFrame) -> dict[str, Any]:
        df = df.dropna(subset=[self.target_column, *self.exog_columns]).copy()
        df = df[(df["lead_time_hours"] >= 0.5) & (df["lead_time_hours"] <= 4.0)].copy()

        df["error"] = df[self.target_column] - df["p50"]

        return {
            "X": df[self.exog_columns],
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

        df["gti_lag1"] = df.groupby("time")["gti"].shift(1)
        df["gti_delta"] = (df["gti"] - df["gti_lag1"]).fillna(0)

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
        refit_hours: int = 24,
        max_train_days: int = 30,
    ) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame, bool]]:

        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

        time_index = pd.DatetimeIndex(df_sorted["time"])
        update_times = df_sorted["time"].unique()

        starts = time_index.searchsorted(update_times, side="left")
        ends = time_index.searchsorted(update_times, side="right")

        retrain_every = pd.Timedelta(hours=refit_hours)
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

        windows = [
            (0.5, 2.0),
            (2.0, 4.0),
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 24.0),
            (24.0, 48.0),
        ]

        window_errors = {f"{start:g}-{end:g}h": [] for start, end in windows}
        window_baseline_errors = {f"{start:g}-{end:g}h": [] for start, end in windows}

        folds = self.generate_walk_forward_folds(
            df_sorted,
            steps=steps,
        )

        total_updates = df_sorted["time"].nunique()

        progress = tqdm(folds, total=total_updates, desc="Solar backtest")

        for update_time, train_df, test_df, need_retrain in progress:
            train_day = (
                train_df[train_df["p50"] > 1] if not train_df.empty else pd.DataFrame()
            )

            if need_retrain and not train_day.empty:
                train_clean = train_day.dropna(subset=self.exog_columns)
                if not train_clean.empty:
                    model = self.create(**self.best_params)
                    model.fit(
                        train_clean[self.exog_columns], train_clean["error_target"]
                    )

            if model is None:
                continue

            test_clean = test_df.dropna(subset=self.exog_columns).copy()
            if test_clean.empty:
                continue

            day_mask = test_clean["p50"] > 1
            test_clean["pred"] = test_clean["p50"].copy()

            if day_mask.any() and model is not None:
                day_features = test_clean.loc[day_mask, self.exog_columns]
                error_pred = model.predict(day_features)
                test_clean.loc[day_mask, "pred"] = (
                    test_clean.loc[day_mask, "p50"] + error_pred
                ).clip(lower=0)
            else:
                test_clean["pred"] = test_clean["pred"].clip(lower=0)

            baseline_errors.extend(
                (test_clean["P_solar"] - test_clean["p50"]).abs().tolist()
            )
            ml_errors.extend(
                (test_clean["P_solar"] - test_clean["pred"]).abs().tolist()
            )

            for start, end in windows:
                label = f"{start:g}-{end:g}h"

                mask = (test_clean["lead_time_hours"] >= start) & (
                    test_clean["lead_time_hours"] < end
                )

                window_test = test_clean.loc[mask]

                if window_test.empty:
                    continue

                window_baseline_errors[label].extend(
                    (window_test["P_solar"] - window_test["p50"]).abs().tolist()
                )

                window_errors[label].extend(
                    (window_test["P_solar"] - window_test["pred"]).abs().tolist()
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

        logging.info("Solar MAE per lead-time window:")

        for start, end in windows:
            label = f"{start:g}-{end:g}h"

            baseline_window = window_baseline_errors[label]
            ml_window = window_errors[label]

            if not ml_window:
                continue

            baseline = float(np.mean(baseline_window))
            ml = float(np.mean(ml_window))

            improvement = 100 * (baseline - ml) / baseline if baseline else 0.0

            logging.info(
                "  %8s: baseline=%.2f W | ML=%.2f W | improvement=%+.1f%% | n=%d",
                label,
                baseline,
                ml,
                improvement,
                len(ml_window),
            )

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=ml_mae,
            points=backtest_points,
        )

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 12,
        n_trials: int = 30,
        study_storage: str | Path | None = None,
        refit_hours: int = 24 * 7,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df).dropna(subset=[self.target_column, "p50"])
        df["error_target"] = df[self.target_column] - df["p50"]
        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

        folds = list(
            self.generate_walk_forward_folds(
                df_sorted,
                steps=steps,
                refit_hours=refit_hours,
                max_train_days=30,
            )
        )

        logging.info(
            "Tune folds: %d totaal, %d retrain-momenten (retrain_every=%dh)",
            len(folds),
            sum(1 for _, _, _, need_retrain in folds if need_retrain),
            refit_hours,
        )

        def objective(trial: Trial) -> float:
            params = self.search_space(trial)
            features = self.exog_columns

            ml_errors: list[float] = []
            model = None

            for update_time, train_df, test_df, need_retrain in folds:
                train_day = (
                    train_df[train_df["p50"] > 1]
                    if not train_df.empty
                    else pd.DataFrame()
                )

                if need_retrain and not train_day.empty:
                    train_clean = train_day.dropna(subset=features)
                    if not train_clean.empty:
                        model = self.create(**params)
                        model.fit(train_clean[features], train_clean["error_target"])

                if model is None:
                    continue

                test_clean = test_df.dropna(subset=features).copy()
                if test_clean.empty:
                    continue

                day_mask = test_clean["p50"] > 1
                pred = test_clean["p50"].copy()

                if day_mask.any() and model is not None:
                    day_features = test_clean.loc[day_mask, features]
                    error_pred = model.predict(day_features)
                    pred.loc[day_mask] = (
                        test_clean.loc[day_mask, "p50"] + error_pred
                    ).clip(lower=0)
                else:
                    pred = pred.clip(lower=0)

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

        self.forecaster = self.create(**study.best_params)

        trials_df = study.trials_dataframe()

        logging.info(
            "Tune finished - best MAE: %.2f | params: %s",
            study.best_value,
            study.best_params,
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
                "solcast",
                config.forecast.solcast,
                attributes=["p10", "p50", "p90"],
                interval="30m",
                aggregation="last",
            )
            .attribute_timeseries(
                "open_meteo",
                config.forecast.open_meteo,
                attributes=["gti", "temperature", "is_day"],
                interval="30m",
                aggregation="last",
            )
            .join(
                left="solcast",
                right="P_solar",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .join(
                left="solcast",
                right="open_meteo",
                on=("time", "target_time"),
                how="left",
            )
            .build()
        )
