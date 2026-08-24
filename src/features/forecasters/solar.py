import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd
from optuna import Study, Trial, create_study
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
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

MIN_SOLAR_IRRADIANCE = 100
RETRAIN_INTERVAL_HOURS = 6
MAX_TRAIN_WINDOW_DAYS = 30


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
            "p50",
            "p90",
            "spread_upper",
            "spread_lower",
            "gti",
            "temperature",
            "wind_speed",
            "precipitation",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "lead_time_hours",
            "hour",
            "season_phase",
            "lag_30m_mean",
            "lag_30m_max",
            "lag_30m_std",
            "lag_30m_trend",
            "lag_1h_trend",
            "lag_2h_mean",
            "lag_2h_max",
            "lag_2h_std",
            "lag_24h_mean",
        ]

    def search_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.1, 100.0, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
        }

    def create(self, **overrides: Any) -> HistGradientBoostingRegressor:
        params: dict[str, Any] = dict(
            loss="absolute_error",
            max_iter=150,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=90,
            max_depth=4,
            random_state=42,
            early_stopping=False,
            validation_fraction=0.15,
            n_iter_no_change=15,
        )
        params.update(overrides)

        return HistGradientBoostingRegressor(**params)

    def predict_arguments(self, df: pd.DataFrame, steps: int = 24) -> pd.DataFrame:
        now = datetime.now(UTC)

        future = (
            df[(df["target_time"] > now) & (df["time"] <= now)]
            .sort_values(["target_time", "time"])
            .drop_duplicates("target_time", keep="last")
            .sort_values("target_time")
            .iloc[:steps]
        )

        return future[self.exog_columns]

    def arguments(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.dropna(subset=[self.target_column, *self.exog_columns]).copy()

        df = df[
            (df["p50"] > MIN_SOLAR_IRRADIANCE)
            & (df["lead_time_hours"] >= 0.0)
            & (df["lead_time_hours"] <= 6.0)
        ].copy()

        y_target = df[self.target_column] - df["p50"]

        return df[self.exog_columns], y_target

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["lead_time_hours"] = (
            df["target_time"] - df["time"]
        ).dt.total_seconds() / 3600.0

        df["spread_upper"] = (df["p90"] - df["p50"]).clip(lower=0)
        df["spread_lower"] = (df["p50"] - df["p10"]).clip(lower=0)

        df["hour"] = df["target_time"].dt.hour + df["target_time"].dt.minute / 60.0

        df["season_phase"] = np.cos(
            2 * np.pi * (df["target_time"].dt.dayofyear - 172) / 365.25
        )

        actuals = (
            df[["target_time", "P_solar", "P_max", "P_std"]]
            .drop_duplicates("target_time")
            .set_index("target_time")
            .sort_index()
        )

        lag_1 = actuals.shift(1)
        lag_2 = actuals.shift(2)
        lag_3 = actuals.shift(3)

        df["lag_30m_mean"] = df["time"].map(lag_1["P_solar"]).fillna(0.0)
        df["lag_30m_max"] = df["time"].map(lag_1["P_max"]).fillna(0.0)
        df["lag_30m_std"] = df["time"].map(lag_1["P_std"]).fillna(0.0)
        df["lag_30m_trend"] = (
            df["time"].map(lag_1["P_solar"] - lag_2["P_solar"]).fillna(0.0)
        )

        df["lag_1h_trend"] = (
            df["time"].map(lag_1["P_solar"] - lag_3["P_solar"]).fillna(0.0)
        )

        rolling_2h_solar = actuals["P_solar"].shift(1).rolling(window=4, min_periods=1)
        rolling_2h_max = actuals["P_max"].shift(1).rolling(window=4, min_periods=1)

        df["lag_2h_mean"] = df["time"].map(rolling_2h_solar.mean()).fillna(0.0)
        df["lag_2h_max"] = df["time"].map(rolling_2h_max.max()).fillna(0.0)
        df["lag_2h_std"] = df["time"].map(rolling_2h_solar.std()).fillna(0.0)

        df["lag_24h_mean"] = (
            df["target_time"].map(actuals["P_solar"].shift(48)).fillna(0.0)
        )

        df = df.drop(columns=["P_max", "P_std"])

        return df.sort_values(["time", "target_time"])

    def predict_result(self, prediction: np.ndarray, df: pd.DataFrame) -> pd.Series:
        p50 = df["p50"].to_numpy()

        final_prediction = p50 + np.nan_to_num(prediction, nan=0.0)
        final_prediction = np.where(p50 < MIN_SOLAR_IRRADIANCE, p50, final_prediction)
        final_prediction = np.maximum(final_prediction, 0.0)

        return pd.Series(
            final_prediction,
            index=df["target_time"],
            name="pred",
        )

    def generate_walk_forward_folds(
        self,
        df: pd.DataFrame,
        steps: int,
        refit_hours: int = RETRAIN_INTERVAL_HOURS,
        max_train_days: int = MAX_TRAIN_WINDOW_DAYS,
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

        for update_time, start, end in zip(update_times, starts, ends, strict=True):
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
        last_X_train = None
        last_y_train = None

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

        folds = self.generate_walk_forward_folds(df_sorted, steps=steps)
        total_updates = df_sorted["time"].nunique()
        progress = tqdm(folds, total=total_updates, desc="Solar backtest")

        for update_time, train_df, test_df, need_retrain in progress:
            if need_retrain and not train_df.empty:
                try:
                    X_train, y_train = self.arguments(train_df)
                    if not X_train.empty:
                        best_params = getattr(self, "best_params", {})
                        model = self.create(**best_params)
                        model.fit(X_train, y_train)

                        last_X_train = X_train
                        last_y_train = y_train
                except ValueError:
                    continue

            if model is None:
                continue

            test_clean = test_df.dropna(subset=self.exog_columns).copy()
            if test_clean.empty:
                continue

            error_pred = model.predict(test_clean[self.exog_columns])
            test_clean["pred"] = self.predict_result(error_pred, test_clean).to_numpy()

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

                if not window_test.empty:
                    window_baseline_errors[label].extend(
                        (window_test["P_solar"] - window_test["p50"]).abs().tolist()
                    )
                    window_errors[label].extend(
                        (window_test["P_solar"] - window_test["pred"]).abs().tolist()
                    )

            ts = pd.to_datetime(str(update_time))
            label_ts = to_local_time(ts.to_pydatetime()).strftime("%d-%m %H:%M")

            backtest_points.append(
                BacktestPoint(
                    label=f"ML {label_ts}",
                    group="ML",
                    points=make_points(test_clean.sort_values("target_time"), "pred"),
                )
            )
            backtest_points.append(
                BacktestPoint(
                    label=f"Update {label_ts}",
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

            if ml_window:
                baseline = float(np.mean(baseline_window))
                ml = float(np.mean(ml_window))
                imp = 100 * (baseline - ml) / baseline if baseline else 0.0
                logging.info(
                    "  %8s: baseline=%.2f W | ML=%.2f W | improvement=%+.1f%% | n=%d",
                    label,
                    baseline,
                    ml,
                    imp,
                    len(ml_window),
                )

        if model is not None and last_X_train is not None and not last_X_train.empty:
            result = cast(
                Bunch,
                permutation_importance(
                    model, last_X_train, last_y_train, n_repeats=5, random_state=42
                ),
            )

            top_features = last_X_train.columns[
                result.importances_mean.argsort()[::-1][:5]
            ]
            logging.info(f"Top 5 important features: {list(top_features)}")

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
        refit_hours: int = 24,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df).dropna(subset=[self.target_column, "p50"])
        df_sorted = df.sort_values(["time", "target_time"]).reset_index(drop=True)

        folds = list(
            self.generate_walk_forward_folds(
                df_sorted,
                steps=steps,
                refit_hours=refit_hours,
                max_train_days=MAX_TRAIN_WINDOW_DAYS,
            )
        )

        def objective(trial: Trial) -> float:
            params = self.search_space(trial)
            ml_errors: list[float] = []
            model = None

            for _, train_df, test_df, need_retrain in folds:
                if need_retrain and not train_df.empty:
                    try:
                        X_train, y_train = self.arguments(train_df)
                        if not X_train.empty:
                            model = self.create(**params)
                            model.fit(X_train, y_train)
                    except ValueError:
                        pass

                if model is None:
                    continue

                test_clean = test_df.dropna(subset=self.exog_columns).copy()
                if test_clean.empty:
                    continue

                error_pred = model.predict(test_clean[self.exog_columns])
                pred = self.predict_result(error_pred, test_clean).to_numpy()

                ml_errors.extend((test_clean["P_solar"] - pred).abs().tolist())

            return float(np.mean(ml_errors)) if ml_errors else 9999.0

        storage = str(study_storage) if study_storage else None
        study = create_study(
            study_name=self.name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.forecaster = self.create(**study.best_params)

        logging.info(
            "Tune finished - best MAE: %.2f | params: %s",
            study.best_value,
            study.best_params,
        )

        return study.trials_dataframe(), study

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
            .timeseries(
                "P_max",
                config.solar.production,
                interval="30m",
                aggregation="max",
                fill=0,
            )
            .timeseries(
                "P_std",
                config.solar.production,
                interval="30m",
                aggregation="stddev",
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
                attributes=[
                    "gti",
                    "temperature",
                    "precipitation",
                    "wind_speed",
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                ],
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
                right="P_max",
                left_on=("target_time",),
                right_on=("time",),
                how="left",
            )
            .join(
                left="solcast",
                right="P_std",
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
