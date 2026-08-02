from abc import abstractmethod
from pathlib import Path
from typing import Any, cast

import pandas as pd
from joblib import dump, load
from optuna import Study, Trial
from skforecast.base import ForecasterBase
from skforecast.model_selection import (
    TimeSeriesFold,
    backtesting_forecaster,
    bayesian_search_forecaster,
)
from skforecast.utils import load_forecaster, save_forecaster
from sklearn.ensemble import HistGradientBoostingRegressor

from domain.models.interface import Forecaster
from domain.models.state import BacktestResult


class SkforecastForecaster(Forecaster):
    def __init__(self):
        self.forecaster = self.create()

    @abstractmethod
    def create(self) -> ForecasterBase: ...

    def arguments(self, df: pd.DataFrame):
        return {
            "y": df[self.target_column],
            "exog": df[self.exog_columns],
        }

    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ):
        return {"last_window": last_window[self.exog_columns]}

    @abstractmethod
    def search_space(self, trial: Trial) -> dict[str, Any]: ...

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().set_index("time").sort_index().asfreq("15min")

    def fit(self, df: pd.DataFrame):
        df = self.prepare(df)

        self.forecaster.fit(**self.arguments(df))

    def predict(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
        steps: int = 24,
    ) -> pd.Series:
        if df is None:
            df = pd.DataFrame()

        df = self.prepare(df)

        return self.forecaster.predict(
            steps=steps,
            last_window=last_window,
            **self.predict_arguments(df),
        )

    def backtest(
        self,
        df: pd.DataFrame,
        steps: int = 24,
    ) -> BacktestResult:
        df = self.prepare(df)

        metric, result = backtesting_forecaster(
            n_jobs=1,
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps),
            **self.arguments(df),
        )

        return self.backtest_result(
            df=df,
            metric=metric,
            result=result,
        )

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]:
        df = self.prepare(df)

        result, study = bayesian_search_forecaster(
            n_jobs=1,
            metric="mean_absolute_error",
            forecaster=self.forecaster,
            cv=self.create_cv(df, steps),
            search_space=self.search_space,
            n_trials=n_trials,
            random_state=42,
            return_best=True,
            kwargs_create_study={
                "study_name": self.name,
                "storage": study_storage,
                "load_if_exists": True,
                "direction": "minimize",
            },
            **self.arguments(df),
        )

        return result, cast(Study, study)

    def create_cv(self, df: pd.DataFrame, steps: int) -> TimeSeriesFold:
        return TimeSeriesFold(
            steps=steps,
            initial_train_size=int(len(df) * 0.7),
            refit=True,
            fixed_train_size=False,
        )

    def backtest_result(
        self,
        df: pd.DataFrame,
        metric: pd.DataFrame,
        result: pd.DataFrame,
    ) -> BacktestResult:
        result = result.copy()

        result["actual"] = df.loc[result.index, self.target_column]

        return BacktestResult(
            name=self.name,
            y_axis=self.y_axis,
            unit=self.unit,
            mae=float(metric["mean_absolute_error"].iloc[0]),
            points=[
                {
                    "time": str(index),
                    **{str(key): float(value) for key, value in row.items()},
                }
                for index, row in result.iterrows()
            ],
        )

    def save(self, path: Path) -> None:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_forecaster(
            self.forecaster,
            str(path / f"{self.name}.joblib"),
        )

    def load(self, path: Path) -> None:
        file_name = path / f"{self.name}.joblib"

        if not file_name.exists():
            return

        self.forecaster = cast(
            ForecasterBase,
            load_forecaster(str(file_name)),
        )


# class SklearnForecaster(Forecaster):
#     def __init__(self):
#         self.forecaster = self.create()
#
#     @abstractmethod
#     def create(self) -> HistGradientBoostingRegressor: ...
#
#     @property
#     def evaluation_column(self) -> str:
#         return self.target_column
#
#     @abstractmethod
#     def search_space(self, trial: Trial) -> dict[str, Any]: ...
#
#     def arguments(self, df: pd.DataFrame) -> dict[str, Any]:
#         return {
#             "X": df[self.exog_columns],
#             "y": df[self.target_column],
#         }
#
#     def predict_arguments(
#         self,
#         last_window: pd.DataFrame,
#         df: pd.DataFrame | None = None,
#     ):
#         return {
#             "last_window": last_window,
#             "exog": df[self.exog_columns] if df is not None else None,
#         }
#
#     def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
#         return df.copy()
#
#     def fit(self, df: pd.DataFrame):
#         df = self.prepare(df)
#
#         args = self.arguments(df)
#
#         self.forecaster.fit(
#             args["X"],
#             args["y"],
#         )
#
#     def predict(
#         self,
#         last_window: pd.DataFrame,
#         df: pd.DataFrame | None = None,
#         steps: int = 24,
#     ) -> pd.Series:
#         if df is None:
#             df = pd.DataFrame()
#
#         df = self.prepare(df)
#
#         exog = df[self.exog_columns].iloc[:steps]
#
#         prediction = pd.Series(
#             self.forecaster.predict(exog),
#             index=exog.index,
#             name="error",
#         )
#
#         return self.predict_result(
#             prediction,
#             df.iloc[:steps],
#         )
#
#     def backtest(
#         self,
#         df: pd.DataFrame,
#         steps: int = 24,
#     ) -> BacktestResult:
#         df = self.prepare(df)
#
#         predictions = []
#
#         for issue_time in sorted(df["time"].unique()):
#             train = df[df["target_time"] < issue_time]
#
#             if train.empty:
#                 continue
#
#             if issue_time - train["target_time"].min() < pd.Timedelta(days=7):
#                 continue
#
#             test = df[
#                 (df["time"] == issue_time) & (df["target_time"] >= issue_time)
#             ].head(steps)
#
#             if train.empty or test.empty:
#                 continue
#
#             model = self.create()
#
#             arguments = self.arguments(train)
#
#             model.fit(
#                 arguments["X"],
#                 arguments["y"],
#             )
#
#             test = test.copy()
#
#             prediction = pd.Series(
#                 model.predict(test[self.exog_columns]),
#                 index=test.index,
#             )
#
#             test["pred"] = self.predict_result(
#                 prediction,
#                 test,
#             )
#
#             predictions.append(test)
#
#             print("pred next")
#
#         if not predictions:
#             raise ValueError("No backtest predictions generated.")
#
#         result = pd.concat(
#             predictions,
#             ignore_index=True,
#         )
#
#         return self.backtest_result(result)
#
#     def backtest_result(
#         self,
#         result: pd.DataFrame,
#     ) -> BacktestResult:
#         actual = result[self.evaluation_column]
#
#         mae = (actual - result["pred"]).abs().mean()
#
#         return BacktestResult(
#             name=self.name,
#             y_axis=self.y_axis,
#             unit=self.unit,
#             mae=float(mae),
#             points=[
#                 {
#                     "time": str(row["target_time"]),
#                     "actual": float(row[self.evaluation_column]),
#                     "pred": float(row["pred"]),
#                     "p50": float(row["p50"]),
#                 }
#                 for _, row in result.iterrows()
#             ],
#         )
#
#     def predict_result(self, prediction: pd.Series, df: pd.DataFrame) -> pd.Series: ...
#
#     def save(self, path: Path) -> None:
#         path.mkdir(
#             parents=True,
#             exist_ok=True,
#         )
#
#         dump(self.forecaster, path / f"{self.name}.joblib")
#
#     def load(self, path: Path) -> None:
#         file_name = path / f"{self.name}.joblib"
#
#         if not file_name.exists():
#             return
#
#         self.forecaster = load(file_name)
