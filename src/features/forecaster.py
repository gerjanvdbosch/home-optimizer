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
from domain.models.state import BacktestPoint, BacktestResult


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
        return {
            "last_window": last_window[self.target_column],
            "exog": df[self.exog_columns] if df is not None else None,
        }

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
            **self.predict_arguments(last_window, df),
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

        def make_point(label: str, column: str) -> BacktestPoint:
            points = (
                result[[column]]
                .rename(columns={column: "value"})
                .rename_axis("time")
                .reset_index()
                .to_dict("records")
            )

            return BacktestPoint(
                label=label,
                points=cast(list[dict[str, object]], points),
            )

        return BacktestResult(
            name=self.name,
            label=self.label,
            unit=self.unit,
            mae=float(metric["mean_absolute_error"].iloc[0]),
            points=[
                make_point("Actual", "actual"),
                make_point("Prediction", "pred"),
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


class SklearnForecaster(Forecaster):
    def __init__(self):
        self.forecaster = self.create()

    @abstractmethod
    def create(self) -> HistGradientBoostingRegressor: ...

    def search_space(self, trial: Trial) -> dict[str, Any]:
        raise NotImplementedError()

    def arguments(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "X": df[self.exog_columns],
            "y": df[self.target_column],
        }

    def predict_arguments(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ):
        return {
            "last_window": last_window,
            "exog": df[self.exog_columns] if df is not None else None,
        }

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def fit(self, df: pd.DataFrame):
        df = self.prepare(df)

        args = self.arguments(df)

        self.forecaster.fit(
            args["X"],
            args["y"],
        )

    def predict(
        self,
        last_window: pd.DataFrame,
        df: pd.DataFrame | None = None,
        steps: int = 24,
    ) -> pd.Series:
        if df is None:
            df = pd.DataFrame()

        df = self.prepare(df)

        exog = df[self.exog_columns].iloc[:steps]

        prediction = pd.Series(
            self.forecaster.predict(exog),
            index=exog.index,
            name="error",
        )

        return self.predict_result(
            prediction,
            df.iloc[:steps],
        )

    def predict_result(self, prediction: pd.Series, df: pd.DataFrame) -> pd.Series: ...

    def backtest(self, df: pd.DataFrame, steps: int = 24) -> BacktestResult:
        raise NotImplementedError()

    def tune(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        n_trials: int = 10,
        study_storage: str | Path | None = None,
    ) -> tuple[pd.DataFrame, Study]:
        raise NotImplementedError()

    def save(self, path: Path) -> None:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        dump(self.forecaster, path / f"{self.name}.joblib")

    def load(self, path: Path) -> None:
        file_name = path / f"{self.name}.joblib"

        if not file_name.exists():
            return

        self.forecaster = load(file_name)
