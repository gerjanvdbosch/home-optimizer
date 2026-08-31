from typing import Any

import pandas as pd
from optuna import Trial
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge

from domain.models import BoilerThermalModel, Config, DatasetDefinition, ForecasterType
from features.dataset import DatasetBuilder
from features.forecaster import SkforecastForecaster, SklearnForecaster


class BoilerForecaster(SklearnForecaster):
    @property
    def name(self) -> ForecasterType:
        return "boiler"

    @property
    def label(self) -> str:
        return "Temperature"

    @property
    def unit(self) -> str:
        return "°C"

    @property
    def target_column(self) -> str:
        return "dT_top"

    @property
    def exog_columns(self) -> list[str]:
        return ["T_diff_top_amb", "T_diff_bottom_amb", "compressor_dhw"]

    def create(self, **overrides: Any) -> Ridge:
        # fit_intercept=False omdat bij T_boiler == T_amb het verlies exact 0 moet zijn
        return Ridge(alpha=0.01, fit_intercept=False, **overrides)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "time" in df.columns:
            df = df.set_index("time")
        df = df.sort_index().asfreq("15min")

        required = ["T_top", "T_bottom", "compressor_freq", "state"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset mist vereiste kolom(men): {missing}")

        # 1. Omgevingstemperatuur
        t_ambient = df["T_ambient"] if "T_ambient" in df.columns else 20.0
        df["T_diff_top_amb"] = df["T_top"] - t_ambient
        df["T_diff_bottom_amb"] = df["T_bottom"] - t_ambient

        # 2. SWW Compressor
        is_sww = df["state"].astype(str).str.strip().str.upper() == "SWW"
        df["compressor_dhw"] = df["compressor_freq"].where(is_sww, 0.0)

        # 3. dT voor BEIDE lagen over 15m
        df["dT_top"] = df["T_top"].shift(-1) - df["T_top"]
        df["dT_bottom"] = df["T_bottom"].shift(-1) - df["T_bottom"]

        targets = ["dT_top", "dT_bottom"]
        df = df.dropna(subset=self.exog_columns + targets)

        # Filter douchen en meetruis
        valid_idle = (
            (df["compressor_dhw"] == 0)
            & (df["dT_top"] <= 0.0)
            & (df["dT_top"] > -0.5)
            & (df["dT_bottom"] <= 0.0)
            & (df["dT_bottom"] > -0.5)
        )
        valid_sww = (
            (df["compressor_dhw"] > 0) & (df["dT_top"] >= 0.0) & (df["dT_bottom"] > 0.0)
        )

        clean_df = df[valid_idle | valid_sww].copy()
        if clean_df.empty:
            raise ValueError("Geen geldige trainingsdata overgebleven na filteren.")

        return clean_df

    def fit(self, df: pd.DataFrame) -> None:
        df_clean = self.prepare(df)

        sww_data = df_clean[df_clean["compressor_dhw"] > 0]
        if sww_data.empty:
            raise ValueError("Training mislukt: Geen SWW data gevonden.")

        typical_dhw_freq = float(sww_data["compressor_dhw"].median())

        X = df_clean[self.exog_columns]
        Y = df_clean[["dT_top", "dT_bottom"]]

        self.forecaster = self.create()
        self.forecaster.fit(X, Y)
        self.forecaster.dhw_freq_ = typical_dhw_freq

    def to_thermal_model(self) -> BoilerThermalModel:
        if not hasattr(self.forecaster, "coef_") or not hasattr(
            self.forecaster, "dhw_freq_"
        ):
            raise ValueError("Model is nog niet getraind.")

        # coef_ matrix heeft vorm (2, 3):
        # Rij 0 = dT_top parameters: [beta_top_top, beta_top_bottom, c_top]
        # Rij 1 = dT_bottom parameters: [beta_bottom_top, beta_bottom_bottom, c_bottom]
        coef = self.forecaster.coef_

        return BoilerThermalModel(
            # T_top[t+1] = (1 + beta_11)*T_top + beta_12*T_bottom + ...
            a_top_top=1.0 + float(coef[0, 0]),
            a_top_bottom=float(coef[0, 1]),
            c_top=max(0.0, float(coef[0, 2])),
            # T_bottom[t+1] = beta_21*T_top + (1 + beta_22)*T_bottom + ...
            a_bottom_top=float(coef[1, 0]),
            a_bottom_bottom=1.0 + float(coef[1, 1]),
            c_bottom=max(0.0, float(coef[1, 2])),
            dhw_freq=float(self.forecaster.dhw_freq_),
            t_ambient=20.0,
        )

    def dataset(self, config: Config) -> DatasetDefinition:
        return (
            DatasetBuilder()
            .timeseries(
                "T_top",
                config.heat_pump.boiler.top_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "T_bottom",
                config.heat_pump.boiler.bottom_temperature,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "compressor_freq",
                config.heat_pump.compressor_frequency,
                interval="15m",
                aggregation="mean",
                fill="previous",
            )
            .timeseries(
                "state",
                config.heat_pump.state,
                interval="15m",
                aggregation="first",
                fill="previous",
            )
            .build()
        )


# class BoilerForecaster(SkforecastForecaster):
#     @property
#     def name(self) -> ForecasterType:
#         return "boiler"
#
#     @property
#     def label(self) -> str:
#         return "Temperature"
#
#     @property
#     def unit(self) -> str:
#         return "°C"
#
#     @property
#     def target_column(self) -> str:
#         return "T_top"
#
#     @property
#     def exog_columns(self) -> list[str]:
#         return [
#             "T_bottom",
#             "state",
#             "compressor_freq",
#             "T_setpoint",
#             "T_supply",
#         ]
#
#     def create(self, **overrides: Any):
#         return ForecasterRecursive(
#             forecaster_id=self.name,
#             estimator=HistGradientBoostingRegressor(
#                 learning_rate=0.031,
#                 max_depth=5,
#                 max_iter=250,
#                 min_samples_leaf=16,
#                 random_state=42,
#             ),
#             lags=48,
#             calendar_features=CalendarFeatures(
#                 features=[
#                     "minute",
#                     "hour",
#                     "week",
#                     "month",
#                     "quarter",
#                     "day_of_week",
#                     "weekend",
#                 ],
#                 encoding="cyclical",
#             ),
#         )
#
#     def search_space(self, trial: Trial) -> dict[str, Any]:
#         return {
#             "lags": trial.suggest_categorical(
#                 "lags",
#                 [
#                     48,
#                     96,
#                     192,
#                 ],
#             ),
#             "learning_rate": trial.suggest_float(
#                 "learning_rate",
#                 0.03,
#                 0.07,
#             ),
#             "max_depth": trial.suggest_int(
#                 "max_depth",
#                 2,
#                 5,
#             ),
#             "max_iter": trial.suggest_int(
#                 "max_iter",
#                 50,
#                 250,
#                 step=50,
#             ),
#             "min_samples_leaf": trial.suggest_int(
#                 "min_samples_leaf",
#                 10,
#                 40,
#             ),
#         }
#
#     def dataset(self, config: Config) -> DatasetDefinition:
#         return (
#             DatasetBuilder()
#             .timeseries(
#                 "state",
#                 config.heat_pump.state,
#                 interval="15m",
#                 aggregation="first",
#                 fill="previous",
#             )
#             .timeseries(
#                 "compressor_freq",
#                 config.heat_pump.compressor_frequency,
#                 aggregation="mean",
#                 interval="15m",
#                 fill="previous",
#             )
#             .timeseries(
#                 "T_setpoint",
#                 config.heat_pump.boiler.setpoint,
#                 aggregation="mean",
#                 interval="15m",
#                 fill="previous",
#             )
#             .timeseries(
#                 "T_top",
#                 config.heat_pump.boiler.top_temperature,
#                 aggregation="mean",
#                 interval="15m",
#                 fill="previous",
#             )
#             .timeseries(
#                 "T_bottom",
#                 config.heat_pump.boiler.bottom_temperature,
#                 aggregation="mean",
#                 interval="15m",
#                 fill="previous",
#             )
#             .timeseries(
#                 "T_supply",
#                 config.heat_pump.supply_temperature,
#                 aggregation="mean",
#                 interval="15m",
#                 fill="previous",
#             )
#             .build()
#         )
