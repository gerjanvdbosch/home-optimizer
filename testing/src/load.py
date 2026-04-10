import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from utils import add_cyclic_time_features
from config import Config
from context import Context
from database import Database

logger = logging.getLogger(__name__)

# Lag-definities op één plek zodat training en predict altijd synchroon lopen
LAG_DEFINITIONS = {
    "lag_24h": timedelta(hours=24),
    "lag_7d": timedelta(days=7),
}


class NowCaster:
    def __init__(self, decay_hours: float = 1.0):
        self.decay_hours = decay_hours
        self.current_ratio = 0.0

    def update(self, actual_kw: float, forecasted_kw: float):
        if actual_kw is None or forecasted_kw is None:
            return

        error = actual_kw - forecasted_kw

        if error <= 0:
            # Verbruik valt mee (actual < forecast) -> Snel aanpassen (agressief)
            alpha = 0.25
        else:
            # Verbruik hoger dan verwacht -> Voorzichtig aanpassen (kan ruis/piek zijn)
            # Gaussian decay: hoe groter de fout, hoe minder we de bias geloven als structureel
            decay_factor = np.exp(-((error / 1.5) ** 2))
            alpha = 0.15 * decay_factor
            alpha = max(0.01, alpha)

        # Update de bias (Exponential Moving Average)
        self.current_ratio = (1 - alpha) * self.current_ratio + (alpha * error)

    def apply(
        self, df: pd.DataFrame, current_time: datetime, col_name: str
    ) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        # 1. Bereken verschil in uren
        delta_hours = (df["timestamp"] - current_time).dt.total_seconds() / 3600.0

        # 2. Bepaal wat toekomst is (voor stap 3)
        is_future = df["timestamp"] >= current_time

        # 3. Decay berekenen
        # We clippen hier wel op 0 om math errors (overflow) te voorkomen bij oude data,
        # maar we lossen het toepassen op het verleden op in de volgende stap.
        # (Voor het verleden wordt de factor hier 1.0, maar die zetten we zo op 0)
        delta_hours_math = delta_hours.clip(lower=0)
        decay_factors = np.exp(-delta_hours_math / self.decay_hours)

        correction_vector = self.current_ratio * decay_factors

        # 4. FIX: Zet correctie op 0.0 voor alles wat NIET in de toekomst ligt
        correction_vector = correction_vector.where(is_future, 0.0)

        corrected_series = df[col_name] + correction_vector
        return corrected_series.clip(lower=0.05)


class LoadModel:
    """
    Het Machine Learning model (HistGradientBoosting).
    Leert de basislijn van het huis (zonder WP).
    """

    def __init__(self, path: Path):
        self.path = path
        self.model: Optional[BaseEstimator] = None
        self.profile: Optional[pd.DataFrame] = None  # opgeslagen bij training
        self.is_fitted = False
        self.feature_cols = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "temp",
            # Lag features: altijd beschikbaar (liggen in het verleden)
            "lag_24h",  # verbruik gisteren zelfde uur
            "lag_7d",  # verbruik vorige week zelfde uur
            # Profiel features: gemiddeld gedrag per uur/weekdag uit history
            "profile_mean",  # verwacht verbruik dit tijdslot
            "profile_std",  # variabiliteit → helpt de quantile schatting
        ]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.model = data.get("model")
                self.profile = data.get("profile")  # <-- profiel meeladen
                self.is_fitted = True
            except Exception:
                logger.error("[Load] Model corrupt.")

    def _build_profile(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Berekent gemiddeld verbruiksprofiel per (weekdag, uur) uit de history.
        Wordt eenmalig bij training berekend en opgeslagen naast het model.
        """
        df = history_df.copy()
        df["base_load"] = (df["load_actual"] - df["wp_actual"].fillna(0)).clip(lower=0.05)
        df["_hour"] = df["timestamp"].dt.hour
        df["_dow"] = df["timestamp"].dt.dayofweek
        profile = (
            df.groupby(["_dow", "_hour"])["base_load"]
            .agg(profile_mean="mean", profile_std="std")
            .reset_index()
            .rename(columns={"_dow": "dow", "_hour": "hour"})
        )
        return profile

    def _apply_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Voegt profile_mean en profile_std toe aan df via het opgeslagen profiel.
        Geen database-call nodig — profiel zit in geheugen.
        """
        if self.profile is None:
            df["profile_mean"] = np.nan
            df["profile_std"] = np.nan
            return df

        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["dow"] = df["timestamp"].dt.dayofweek
        df = df.merge(self.profile, on=["dow", "hour"], how="left")
        return df.drop(columns=["hour", "dow"])

    def _apply_lags(
        self, df: pd.DataFrame, lag_lookup: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Voegt lag-kolommen toe vanuit een vooraf opgehaalde lookup-dict.

        lag_lookup: {lag_naam: pd.Series met forecast_timestamp als index}
        De caller (LoadForecaster) is verantwoordelijk voor het ophalen
        van alleen de benodigde timestamps — niet de volledige history.
        """
        df = df.copy()
        for lag_name, series in lag_lookup.items():
            if series is not None and not series.empty:
                df[lag_name] = df["timestamp"].map(series)
            else:
                df[lag_name] = np.nan
        return df

    def _prepare_features(
        self,
        df: pd.DataFrame,
        lag_lookup: Optional[dict] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        df = add_cyclic_time_features(df, col_name="timestamp")
        df = self._apply_profile(df)

        if lag_lookup:
            df = self._apply_lags(df, lag_lookup)
        else:
            for lag_name in LAG_DEFINITIONS:
                df[lag_name] = np.nan

        X = df.reindex(columns=self.feature_cols)

        return X.apply(pd.to_numeric, errors="coerce")

    def train(self, df_history: pd.DataFrame):
        df_train = df_history.copy()
        df_train = df_train.dropna(subset=["wp_actual"])

        if len(df_train) < 10:
            logger.warning("[Load] Niet genoeg data om model te trainen.")
            return

        # Base Load berekening
        df_train["base_load"] = (df_train["load_actual"] - df_train["wp_actual"].fillna(0)).clip(lower=0.05)

        df_train = df_train.dropna(subset=["base_load"]).reset_index(drop=True)

        # Profiel bouwen uit volledige history en opslaan in geheugen + op disk
        self.profile = self._build_profile(df_history)

        # Lags direct uit de traindata zelf berekenen (geen aparte DB-call nodig)
        lag_lookup = {}
        load_series = df_train.set_index("timestamp")["base_load"]
        for lag_name, delta in LAG_DEFINITIONS.items():
            shifted = load_series.copy()
            shifted.index = shifted.index + delta
            lag_lookup[lag_name] = shifted

        self.model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.9,
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            early_stopping=True,
            random_state=42,
        )

        X = self._prepare_features(df_train, lag_lookup=lag_lookup)
        y = df_train["base_load"]

        self.model.fit(X, y)
        self.mae = mean_absolute_error(y, self.model.predict(X))
        joblib.dump({"model": self.model, "mae": self.mae}, self.path)
        self.is_fitted = True

        logger.info(
            f"[Load] Model getraind op {len(df_train)} records. MAE={self.mae:.2f}kW"
        )

    def predict(
        self,
        df_forecast: pd.DataFrame,
        lag_lookup: Optional[dict] = None,
        fallback_kw: float = 0.05,
    ) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(fallback_kw, index=df_forecast.index)

        X = self._prepare_features(df_forecast, lag_lookup=lag_lookup)
        pred = self.model.predict(X)
        return np.maximum(pred, 0.05)


class LoadForecaster:
    def __init__(self, config: Config, context: Context, database: Database):
        self.config = config
        self.context = context
        self.database = database
        self.model = LoadModel(Path(config.load_model_path))
        self.nowcaster = NowCaster()

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)

        # 1. Haal samengevoegde data (Measurements + Forecast)
        df = self.database.get_history(cutoff_date=cutoff)

        if df.empty:
            logger.warning("[Load] Geen data gevonden voor training.")
            return

        self.model.train(df)

    def _get_lag_lookup(self, timestamps: pd.Series) -> dict[str, pd.Series]:
        max_delta = max(LAG_DEFINITIONS.values())
        cutoff = timestamps.min() - max_delta - timedelta(hours=1)

        df_history = self.database.get_history(cutoff_date=cutoff)

        lag_lookup = {}

        if df_history.empty:
            for lag_name in LAG_DEFINITIONS:
                lag_lookup[lag_name] = pd.Series(dtype=float)
            return lag_lookup

        df_history["base_load"] = (
            df_history["load_actual"] - df_history["wp_actual"].fillna(0)
        ).clip(lower=0.05)

        history_sorted = (
            df_history[["timestamp", "base_load"]]
            .dropna(subset=["base_load"])
            .sort_values("timestamp")
        )

        for lag_name, delta in LAG_DEFINITIONS.items():
            lookup_df = pd.DataFrame(
                {
                    "forecast_ts": timestamps,
                    "timestamp": timestamps - delta,
                }
            ).sort_values("timestamp")

            merged = pd.merge_asof(
                lookup_df,
                history_sorted,
                on="timestamp",
                tolerance=pd.Timedelta(minutes=15),
                direction="nearest",
            )

            series = merged.set_index("forecast_ts")["base_load"]
            lag_lookup[lag_name] = series

        return lag_lookup

    def update_nowcast(self, df):
        if "load_ml" not in df.columns:
            return df

        idx_now = df["timestamp"].searchsorted(self.context.now)
        row_now = df.iloc[min(idx_now, len(df) - 1)]

        self.nowcaster.update(
            actual_kw=self.context.stable_load,
            forecasted_kw=row_now["load_ml"],
        )

        df["load_corrected"] = self.nowcaster.apply(df, self.context.now, "load_ml")

        self.context.load_bias = round(self.nowcaster.current_ratio, 3)

        return df

    def update_forecast(self, df):
        lag_lookup = self._get_lag_lookup(df["timestamp"])

        df["load_ml"] = self.model.predict(df, lag_lookup=lag_lookup)
        df["load_corrected"] = self.nowcaster.apply(df, self.context.now, "load_ml")

        self.context.load_bias = round(self.nowcaster.current_ratio, 3)

        return df
