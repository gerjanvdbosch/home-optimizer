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


class NowCaster:
    def __init__(self, decay_hours: float = 1.0):
        self.decay_hours = decay_hours
        self.current_ratio = 0.0

    def update(self, actual_kw: float, forecasted_kw: float):
        if actual_kw is None or forecasted_kw is None:
            return

        error = actual_kw - forecasted_kw
        alpha = 0.0

        if error <= 0:
            # Load is lager dan verwacht -> Snel aanpassen (agressief)
            alpha = 0.25
        else:
            # Load is hoger dan verwacht -> Voorzichtig aanpassen (kan ruis/piek zijn)
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
        self.is_fitted = False
        self.feature_cols = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "temp",
        ]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.model = data.get("model") if isinstance(data, dict) else data
                self.is_fitted = True
            except Exception:
                logger.error("[Load] Model corrupt.")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = add_cyclic_time_features(df, col_name="timestamp")
        X = df.reindex(columns=self.feature_cols)

        return X.apply(pd.to_numeric, errors="coerce")

    def train(self, df_history: pd.DataFrame):
        df_train = df_history.copy()
        df_train = df_train.dropna(subset=["pv_actual", "wp_actual"])

        if len(df_train) < 10:
            logger.warning("[Load] Niet genoeg data om model te trainen.")
            return

        # Base Load berekening
        df_train["target_load"] = df_train["load_actual"] - df_train["wp_actual"]
        df_train["target_load"] = df_train["target_load"].clip(lower=0.05)

        df_train = (
            df_train
            .dropna(subset=["target_load"])
            .reset_index()
        )

        # AANPASSING: Quantile Regression
        # We voorspellen het 90e percentiel (bovengrens).
        # Dit zorgt dat de optimizer "ruimte" houdt voor het huishouden.
        self.model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            early_stopping=True,
            random_state=42,
        )

        X = self._prepare_features(df_train)
        y = df_train["target_load"]

        self.model.fit(X, y)
        self.mae = mean_absolute_error(y, self.model.predict(X))
        joblib.dump({"model": self.model, "mae": self.mae}, self.path)
        self.is_fitted = True

        logger.info(f"[Load] Model getraind op {len(df_train)} records. MAE={self.mae:.2f}kW")

    def predict(
        self, df_forecast: pd.DataFrame, fallback_kw: float = 0.05
    ) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(fallback_kw, index=df_forecast.index)

        X = self._prepare_features(df_forecast)
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

    def update(self, current_time: datetime, current_load_kw: float):
        forecast_df = self.context.forecast_df
        df_calc = forecast_df.copy()

        # 1. Base Prediction (Machine Learning)
        df_calc["load_ml"] = self.model.predict(df_calc)

        # 2. Update NowCaster state (Bias berekenen)
        # We halen de voorspelling voor 'nu' op om de error te bepalen
        idx_now = df_calc["timestamp"].searchsorted(current_time)
        row_now = df_calc.iloc[min(idx_now, len(df_calc) - 1)]

        predicted_now = row_now["load_ml"]

        self.nowcaster.update(actual_kw=current_load_kw, forecasted_kw=predicted_now)

        # 3. Apply NowCaster (Correctie projecteren over de tijd)
        df_calc["load_corrected"] = self.nowcaster.apply(
            df_calc, current_time, "load_ml"
        )

        self.context.load_bias = round(self.nowcaster.current_ratio, 3)
        self.context.forecast_df = df_calc

        return df_calc
