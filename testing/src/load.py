import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator
from utils import add_cyclic_time_features
from config import Config
from context import Context

logger = logging.getLogger(__name__)

class NowCaster:
    def __init__(self, decay_hours: float = 2.0):
        self.decay_hours = decay_hours
        self.current_ratio = 0.0

    def update(self, actual_kw: float, forecasted_kw: float):
        """
        Update de bias met een dynamisch berekende alpha.
        Dit zorgt voor vloeiende overgangen en filtert pieken wiskundig weg.
        """
        if actual_kw is None or forecasted_kw is None:
            return

        # Bereken de ruwe fout
        error = actual_kw - forecasted_kw
        alpha = 0.0

        if error <= 0:
            # SCENARIO 1: Load is lager dan verwacht (Apparaat uit)
            # We volgen dit agressief, want verbruik 'zakt' niet zomaar als ruis.
            alpha = 0.25

        else:
            # SCENARIO 2: Load is hoger dan verwacht (Mogelijk een piek)
            # We gebruiken een Gaussian Decay formule.
            # - Bij kleine error (0.1) is exp bijna 1 -> alpha ≈ 0.15
            # - Bij grote error (2.0) is exp bijna 0 -> alpha ≈ 0.00

            decay_factor = np.exp(- (error / 1.5) ** 2)
            alpha = 0.15 * decay_factor

            # Zorg dat alpha nooit helemaal 0 wordt (blijf altijd een beetje leren)
            alpha = max(0.01, alpha)

        # Update de bias met de berekende alpha
        self.current_ratio = (1 - alpha) * self.current_bias_kw + (alpha * error)

    def apply(
        self, df: pd.DataFrame, current_time: datetime, col_name: str
    ) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        # Tijdsverschil
        delta_hours = (df["timestamp"] - current_time).dt.total_seconds() / 3600.0
        delta_hours = delta_hours.clip(lower=0)

        # Decay: De bias sterft uit naar 0 naarmate we verder in de toekomst kijken
        decay_factors = np.exp(-delta_hours / self.decay_hours)

        # Correctie vector (Additive)
        correction_vector = self.current_bias_kw * decay_factors

        # Pas toe op de voorspelling
        corrected_series = df[col_name] + correction_vector

        # Veiligheidsfloor: Verbruik kan nooit negatief zijn.
        # 0.05 kW (50W) is een redelijke 'huis slaapt' ondergrens.
        return corrected_series.clip(lower=0.05)

class LoadModel:
    def __init__(self, path: Path):
        self.path = path
        self.model: Optional[BaseEstimator] = None
        self.is_fitted = False
        self.feature_cols = [
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
            "doy_sin", "doy_cos",
            "temp"
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
        # Filter rijen waar we geen load of temp data hebben
        df_train = df_history.dropna(subset=["load_actual"]).copy()

        X = self._prepare_features(df_train)
        y = df_train["load_actual"]

        # HistGradientBoosting is vaak sneller en beter dan RandomForest voor tijdreeksen
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            early_stopping=True,
            random_state=42
        )

        self.model.fit(X, y)
        joblib.dump({"model": self.model}, self.path)
        self.is_fitted = True
        logger.info(f"[Load] Model getraind.")

    def predict(self, df_forecast: pd.DataFrame, fallback_kw: float = 0.15) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(fallback_kw, index=df_forecast.index)

        X = self._prepare_features(df_forecast)
        pred = self.model.predict(X)
        return np.maximum(pred, 0.05)


class LoadForecaster:
    def __init__(self, config: Config, context: Context):
        self.config = config
        self.context = context
        self.model = LoadModel(Path(config.load_model_path))
        self.nowcaster = LoadNowCaster(decay_hours=2.0)

    def update(self, current_time: datetime, current_load_kw: float) -> pd.DataFrame:
        """
        current_load_kw: Bij voorkeur een gemiddelde van de laatste 60-180 seconden!
        """
        forecast_df = self.context.forecast_df
        df_calc = forecast_df.copy()

        # 1. Base Prediction (Machine Learning)
        df_calc["load_ml"] = self.model.predict(df_calc)

        # 2. Update NowCaster state (Bias berekenen)
        # We halen de voorspelling voor 'nu' op om de error te bepalen
        idx_now = df_calc["timestamp"].searchsorted(current_time)
        row_now = df_calc.iloc[min(idx_now, len(df_calc) - 1)]

        predicted_now = df_calc.iloc[idx_now]["load_ml"]

        self.nowcaster.update(actual_kw=current_load_kw, forecasted_kw=predicted_now)

        # 3. Apply NowCaster (Correctie projecteren over de tijd)
        df_calc["load_corrected"] = self.nowcaster.apply(
            df_calc, current_time, "load_ml"
        )

        self.context.load_bias = round(self.nowcaster.current_ratio, 3)
        self.context.forecast_df = df_calc

        return df_calc
