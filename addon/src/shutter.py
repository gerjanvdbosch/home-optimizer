import logging
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from utils import add_cyclic_time_features

logger = logging.getLogger(__name__)


class ShutterPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None
        self.is_fitted = False
        # Het model leert jouw gedrag op basis van tijd, buitentemperatuur en zon
        self.features = [
            "temp",
            "solar",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "doy_sin",
            "doy_cos",
        ]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
                self.is_fitted = True
                logger.info("[Shutter] Model geladen.")
            except Exception as e:
                logger.warning(f"[Shutter] Model laden mislukt: {e}")

    def train(self, df: pd.DataFrame):
        df_train = df.copy().dropna(subset=["shutter_room"])

        if len(df_train) < 10:
            logger.info("[Shutter] Te weinig data om te trainen.")
            return

        df_train = add_cyclic_time_features(df_train, "timestamp")
        df_train["solar"] = df_train.get("pv_actual", 0.0)

        X = df_train[self.features].fillna(0)
        y = df_train["shutter_room"]

        # Gebruik een RandomForestRegressor om het percentage (0-100) te voorspellen
        self.model = RandomForestRegressor(
            n_estimators=50, max_depth=8, min_samples_leaf=10, random_state=42
        ).fit(X, y)

        self.is_fitted = True
        joblib.dump(self.model, self.path)
        logger.info("[Shutter] Succesvol getraind op historische data.")

    def predict(self, forecast_df, current_shutter_open):
        """Voorspelt wat het rolluik de komende 24 uur gaat doen."""
        if forecast_df is None:
            return np.array([current_shutter_open])

        if not self.is_fitted or self.model is None:
            # Fallback: als we niks weten, gokken we dat hij z'n huidige stand behoudt
            return np.full(len(forecast_df), current_shutter_open)

        df_feat = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        df_feat["solar"] = df_feat.get(
            "power_corrected", df_feat.get("pv_estimate", 0.0)
        )

        for col in self.features:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        # Voorspel het percentage (0 tot 100)
        predicted_shutters = self.model.predict(df_feat[self.features])
        predicted_shutters = np.clip(predicted_shutters, 0.0, 100.0)

        # ── Anker korte-termijn voorspelling aan huidige gemeten rolluikstand ──
        # Het ML-model is getraind op historisch gemiddeld gedrag. Bij de huidige
        # meting weten we de werkelijke stand met zekerheid; gebruik die als
        # startpunt en laat de voorspelling exponentieel vervagen naar het model.
        # τ = 4 stappen ≈ 1 uur (typische reactietijd bij veranderende zon).
        n = len(predicted_shutters)
        tau = 4.0  # stappen
        t = np.arange(n, dtype=float)
        anchor_weight = np.exp(-t / tau)
        blended = (
            anchor_weight * float(current_shutter_open)
            + (1.0 - anchor_weight) * predicted_shutters
        )
        return np.clip(blended, 0.0, 100.0)
