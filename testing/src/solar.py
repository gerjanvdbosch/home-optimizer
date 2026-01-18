import numpy as np
import pandas as pd
import joblib
import shap
import logging

from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from utils import add_cyclic_time_features
from typing import Dict
from config import Config
from context import Context

logger = logging.getLogger(__name__)


class NowCaster:
    def __init__(self, model_mae: float, pv_max_kw: float, decay_hours: float = 3.0):
        self.decay_hours = decay_hours
        self.pv_max_kw = pv_max_kw
        self.current_ratio = 1.0

        # Veiligheidsmarges
        error_margin = (2.5 * model_mae) / (pv_max_kw + 0.1)
        self.max_ratio = 1.0 + error_margin
        self.min_ratio = max(0.2, 1.0 - error_margin)

    def update(self, actual_kw: float, forecasted_kw: float):
        # 1. Gating: Negeer ruis in de nacht/ochtend (<5% van systeem max)
        # We laten de ratio langzaam 'uitdoven' naar 1.0 als er geen zon is.
        if forecasted_kw < 0.05 * self.pv_max_kw:
            self.current_ratio = (0.95 * self.current_ratio) + (0.05 * 1.0)
            return

        # 2. Ratio met floor
        # Voorkom delen door 0, maar gebruik max(0.05) om extreme ratio's te dempen
        raw_ratio = actual_kw / max(forecasted_kw, 0.05)
        raw_ratio = np.clip(raw_ratio, self.min_ratio, self.max_ratio)

        # 3. Confidence Weighting (Cruciaal voor ochtendstabiliteit)
        # Bij laag vermogen (ochtend) vertrouwen we de meting minder -> lage alpha
        confidence = np.clip(forecasted_kw / (self.pv_max_kw * 0.4), 0.1, 1.0)
        alpha = 0.3 * confidence

        # Update (Low pass filter)
        self.current_ratio = (1 - alpha) * self.current_ratio + alpha * raw_ratio

    def apply(
        self, df: pd.DataFrame, now: datetime, col_name: str, actual_pv: float = None
    ) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        # Tijdsverschil
        delta_hours = (df["timestamp"] - now).dt.total_seconds() / 3600.0
        delta_hours = delta_hours.clip(lower=0)

        # 1. Asymmetrische Decay
        # Negatieve bias (wolk) vergeten we sneller dan positieve bias
        effective_decay = (
            self.decay_hours if self.current_ratio >= 1.0 else (self.decay_hours * 0.5)
        )
        decay_factors = np.exp(-delta_hours / effective_decay)

        # Basis correctie (vermenigvuldigen)
        correction_vector = 1.0 + (self.current_ratio - 1.0) * decay_factors
        corrected_series = df[col_name] * correction_vector

        # 2. Short-term Persistence (De '0-Forecast' Fix)
        # Dit repareert het gat als forecast=0 maar actual=2kW
        if actual_pv is not None and actual_pv > 0.1:
            idx_now = df["timestamp"].searchsorted(now)
            idx_now = min(idx_now, len(df) - 1)

            # Vergelijk kW met kW
            current_model_val = corrected_series.iloc[idx_now]
            error_gap = max(0, actual_pv - current_model_val)

            if error_gap > 0.1:
                # Voeg het verschil toe, uitstervend over ~1 uur
                steps = np.maximum(0, np.arange(len(df)) - idx_now)
                boost_vector = error_gap * np.exp(-steps / 4.0)
                corrected_series += boost_vector

        return corrected_series.clip(0, self.pv_max_kw)


class SolarModel:
    def __init__(self, path: Path):
        self.path = path
        self.model: Optional[BaseEstimator] = None
        self.mae = 0.2
        self.feature_cols = [
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "pv_estimate",
            "pv_estimate10",
            "pv_estimate90",
            "uncertainty",
            "temp",
            "cloud",
            "wind",
            "radiation",
            "diffuse",
            "tilted",
        ]
        self.is_fitted = False
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                if isinstance(data, dict):
                    self.model = data.get("model")
                    self.mae = data.get("mae", 0.2)
                else:
                    self.model = data
                self.is_fitted = True
            except Exception:
                logger.error("[Solar] Model corrupt.")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = add_cyclic_time_features(df, col_name="timestamp")
        df["uncertainty"] = df["pv_estimate90"] - df["pv_estimate10"]
        X = df.reindex(columns=self.feature_cols)

        return X.apply(pd.to_numeric, errors="coerce")

    def train(self, df_history: pd.DataFrame, system_max: float):
        df_train = df_history.dropna(subset=["pv_actual"]).copy()

        # Target berekenen op de UUR data
        # Dit is veel stabieler dan op kwartierdata
        df_hourly = df_train.set_index("timestamp").resample("1H").mean().reset_index()

        X = self._prepare_features(df_hourly)
        y = df_hourly["pv_actual"].clip(0, system_max)

        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )

        self.model.fit(X, y)
        self.mae = mean_absolute_error(y, self.model.predict(X))
        joblib.dump({"model": self.model, "mae": self.mae}, self.path)
        self.is_fitted = True

        logger.info(f"[Solar] Model getraind met MAE: {self.mae:.3f} kW")

    def predict(self, df_forecast: pd.DataFrame, model_ratio: float = 0.7):
        raw_solcast = df_forecast["pv_estimate"].fillna(0)

        if not self.is_fitted:
            return pd.DataFrame(
                {"prediction": raw_solcast, "prediction_raw": raw_solcast}
            )

        X = self._prepare_features(df_forecast)
        # 1. De "Pure" ML voorspelling
        pred_ml = np.maximum(self.model.predict(X), 0)
        # 2. De "Blended" veiligheidsmix
        solcast_ratio = 1.0 - model_ratio
        pred_final = (pred_ml * model_ratio) + (raw_solcast * solcast_ratio)

        return pd.DataFrame({"prediction": pred_final, "prediction_raw": pred_ml})

    def explain(self, df_row: pd.DataFrame) -> Dict[str, str]:
        if not self.is_fitted:
            return {"Info": "No model"}

        try:
            X = self._prepare_features(df_row)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            base = (
                explainer.expected_value
                if np.isscalar(explainer.expected_value)
                else explainer.expected_value[0]
            )

            y_pred = self.model.predict(X)
            result = {"base": f"{base:.2f}", "prediction": f"{y_pred[0]:.2f}"}

            for col, val in zip(self.feature_cols, shap_values[0]):
                result[col] = f"{val:+.2f}"

            return result
        except Exception:
            return {"Solar": "Shap failed"}


class SolarForecaster:
    def __init__(self, config: Config, context: Context, database: 'Database'):
        self.model = SolarModel(Path(config.solar_model_path))
        self.context = context
        self.config = config
        self.database = database
        self.nowcaster = NowCaster(model_mae=self.model.mae, pv_max_kw=config.pv_max_kw)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)

        df = self.database.get_history(cutoff_date=cutoff)

        if df.empty:
            logger.warning("[Solar] Geen historische data om model te trainen.")
            return

        self.model.train(df, system_max=self.config.pv_max_kw)

    def update(self, current_time: datetime, actual_pv: float):
        forecast_df = self.context.forecast_df
        df_calc = forecast_df.copy()

        preds = self.model.predict(df_calc, self.config.solar_model_ratio)

        df_calc["power_ml"] = preds["prediction"]
        df_calc["power_ml_raw"] = preds["prediction_raw"]

        # Bias ankerpunt (nu)
        idx_now = df_calc["timestamp"].searchsorted(current_time)
        row_now = df_calc.iloc[min(idx_now, len(df_calc) - 1)]

        self.nowcaster.update(self.context.stable_pv, row_now["power_ml"])

        df_calc["power_corrected"] = self.nowcaster.apply(
            df_calc, current_time, "power_ml", actual_pv=self.context.stable_pv
        )
        df_calc["power_corrected"] = df_calc["power_corrected"].clip(
            0, self.config.pv_max_kw
        )

        self.context.solar_bias = round(self.nowcaster.current_ratio, 3)
        self.context.forecast_df = df_calc

        return df_calc
