import numpy as np
import pandas as pd
import joblib
import logging

from client import HAClient
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from utils import add_cyclic_time_features
from context import Context, HvacMode

logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, client: HAClient, context: Context):
        self.client = client
        self.context = context

    def get_forecast(self):
        minutely = self.client.get_weather()

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(minutely["time"], utc=True),
                "temp": minutely["temperature_2m"],
                "cloud": minutely["cloud_cover"],
                "wind": minutely["wind_speed_10m"],
                "radiation": minutely["shortwave_radiation_instant"],
                "diffuse": minutely["diffuse_radiation_instant"],
                "tilted": minutely["global_tilted_irradiance_instant"],
            }
        )

        logger.debug("[Weather] API-update succesvol")
        return df


class TempModel:
    """Het ML model dat lokale afwijkingen leert."""

    def __init__(self, path: Path):
        self.path = path
        self.model = None
        self.is_fitted = False
        self.feature_cols = [
            "temp_api",
            "solar",
            "cloud",
            "wind",
            "hvac_mode",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "doy_sin",
            "doy_cos",
        ]

    def load(self):
        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
                self.is_fitted = True
                logger.info("[TempML] Model geladen.")
            except Exception:
                logger.error("[TempML] Model corrupt.")

    def train(self, df_history: pd.DataFrame):
        df_train = df_history.dropna(
            subset=["temp", "outside_temp", "pv_actual"]
        ).copy()
        if len(df_train) < 50:
            return

        df_train = add_cyclic_time_features(df_train, "timestamp")
        df_train["temp_api"] = df_train["temp"]
        df_train["solar"] = df_train["pv_actual"]

        X = df_train[self.feature_cols].fillna(0)
        y = df_train["outside_temp"]

        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, early_stopping=True, random_state=42
        ).fit(X, y)

        joblib.dump(self.model, self.path)
        self.is_fitted = True
        logger.info(f"[TempML] Getraind op {len(df_train)} records.")

    def predict(
        self, df_forecast: pd.DataFrame, mode_override: int = None
    ) -> pd.Series:
        if not self.is_fitted:
            return df_forecast["temp"]

        df_feat = add_cyclic_time_features(df_forecast.copy(), "timestamp")
        df_feat["temp_api"] = df_feat["temp"]
        df_feat["solar"] = df_feat.get(
            "power_corrected", df_feat.get("pv_estimate", 0.0)
        )

        # Als we een override opgeven (bijv. 0 voor 'schoon weer'), gebruiken we die.
        # Anders gebruiken we de hvac_mode die al in de df staat.
        if mode_override is not None:
            df_feat["hvac_mode"] = mode_override
        else:
            df_feat["hvac_mode"] = df_feat.get("hvac_mode", HvacMode.OFF.value)

        X = df_feat[self.feature_cols].fillna(0)
        return pd.Series(self.model.predict(X), index=df_forecast.index)


class TemperatureForecaster:
    def __init__(self, config, context, database):
        self.path = Path(config.temp_model_path)
        self.context = context
        self.database = database
        self.model = None
        self.is_fitted = False

        # Nowcast state
        self.current_bias = 0.0
        self.decay_hours = 4.0

        self.features = [
            "temp_api",
            "solar",
            "cloud",
            "wind",
            "hvac_mode",
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
                logger.info("[Temperature] Model geladen.")
            except Exception as e:
                logger.warning(f"[Temperature] Model laden mislukt: {e}")

    def train(self, days_back: int = 730):
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        df = self.database.get_history(cutoff_date=cutoff)

        df_train = df.copy().dropna(subset=["temp", "outside_temp", "pv_actual"])
        if len(df_train) < 50:
            logger.info("[Temperature] Te weinig data om te trainen.")
            return

        df_train = add_cyclic_time_features(df_train, "timestamp")
        df_train["temp_api"] = df_train["temp"]
        df_train["solar"] = df_train["pv_actual"]

        X = df_train[self.features].fillna(0)
        y = df_train["outside_temp"]

        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, early_stopping=True, random_state=42
        ).fit(X, y)

        self.is_fitted = True
        joblib.dump(self.model, self.path)
        logger.info(f"[Temperature] Getraind op {len(df_train)} records.")

    def _predict_with_mode(self, df, mode_value):
        """Helper om predictie te doen met een specifieke HVAC mode."""
        df_feat = add_cyclic_time_features(df.copy(), "timestamp")
        df_feat["temp_api"] = df_feat["temp"]
        df_feat["solar"] = df_feat.get(
            "power_corrected", df_feat.get("pv_estimate", 0.0)
        )
        df_feat["hvac_mode"] = mode_value

        return self.model.predict(df_feat[self.features].fillna(0))

    def update_nowcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or "temp_ml" not in df.columns:
            return df

        current_time = self.context.now
        idx_now = df["timestamp"].searchsorted(current_time)
        row_now = df.iloc[[min(idx_now, len(df) - 1)]]

        # 1. Bepaal wat de sensor NU hoort te meten (inclusief huidige hitte van HP)
        current_mode = getattr(self.context, "hvac_mode", HvacMode.OFF).value
        expected_sensor_now = self._predict_with_mode(row_now, current_mode)[0]

        # 2. Bereken weers-bias (EMA update)
        actual_temp = getattr(self.context, "outside_temp", None)
        if actual_temp is not None:
            error = actual_temp - expected_sensor_now
            alpha = 0.1  # Lage alpha voor stabiliteit
            self.current_bias = (1 - alpha) * self.current_bias + (alpha * error)

        # 3. Pas bias toe op de SCHONE voorspelling (temp_ml is al met mode 0 berekend)
        delta_hours = (df["timestamp"] - current_time).dt.total_seconds().clip(
            lower=0
        ) / 3600.0
        decay_factors = np.exp(-delta_hours / self.decay_hours)

        df["temp"] = df["temp_ml"] + (self.current_bias * decay_factors)

        self.context.temp_bias = round(self.current_bias, 2)
        return df

    def update_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not self.is_fitted:
            return df

        # Bereken de basis-voorspelling ALTIJD met mode 0 (omgevingstemp zonder HP invloed)
        df["temp_ml"] = self._predict_with_mode(df, mode_value=HvacMode.OFF)

        return self.update_nowcast(df)
