import numpy as np
import pandas as pd
import joblib
import logging

from client import HAClient
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from utils import add_cyclic_time_features
from context import Context

logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, client: HAClient, context: Context):
        self.client = client

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


class TempNowCaster:
    """Real-time correctie voor temperatuur (additieve bias)."""

    def __init__(self, decay_hours: float = 4.0):
        self.decay_hours = decay_hours
        self.current_bias = 0.0

    def update(self, actual_temp: float, forecasted_temp: float):
        if actual_temp is None or forecasted_temp is None:
            return
        error = actual_temp - forecasted_temp
        # Low-pass filter voor stabiliteit
        alpha = 0.15 if abs(error) > 1.0 else 0.05
        self.current_bias = (1 - alpha) * self.current_bias + (alpha * error)

    def apply(
        self, df: pd.DataFrame, current_time: pd.Timestamp, col_name: str
    ) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        delta_hours = (df["timestamp"] - current_time).dt.total_seconds().clip(
            lower=0
        ) / 3600.0
        decay_factors = np.exp(-delta_hours / self.decay_hours)
        # Pas bias toe op de toekomst, uitstervend over tijd
        return df[col_name] + (self.current_bias * decay_factors)


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

    def predict(self, df_forecast: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            return df_forecast["temp"]

        df_feat = add_cyclic_time_features(df_forecast.copy(), "timestamp")
        df_feat["temp_api"] = df_feat["temp"]
        df_feat["solar"] = df_feat.get(
            "power_corrected", df_feat.get("pv_estimate", 0.0)
        )

        X = df_feat[self.feature_cols].fillna(0)
        return pd.Series(self.model.predict(X), index=df_forecast.index)


class TemperatureForecaster:
    """Orkestrator voor Temperatuur ML + Nowcasting."""

    def __init__(self, config, context, database):
        self.context = context
        self.database = database
        self.model = TempModel(Path(config.temp_model_path))
        self.nowcaster = TempNowCaster(decay_hours=4.0)
        self.model.load()

    def train(self, days_back: int = 730):
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        df = self.database.get_history(cutoff_date=cutoff)
        self.model.train(df)

    def update_nowcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if "temp_ml" not in df.columns:
            return df

        # 1. Update bias op basis van huidige meting
        idx_now = df["timestamp"].searchsorted(self.context.now)
        row_now = df.iloc[min(idx_now, len(df) - 1)]

        self.nowcaster.update(
            actual_temp=getattr(self.context, "outside_temp", None),
            forecasted_temp=row_now["temp_ml"],
        )

        # 2. Pas bias toe op de hele horizon
        df["temp"] = self.nowcaster.apply(df, self.context.now, "temp_ml")

        # 3. Update dashboard info
        self.context.temp_bias = round(self.nowcaster.current_bias, 2)
        return df

    def update_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # 1. ML basis-voorspelling
        df["temp_ml"] = self.model.predict(df)
        # 2. Direct door naar nowcasting voor sensor-aansluiting
        return self.update_nowcast(df)
