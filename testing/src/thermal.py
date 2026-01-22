import pandas as pd
import numpy as np
import joblib
import logging
import os
from pathlib import Path
from datetime import timedelta, datetime
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

# =========================================================
# 1. NOWCASTER (Korte termijn correctie)
# =========================================================
class ThermalNowCaster:
    def __init__(self):
        self.bias = 0.0
        # Hoe snel vergeten we de afwijking?
        # 0.8 betekent: na 1 stap telt hij nog voor 80% mee, na 2 voor 64%, etc.
        # Dit is belangrijk omdat een open raam meestal tijdelijk is.
        self.decay = 0.8

    def update(self, predicted_delta: float, actual_delta: float):
        """
        Update de bias op basis van realiteit.
        predicted_delta: Wat het model dacht dat er zou gebeuren.
        actual_delta: Wat er echt gebeurde (huidige temp - vorige temp).
        """
        error = actual_delta - predicted_delta
        # Low pass filter: We nemen 50% van de nieuwe fout over
        self.bias = (self.bias * 0.5) + (error * 0.5)
        # logger.debug(f"[ThermalNowCaster] Bias updated to: {self.bias:.4f}")

# =========================================================
# 2. THERMAL MODEL (Het Fysica Brein)
# =========================================================
class ThermalModel:
    def __init__(self, path: Path):
        self.path = path
        self.model = None
        self.is_fitted = False

        # FEATURES:
        # 1. inside: Hoe warmer binnen, hoe langzamer het opwarmt (Newton's cooling law)
        # 2. outside: Hoe kouder buiten, hoe groter het verlies
        # 3. compressor_freq: Hoe hard werkt de warmtepomp? (0 - 100 Hz)
        # 4. supply_temp: Hoe warm is het water? (30 - 50 °C)
        # 5. solar: Zoninstraling (W/m2)
        # 6. wind: Windkracht (m/s)
        # 7. prev_delta: Momentum van de vloer
        # 8. hvac_mode: Verwarmen / SWW / Uit
        self.features = ["inside", "outside", "compressor_freq", "supply_temp", "solar", "wind", "prev_delta", "hvac_mode"]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.model = data.get("model")
                self.is_fitted = True
            except Exception as e:
                logger.error(f"[Thermal] Kon model niet laden: {e}")

    def train(self, df_history: pd.DataFrame):
        """Traint het model op historische data."""
        if df_history.empty: return

        df = df_history.copy().sort_values('timestamp')

        # Feature Engineering
        df['next_temp'] = df['inside'].shift(-1)
        df['target_delta'] = df['next_temp'] - df['inside']
        df['prev_delta'] = (df['inside'] - df['inside'].shift(1)).fillna(0)

        # Filter: Alleen stappen van ~15 min
        df['dt_min'] = (df['timestamp'].shift(-1) - df['timestamp']).dt.total_seconds() / 60
        df = df[(df['dt_min'] >= 10) & (df['dt_min'] <= 20)].dropna()

        if len(df) < 500:
            logger.warning("[Thermal] Te weinig data (<500 samples) om te trainen.")
            return

        X = df[self.features]
        y = df['target_delta']

        # Constraints (Natuurwetten afdwingen):
        # inside(-), outside(+), freq(+), supply(+), solar(+), wind(-), prev(+), hvac(+)
        # inside is min, want hoe warmer binnen, hoe moeilijker het is om NOG warmer te worden
        monotonic_cst = [-1, 1, 1, 1, 1, -1, 1, 1]

        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            monotonic_cst=monotonic_cst,
            learning_rate=0.05,
            max_leaf_nodes=31,
            early_stopping=True
        )
        self.model.fit(X, y)

        joblib.dump({"model": self.model}, self.path)
        self.is_fitted = True
        logger.info(f"[Thermal] Model getraind op {len(df)} samples.")

    def predict_step(self, inside, outside, freq, supply_temp, solar, wind, prev_delta, hvac_mode=1):
        """Voorspelt de temperatuurverandering in 15 minuten."""
        if not self.is_fitted:
            # Fallback Physics (Isolatiewaarde gok)
            loss = (inside - outside) * 0.015
            gain = (freq / 100.0) * 0.35 * hvac_mode
            return (gain - loss) + (0.0001 * solar)

        X = pd.DataFrame([[inside, outside, float(freq), float(supply_temp), solar, wind, prev_delta, int(hvac_mode)]], columns=self.features)
        return self.model.predict(X)[0]

# =========================================================
# 3. THERMAL PLANNER (De Simulator)
# =========================================================
class ThermalPlanner:
    def __init__(self, model: ThermalModel, nowcaster: ThermalNowCaster):
        self.model = model
        self.nowcaster = nowcaster

    def calculate_start_time(self,
                             current_temp: float,
                             target_temp: float,
                             deadline: datetime,
                             df_forecast: pd.DataFrame) -> tuple[datetime, int]:
        """
        Wrappertje: Berekent hoe laat je moet beginnen om de deadline te halen.
        """
        minutes_needed = self.calculate_time_to_heat(
            current_temp, target_temp, df_forecast
        )
        start_time = deadline - timedelta(minutes=minutes_needed)
        return start_time, minutes_needed

    def calculate_run_profile(self,
                              start_temp: float,
                              target_temp: float,
                              df_forecast: pd.DataFrame,
                              is_dhw: bool = False) -> tuple[int, np.ndarray]:
        """
        COMBINATIE FUNCTIE:
        Simuleert het verloop van A naar B met behulp van Machine Learning.

        Returns:
            duration (int): Aantal minuten
            profile (np.array): Array met kW verbruik per kwartier (voor de Optimizer)
        """
        if start_temp >= target_temp:
            return 0, np.array([])

        sim_temp = start_temp
        bias = self.nowcaster.bias
        curr_prev_delta = 0.0

        # We bouwen het profiel op terwijl we simuleren
        power_profile = []

        # Maximaal 5 uur simuleren (beveiliging)
        max_steps = int(300 / 15)

        for i in range(max_steps):
            # 1. Haal weerdata
            idx = min(i, len(df_forecast)-1)
            row = df_forecast.iloc[idx]
            outside_t = row.get('temp', 10)

            # 2. Bepaal Power Strategie (Boost curve)
            # Professioneel: COP is afhankelijk van temperatuurverschil (Lift)
            # Dit maakt de schatting van het elektriciteitsverbruik (kW) nauwkeuriger.
            if is_dhw:
                # Boiler: Hoe warmer het water, hoe lager de COP, hoe meer stroom nodig voor zelfde warmte
                # Simpele curve: begint op 2.2kW, eindigt op 3.0kW
                progress = min(1.0, (sim_temp - 40) / (target_temp - 40)) if target_temp > 40 else 0
                kw_input = 2.2 + (0.8 * progress)

                # Vertaal kW naar Frequency (schatting voor ML model input)
                # Stel 3.5kW = 100Hz
                boost_freq = min(100.0, (kw_input / 3.5) * 100.0)
                boost_supply = 55.0 # Boiler stookt altijd heet
            else:
                # Vloer: Stooklijn afhankelijk van buiten
                kw_input = 1.5 # Vloer is vaak stabieler vermogen
                boost_supply = 40.0 if outside_t < 5 else 30.0
                boost_freq = 60.0

            # 3. Voorspel Temperatuur Delta (ML Model)
            delta = self.model.predict_step(
                inside=sim_temp,
                outside=outside_t,
                freq=boost_freq,
                supply_temp=boost_supply,
                solar=row.get('pv_estimate', 0),
                wind=row.get('wind', 0),
                prev_delta=curr_prev_delta,
                hvac_mode=1
            )

            # 4. Update Simulatie State
            sim_temp += (delta + bias)
            curr_prev_delta = delta
            bias *= self.nowcaster.decay

            # Voeg berekend vermogen toe aan profiel
            power_profile.append(kw_input)

            # 5. Check of we er zijn
            if sim_temp >= target_temp:
                break

        # Converteer lijst naar numpy array voor CVXPY optimizer
        return len(power_profile) * 15, np.array(power_profile)
