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
        self.features = ["inside", "outside", "compressor_freq", "supply_temp", "solar", "wind", "prev_delta"]
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
        # inside(-), outside(+), freq(+), supply(+), solar(+), wind(-), prev(+)
        # inside is min, want hoe warmer binnen, hoe moeilijker het is om NOG warmer te worden
        monotonic_cst = [-1, 1, 1, 1, 1, -1, 1]

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

    def predict_step(self, inside, outside, freq, supply_temp, solar, wind, prev_delta):
        """Voorspelt de temperatuurverandering in 15 minuten."""
        if not self.is_fitted:
            # Fallback Physics (Isolatiewaarde gok)
            loss = (inside - outside) * 0.015
            # Gain is afhankelijk van frequentie en aanvoer
            power_factor = freq / 100.0
            gain = 0.35 * power_factor
            return (gain - loss) + (0.0001 * solar)

        X = pd.DataFrame(
            [[inside, outside, float(freq), float(supply_temp), solar, wind, prev_delta]],
            columns=self.features
        )
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

    def calculate_time_to_heat(self,
                               start_temp: float,
                               target_temp: float,
                               df_forecast: pd.DataFrame,
                               max_minutes: int = 300) -> int:
        """
        Simuleert: Hoe lang duurt het om van A naar B te komen?
        Houdt rekening met de huidige NowCaster bias!
        """
        if start_temp >= target_temp:
            return 0

        sim_temp = start_temp

        # HIER GEBRUIKEN WE DE NOWCASTER
        # We starten met de huidige afwijking (bijv. tocht)
        bias = self.nowcaster.bias

        curr_prev_delta = 0.0 # Start vanuit stilstand aanname
        minutes_needed = 0

        # We simuleren kwartier voor kwartier
        steps = int(max_minutes / 15)

        for i in range(steps):
            # Pak weerdata voor dit tijdstip in de toekomst
            idx = min(i, len(df_forecast)-1)
            row = df_forecast.iloc[idx]

            # --- BOOST LOGICA ---
            # Als we moeten inhalen, doen we dat op hoog vermogen.
            # Hoe kouder buiten, hoe hoger de benodigde aanvoer (stooklijn).
            # Dit is een simpele stooklijn-gok voor de simulatie:
            boost_freq = 75.0 # Hz
            boost_supply = 40.0 if row.get('temp', 0) < 5 else 35.0

            # Vraag het model: Wat gebeurt er als we VOL GAS geven?
            delta = self.model.predict_step(
                inside=sim_temp,
                outside=row.get('temp', 0),
                freq=boost_freq,
                supply_temp=boost_supply,
                solar=row.get('pv_estimate', 0),
                wind=row.get('wind', 0),
                prev_delta=curr_prev_delta
            )

            # Update simulatie
            # We tellen de bias (het open raam) erbij op
            sim_temp += (delta + bias)
            curr_prev_delta = delta

            # De bias dooft langzaam uit.
            # We gaan ervan uit dat je dat open raam straks wel dicht doet.
            bias *= self.nowcaster.decay

            minutes_needed += 15

            if sim_temp >= target_temp:
                return minutes_needed

        return max_minutes # Niet gelukt binnen de tijd

    def can_pause_heating(self, current_temp, min_temp, duration_minutes, df_forecast) -> bool:
        """
        Simuleert: Als de WP uit gaat (voor boiler), zakt de temp dan onder het minimum?
        """
        sim_temp = current_temp
        bias = self.nowcaster.bias # Ook hier bias gebruiken!
        curr_prev_delta = 0.0
        steps = int(duration_minutes / 15)

        for i in range(steps):
            idx = min(i, len(df_forecast)-1)
            row = df_forecast.iloc[idx]

            # Simuleer met UIT (freq=0)
            # Supply temp zakt langzaam, maar voor model simulatie: laag houden
            delta = self.model.predict_step(
                inside=sim_temp,
                outside=row.get('temp', 0),
                freq=0.0,
                supply_temp=20.0, # Kamertemperatuur water
                solar=row.get('pv_estimate', 0),
                wind=row.get('wind', 0),
                prev_delta=curr_prev_delta
            )

            sim_temp += (delta + bias)
            curr_prev_delta = delta
            bias *= self.nowcaster.decay

            if sim_temp < min_temp:
                return False # Nee, huis koelt te snel af

        return True # Ja, pauzeren is veilig