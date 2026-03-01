import numpy as np
import pandas as pd

from datetime import datetime, timedelta


def round_half(x):
    return round(x * 2) / 2


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def to_kw(watts):
    try:
        value = float(watts)
        return value / 1000.0
    except (TypeError, ValueError):
        return 0.0


def add_cyclic_time_features(df: pd.DataFrame, col_name="timestamp") -> pd.DataFrame:
    """
    Voegt cyclische tijd-features toe (hour, day, doy) als sin/cos paren.
    Neemt minuten mee voor hogere precisie.
    """
    if df is None or col_name not in df.columns:
        return df

    # Huidige tijdzone van de omgeving
    tz = datetime.now().astimezone().tzinfo

    df = df.copy()
    dt = df[col_name].dt.tz_convert(tz).dt

    # 1. Tijd van de dag (0..24 uur)
    # We voegen minuten toe voor precisie (bv. 14:30 wordt 14.5)
    precise_hour = dt.hour + (dt.minute / 60.0)

    df["hour_sin"] = np.sin(2 * np.pi * precise_hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * precise_hour / 24.0)

    # 2. Dag van de week (0..6, Maandag=0)
    df["day_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7.0)

    # 3. Dag van het jaar (1..366) - Seizoenen
    df["doy_sin"] = np.sin(2 * np.pi * dt.dayofyear / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * dt.dayofyear / 366.0)

    return df


def generate_sunny_forecast(start_time=None):
    if start_time is None:
        # Start op het hele uur van vandaag
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    # 3 dagen in kwartieren = 3 * 24 * 4 = 288 stappen
    periods = 288
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(periods)]

    data = []

    for ts in timestamps:
        # Uur van de dag als float (bijv 14.25 voor 14:15)
        hour = ts.hour + ts.minute / 60.0

        # --- PV MODEL (Zonnig) ---
        # Zon tussen 07:00 en 19:00, piek om 13:00
        if 7.0 <= hour <= 19.0:
            # Sinus curve voor zonlicht
            solar_intensity = np.sin(np.pi * (hour - 7) / 12)
            pv = round(2.0 * solar_intensity, 3)  # Piek van 3.5 kW
        else:
            pv = 0.0

        # --- TEMP MODEL ---
        # Koudst om 05:00, warmst om 15:00
        temp = round(8 + 7 * np.sin(np.pi * (hour - 9) / 12), 1)

        # --- LOAD MODEL (Huisverbruik) ---
        # Basislast 0.15kW + piek om 08:00 en 18:00
        load = 0.15
        if 7.0 <= hour <= 9.0:
            load += 0.4 * np.sin(np.pi * (hour - 7) / 2)
        if 17.0 <= hour <= 20.0:
            load += 0.8 * np.sin(np.pi * (hour - 17) / 3)
        load = round(load + np.random.uniform(0, 0.05), 3)

        # --- DATA RIJ ---
        row = {
            "timestamp": ts,
            "period_start": 0.0,
            "pv_estimate": pv,
            "pv_estimate10": round(pv * 0.8, 3),
            "pv_estimate90": round(pv * 1.2, 3),
            "temp": temp,
            "cloud": 0.0 if pv > 0 else 10.0,  # Helder
            "wind": round(np.random.uniform(2, 5), 1),
            "radiation": round(pv * 250, 1),
            "diffuse": round(pv * 50, 1),
            "tilted": round(pv * 300, 1),
            "power_ml": pv,
            "power_ml_raw": pv,
            "power_corrected": pv,
            "load_ml": load,
            "load_corrected": load
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Zorg dat de timestamp kolom echt datetime objecten bevat met UTC (of jouw locale zone)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    return df
