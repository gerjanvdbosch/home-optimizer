import numpy as np
import pandas as pd
import tzlocal

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone


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


def add_cyclic_time_features(
    df: pd.DataFrame, col_name="timestamp", local_tz=True
) -> pd.DataFrame:
    """
    Voegt cyclische tijd-features toe.

    local_tz=True:  Gebruikt lokale tijd (geschikt voor Load/Base verbruik).
    local_tz=False: Gebruikt UTC (geschikt voor PV/Zon ivm voorkomen van zomertijd-sprongen).
    """
    if df is None or col_name not in df.columns:
        return df

    # 1. Converteer naar de juiste tijdzone
    if local_tz:
        tz = ZoneInfo(tzlocal.get_localzone_name())
    else:
        tz = timezone.utc

    df = df.copy()
    dt = df[col_name].dt.tz_convert(tz).dt

    # 2. Tijd van de dag (0..24 uur) met minuten precisie
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
        # We gebruiken ts.hour % 24 voor het geval we over meerdere dagen itereren
        hour = ts.hour + ts.minute / 60.0

        # --- PV MODEL (Afgestemd op log) ---
        # Zon tussen 07:15 en 19:00, piek van 1.16 kW rond 13:00
        if 7.25 <= hour <= 19.0:
            # Piek in het midden van de dag (11.75 uren zonlicht)
            solar_intensity = np.sin(np.pi * (hour - 7.25) / 11.75)
            pv = round(1.16 * solar_intensity, 2)
        else:
            pv = 0.0

        # --- TEMP MODEL (Afgestemd op log) ---
        # Koudst rond 07:00 (2.6°C), warmst rond 14:00 (10.3°C)
        # Midden is 6.45°C, amplitude is 3.85°C.
        temp = round(6.45 + 3.85 * np.sin(np.pi * (hour - 10.5) / 12), 1)

        # --- LOAD MODEL (Huisverbruik afgestemd op log) ---
        # Basislast ~0.18kW + ochtendpiek + avondpiek
        load = 0.18
        if 7.0 <= hour <= 9.0:
            # Ochtendpiek van +0.20 kW (tot ~0.38 kW)
            load += 0.20 * np.sin(np.pi * (hour - 7.0) / 2.0)
        if 17.0 <= hour <= 20.0:
            # Avondpiek van +0.53 kW (tot ~0.71 kW)
            load += 0.53 * np.sin(np.pi * (hour - 17.0) / 3.0)

        # Voeg hele lichte natuurlijke ruis toe (tussen -0.02 en +0.02)
        load = round(load + np.random.uniform(-0.02, 0.02), 2)
        # Zorg dat load nooit onrealistisch laag wordt
        load = max(0.12, load)

        # --- DATA RIJ ---
        row = {
            "timestamp": ts,
            "period_start": 0.0,
            "pv_estimate": pv,
            "pv_estimate10": round(pv * 0.8, 2),
            "pv_estimate90": round(pv * 1.2, 2),
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
            "load_corrected": load,
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Zorg dat de timestamp kolom echt datetime objecten bevat met UTC (of jouw locale zone)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df
