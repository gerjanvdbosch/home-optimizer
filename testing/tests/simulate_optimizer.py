import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from optimizer import Optimizer

# --- MOCK CLASSES (zodat we geen dependencies missen) ---
class SolarStatus:
    WAIT = "WAIT"
    START = "START"

class SolarContext:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# --- SIMULATIE SCRIPT ---

def generate_dummy_data(start_time, hours=24):
    """Maakt een dataframe met een mooie zonne-curve."""
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(hours*4)]
    df = pd.DataFrame({"timestamp": timestamps})

    # Maak een PV curve (Sinus golf tussen 06:00 en 20:00)
    # Piek om 13:00
    x = np.array([(t.hour + t.minute/60) for t in timestamps])

    # Gaussian bell curve voor zon
    mu = 13.0 # Piek uur
    sig = 2.5 # Breedte
    pv_raw = 2.0 * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # Voeg wat ruis toe
    noise = np.random.normal(0, 0.1, len(pv_raw))
    df["power_corrected"] = (pv_raw + noise).clip(0)

    # Scenario: Wat als er HEEL weinig zon is? (Zet uncomment hieronder om te testen)
    # df["power_corrected"] = df["power_corrected"] * 0.1

    return df

def run_simulation():
    # Setup
    now = datetime(2024, 6, 21, 7, 0, 0) # 7 uur 's ochtends
    df = generate_dummy_data(now)

    # Optimizer config
    dhw_duration = 1.0 # uur
    dhw_power = 2.7    # kW
    optimizer = Optimizer(pv_max_kw=2.0, duration_hours=dhw_duration, dhw_power_kw=dhw_power)

    print(f"--- Start Simulatie @ {now} ---")
    status, context = optimizer.optimize(df, now)

    print(f"Besluit: {status}")
    print(f"Reden: {context.reason}")
    print(f"Geplande start: {context.planned_start}")
    print(f"Verwachte zonne-energie in boiler: {context.energy_best:.2f} kWh")

    # --- PLOTTEN ---
    plt.figure(figsize=(12, 6))

    # 1. Zon
    plt.plot(df["timestamp"], df["power_corrected"], label="Verwachte PV (kW)", color="orange", alpha=0.7)

    # 2. Het geplande blok
    if context.planned_start:
        start_idx = df[df["timestamp"] == context.planned_start].index[0]
        steps = int(dhw_duration * 4)

        dhw_profile = np.zeros(len(df))
        dhw_profile[start_idx : start_idx + steps] = dhw_power

        # Plot de DHW lijn
        plt.plot(df["timestamp"], dhw_profile, label="Geplande DHW Run (kW)", color="blue", linewidth=2, linestyle="--")

        # Vul de oppervlakte in die overlapt met zon (Self Consumption)
        overlap = np.minimum(df["power_corrected"], dhw_profile)
        plt.fill_between(df["timestamp"], 0, overlap, color="green", alpha=0.3, label="Direct Zonneverbruik")

        # Vul de oppervlakte in die Grid is (Grijs)
        grid_needed = np.maximum(0, dhw_profile - df["power_corrected"])
        # We plotten dit "bovenop" de zon visueel, of gewoon apart
        # Hieronder: simpel visueel trucje, vul rood in waar DHW > PV
        plt.fill_between(df["timestamp"], df["power_corrected"], dhw_profile, where=(dhw_profile > df["power_corrected"]), color="red", alpha=0.3, label="Grid Import")

    plt.title("Solar DHW Optimalisatie (CVXPY)")
    plt.ylabel("Vermogen (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Zorg dat je de optimizer class hierboven ook in dit bestand plakt of importeert
    run_simulation()