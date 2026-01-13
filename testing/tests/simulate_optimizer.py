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

    # 1. Input parameters
    current_water_temp = 30.0 # Koud water
    target_water_temp = 55.0  # Heet water
    outside_temp = 1.0

    # 2. Initialiseer optimizer en BEREKEN HET PROFIEL
    optimizer = Optimizer(pv_max_kw=4.0)

    # Dit berekent nu: [1.7, 2.0, 2.3, 2.7, 2.7, ...] afhankelijk van temperatuur
    profile = optimizer.calculate_profile(current_water_temp, target_water_temp, outside_temp=outside_temp)

    print(f"--- Start Simulatie @ {now} ---")
    print(f"Berekend profiel (kW): {profile}")
    print(f"Totale duur: {len(profile) * 15} minuten")

    status, context = optimizer.optimize(df, now, profile)

    print(f"Besluit: {status}")
    print(f"Reden: {context.reason}")
    if context.planned_start:
        print(f"Geplande start: {context.planned_start}")
        print(f"Verwachte zonne-energie in boiler: {context.energy_best:.2f} kWh")

    # --- PLOTTEN ---
    plt.figure(figsize=(12, 6))

    # 1. Zon
    plt.plot(df["timestamp"], df["power_corrected"], label="Verwachte PV (kW)", color="orange", alpha=0.7)

    # 2. Het geplande blok
    if context.planned_start:
        # Zoek waar de starttijd zit in de dataframe
        mask = df["timestamp"] == context.planned_start
        if mask.any():
            start_idx = df[mask].index[0]

            # AANPASSING: Gebruik de lengte en waarden van 'profile'
            steps = len(profile)

            # Maak een array met nullen ter grootte van de hele grafiek
            dhw_profile_plot = np.zeros(len(df))

            # Zorg dat we niet buiten de array schrijven als de start heel laat is
            end_idx = start_idx + steps
            if end_idx > len(df):
                # Knip het profiel af als het buiten de grafiek valt
                actual_steps = len(df) - start_idx
                dhw_profile_plot[start_idx:] = profile[:actual_steps]
            else:
                # Plak het berekende profiel (de ramp-up) in de plot array
                dhw_profile_plot[start_idx : end_idx] = profile

            # Plot de DHW lijn (Blauw stippellijn)
            plt.plot(df["timestamp"], dhw_profile_plot, label="Geplande DHW Profiel (kW)", color="blue", linewidth=2, linestyle="--")

            # Vul de oppervlakte in die overlapt met zon (Groen - Gratis energie)
            overlap = np.minimum(df["power_corrected"], dhw_profile_plot)
            plt.fill_between(df["timestamp"], 0, overlap, color="green", alpha=0.3, label="Direct Zonneverbruik")

            # Vul de oppervlakte in die Grid is (Rood - Import)
            # Waar DHW > Zon, moeten we importeren
            plt.fill_between(df["timestamp"], df["power_corrected"], dhw_profile_plot,
                             where=(dhw_profile_plot > df["power_corrected"]),
                             color="red", alpha=0.3, label="Grid Import")
        else:
            print("Waarschuwing: Geplande starttijd valt buiten plot bereik.")

    plt.title("Solar DHW Optimalisatie (CVXPY)")
    plt.ylabel("Vermogen (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Zorg dat je de optimizer class hierboven ook in dit bestand plakt of importeert
    run_simulation()