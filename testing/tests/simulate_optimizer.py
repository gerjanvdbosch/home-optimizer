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

    # --- TOEVOEGING: Temperatuur Curve ---
    # Temperatuur loopt vaak iets achter op de zon (piek rond 15:00 - 16:00)
    # Basis: 12 graden, Variatie: +/- 6 graden
    temp_base = 12
    temp_amp = 6
    # Sinus golf verschoven zodat piek rond 15u ligt
    temp_raw = temp_base + temp_amp * np.sin((x - 9) * (np.pi / 12))

    # Ruis op temperatuur
    temp_noise = np.random.normal(0, 0.2, len(temp_raw))
    df["temp"] = temp_raw + temp_noise

    # Scenario: Wat als er HEEL weinig zon is? (Zet uncomment hieronder om te testen)
    # df["power_corrected"] = df["power_corrected"] * 0.1

    return df

def run_simulation():
    # Setup
    now = datetime(2024, 6, 21, 7, 0, 0).replace(tzinfo=datetime.now().astimezone().tzinfo)  # Zomertijd
    df = generate_dummy_data(now)

    # 1. Input parameters
    current_water_temp = 28.0 # Koud water
    target_water_temp = 51.5  # Heet water
    outside_temp = 9.0

    # We pakken de start temperatuur uit de dataframe voor consistentie
    outside_temp_start = df.iloc[0]["temp"]

    # 2. Initialiseer optimizer en BEREKEN HET PROFIEL
    optimizer = Optimizer(pv_max_kw=2.0)

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

    # --- PLOTTEN (AANGEPAST MET DUBBELE AS) ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # AS 1 (Links): Vermogen (kW)
    ax1.set_xlabel("Tijd")
    ax1.set_ylabel("Vermogen (kW)", color="black")

    # 1. Zon
    l1 = ax1.plot(df["timestamp"], df["power_corrected"], label="Verwachte PV (kW)", color="orange", alpha=0.8)

    # 2. Het geplande blok
    l2 = [] # Placeholder voor legend
    l3 = []
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

            # Plot DHW
            l2 = ax1.plot(df["timestamp"], dhw_profile_plot, label="Geplande DHW Profiel (kW)", color="blue", linewidth=2, linestyle="--")

            # Vul gebieden
            overlap = np.minimum(df["power_corrected"], dhw_profile_plot)
            ax1.fill_between(df["timestamp"], 0, overlap, color="green", alpha=0.3, label="Direct Zonneverbruik")

            ax1.fill_between(df["timestamp"], df["power_corrected"], dhw_profile_plot,
                             where=(dhw_profile_plot > df["power_corrected"]),
                             color="red", alpha=0.3, label="Grid Import")

    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True, alpha=0.3)

    # --- AS 2 (Rechts): Temperatuur (C) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperatuur (°C)", color="tab:red")

    # Plot Temp
    l3 = ax2.plot(df["timestamp"], df["temp"], label="Buitentemperatuur (°C)", color="tab:red", linestyle=":", linewidth=2)
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # Legenda samenvoegen
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Solar DHW Optimalisatie (Grid & Temperatuur)")
    plt.show()

if __name__ == "__main__":
    # Zorg dat je de optimizer class hierboven ook in dit bestand plakt of importeert
    run_simulation()