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

from mpc import BoilerMPC, BoilerConfig

# ==========================================
# 1. MAAK NEP DATA (Forecast)
# ==========================================
def generate_scenario(date_str="2024-06-21"):
    # Maak tijdstippen voor 24 uur (per kwartier)
    start = datetime.strptime(date_str, "%Y-%m-%d")
    timestamps = [start + timedelta(minutes=15*i) for i in range(96)]

    df = pd.DataFrame({"timestamp": timestamps})

    # Simuleer Zon (Gauss curve rond 13:00)
    # Zomer scenario: Piek van 3.0 kW
    hour_float = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["power_corrected"] = 2.0 * np.exp(-((hour_float - 13) ** 2) / 4)
    df["power_corrected"] = df["power_corrected"].clip(lower=0)

    # Simuleer Huisverbruik (Beetje ruis rond 0.2 kW)
    df["load"] = 0.2 + np.random.normal(0, 0.05, len(df))
    df["load"] = df["load"].clip(lower=0.1)

    return df

# ==========================================
# 2. DE SIMULATIE LOOP
# ==========================================
def run_simulation():
    # Configureren
    config = BoilerConfig(
        volume_liters=200,
        power_kw=2.2,

        # 1. DEADLINE FOCUS:
        target_temp=50,      # Doel om 17:00

        # 2. NACHT-STOP:
        # Zet min_temp laag (bv 20°C). Hierdoor hoeft hij 's nachts
        # niet te stoken om een ondergrens te bewaken.
        min_temp=20,
        max_temp=70,

        loss_coef=0.5,
        deadline_hour=17,

        # 3. PRIJS PRIKKELS (Sturen op Solar):
        # Maak netstroom kunstmatig duurder en zonnestroom gratis/goedkoop.
        # Dit dwingt de MPC om elk beetje zon te pakken.
        grid_price=0.22,     # Hoge straf op netstroom
        solar_price=-0.22     # Gratis eigen stroom (of zelfs negatief om te stimuleren)
    )

    mpc = BoilerMPC(config)
    df_scenario = generate_scenario()

    # Start condities (Laten we koud beginnen)
    current_temp = 40.0
    results = []

    print("--- Start Geoptimaliseerde Simulatie ---")

    for i in range(len(df_scenario)):
        now = df_scenario.iloc[i]["timestamp"]

        if i >= len(df_scenario) - 1:
            break

        df_future = df_scenario.iloc[i:].copy()

        # Oplossen
        solution = mpc.solve(df_future, current_temp, df_scenario.iloc[i]["load"])

        if solution is not None:
            power_cmd = solution.iloc[0]["mpc_power_kw"]

            # Slimme schakelaar logica:
            # Alleen aan als MPC echt vermogen vraagt (> 0.1 kW)
            is_heating = 1 if power_cmd > 0.1 else 0
            actual_power = config.power_kw if is_heating else 0.0
        else:
            print(f"{now.time()} - Solver failed!")
            actual_power = 0.0
            is_heating = 0

        # Fysica update
        dt = 0.25 # uur
        heat_added = (actual_power * dt) / (config.volume_liters * 0.001163)
        heat_lost = config.loss_coef * dt
        new_temp = current_temp + heat_added - heat_lost

        results.append({
            "timestamp": now,
            "temp": current_temp,
            "solar": df_scenario.iloc[i]["power_corrected"],
            "heating": actual_power,
            "mpc_cmd": power_cmd if solution is not None else 0
        })

        if is_heating:
             print(f"{now.strftime('%H:%M')} | Solar: {df_scenario.iloc[i]['power_corrected']:.2f}kW -> BOILER AAN")

        current_temp = new_temp

    return pd.DataFrame(results)

# ==========================================
# 3. VISUALISATIE
# ==========================================
if __name__ == "__main__":
    res = run_simulation()

    # Plotten
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # As 1: Temperatuur
    ax1.set_xlabel('Tijd')
    ax1.set_ylabel('Temperatuur (°C)', color='tab:red')
    ax1.plot(res['timestamp'], res['temp'], color='tab:red', label='Boiler Temp', linewidth=2)
    ax1.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='Doel (60°C)')
    ax1.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Min (40°C)')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(20, 80)

    # As 2: Vermogen (Solar & Boiler)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Vermogen (kW)', color='tab:blue')

    # Solar Area
    ax2.fill_between(res['timestamp'], res['solar'], color='yellow', alpha=0.3, label='Zonne-energie')

    # Boiler AAN blokken
    ax2.bar(res['timestamp'], res['heating'], width=0.01, color='blue', alpha=0.3, label='Boiler AAN')

    # MPC Wens lijn
    ax2.plot(res['timestamp'], res['mpc_cmd'], color='blue', linestyle=':', label='MPC Wens')

    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 4)

    # Titel & Grid
    plt.title('MPC Simulatie: Solar Optimalisatie & Deadline')
    fig.tight_layout()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Legenda combineren
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.show()