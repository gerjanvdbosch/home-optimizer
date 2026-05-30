import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================================
# CONFIG
# ==========================================================

dt = 1.0
RESAMPLE = "5min"

# ==========================================================
# LOAD DATA
# ==========================================================

df_temp = pd.read_csv("data/temp.csv")
df_temp["time"] = pd.to_datetime(df_temp["time"])
df_temp["temp"] = pd.to_numeric(df_temp["°C.value"], errors="coerce")

df = df_temp.pivot_table(
    index="time",
    columns="entity_id",
    values="temp",
    aggfunc="last"
)

df = df.rename(columns={
    "danfoss_15_temperature": "room_temp",
    "ecodan_heatpump_ca09ec_buiten_temp": "outside_temp",
})

df = df.sort_index()

# ==========================================================
# PV
# ==========================================================

pv = pd.read_csv("data/pv.csv")
pv["time"] = pd.to_datetime(pv["time"])
pv["pv_power"] = pd.to_numeric(pv["W.value"], errors="coerce")
pv = pv.set_index("time").sort_index()

# ==========================================================
# SHUTTER
# ==========================================================

sh = pd.read_csv("data/shutter.csv")
sh["time"] = pd.to_datetime(sh["time"])
sh["shutter"] = pd.to_numeric(sh["cover.woonkamer.current_position"], errors="coerce")
sh = sh.set_index("time").sort_index()

# ==========================================================
# ALIGN
# ==========================================================

df = df.resample(RESAMPLE).mean().ffill()
pv = pv.resample(RESAMPLE).mean().ffill()
sh = sh.resample(RESAMPLE).mean().ffill()

df = df.join(pv[["pv_power"]], how="left")
df = df.join(sh[["shutter"]], how="left")

df = df.ffill().dropna()

# not needed now
df["heating"] = 0.0

# solar proxy
pv_max = df["pv_power"].quantile(0.99)
df["solar"] = np.clip(np.maximum(0, -df["pv_power"]) / pv_max, 0, 1)

# ==========================================================
# MODEL (2-state RC)
# ==========================================================

def simulate(params, df):

    (
        k_room_mass,
        k_mass_room,
        k_outside,
        k_heating,
        k_solar
    ) = params

    n = len(df)

    T_room = np.zeros(n)
    T_mass = np.zeros(n)

    T_room[0] = df["room_temp"].iloc[0]
    T_mass[0] = df["room_temp"].iloc[0]

    for t in range(n - 1):

        room = T_room[t]
        mass = T_mass[t]

        outside = df["outside_temp"].iloc[t]
        heating = df["heating"].iloc[t]

        solar = df["solar"].iloc[t] * (df["shutter"].iloc[t] / 100.0)

        # ==================================================
        # MASS DYNAMICS
        # ==================================================
        d_mass = (
            k_room_mass * (room - mass)
            + k_heating * heating
        )

        T_mass[t + 1] = mass + d_mass * dt

        # ==================================================
        # ROOM DYNAMICS
        # ==================================================
        d_room = (
            k_mass_room * (mass - room)
            + k_outside * (outside - room)
            + k_solar * solar
        )

        T_room[t + 1] = room + d_room * dt

    return T_room, T_mass

# ==========================================================
# LOSS
# ==========================================================

def loss(params):
    pred, _ = simulate(params, df)
    return np.mean((pred - df["room_temp"].values) ** 2)

# ==========================================================
# FIT
# ==========================================================

x0 = np.array([
    0.001,  # room <- mass
    0.01,   # mass <- room
    0.001,  # outside
    0.0,    # heating
    0.1     # solar
])

bounds = [
    (0, 0.05),
    (0, 0.2),
    (0, 0.05),
    (0, 2.0),
    (0, 2.0)
]

result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)

print("Fitted parameters:")
print(result.x)

# ==========================================================
# SIMULATIE
# ==========================================================

best = result.x
pred, mass = simulate(best, df)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(12, 6))

plt.plot(df.index, df["room_temp"], label="Measured room")
plt.plot(df.index, pred, label="Model room")
plt.plot(df.index, mass, label="Estimated mass temperature")

# plt.plot(df.index, df["outside_temp"], label="Outside", alpha=0.3)

plt.title("2-State RC Floor Heating Model (no upstairs)")
plt.legend()
plt.grid()
plt.show()