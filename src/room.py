import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca

# ==========================================================
# CONFIG
# ==========================================================

dt = 1.0
RESAMPLE = "5min"

# ==========================================================
# LOAD DATA  (identiek aan origineel)
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
    "danfoss_15_temperature":               "room_temp",
    "ecodan_heatpump_ca09ec_buiten_temp":   "outside_temp",
})

df = df.sort_index()

pv = pd.read_csv("data/pv.csv")
pv["time"] = pd.to_datetime(pv["time"])
pv["pv_power"] = pd.to_numeric(pv["W.value"], errors="coerce")
pv = pv.set_index("time").sort_index()

sh = pd.read_csv("data/shutter.csv")
sh["time"] = pd.to_datetime(sh["time"])
sh["shutter"] = pd.to_numeric(sh["cover.woonkamer.current_position"], errors="coerce")
sh = sh.set_index("time").sort_index()

df = df.resample(RESAMPLE).mean().ffill()
pv = pv.resample(RESAMPLE).mean().ffill()
sh = sh.resample(RESAMPLE).mean().ffill()

df = df.join(pv[["pv_power"]], how="left")
df = df.join(sh[["shutter"]], how="left")
df = df.ffill().dropna()

df["heating"] = 0.0

pv_max = df["pv_power"].quantile(0.99)
df["solar"] = np.clip(np.maximum(0, -df["pv_power"]) / pv_max, 0, 1)

# ==========================================================
# CASADI: definieer één stap als symbolische Function
# ==========================================================

# State: [T_room, T_mass]
state = ca.MX.sym("state", 2)
T_room_s, T_mass_s = state[0], state[1]

# Parameters: [k_room_mass, k_mass_room, k_outside, k_heating, k_solar]
p = ca.MX.sym("p", 5)
k_room_mass, k_mass_room, k_outside, k_heating, k_solar = \
    p[0], p[1], p[2], p[3], p[4]

# Inputs voor één tijdstap: [outside, heating, solar]
inp = ca.MX.sym("inp", 3)
outside_s, heating_s, solar_s = inp[0], inp[1], inp[2]

# ==================================================
# MASS DYNAMICS
# ==================================================
d_mass = k_room_mass * (T_room_s - T_mass_s) + k_heating * heating_s
T_mass_next = T_mass_s + d_mass * dt

# ==================================================
# ROOM DYNAMICS
# ==================================================
d_room = (
    k_mass_room * (T_mass_s - T_room_s)
    + k_outside  * (outside_s - T_room_s)
    + k_solar    * solar_s
)
T_room_next = T_room_s + d_room * dt

state_next = ca.vertcat(T_room_next, T_mass_next)

# Herbruikbare Function — CasADi optimaliseert de graph intern
step_fn = ca.Function(
    "step",
    [state, inp, p],
    [state_next],
    ["state", "inp", "p"],
    ["state_next"],
)

# ==========================================================
# OPTIMALISATIE met Opti stack + IPOPT
# ==========================================================

opti = ca.Opti()

params = opti.variable(5)

# Bounds (zelfde als origineel)
opti.subject_to(opti.bounded(0,    params[0], 0.05))   # k_room_mass
opti.subject_to(opti.bounded(0,    params[1], 0.20))   # k_mass_room
opti.subject_to(opti.bounded(0,    params[2], 0.05))   # k_outside
# opti.subject_to(opti.bounded(0,    params[3], 2.00))   # k_heating
opti.subject_to(opti.bounded(0,    params[4], 2.00))   # k_solar

# Beginschatting
opti.set_initial(params, [0.001, 0.01, 0.001, 0.0, 0.1])

# Data als numpy arrays (geen CasADi-overhead per rij)
T_meas   = df["room_temp"].values
T_out    = df["outside_temp"].values
H        = df["heating"].values
S        = (df["solar"] * df["shutter"] / 100.0).values
n        = len(df)

# Symbolic rollout (single shooting)
state_k  = ca.vertcat(T_meas[0], T_meas[0])  # beginwaarden
residuals = []

for t in range(n - 1):
    inp_t   = ca.vertcat(T_out[t], H[t], S[t])
    state_k = step_fn(state=state_k, inp=inp_t, p=params)["state_next"]
    residuals.append(state_k[0] - T_meas[t + 1])   # T_room residu

residuals = ca.vertcat(*residuals)
mse = ca.dot(residuals, residuals) / (n - 1)

opti.minimize(mse)

# IPOPT opties — zet print_level op 0 voor stille run
opti.solver("ipopt", {}, {"print_level": 5})

sol = opti.solve()

best = sol.value(params)
print("\nFitted parameters:")
labels = ["k_room_mass", "k_mass_room", "k_outside", "k_heating", "k_solar"]
for lbl, val in zip(labels, best):
    print(f"  {lbl:15s} = {val:.6f}")

# ==========================================================
# SIMULATIE voor plot (numpy, zelfde als origineel)
# ==========================================================

def simulate_numpy(params, df):
    k_room_mass, k_mass_room, k_outside, k_heating, k_solar = params
    n = len(df)
    T_room = np.zeros(n)
    T_mass = np.zeros(n)
    T_room[0] = T_mass[0] = df["room_temp"].iloc[0]

    for t in range(n - 1):
        room    = T_room[t]
        mass    = T_mass[t]
        outside = df["outside_temp"].iloc[t]
        heating = df["heating"].iloc[t]
        solar   = df["solar"].iloc[t] * (df["shutter"].iloc[t] / 100.0)

        d_mass  = k_room_mass * (room - mass) + k_heating * heating
        T_mass[t + 1] = mass + d_mass * dt

        d_room  = (k_mass_room * (mass - room)
                   + k_outside  * (outside - room)
                   + k_solar    * solar)
        T_room[t + 1] = room + d_room * dt

    return T_room, T_mass

pred, mass_traj = simulate_numpy(best, df)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["room_temp"], label="Measured room")
plt.plot(df.index, pred,            label="Model room")
plt.plot(df.index, mass_traj,       label="Estimated mass temperature")
plt.title("2-State RC Floor Heating Model — CasADi / IPOPT fit")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()