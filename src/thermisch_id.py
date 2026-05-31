"""
thermisch_id.py — Thermische parameter identificatie voor een middenwoning
==========================================================================

Model:  2R2C grey-box (2 thermische weerstanden, 2 capaciteiten)
Solver: CasADi + IPOPT  (multiple shooting)

Fysisch schema:
                    g_solar·P_pv (zon door ramen → lucht)
                          ↓
  T_out ──[R_env]── T_mass ──[R_int]── T_room ── (meting)
                      ↑
               g_mass·P_pv  (zon door ramen → betonvloer)

States:  x = [T_room [°C],  T_mass [°C]]
Inputs:  u = [T_out  [°C],  P_pv  [W]]
Params:  θ = [R_env, R_int, C_air, C_mass, g_solar, g_mass]

ODE:
  C_air  · dT_room/dt = (T_mass − T_room)/R_int  + g_solar · P_pv
  C_mass · dT_mass/dt = (T_out  − T_mass)/R_env
                      + (T_room − T_mass)/R_int
                      + g_mass  · P_pv

Typische waarden — middenwoning 2023 (energielabel A/A+):
  UA_env  ≈  40–80 W/K   →  R_env  ≈  0.012–0.025 K/W
  C_mass  ≈  8–20 MJ/K   (80m² betonvloer 10cm + muren)
  C_air   ≈  0.1–0.5 MJ/K
  g_solar ≈  0.1–0.5     (PV-proxy → warmtewinst kamer)
  τ_massa ≈  40–200 uur  (traag reagerend systeem → vloerverwarming!)

Datavereisten voor goede identificatie:
  - Min. 10 dagen, bij voorkeur 3–4 weken
  - Variatie in T_out (dag/nacht, liefst ook koude periodes)
  - Periodes zonder actief verwarmen of koelen
  - Tijdstap 15 min (900 s) werkt goed
"""

import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATIE  —  pas hier aan voor jouw huis
# ══════════════════════════════════════════════════════════════════════

DT = 900          # tijdstap [s]  (15 min)
N_SHOOT = 12      # multiple-shooting nodes (meer = robuuster, trager)

P0 = {
    "log_R_env":  np.log(1/55),    # UA ≈ 55 W/K (realistisch A-label)
    "log_R_int":  np.log(1/400),   # UA_int ≈ 400 W/K
    "log_C_air":  np.log(4e5),     # 400 kJ/K
    "log_C_mass": np.log(1.5e7),   # 15 MJ/K
    "g_solar":    0.4,
    "g_mass":     0.1,
    "Q_base":     200.0,
}

# Fysisch realistische bounds (log voor R en C)
BOUNDS = {
    # UA_env = 1/R_env: 30-90 W/K voor goede isolatie
    "log_R_env": (np.log(1 / 90), np.log(1 / 30)),  # ≈ 0.011–0.033 K/W

    # R_int: warmteovergang vloer↔lucht, h≈6-12 W/m²K, A≈60m² → UA≈360-720 W/K
    "log_R_int": (np.log(1 / 800), np.log(1 / 200)),  # ≈ 0.00125–0.005 K/W

    # C_air: lucht + lichte inboedel, 0.2–0.8 MJ/K
    "log_C_air": (np.log(2e5), np.log(8e5)),

    # C_mass: betonvloer 10cm + binnenmuren, 8–25 MJ/K
    "log_C_mass": (np.log(8e6), np.log(2.5e7)),

    # Zonnewinst: PV is proxy, dus factor mag >1 maar niet extreem
    "g_solar": (0.0, 2.0),
    "g_mass": (0.0, 0.5),  # vloer krijgt minder directe zon

    # Interne winsten: 100-600 W typisch (2 personen + apparaten)
    "Q_base": (50.0, 600.0),
}

# ══════════════════════════════════════════════════════════════════════
#  1. DATA  —  laad CSV of gebruik synthetische testdata
# ══════════════════════════════════════════════════════════════════════

def load_csv(dt_resample: int = DT):
    """
    Laad meetdata uit Home Assistant CSV-exports.

    Verwacht:
        data/temp.csv  — kolommen: time, entity_id, °C.value
        data/pv.csv    — kolommen: time, W.value
                         (PV-productie als negatief getal, zoals HA het exporteert)

    Tip: verwijder periodes met actieve verwarming/koeling vóór het inladen.
    """
    RESAMPLE = "5min"

    # ── Temperatuurdata ───────────────────────────────────────────────
    df_temp = pd.read_csv("data/temp.csv")
    df_temp["time"] = pd.to_datetime(df_temp["time"])
    df_temp["temp"] = pd.to_numeric(df_temp["°C.value"], errors="coerce")

    df = df_temp.pivot_table(
        index="time",
        columns="entity_id",
        values="temp",
        aggfunc="last",
    )
    df.columns.name = None   # verwijder MultiIndex-naam

    df = df.rename(columns={
        "danfoss_15_temperature":            "T_room",
        "ecodan_heatpump_ca09ec_buiten_temp": "T_out",
    })
    df = df.sort_index().resample(RESAMPLE).mean().ffill()

    # ── PV-data ───────────────────────────────────────────────────────
    pv = pd.read_csv("data/pv.csv")
    pv["time"] = pd.to_datetime(pv["time"])
    pv["P_pv"] = pd.to_numeric(pv["W.value"], errors="coerce")
    pv = pv.set_index("time").sort_index()
    pv = pv[["P_pv"]].resample(RESAMPLE).mean().ffill()  # BUG-FIX 1: was "pv_power"

    # ── Rolluiken (Shutters) ──────────────────────────────────────────
    sh = pd.read_csv("data/shutter.csv")
    sh["time"] = pd.to_datetime(sh["time"])
    # Let op de kolomnaam in jouw csv, waarschijnlijk 'cover.woonkamer.current_position'
    sh["shutter"] = pd.to_numeric(sh["cover.woonkamer.current_position"], errors="coerce")
    sh = sh.set_index("time").sort_index()
    sh = sh[["shutter"]].resample(RESAMPLE).mean().ffill()

    # ── Samenvoegen ───────────────────────────────────────────────────
    df = df.join(pv, how="left")
    df = df.join(sh, how="left").ffill().dropna(subset=["T_room", "T_out", "P_pv"])

    # Normaliseer shutter naar [0, 1] (Ervan uitgaande dat 100 = open, 0 = dicht)
    # Als er geen data is, gaan we er voor de veiligheid vanuit dat hij open is (1.0)
    df["shutter"] = df["shutter"].fillna(100.0) / 100.0

    # PV fix
    df["P_pv"] = np.maximum(0.0, df["P_pv"].values)

    df = (
        df[["T_room", "T_out", "P_pv", "shutter"]]  # <--- voeg shutter toe
        .resample(f"{dt_resample}s")
        .mean()
        .dropna()
    )
    # Geef nu 4 arrays terug in plaats van 3
    return df["T_room"].values, df["T_out"].values, df["P_pv"].values, df["shutter"].values

# ══════════════════════════════════════════════════════════════════════
#  2. THERMISCH MODEL IN CASADI
# ══════════════════════════════════════════════════════════════════════

def make_rk4_step(dt: float) -> ca.Function:
    """
    Bouw één Runge-Kutta-4 integratiestap van het 2R2C model.

    Returns
    -------
    F : ca.Function  met signatuur  F(x, u, p) → x_next
        x = [T_room, T_mass]     state
        u = [T_out,  P_pv ]      invoer
        p = [log_R_env, log_R_int, log_C_air, log_C_mass, g_solar, g_mass]
    """
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u", 3)
    p = ca.MX.sym("p", 7)   # <--- WAS 6, NU 7

    T_room, T_mass = x[0], x[1]
    T_out_v, P_pv_v, shutter_v = u[0], u[1], u[2]  # <--- Shutter uitlezen

    # Exp-transformatie: paramters altijd positief
    R_env   = ca.exp(p[0])
    R_int   = ca.exp(p[1])
    C_air   = ca.exp(p[2])
    C_mass  = ca.exp(p[3])
    g_solar = p[4]   # kan in principe 0 zijn (noordgerichte woning)
    g_mass  = p[5]
    Q_base = p[6]  # <--- Uitlezen

    effective_solar = P_pv_v * shutter_v

    # ODE rechterhand
    dT_room = (1 / C_air) * (
        (T_mass - T_room) / R_int
        + g_solar * effective_solar   # <--- Gebruik de effectieve zon
        + Q_base  # <--- Interne warmte warmt de lucht op!
    )
    dT_mass = (1 / C_mass) * (
        (T_out_v - T_mass) / R_env
        + (T_room  - T_mass) / R_int
        + g_mass * effective_solar  # <--- Gebruik de effectieve zon
    )

    f = ca.Function("f", [x, u, p], [ca.vertcat(dT_room, dT_mass)])

    # Klassieke RK4 — nauwkeuriger dan Euler, zeker bij 15 min tijdstap
    k1 = f(x,             u, p)
    k2 = f(x + dt/2 * k1, u, p)
    k3 = f(x + dt/2 * k2, u, p)
    k4 = f(x + dt   * k3, u, p)
    xn = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return ca.Function("F", [x, u, p], [xn], ["x", "u", "p"], ["xn"])


# ══════════════════════════════════════════════════════════════════════
#  3. PARAMETER IDENTIFICATIE — MULTIPLE SHOOTING
# ══════════════════════════════════════════════════════════════════════

def identify(
    T_room_meas: np.ndarray,
    T_out_data:  np.ndarray,
    P_pv_data:   np.ndarray,
    shutter_data: np.ndarray,
    dt:       float = DT,
    n_shoot:  int   = N_SHOOT,
    verbose:  bool  = True,
):
    """
    Identificeer thermische parameters via multiple shooting + IPOPT.

    Multiple shooting:
      - Tijdreeks opgedeeld in n_shoot gelijke segmenten
      - Per segment vrije initiële toestand [T_room_0, T_mass_0]
      - Harde continuïteitsconstraint koppelt opeenvolgende segmenten
      - Voordeel vs. single-shooting: betere conditionering voor lange datasets

    Parameters
    ----------
    T_room_meas : (N,) gemeten kamertemperatuur [°C]
    T_out_data  : (N,) buitentemperatuur [°C]
    P_pv_data   : (N,) PV-vermogen [W]
    dt          : tijdstap [s]
    n_shoot     : aantal shooting-nodes

    Returns
    -------
    params : dict   — fysische parameters + afgeleide grootheden
    p_opt  : DM     — optimale parametersvector (log-ruimte)
    """
    N   = len(T_room_meas)
    seg = N // n_shoot          # tijdstappen per segment
    F   = make_rk4_step(dt)

    opti = ca.Opti()

    # ── 3a. Parametersvector ─────────────────────────────────────────
    p_keys = list(P0.keys())
    p = opti.variable(len(p_keys))  # <--- NIEUW: past zich automatisch aan
    opti.set_initial(p, list(P0.values()))

    for i, key in enumerate(p_keys):
        lo, hi = BOUNDS[key]
        opti.subject_to(p[i] >= lo)
        opti.subject_to(p[i] <= hi)

    # ── 3b. Shooting-states  X[:, m] = [T_room, T_mass] bij t = m·seg·dt ──
    X = opti.variable(2, n_shoot)
    for m in range(n_shoot):
        idx0 = m * seg
        T0 = float(T_room_meas[min(idx0, N - 1)])
        opti.set_initial(X[:, m], [T0, T0 - 0.5])
        # Temperatuurbounds — ruim maar fysisch
        opti.subject_to(X[0, m] >= 5);  opti.subject_to(X[0, m] <= 35)
        opti.subject_to(X[1, m] >= 5);  opti.subject_to(X[1, m] <= 35)

    # ── 3c. Kostfunctie + continuïteitsconstraints ───────────────────
    J = 0.0

    for m in range(n_shoot):
        i0  = m * seg
        i1  = min(i0 + seg, N)
        xk  = X[:, m]

        for k in range(i1 - i0 - 1):
            idx = i0 + k
            u_k = ca.vertcat(
                float(T_out_data[idx]),
                float(P_pv_data[idx]),
                float(shutter_data[idx])
            )
            xk = F(x=xk, u=u_k, p=p)["xn"]

            # Gewogen kwadratensom op T_room (enige meting)
            J += (xk[0] - float(T_room_meas[idx + 1])) ** 2

        # Continuïteitsconstraint: einde segment m = begin segment m+1
        if m < n_shoot - 1:
            opti.subject_to(X[:, m + 1] == xk)

    # Zachte regularisatie — trekt parameters naar startwaarden als data
    # onvoldoende informatief zijn (bijv. weinig T_out variatie)
    p0_dm = ca.DM(list(P0.values()))
    J += 5e-4 * ca.sumsqr(p - p0_dm)

    opti.minimize(J)

    # ── 3d. IPOPT configuratie ────────────────────────────────────────
    opts = {
        "ipopt.print_level":     5 if verbose else 0,
        "ipopt.max_iter":        800,
        "ipopt.tol":             1e-7,
        "ipopt.acceptable_tol":  1e-5,
        "ipopt.mu_strategy":     "adaptive",
        "print_time":            verbose,
    }
    opti.solver("ipopt", opts)

    # ── 3e. Oplossen ─────────────────────────────────────────────────
    try:
        sol = opti.solve()
        p_opt = sol.value(p)
    except Exception as exc:
        print(f"\n⚠  Oplossen mislukt: {exc}")
        print("   Probeer: meer data, kleinere n_shoot, of betere startwaarden.")
        raise

    # ── 3f. Fysische parameters berekenen ────────────────────────────
    R_env   = float(np.exp(p_opt[0]))
    R_int   = float(np.exp(p_opt[1]))
    C_air   = float(np.exp(p_opt[2]))
    C_mass  = float(np.exp(p_opt[3]))
    g_solar = float(p_opt[4])
    g_mass  = float(p_opt[5])
    Q_base = float(p_opt[6])

    params = {
        # Primaire thermische parameters
        "R_env":    R_env,
        "R_int":    R_int,
        "C_air":    C_air,
        "C_mass":   C_mass,
        "g_solar":  g_solar,
        "g_mass":   g_mass,
        "Q_base": Q_base,  # <--- Toevoegen aan dict
        # Afgeleide / meer intuïtieve grootheden
        "UA_env":       1.0 / R_env,                 # transmissieverlies  [W/K]
        "UA_int":       1.0 / R_int,                 # vloer↔lucht warmteovergang [W/K]
        "tau_air_h":    C_air  * R_int / 3600,       # tijdconstante lucht  [uur]
        "tau_mass_h":   C_mass * R_env / 3600,       # tijdconstante massa  [uur]
    }
    # Haal de geoptimaliseerde T_mass van de allereerste tijdstap op:
    T_mass_opt = float(sol.value(X[1, 0]))

    return params, ca.DM(p_opt), T_mass_opt  # <--- return erbij


# ══════════════════════════════════════════════════════════════════════
#  4. SIMULATIE & VOORSPELLING
# ══════════════════════════════════════════════════════════════════════

def simulate(
    p_opt:      ca.DM,
    T_room_0:   float,
    T_mass_0:     float,   # <--- NIEUW
    T_out_data: np.ndarray,
    P_pv_data:  np.ndarray,
    shutter_data: np.ndarray,
    dt:         float = DT,
):
    """
    Open-loop simulatie met gevonden parameters.
    Gebruik voor validatie én voor toekomstige voorspellingen.
    """
    F = make_rk4_step(dt)
    N = len(T_out_data)

    xk = ca.DM([T_room_0, T_mass_0])  # <--- AANGEPAST
    T_room_sim = np.empty(N)
    T_mass_sim = np.empty(N)
    T_room_sim[0] = float(xk[0])
    T_mass_sim[0] = float(xk[1])

    for k in range(N - 1):
        u_k = ca.DM([float(T_out_data[k]), float(P_pv_data[k]), float(shutter_data[k])])
        xk  = F(x=xk, u=u_k, p=p_opt)["xn"]
        T_room_sim[k + 1] = float(xk[0])
        T_mass_sim[k + 1] = float(xk[1])

    return T_room_sim, T_mass_sim


# ══════════════════════════════════════════════════════════════════════
#  5. VISUALISATIE & DIAGNOSE
# ══════════════════════════════════════════════════════════════════════

def _fmt_param_rows(params: dict) -> list:
    rows = [
        ["R_env",        f"{params['R_env'] * 1000:.2f} mK/W"],
        ["UA_env",       f"{params['UA_env']:.1f} W/K"],
        ["R_int",        f"{params['R_int'] * 1000:.2f} mK/W"],
        ["UA_int",       f"{params['UA_int']:.0f} W/K"],
        ["C_air",        f"{params['C_air'] / 1e5:.2f} ×10⁵ J/K"],
        ["C_mass",       f"{params['C_mass'] / 1e6:.2f} MJ/K"],
        ["g_solar",      f"{params['g_solar']:.4f}"],
        ["g_mass",       f"{params['g_mass']:.4f}"],
        ["Q_base",       f"{params['Q_base']:.0f} W"],
        ["τ_lucht",      f"{params['tau_air_h']:.2f} uur"],
        ["τ_massa",      f"{params['tau_mass_h']:.1f} uur"],
    ]
    return rows


def plot_results(
    t:            np.ndarray,
    T_room_meas:  np.ndarray,
    T_room_sim:   np.ndarray,
    T_mass_sim:   np.ndarray,
    T_out:        np.ndarray,
    P_pv:         np.ndarray,
    params:       dict,
    split_idx:    int   = None,
    save_path:    str   = "thermisch_id.png",
    true_params:  dict  = None,
):
    t_h = t / 3600
    res = T_room_meas - T_room_sim

    rmse_train = float(np.sqrt(np.mean(res[:split_idx] ** 2))) if split_idx else None
    rmse_val   = float(np.sqrt(np.mean(res[split_idx:] ** 2))) if split_idx else None
    rmse_all   = float(np.sqrt(np.mean(res ** 2)))

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)
    ax_t   = fig.add_subplot(gs[0, :])
    ax_res = fig.add_subplot(gs[1, :])
    ax_pv  = fig.add_subplot(gs[2, :2])
    ax_tbl = fig.add_subplot(gs[2, 2])

    # ── Temperaturen ──────────────────────────────────────────────────
    ax_t.plot(t_h, T_room_meas, "k.", ms=1.5, alpha=0.5, label="Meting T_kamer", zorder=3)
    ax_t.plot(t_h, T_room_sim,  "r-", lw=1.8, label="Model T_kamer", zorder=4)
    ax_t.plot(t_h, T_mass_sim,  "b--", lw=1.2, alpha=0.75, label="Model T_massa (vloer)")
    ax_t.plot(t_h, T_out,       "g:", lw=1.0, alpha=0.6, label="T_buiten")
    if split_idx:
        ax_t.axvline(t_h[split_idx], color="purple", ls="--", lw=1.5, label="train|validatie")
    ax_t.set_ylabel("Temperatuur [°C]")
    ax_t.legend(ncol=5, fontsize=8.5, loc="upper right")
    ax_t.grid(alpha=0.25)
    ax_t.set_title("Kamertemperatuur — meting vs. 2R2C model", fontsize=11)

    # ── Residuen ──────────────────────────────────────────────────────
    rmse_label = (
        f"Residu  |  RMSE train = {rmse_train:.3f}°C    "
        f"RMSE validatie = {rmse_val:.3f}°C"
        if split_idx
        else f"Residu  |  RMSE = {rmse_all:.3f}°C"
    )
    ax_res.plot(t_h, res, "k-", lw=0.8, alpha=0.6)
    ax_res.fill_between(t_h, res, alpha=0.15, color="steelblue")
    ax_res.axhline(0, color="r", lw=1.2)
    if split_idx:
        ax_res.axvline(t_h[split_idx], color="purple", ls="--", lw=1.5)
    ax_res.set_ylabel("T_meting − T_model [°C]")
    ax_res.set_xlabel("Tijd [uur]")
    ax_res.set_title(rmse_label, fontsize=10)
    ax_res.grid(alpha=0.25)

    # ── PV vermogen ───────────────────────────────────────────────────
    ax_pv.fill_between(t_h, P_pv / 1000, alpha=0.55, color="orange", label="PV [kW]")
    ax_pv.set_ylabel("PV vermogen [kW]")
    ax_pv.set_xlabel("Tijd [uur]")
    ax_pv.grid(alpha=0.25)
    ax_pv.legend()
    ax_pv.set_title("Zonne-energie invoer")

    # ── Parametertabel ────────────────────────────────────────────────
    ax_tbl.axis("off")
    rows = _fmt_param_rows(params)
    if true_params:
        col_labels = ["Parameter", "Gevonden", "Waar"]
        true_vals  = {
            "R_env":   f"{true_params['R_env']*1000:.2f}",
            "UA_env":  f"{1/true_params['R_env']:.1f}",
            "R_int":   f"{true_params['R_int']*1000:.2f}",
            "UA_int":  f"{1/true_params['R_int']:.0f}",
            "C_air":   f"{true_params['C_air']/1e5:.2f}",
            "C_mass":  f"{true_params['C_mass']/1e6:.2f}",
            "g_solar": f"{true_params['g_solar']:.4f}",
            "g_mass":  f"{true_params['g_mass']:.4f}",
        }
        extended = []
        keys_map = ["R_env","UA_env","R_int","UA_int","C_air","C_mass",
                    "g_solar","g_mass","τ_lucht","τ_massa"]
        for row, key in zip(rows, keys_map):
            tv = true_vals.get(key, "—")
            extended.append(row + [tv])
        rows = extended
    else:
        col_labels = ["Parameter", "Waarde"]

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.45)
    ax_tbl.set_title("Parameters", fontsize=10)

    plt.suptitle("Thermische identificatie — middenwoning 2023", fontsize=13, y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Grafiek opgeslagen: {save_path}")
    plt.show()
    return rmse_train, rmse_val, rmse_all


# ══════════════════════════════════════════════════════════════════════
#  6. DIAGNOSE-FUNCTIES
# ══════════════════════════════════════════════════════════════════════

def diagnose(params: dict, true_params: dict = None):
    """Druk een leesbaar overzicht van de gevonden parameters af."""
    sep = "─" * 56
    print(f"\n{sep}")
    print("  Thermische parameters — gevonden door identificatie")
    print(sep)

    lines = [
        ("R_env  (gevelweerstand)",  f"{params['R_env']*1000:.3f} mK/W"),
        ("UA_env (transmissieverlies)", f"{params['UA_env']:.1f} W/K"),
        ("R_int  (vloer↔lucht)",     f"{params['R_int']*1000:.3f} mK/W"),
        ("UA_int",                   f"{params['UA_int']:.0f} W/K"),
        ("C_air  (luchtmassa)",      f"{params['C_air']/1e5:.2f} ×10⁵ J/K"),
        ("C_mass (vloermassa)",      f"{params['C_mass']/1e6:.2f} MJ/K"),
        ("g_solar",                  f"{params['g_solar']:.4f}  (kamer-zonnwinst)"),
        ("g_mass",                   f"{params['g_mass']:.4f}  (vloer-zonnwinst)"),
        ("Q_base (interne winst)",   f"{params['Q_base']:.0f} W"),
        ("τ_lucht",                  f"{params['tau_air_h']:.2f} uur  (tijdconstante lucht)"),
        ("τ_massa",                  f"{params['tau_mass_h']:.1f} uur  (tijdconstante vloer)"),
    ]
    for name, val in lines:
        print(f"  {name:<30s} {val}")

    if true_params:
        print(f"\n{sep}")
        print("  Verificatie vs. synthese-waarden")
        print(sep)
        checks = [
            ("R_env", true_params["R_env"], params["R_env"]),
            ("R_int", true_params["R_int"], params["R_int"]),
            ("C_air", true_params["C_air"], params["C_air"]),
            ("C_mass",true_params["C_mass"],params["C_mass"]),
            ("g_solar",true_params["g_solar"],params["g_solar"]),
            ("g_mass", true_params["g_mass"], params["g_mass"]),
        ]
        for name, tv, fv in checks:
            err = abs(fv - tv) / tv * 100
            mark = "✓" if err < 10 else ("~" if err < 25 else "✗")
            print(f"  {mark} {name:<10s}  waar={tv:.4g}  gevonden={fv:.4g}  fout={err:.1f}%")

    print(sep)

    # Sanity-checks
    issues = []

    # --- UA_env: De "Isolatieschil" ---
    if params["UA_env"] < 15:
        issues.append(
            f"UA_env ({params['UA_env']:.1f}) < 15 W/K: Onrealistisch goed geïsoleerd. Is er stiekem een warmtebron die niet in de data staat?")
    if params["UA_env"] > 120:
        issues.append(
            f"UA_env ({params['UA_env']:.1f}) > 120 W/K: Voor een middenwoning is dit erg hoog (label E/F-niveau). Staat er ergens een raam open?")

    # --- C_air: De "Snelle Massa" (Lucht + Meubels) ---
    # Een kamer van 40m2 (100m3) heeft ~1.2e5 J/K aan lucht. Met meubels/gips erbij is 2e5 - 5e5 normaal.
    if params["C_air"] < 1e5:
        issues.append(
            f"C_air ({params['C_air'] / 1e5:.1f}e5) is erg laag: De kamer zou bij de kleinste zonnestraal al oververhitten.")
    if params["C_air"] > 1e6:
        issues.append(
            f"C_air ({params['C_air'] / 1e5:.1f}e5) is te hoog: De 'snelle' luchttemperatuur reageert nu even traag als een betonmuur.")

    # --- C_mass: De "Trage Massa" (Vloer + Constructie) ---
    if params["C_mass"] < 5e6:
        issues.append(
            f"C_mass ({params['C_mass'] / 1e6:.1f} MJ/K) is laag: De vloer lijkt geen betonvloer te zijn, of de thermische koppeling is slecht.")
    if params["C_mass"] > 5e7:
        issues.append(
            f"C_mass ({params['C_mass'] / 1e6:.1f} MJ/K) is extreem hoog: Dit komt overeen met >40 cm massief beton over het hele oppervlak.")

    # --- UA_int: Koppeling tussen vloer en lucht ---
    # Typische warmteoverdrachtscoëfficiënt vloer -> lucht is ~6-10 W/m2K.
    # Bij 50m2 vloer verwacht je UA_int rond de 300-500 W/K.
    if params["UA_int"] < 50:
        issues.append(
            f"UA_int ({params['UA_int']:.0f}) is erg laag: De vloer en de lucht zijn 'ontkoppeld'. De vloer doet niet mee als buffer.")
    if params["UA_int"] > 2000:
        issues.append(
            f"UA_int ({params['UA_int']:.0f}) is extreem hoog: De lucht en vloer hebben vrijwel altijd exact dezelfde temperatuur (1C model).")

    # --- Q_base: Interne winsten (Apparaten + Mensen) ---
    if params["Q_base"] < 50:
        issues.append(f"Q_base ({params['Q_base']:.0f} W) is erg laag: Geen koelkast, router of mensen aanwezig?")
    if params["Q_base"] > 800:
        issues.append(
            f"Q_base ({params['Q_base']:.0f} W) is erg hoog: Dit compenseert mogelijk voor een verwarming die stiekem aanstond.")

    # --- Zonne-winsten (g_solar en g_mass) ---
    # De PV-installatie is een proxy. Als g > 1.0, is de zonkracht op de ramen veel groter dan de PV-opwek (onwaarschijnlijk).
    if params["g_solar"] + params["g_mass"] > 2.0:
        issues.append(
            f"Gezamenlijke zonne-winst ({params['g_solar'] + params['g_mass']:.2f}) is erg hoog: De PV-proxy wordt overschat.")
    if params["g_solar"] < 0.01 and params["g_mass"] < 0.01:
        issues.append("Zonne-winst is bijna nul: Schijnt de zon wel naar binnen, of zijn de rolluiken altijd dicht?")

    # --- Tijdsconstante (Tau) ---
    if params["tau_mass_h"] > 300:
        issues.append(
            f"τ_massa ({params['tau_mass_h']:.0f} uur) is > 12 dagen: Het model is 'verlamd' en reageert nauwelijks op buitentemperatuur.")
    if params["tau_air_h"] > 10:
        issues.append(
            f"τ_lucht ({params['tau_air_h']:.1f} uur) is te traag: De kamertemperatuur moet sneller kunnen reageren op interne winsten.")

    # --- Verhouding Massa vs Lucht ---
    if params["C_mass"] < params["C_air"]:
        issues.append(
            "Fysische inconsistentie: De luchtcapaciteit is groter dan de masscapaciteit. Dit is onmogelijk in een woning.")

    if issues:
        print("\n  ⚠  Waarschuwingen:")
        for w in issues:
            print(f"     • {w}")
    else:
        print("\n  ✓ Parameters zien er fysisch redelijk uit.")
    print()


# ══════════════════════════════════════════════════════════════════════
#  7. HOOFDPROGRAMMA
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Kies databron ─────────────────────────────────────────────────
    USE_SYNTHETIC = False
    # Eigen CSV:  timestamp, T_room, T_out, P_pv
    T_room, T_out, P_pv, shutter = load_csv(dt_resample=DT)
    N = len(T_room)
    t = np.arange(N) * DT
    TRUE_PARAMS = None
    print(f"Data geladen: {N} punten ({N * DT / 86400:.1f} dagen)")

    # Voeg dit toe na load_csv() om te debuggen:
    print(f"P_pv stats: min={P_pv.min():.1f}W, max={P_pv.max():.1f}W, mean={P_pv.mean():.1f}W")
    print(f"Aantal nul-waarden: {(P_pv == 0).sum()} / {len(P_pv)} ({(P_pv == 0).mean() * 100:.1f}%)")
    print(f"Aantal >100W: {(P_pv > 100).sum()} / {len(P_pv)}")

    print(f"Shutter stats: min={shutter.min():.2f}, max={shutter.max():.2f}, mean={shutter.mean():.2f}")
    print(f"Shutter >0.5: {(shutter > 0.5).mean() * 100:.1f}% van de tijd")

    N = len(T_room)

    # ── Train/validatie split ──────────────────────────────────────────
    split = int(0.72 * N)   # 72% trainen, 28% valideren
    print(f"Totaal: {N} punten  ({N * DT / 86400:.1f} d)  "
          f"→ train: {split} / val: {N - split}\n")

    # ── Parameter identificatie ────────────────────────────────────────
    print("Parameter identificatie starten...")
    params, p_opt, T_mass_0_opt = identify(
        T_room[:split], T_out[:split], P_pv[:split], shutter[:split],
        dt=DT, n_shoot=N_SHOOT, verbose=False,
    )

    diagnose(params, true_params=TRUE_PARAMS if USE_SYNTHETIC else None)

    # ── Simuleer over VOLLEDIGE periode (incl. validatie) ─────────────
    print("Simulatie over volledige periode...")
    T_room_sim, T_mass_sim = simulate(p_opt, T_room[0], T_mass_0_opt, T_out, P_pv, shutter, dt=DT)

    rmse_train, rmse_val, rmse_all = plot_results(
        t, T_room, T_room_sim, T_mass_sim,
        T_out, P_pv, params,
        split_idx=split,
        save_path="thermisch_id.png",
        true_params=TRUE_PARAMS if USE_SYNTHETIC else None,
    )

    print("\nPrestatie:")
    print(f"  RMSE train      : {rmse_train:.3f}°C")
    print(f"  RMSE validatie  : {rmse_val:.3f}°C")
    print(f"  RMSE totaal     : {rmse_all:.3f}°C")

    if rmse_val < 0.30:
        print("→ uitstekend ✓✓")
    elif rmse_val < 0.60:
        print("→ goed ✓")
    elif rmse_val < 1.00:
        print("→ redelijk ~")
    else:
        print("→ verbetering gewenst ⚠")