"""
optimizer.py

Bevat:
  - ThermalMPC   — MILP via CVXPY + SLP-iteraties via PhysicsLinearizer
  - Optimizer    — orkestratie: training, resolve, get_plan
"""

import logging
import numpy as np
import pandas as pd
import cvxpy as cp

from datetime import datetime, timedelta
from context import Context, HvacMode
from thermal import (
    SystemIdentificator,
    HPPerformanceMap,
    HydraulicPredictor,
    UfhResidualPredictor,
    DhwResidualPredictor,
    PhysicsLinearizer,
    ComfortCostCalculator,
)
from shutter import ShutterPredictor

logger = logging.getLogger(__name__)


# =========================================================
# THERMAL MPC (DPP compliant)
# =========================================================

CLIMATE_CONFIG = {
    "room": {
        "target": [
            ("00:00", 19.0, 0.5, 1.5),
            ("06:00", 19.0, 0.5, 1.5),
            ("09:00", 19.5, 0.5, 1.5),
            ("17:00", 20.0, 0.5, 1.5),
            ("22:00", 19.0, 0.5, 1.5),
        ]
    },
    "dhw": {
        "target": [
            ("00:00", 20.0, 5.0, 30.0),
            ("19:59", 20.0, 5.0, 35.0),
            ("20:00", 50.0, 2.0, 5.0),
            ("21:00", 50.0, 2.0, 5.0),
            ("21:01", 20.0, 5.0, 30.0),
        ]
    },
}


class ThermalMPC:
    def __init__(self, ident, perf_map, hydraulic, res_dhw):
        self.ident = ident
        self.perf_map = perf_map
        self.hydraulic = hydraulic
        self.res_dhw = res_dhw
        self.horizon = 96
        self.dt = 0.25

        self._last_t_mass_model = None
        self._last_t_air_pred = None

        self.plan_t_sup_ufh = np.zeros(self.horizon)
        self.plan_t_sup_dhw = np.zeros(self.horizon)
        self.plan_p_el_ufh = np.zeros(self.horizon)
        self.plan_p_el_dhw = np.zeros(self.horizon)

        self._build_problem()

    def _build_problem(self):
        T = self.horizon
        C_air = self.ident.C_air  # snelle massa [kWh/K]
        C_mass = self.ident.C_mass  # trage massa  [kWh/K]
        R_im = self.ident.R_im  # massa ↔ lucht [K/kW]
        R_oa = self.ident.R_oa  # lucht → buiten [K/kW]

        # ── Parameters ────────────────────────────────────────────────────
        self.L = self.ident.ufh_lag_steps
        self.P_t_air_init = cp.Parameter()
        self.P_t_mass_init = cp.Parameter()
        self.P_t_dhw_init = cp.Parameter()
        self.P_init_ufh = cp.Parameter(nonneg=True)
        self.P_init_dhw = cp.Parameter(nonneg=True)
        self.P_comp_on_init = cp.Parameter(nonneg=True)

        self.P_prices = cp.Parameter(T, nonneg=True)
        self.P_export_prices = cp.Parameter(T, nonneg=True)
        self.P_solar = cp.Parameter(T, nonneg=True)
        self.P_base_load = cp.Parameter(T, nonneg=True)

        self.P_temp_out = cp.Parameter(T)
        self.P_room_min = cp.Parameter(T, nonneg=True)
        self.P_room_max = cp.Parameter(T, nonneg=True)
        self.P_dhw_min = cp.Parameter(T, nonneg=True)
        self.P_dhw_max = cp.Parameter(T, nonneg=True)
        self.P_solar_gain = cp.Parameter(T)
        self.P_strictness = cp.Parameter(T, nonneg=True)

        self.P_cost_room_under = cp.Parameter(nonneg=True)
        self.P_cost_room_over = cp.Parameter(nonneg=True)
        self.P_cost_dhw_under = cp.Parameter(nonneg=True)
        self.P_cost_dhw_over = cp.Parameter(nonneg=True)
        self.P_val_terminal_room = cp.Parameter(nonneg=True)
        self.P_val_terminal_dhw = cp.Parameter(nonneg=True)

        # Bevroren vermogen en COP per tijdstap (gevuld door SLP-iteraties)
        self.P_cop_ufh = cp.Parameter(T, nonneg=True)
        self.P_cop_dhw = cp.Parameter(T, nonneg=True)
        self.P_fixed_pel_ufh = cp.Parameter(T, nonneg=True)
        self.P_fixed_pel_dhw = cp.Parameter(T, nonneg=True)
        self.P_ufh_p_th_hist = cp.Parameter(self.L, nonneg=True)

        self.P_dhw_demand = cp.Parameter(T, nonneg=True)

        # ── Variabelen ────────────────────────────────────────────────────
        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)
        self.comp_start = cp.Variable(T, nonneg=True)

        self.p_el_ufh = cp.Variable(T, nonneg=True)
        self.p_el_dhw = cp.Variable(T, nonneg=True)

        self.p_grid = cp.Variable(T, nonneg=True)
        self.p_export = cp.Variable(T, nonneg=True)
        self.p_solar_self = cp.Variable(T, nonneg=True)

        self.t_air = cp.Variable(T + 1)  # luchttemperatuur (snel, comfort)
        self.t_mass = cp.Variable(T + 1)  # vloeremperatuur  (traag, buffer)
        self.t_dhw = cp.Variable(T + 1, nonneg=True)

        self.s_room_low = cp.Variable(T, nonneg=True)
        self.s_room_high = cp.Variable(T, nonneg=True)
        self.s_dhw_low = cp.Variable(T, nonneg=True)
        self.s_dhw_high = cp.Variable(T, nonneg=True)

        self.P_terminal_offset = cp.Parameter()

        # ── Constraints ───────────────────────────────────────────────────
        constraints = [
            self.t_air[0] == self.P_t_air_init,
            self.t_mass[0] == self.P_t_mass_init,
            self.t_dhw[0] == self.P_t_dhw_init,
        ]

        # Compressorstarts detecteren (niet klep-wissels)
        comp_on = self.ufh_on + self.dhw_on
        constraints += [self.comp_start[0] >= comp_on[0] - self.P_comp_on_init]
        for t in range(1, T):
            constraints += [self.comp_start[t] >= comp_on[t] - comp_on[t - 1]]

        for t in range(T):
            p_el_wp = self.p_el_ufh[t] + self.p_el_dhw[t]
            p_th_dhw_now = self.p_el_dhw[t] * self.P_cop_dhw[t]

            if t < self.L:
                # Pak waarde uit historie
                # We mappen t=0 op hist[L-1], t=1 op hist[L-2], etc.
                p_th_delayed = self.P_ufh_p_th_hist[self.L - 1 - t]
            else:
                # Pak de beslissing van (t - lag) geleden
                p_th_delayed = self.p_el_ufh[t - self.L] * self.P_cop_ufh[t - self.L]

            constraints += [
                # Stroombalans
                p_el_wp + self.P_base_load[t] == self.p_grid[t] + self.p_solar_self[t],
                self.P_solar[t] == self.p_solar_self[t] + self.p_export[t],
                # Luchttemperatuur
                self.t_air[t + 1]
                == self.t_air[t]
                + (
                    (self.t_mass[t] - self.t_air[t]) / R_im  # warmte van vloer
                    - (self.t_air[t] - self.P_temp_out[t]) / R_oa  # verlies naar buiten
                )
                * self.dt
                / C_air
                + self.P_solar_gain[t] * self.dt,  # zon treft ALLEEN lucht
                # Vloermassa
                self.t_mass[t + 1]
                == self.t_mass[t]
                + (p_th_delayed - (self.t_mass[t] - self.t_air[t]) / R_im)
                * self.dt
                / C_mass,
                # Thermische balans boiler
                self.t_dhw[t + 1]
                == self.t_dhw[t]
                + (
                    (p_th_dhw_now * self.dt) / self.ident.C_tank
                    - (self.t_dhw[t] - self.t_air[t])
                    * (self.ident.K_loss_dhw * self.dt)
                )
                - self.P_dhw_demand[t],
                # Niet tegelijk vloer en boiler
                self.ufh_on[t] + self.dhw_on[t] <= 1,
                # Vermogen is exact het geleerde profiel als hij aan staat
                self.p_el_ufh[t] == self.ufh_on[t] * self.P_fixed_pel_ufh[t],
                self.p_el_dhw[t] == self.dhw_on[t] * self.P_fixed_pel_dhw[t],
                # Comfort wordt gemeten in de lucht (thermostaat)
                self.t_air[t + 1] + self.s_room_low[t] >= self.P_room_min[t],
                self.t_air[t + 1] - self.s_room_high[t] <= self.P_room_max[t],
                # Comfortlimieten met slack
                self.t_dhw[t + 1] + self.s_dhw_low[t] >= self.P_dhw_min[t],
                self.t_dhw[t + 1] - self.s_dhw_high[t] <= self.P_dhw_max[t],
            ]

        # ── Doelfunctie ───────────────────────────────────────────────────
        net_cost = (
            cp.sum(
                cp.multiply(self.p_grid, self.P_prices)
                - cp.multiply(self.p_export, self.P_export_prices)
            )
            * self.dt
        )

        extra_penalty = cp.multiply(cp.pos(self.s_room_low - 0.25), self.P_strictness)
        comfort = cp.sum(
            self.s_room_low * self.P_cost_room_under
            + self.s_room_high * self.P_cost_room_over
            + self.s_dhw_low * self.P_cost_dhw_under
            + self.s_dhw_high * self.P_cost_dhw_over
            + extra_penalty
        )

        # Hoge straf op compressorstart, lage straf op klep-wissel
        valve_switches = (
            cp.pos(self.ufh_on[0] - self.P_init_ufh)
            + cp.sum(cp.pos(self.ufh_on[1:] - self.ufh_on[:-1]))
            + cp.pos(self.dhw_on[0] - self.P_init_dhw)
            + cp.sum(cp.pos(self.dhw_on[1:] - self.dhw_on[:-1]))
        )
        switching = (cp.sum(self.comp_start) * 0.15) + (valve_switches * 0.05)

        # Terminal value op t_mass: vloer houdt warmte vast na horizoneinde.
        # t_air aan het einde is vluchtig (lekt snel weg via R_oa).
        stored_heat = (
            self.t_mass[T] * self.P_val_terminal_room
            + self.t_dhw[T] * self.P_val_terminal_dhw
            - self.P_terminal_offset
        )

        self.problem = cp.Problem(
            cp.Minimize(net_cost + comfort + switching - stored_heat),
            constraints,
        )

    def _get_interpolated_values(self, schedule, times_to_predict):
        sched_mins = []
        sched_vals = []

        for entry in schedule:
            t_str = entry[0]
            vals = entry[1:]  # Pak (temp, low, high)
            h, m = map(int, t_str.split(":"))
            sched_mins.append(h * 60 + m)
            sched_vals.append(vals)

        idx = np.argsort(sched_mins)
        sched_mins = np.array(sched_mins)[idx]
        sched_vals = np.array(sched_vals)[idx]

        # Wrap around logica
        xp = np.concatenate(
            [[sched_mins[-1] - 1440], sched_mins, [sched_mins[0] + 1440]]
        )
        fp = np.vstack([sched_vals[-1], sched_vals, sched_vals[0]])

        query_mins = np.array([(t.hour * 60 + t.minute) for t in times_to_predict])

        # Interpoleer alle 3 de kolommen (Target, Low, High)
        interp_target = np.interp(query_mins, xp, fp[:, 0])
        interp_low = np.interp(query_mins, xp, fp[:, 1])
        interp_high = np.interp(query_mins, xp, fp[:, 2])

        return interp_target, interp_low, interp_high

    def _get_targets(self, now, T):
        local_tz = datetime.now().astimezone().tzinfo
        now_local = now.astimezone(local_tz)
        future_times = [now_local + timedelta(hours=t * self.dt) for t in range(T)]

        # Kamer
        r_t, r_l, r_h = self._get_interpolated_values(
            CLIMATE_CONFIG["room"]["target"], future_times
        )
        r_min = r_t - r_l
        r_max = r_t + r_h

        # Boiler
        d_t, d_l, d_h = self._get_interpolated_values(
            CLIMATE_CONFIG["dhw"]["target"], future_times
        )
        d_min = d_t - d_l
        d_max = d_t + d_h

        return r_min, r_max, d_min, d_max, r_t, d_t

    def _robust_trend(self, recent_df: pd.DataFrame) -> float:
        """Schat dT/dt via Weighted Least Squares over de laatste 2 uur."""
        df = recent_df.tail(8).copy()
        if len(df) < 4:
            logger.warning(
                f"[Optimizer] Te weinig data voor trend (slechts {len(df)} punten)"
            )
            return 0.0

        # Regime-check: negeer data voor een HVAC switch
        if df["hvac_mode"].nunique() > 1:
            df = df[df["hvac_mode"] == df["hvac_mode"].iloc[-1]]
            if len(df) < 3:
                logger.warning(
                    f"[Optimizer] Na regime-wissel slechts {len(df)} punten over"
                )
                return 0.0

        t0 = df["timestamp"].iloc[0]
        x = (df["timestamp"] - t0).dt.total_seconds().values / 3600.0
        y = df["room_temp"].values

        weights = np.exp(x - x[-1])
        W = np.diag(weights)
        A = np.vstack([x, np.ones(len(x))]).T

        try:
            a, _ = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
            logger.info(
                f"[Optimizer] WLS trend: dT/dt={a:.3f} K/h over {len(df)} punten"
            )
            return float(a)
        except Exception as e:
            logger.warning(f"[Optimizer] WLS trend failed: {e}")
            return 0.0

    def _get_steady_state(self, t_air: float, t_out: float) -> float:
        """Bereken T_mass op basis van thermisch evenwicht (geen dynamiek)."""
        q_loss = (t_air - t_out) / self.ident.R_oa
        return float(t_air + (self.ident.R_im * q_loss))

    def update_and_get_initial_state(
        self,
        state: dict,
        recent_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        current_solar_gain: float,
    ) -> float:
        """
        De 'State Observer'. Bepaalt de start-temperatuur van de vloer.
        """
        t_air_meas = float(state["room_temp"])
        t_out_now = float(forecast_df.temp.iloc[0])

        # 1. Bepaal of we een koude start hebben of een observer-update
        if self._last_t_air_pred is None or self._last_t_mass_model is None:
            # KOUDE START: Gebruik Fysische Inversie
            dT_dt = self._robust_trend(recent_df)

            # Q_mass = C_air * (dT/dt - solar) + Q_loss
            q_mass = (self.ident.C_air * (dT_dt - current_solar_gain)) + (
                t_air_meas - t_out_now
            ) / self.ident.R_oa
            t_mass_init = t_air_meas + (self.ident.R_im * q_mass)

            logger.info(
                f"[MPC-Engine] Koude start inversie: T_mass={t_mass_init:.2f}°C"
            )
        else:
            # RUNTIME: Luenberger Observer
            error = t_air_meas - self._last_t_air_pred
            K = np.clip(self.dt / (self.ident.R_im * self.ident.C_air), 0.05, 0.5)

            t_mass_init = self._last_t_mass_model + K * error
            logger.info(
                f"[MPC-Engine] Observer update: K={K:.3f}, Error={error:.3f}K -> T_mass={t_mass_init:.2f}°C"
            )

        # Sanity check
        t_mass_init = float(np.clip(t_mass_init, t_air_meas - 1.0, t_air_meas + 5.0))
        return t_mass_init

    def solve(
        self,
        state: dict,
        forecast_df: pd.DataFrame,
        solar_gains: np.ndarray,
        avg_price: float,
        export_price: float,
        recent_df: pd.DataFrame,
    ):
        """
        Los het MILP op via SLP-iteraties.

        Parameters
        ----------
        state             : sensortoestand (room_temp, dhw_top, dhw_bottom, hvac_mode, now)
        forecast_df       : weersvoorspelling (temp, power_corrected, load_corrected)
        recent_history_df : recente wp_output voor de lag-buffer
        solar_gains       : voorspelde zonopwarming per kwartier [K]
        avg_price         : gemiddelde importprijs [euro/kWh]
        export_price      : teruglevertarief [euro/kWh]
        """
        T = self.horizon

        # ── Toestand initialiseren ────────────────────────────────────────
        r_min, r_max, d_min, d_max, r_t, d_t = self._get_targets(state["now"], T)

        t_room_init = float(state["room_temp"])
        self.P_t_air_init.value = t_room_init

        t_mass_init = self.update_and_get_initial_state(
            state, recent_df, forecast_df, solar_gains[0]
        )

        self.P_t_mass_init.value = t_mass_init
        self.P_t_dhw_init.value = float((state["dhw_top"] + state["dhw_bottom"]) / 2.0)
        self.P_init_ufh.value = (
            1.0 if state["hvac_mode"] == HvacMode.HEATING.value else 0.0
        )
        self.P_init_dhw.value = 1.0 if state["hvac_mode"] == HvacMode.DHW.value else 0.0
        self.P_comp_on_init.value = (
            1.0
            if state["hvac_mode"] in [HvacMode.HEATING.value, HvacMode.DHW.value]
            else 0.0
        )

        self.P_temp_out.value = forecast_df.temp.values[:T].astype(float)
        self.P_solar.value = forecast_df.power_corrected.values[:T].astype(float)
        self.P_base_load.value = forecast_df.load_corrected.values[:T].astype(float)
        self.P_room_min.value = r_min
        self.P_room_max.value = r_max
        self.P_dhw_min.value = d_min
        self.P_dhw_max.value = d_max
        self.P_solar_gain.value = solar_gains[:T].astype(float)
        self.P_prices.value = np.full(T, float(avg_price))
        self.P_export_prices.value = np.full(T, float(export_price))

        surplus = self.P_solar.value - self.P_base_load.value
        effective_prices = np.where(surplus > 0, float(export_price), float(avg_price))

        # Gebruik effective_prices alleen voor switching, strictness en terminal value
        effective_max_price = float(np.max(effective_prices))

        dhw_demand = self.res_dhw.predict(forecast_df)
        self.P_dhw_demand.value = dhw_demand[:T].astype(float)

        hist_p_th = np.zeros(self.L)
        if not recent_df.empty:
            # Sorteer op tijd aflopend (nieuwste eerst)
            df_sorted = recent_df.sort_values("timestamp", ascending=False)

            for i in range(self.L):
                if i < len(df_sorted):
                    row = df_sorted.iloc[i]
                    # Bereken wat het thermisch vermogen was in dat kwartier
                    if row["hvac_mode"] == HvacMode.HEATING.value:
                        # Gebruik supply/return als die er zijn, anders schatting uit perf_map
                        dt = max(0, row["supply_temp"] - row["return_temp"])
                        # Gebruik de factor uit thermal.py (bijv. 1.256)
                        hist_p_th[i] = dt * 1.256
                    else:
                        hist_p_th[i] = 0.0

        self.P_ufh_p_th_hist.value = hist_p_th

        logger.info(
            f"[Debug] DHW demand totaal: {dhw_demand[:T].sum():.2f} K  max: {dhw_demand[:T].max():.2f} K"
        )

        # Strictness: koud buiten = hogere urgentie
        delta_t_env = np.maximum(0.0, r_min + 1.0 - forecast_df.temp.values[:T])
        self.P_strictness.value = (3.0 + 0.10 * (delta_t_env**2)) * effective_prices

        # ── SLP-iteraties met convergentiecheck ──────────────────────────
        linearizer = PhysicsLinearizer(
            perf_map=self.perf_map,
            hydraulic=self.hydraulic,
            horizon=T,
            max_iter=10,
            tol=0.05,
        )

        t_out_arr = forecast_df.temp.values[:T].astype(float)
        guessed_t_mass = np.full(T, t_room_init)
        guessed_t_dhw = np.full(T, float(state["dhw_bottom"]))

        # Sentinel: convergentiecheck slaat eerste iteratie altijd over
        p_el_ufh_prev = np.full(T, -999.0)
        p_el_dhw_prev = np.full(T, -999.0)

        p_el_ufh, cop_ufh, p_el_dhw, cop_dhw, sup_ufh, sup_dhw = linearizer.compute(
            t_out_arr, guessed_t_mass, guessed_t_dhw
        )

        # Terminal value updaten op basis van actuele COP-schatting
        avg_cop_ufh_h = float(np.mean(cop_ufh))
        avg_cop_dhw_h = float(np.mean(cop_dhw))

        # Voor de fysieke waarde en boete telt de VOLLEDIGE op te warmen massa
        C_total_room = self.ident.C_air + self.ident.C_mass

        # ── Comfortboetes en Terminal Values (dimensioneel correct) ──
        # We voeden de maximale toekomstige prijs in, zodat comfort de hoogste economische prioriteit krijgt
        costs = ComfortCostCalculator(
            C_room=C_total_room,
            C_tank=self.ident.C_tank,
            avg_cop_ufh=avg_cop_ufh_h,
            avg_cop_dhw=avg_cop_dhw_h,
        ).compute(effective_max_price)

        self.P_cost_room_under.value = costs["room_under"]
        self.P_cost_room_over.value = costs["room_over"]
        self.P_cost_dhw_under.value = costs["tank_under"]
        self.P_cost_dhw_over.value = costs["tank_over"]

        self.P_val_terminal_room.value = costs["terminal_room"]
        self.P_val_terminal_dhw.value = costs["terminal_tank"]

        self.P_terminal_offset.value = (
            r_t[-1] * costs["terminal_room"] + d_t[-1] * costs["terminal_tank"]
        )

        logger.info(
            f"[Terminal] val_room={costs['terminal_room']:.4f} €/K  "
            f"val_dhw={costs['terminal_tank']:.4f} €/K  "
            f"max_future_price={effective_max_price:.3f} €/kWh"
        )

        for iteration in range(linearizer.max_iter):
            # 1. Bereken bevroren vermogen en COP op basis van huidige schatting
            p_el_ufh, cop_ufh, p_el_dhw, cop_dhw, sup_ufh, sup_dhw = linearizer.compute(
                t_out_arr, guessed_t_mass, guessed_t_dhw
            )

            self.plan_t_sup_ufh = sup_ufh
            self.plan_t_sup_dhw = sup_dhw
            self.plan_p_el_ufh = p_el_ufh
            self.plan_p_el_dhw = p_el_dhw

            # 3. CVXPY-parameters vullen
            self.P_fixed_pel_ufh.value = np.clip(p_el_ufh, 0.0, 5.0)
            self.P_fixed_pel_dhw.value = np.clip(p_el_dhw, 0.0, 5.0)
            self.P_cop_ufh.value = np.clip(cop_ufh, 1.5, 9.0)
            self.P_cop_dhw.value = np.clip(cop_dhw, 1.1, 5.0)

            # 4. Convergentiecheck vóór oplossen
            if iteration > 0 and linearizer.has_converged(
                p_el_ufh_prev,
                p_el_dhw_prev,
                p_el_ufh,
                p_el_dhw,
                ufh_on=self.ufh_on.value,
                dhw_on=self.dhw_on.value,
            ):
                logger.info(f"[SLP] Geconvergeerd na {iteration} iteraties.")
                break

            p_el_ufh_prev = p_el_ufh.copy()
            p_el_dhw_prev = p_el_dhw.copy()

            # 5. Oplossen
            try:
                self.problem.solve(solver=cp.HIGHS, warm_start=True)
            except Exception as e:
                logger.error(f"[SLP] Solver exception in iteratie {iteration}: {e}")
                break

            status = self.problem.status
            logger.info(
                f"[SLP] Iteratie {iteration}: status={status}  "
                f"P_el UFH[0]={p_el_ufh[0]:.3f} kW  DHW[0]={p_el_dhw[0]:.3f} kW"
            )

            if status not in ("optimal", "optimal_inaccurate"):
                logger.warning(
                    f"[SLP] Niet-optimale status ({status}) in iteratie {iteration}"
                )
                break

            # 6. Toestandstrajectorie bijwerken voor volgende iteratie
            if self.t_air.value is not None:
                self._last_t_air_pred = float(self.t_air.value[1])
                self._last_t_mass_model = float(self.t_mass.value[1])

                new_t_mass = self.t_mass.value[:-1].copy()
                new_t_dhw = self.t_dhw.value[:-1].copy()

                if iteration == 0:
                    guessed_t_mass = new_t_mass
                    guessed_t_dhw = new_t_dhw
                else:
                    delta_mass = float(np.mean(np.abs(new_t_mass - guessed_t_mass)))
                    delta_dhw = float(np.mean(np.abs(new_t_dhw - guessed_t_dhw)))

                    alpha_mass = float(np.clip(0.6 / (1.0 + delta_mass), 0.20, 0.70))
                    alpha_dhw = float(np.clip(0.3 / (1.0 + delta_dhw), 0.10, 0.45))

                    guessed_t_mass = (
                        alpha_mass * new_t_mass + (1 - alpha_mass) * guessed_t_mass
                    )
                    guessed_t_dhw = (
                        alpha_dhw * new_t_dhw + (1 - alpha_dhw) * guessed_t_dhw
                    )

        # Update met correct data
        if self.t_mass.value is not None:
            final_t_mass = self.t_mass.value[:-1]
            final_t_dhw = self.t_dhw.value[:-1]
            p_el_ufh, cop_ufh, p_el_dhw, cop_dhw, sup_ufh, sup_dhw = linearizer.compute(
                t_out_arr, final_t_mass, final_t_dhw
            )

            self.plan_p_el_ufh = p_el_ufh
            self.plan_p_el_dhw = p_el_dhw
            self.plan_t_sup_ufh = sup_ufh
            self.plan_t_sup_dhw = sup_dhw
            self.P_cop_ufh.value = cop_ufh
            self.P_cop_dhw.value = cop_dhw

        if self.t_air.value is not None:
            ufh_steps = [t for t in range(T) if self.ufh_on.value[t] > 0.5]
            if ufh_steps:
                for t in [ufh_steps[0], ufh_steps[-1]]:
                    logger.info(
                        f"[Debug] t={t:3d} UFH_on  "
                        f"t_air={self.t_air.value[t + 1]:.2f}  "
                        f"t_mass={self.t_mass.value[t + 1]:.2f}  "
                        f"s_low={self.s_room_low.value[t]:.4f}  "
                        f"r_min={self.P_room_min.value[t]:.1f}"
                    )
                logger.info(
                    f"[Debug] UFH {len(ufh_steps)} stappen  "
                    f"terminal t_mass={self.t_mass.value[T]:.2f}  "
                    f"t_air={self.t_air.value[T]:.2f}"
                )


# =========================================================
# OPTIMIZER
# =========================================================


class Optimizer:
    def __init__(self, config, database):
        self.db = database
        self.perf_map = HPPerformanceMap(config.hp_model_path)
        self.ident = SystemIdentificator(config.rc_model_path, config.tank_liters)
        self.hydraulic = HydraulicPredictor(config.hydraulic_model_path)
        self.res_ufh = UfhResidualPredictor(
            config.ufh_model_path, self.ident.R, self.ident.C
        )
        self.res_dhw = DhwResidualPredictor(config.dhw_model_path)
        self.shutter = ShutterPredictor(config.shutter_model_path)
        self.mpc = ThermalMPC(self.ident, self.perf_map, self.hydraulic, self.res_dhw)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)
        df = self.db.get_history(cutoff_date=cutoff)
        if df.empty:
            logger.warning("[Optimizer] Geen trainingsdata beschikbaar.")
            return

        self.perf_map.train(df)
        self.ident.train(df)
        self.hydraulic.train(df)
        self.res_ufh.train(df)
        self.res_dhw.train(df)
        self.shutter.train(df)

        # Herbouw MPC zodat R, C en lag correct zijn na training
        self.mpc._build_problem()

    def resolve(self, context: Context, avg_price: float, export_price: float):
        # 1. Data verzamelen
        recent_df = self.db.get_history(cutoff_date=context.now - timedelta(hours=2))

        state = {
            "now": context.now,
            "hvac_mode": context.hvac_mode.value,
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        # Zonopwarming voorspellen
        shutter_room = float(getattr(context, "shutter_room", 100.0))
        shutters = self.shutter.predict(context.forecast_df, shutter_room)
        solar_gains = self.res_ufh.predict(context.forecast_df, shutters)

        logger.info(
            f"[Optimizer] Solar gain: max={np.max(solar_gains):.3f}  "
            f"gem={np.mean(solar_gains):.3f} K/uur"
        )

        # Oplossen
        self.mpc.solve(
            state=state,
            forecast_df=context.forecast_df,
            solar_gains=solar_gains,
            avg_price=avg_price,
            export_price=export_price,
            recent_df=recent_df,
        )

        if self.mpc.t_mass.value is not None:
            self._t_mass_estimate = float(self.mpc.t_mass.value[1])

        # Lege uitvoer bij solver-fout
        if self.mpc.p_el_ufh.value is None:
            return {
                "mode": "OFF",
                "status": self.mpc.problem.status,
                "target_pel_kw": 0.0,
                "target_supply_temp": 0.0,
                "steps_remaining": 0,
                "pv_remaining": 0.0,
                "solar_self_remaining": 0.0,
                "export_remaining": 0.0,
                "grid_remaining": 0.0,
                "plan": [],
            }

        # Dagstatistieken
        tz = datetime.now().astimezone().tzinfo
        now_local = context.now.astimezone(tz)
        today = now_local.date()

        steps_remaining = sum(
            1
            for t in range(self.mpc.horizon)
            if (now_local + timedelta(minutes=t * 15)).date() == today
        )

        pv_rem = solar_self_rem = export_rem = grid_rem = 0.0
        if steps_remaining > 0:
            sl = steps_remaining
            pv_rem = float(np.sum(self.mpc.P_solar.value[:sl]) * self.mpc.dt)
            solar_self_rem = float(
                np.sum(self.mpc.p_solar_self.value[:sl]) * self.mpc.dt
            )
            export_rem = float(np.sum(self.mpc.p_export.value[:sl]) * self.mpc.dt)
            grid_rem = float(np.sum(self.mpc.p_grid.value[:sl]) * self.mpc.dt)

        # Beslissing voor tijdstap 0
        p_el_ufh_now = float(self.mpc.p_el_ufh.value[0])
        p_el_dhw_now = float(self.mpc.p_el_dhw.value[0])

        mode = "OFF"
        target_pel = 0.0
        target_supply_temp = 0.0

        if p_el_dhw_now > 0.1:
            mode = "DHW"
            target_pel = p_el_dhw_now
            target_supply_temp = float(self.mpc.plan_t_sup_dhw[0])
        elif p_el_ufh_now > 0.1:
            mode = "UFH"
            target_pel = p_el_ufh_now
            target_supply_temp = float(self.mpc.plan_t_sup_ufh[0])

        plan = self.get_plan(context, shutters)

        total_cost_net = sum(
            float(r["cost_net"]) for r in plan if (r["time"].date() == today)
        )
        total_saving = sum(
            float(r["cost_saving"]) for r in plan if (r["time"].date() == today)
        )

        total_cost_gross = sum(
            float(r["cost_gross"]) for r in plan if r["time"].date() == today
        )
        total_export_revenue = sum(
            float(self.mpc.p_export.value[t]) * float(export_price) * self.mpc.dt
            for t in range(steps_remaining)
        )

        return {
            "status": self.mpc.problem.status,
            "mode": mode,
            "target_pel_kw": round(target_pel, 2),
            "target_supply_temp": round(target_supply_temp, 1),
            "steps_remaining": steps_remaining,
            "pv_remaining": round(pv_rem, 3),
            "solar_self_remaining": round(solar_self_rem, 3),
            "export_remaining": round(export_rem, 3),
            "grid_remaining": round(grid_rem, 3),
            "total_cost_net": round(total_cost_net, 2),
            "total_saving": round(total_saving, 2),
            "total_cost_gross": round(total_cost_gross, 2),
            "total_export_revenue": round(total_export_revenue, 2),
            "plan": plan,
        }

    def get_plan(self, context: Context, shutters: np.ndarray) -> list:
        if self.mpc.p_el_ufh.value is None:
            return []

        T = self.mpc.horizon
        minute = (context.now.minute // 15) * 15
        start = context.now.replace(minute=minute, second=0, microsecond=0)

        u_on = self.mpc.ufh_on.value
        d_on = self.mpc.dhw_on.value
        p_u = self.mpc.plan_p_el_ufh
        p_d = self.mpc.plan_p_el_dhw
        t_r = self.mpc.t_air.value
        t_d = self.mpc.t_dhw.value
        u_cop = self.mpc.P_cop_ufh.value
        d_cop = self.mpc.P_cop_dhw.value
        u_sup = self.mpc.plan_t_sup_ufh
        d_sup = self.mpc.plan_t_sup_dhw
        strict = self.mpc.P_strictness.value

        prices = self.mpc.P_prices.value  # importprijs per kwartier
        export_prices = self.mpc.P_export_prices.value  # exportprijs per kwartier
        grid = self.mpc.p_grid.value
        export = self.mpc.p_export.value

        plan = []
        for t in range(T):
            ts = start + timedelta(minutes=t * 15)

            if d_on[t] > 0.5:
                mode_str = "DHW"
                hvac_mode = HvacMode.DHW
            elif u_on[t] > 0.5:
                mode_str = "UFH"
                hvac_mode = HvacMode.HEATING
            else:
                mode_str = "-"
                hvac_mode = HvacMode.OFF

            shutter_val = float(shutters[t]) if t < len(shutters) else float("nan")

            # Bruto kosten zonder zon = alles tegen importprijs
            solar_self = float(self.mpc.p_solar_self.value[t])
            total_load = p_u[t] + p_d[t] + float(self.mpc.P_base_load.value[t])

            cost_gross_val = total_load * prices[t] * self.mpc.dt
            cost_net_val = (
                grid[t] * prices[t] - export[t] * export_prices[t]
            ) * self.mpc.dt
            cost_saving_val = max(0.0, solar_self * prices[t] * self.mpc.dt)

            plan.append(
                {
                    "time": ts,
                    "mode": mode_str,
                    "hvac_mode": hvac_mode.value,
                    "t_out": f"{context.forecast_df.temp.iloc[t]:.2f}",
                    "p_solar": f"{context.forecast_df.power_corrected.iloc[t]:.2f}",
                    "p_load": f"{context.forecast_df.load_corrected.iloc[t]:.2f}",
                    "t_room": f"{t_r[t]:.2f}",
                    "t_dhw": f"{t_d[t]:.2f}",
                    "p_el_ufh": f"{p_u[t] if mode_str == 'UFH' else 0.0:.2f}",
                    "p_el_dhw": f"{p_d[t] if mode_str == 'DHW' else 0.0:.2f}",
                    "cop_ufh": f"{u_cop[t]:.2f}",
                    "cop_dhw": f"{d_cop[t]:.2f}",
                    "supply_ufh": f"{u_sup[t]:.2f}",
                    "supply_dhw": f"{d_sup[t]:.2f}",
                    "strict": f"{strict[t]:.0f}",
                    "shutter": f"{shutter_val:.0f}",
                    "price": f"{prices[t]:.2f}",
                    "cost_ufh": f"{p_u[t] * prices[t] * self.mpc.dt:.3f}",
                    "cost_dhw": f"{p_d[t] * prices[t] * self.mpc.dt:.3f}",
                    "cost_gross": f"{cost_gross_val:.3f}",
                    "cost_net": f"{cost_net_val:.3f}",
                    "cost_saving": f"{cost_saving_val:.3f}",
                }
            )

        return plan
