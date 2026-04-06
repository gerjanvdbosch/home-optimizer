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
import tzlocal

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from context import Context, HvacMode
from thermal import (
    SystemIdentificator,
    HPPerformanceMap,
    HydraulicPredictor,
    UfhResidualPredictor,
    DhwResidualPredictor,
    ComfortCostCalculator,
    ThermalEKF,
    PWATable,
    FACTOR_UFH,
)
from shutter import ShutterPredictor

logger = logging.getLogger(__name__)


# =========================================================
# THERMAL MPC (DPP compliant)
# =========================================================


class ThermalMPC:
    def __init__(self, config, ident, perf_map, hydraulic, res_dhw):
        self.config = config
        self.ident = ident
        self.perf_map = perf_map
        self.hydraulic = hydraulic
        self.res_dhw = res_dhw

        # Multi-resolutie
        self.dt_steps = np.array(
            [0.25] * 96  # Eerste: 15 min nauwkeurigheid
            + [1.00] * 8  # Volgende: 1 uur stappen
        )
        self.horizon = len(self.dt_steps)

        # EKF vervangt Luenberger observer
        self.ekf = ThermalEKF(ident)

        # PWA vervangt PhysicsLinearizer
        self.pwa = PWATable(perf_map, hydraulic)

        self.plan_t_sup_ufh = np.zeros(self.horizon)
        self.plan_t_sup_dhw = np.zeros(self.horizon)

        # Warm-start state
        self._prev_ufh_on = None
        self._prev_dhw_on = None
        self._prev_t_mass = None
        self._prev_t_dhw = None
        self._prev_obj = None

        self._build_problem()

    def _build_problem(self):
        T = self.horizon
        DT = self.dt_steps
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

        self.P_cost_room_under = cp.Parameter(nonneg=True)
        self.P_cost_room_over = cp.Parameter(nonneg=True)
        self.P_cost_dhw_under = cp.Parameter(nonneg=True)
        self.P_cost_dhw_over = cp.Parameter(nonneg=True)
        self.P_val_terminal_room = cp.Parameter(nonneg=True)
        self.P_val_terminal_dhw = cp.Parameter(nonneg=True)

        # Taylor parameters (Constante + Helling) voor P_el en P_th
        self.P_pel_const_ufh = cp.Parameter(T)
        self.P_pel_slope_ufh = cp.Parameter(T)
        self.P_pth_const_ufh = cp.Parameter(T)
        self.P_pth_slope_ufh = cp.Parameter(T)

        self.P_pel_const_dhw = cp.Parameter(T)
        self.P_pel_slope_dhw = cp.Parameter(T)
        self.P_pth_const_dhw = cp.Parameter(T)
        self.P_pth_slope_dhw = cp.Parameter(T)

        self.P_ufh_p_th_hist = cp.Parameter(self.L, nonneg=True)

        self.P_dhw_demand = cp.Parameter(T, nonneg=True)

        self.P_term_target_mass = cp.Parameter()
        self.P_term_target_dhw = cp.Parameter()

        # ── Variabelen ────────────────────────────────────────────────────
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

        self.s_term_room_under = cp.Variable(nonneg=True)  # Voor de terminal value
        self.s_term_dhw_under = cp.Variable(nonneg=True)  # Voor de terminal value

        # ── Variabelen ────────────────────────────────────────────────────
        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)
        self.z_ufh = cp.Variable(T)  # = ufh_on[t] × t_mass[t]
        self.z_dhw = cp.Variable(T)  # = dhw_on[t] × t_dhw[t]

        # Binaire McCormick: z = on × t_sink
        # Binair × continu geeft exacte (niet-relaxte) envelopen
        T_MASS_MIN, T_MASS_MAX = 10.0, 35.0
        T_DHW_MIN, T_DHW_MAX = 10.0, 75.0

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
            dt_t = float(DT[t])  # Gebruik de constante waarde

            p_el_wp = self.p_el_ufh[t] + self.p_el_dhw[t]

            p_th_dhw_now = (
                self.P_pth_const_dhw[t] * self.dhw_on[t]
                + self.P_pth_slope_dhw[t] * self.z_dhw[t]
            )

            if t < self.L:
                p_th_delayed = self.P_ufh_p_th_hist[self.L - 1 - t]
            else:
                t_lag = t - self.L
                p_th_delayed = (
                    self.P_pth_const_ufh[t_lag] * self.ufh_on[t_lag]
                    + self.P_pth_slope_ufh[t_lag] * self.z_ufh[t_lag]
                )

            constraints += [
                # McCormick envelopen blijven behouden...
                self.z_ufh[t] >= T_MASS_MIN * self.ufh_on[t],
                self.z_ufh[t] <= T_MASS_MAX * self.ufh_on[t],
                self.z_ufh[t] >= self.t_mass[t] - T_MASS_MAX * (1 - self.ufh_on[t]),
                self.z_ufh[t] <= self.t_mass[t] - T_MASS_MIN * (1 - self.ufh_on[t]),
                self.z_dhw[t] >= T_DHW_MIN * self.dhw_on[t],
                self.z_dhw[t] <= T_DHW_MAX * self.dhw_on[t],
                self.z_dhw[t] >= self.t_dhw[t] - T_DHW_MAX * (1 - self.dhw_on[t]),
                self.z_dhw[t] <= self.t_dhw[t] - T_DHW_MIN * (1 - self.dhw_on[t]),
                # ── NIEUW: P_el is nu een continue, berekende variabele gebaseerd op state!
                # P_el = Constante * on + Helling * (on * T_sink)
                self.p_el_ufh[t]
                >= self.P_pel_const_ufh[t] * self.ufh_on[t]
                + self.P_pel_slope_ufh[t] * self.z_ufh[t],
                self.p_el_dhw[t]
                >= self.P_pel_const_dhw[t] * self.dhw_on[t]
                + self.P_pel_slope_dhw[t] * self.z_dhw[t],
                # ── Stroombalans ────────────────────────────────────────────
                p_el_wp + self.P_base_load[t] == self.p_grid[t] + self.p_solar_self[t],
                self.P_solar[t] == self.p_solar_self[t] + self.p_export[t],
                # ── Thermische dynamica — p_th via affiene COP ──────────────
                # t_air: zon is al een totaal-pakket, verlies-termen gaan per uur (* dt)
                self.t_air[t + 1]
                == self.t_air[t]
                + (
                    (self.t_mass[t] - self.t_air[t]) / R_im
                    - (self.t_air[t] - self.P_temp_out[t]) / R_oa
                )
                * (dt_t / C_air)
                + self.P_solar_gain[t],
                self.t_mass[t + 1]
                == self.t_mass[t]
                + (p_th_delayed - (self.t_mass[t] - self.t_air[t]) / R_im)
                * (dt_t / C_mass),
                self.t_dhw[t + 1]
                == self.t_dhw[t]
                + (
                    p_th_dhw_now * (dt_t / self.ident.C_tank)
                    - (self.t_dhw[t] - self.t_air[t]) * (self.ident.K_loss_dhw * dt_t)
                )
                - self.P_dhw_demand[t],
                self.ufh_on[t] + self.dhw_on[t] <= 1,
                self.t_air[t + 1] + self.s_room_low[t] >= self.P_room_min[t],
                self.t_air[t + 1] - self.s_room_high[t] <= self.P_room_max[t],
                self.t_dhw[t + 1] + self.s_dhw_low[t] >= self.P_dhw_min[t],
                self.t_dhw[t + 1] - self.s_dhw_high[t] <= self.P_dhw_max[t],
                self.t_mass[T] + self.s_term_room_under >= self.P_term_target_mass,
                self.t_dhw[T] + self.s_term_dhw_under >= self.P_term_target_dhw,
                # Ondergrens (modulatie-bodem)
                self.p_el_ufh[t] >= self.perf_map._pel_min_ufh * self.ufh_on[t],
                self.p_el_dhw[t] >= self.perf_map._pel_min_dhw * self.dhw_on[t],
                # Bovengrens (fysiek maximum van de pomp)
                self.p_el_ufh[t] <= self.perf_map._pel_max_ufh * self.ufh_on[t],
                self.p_el_dhw[t] <= self.perf_map._pel_max_dhw * self.dhw_on[t],
            ]

        # ── Doelfunctie ───────────────────────────────────────────────────
        # Kosten = Som ( (P_grid * Prijs - P_export * ExportPrijs) * dt_stap )
        net_cost = cp.sum(
            cp.multiply(self.p_grid, self.P_prices)
            - cp.multiply(self.p_export, self.P_export_prices)
        )

        comfort = cp.sum(
            self.s_room_low * self.P_cost_room_under
            + self.s_room_high * self.P_cost_room_over
            + self.s_dhw_low * self.P_cost_dhw_under
            + self.s_dhw_high * self.P_cost_dhw_over
        )

        # Hoge straf op compressorstart, lage straf op klep-wissel
        valve_switches = (
            cp.pos(self.ufh_on[0] - self.P_init_ufh)
            + cp.sum(cp.pos(self.ufh_on[1:] - self.ufh_on[:-1]))
            + cp.pos(self.dhw_on[0] - self.P_init_dhw)
            + cp.sum(cp.pos(self.dhw_on[1:] - self.dhw_on[:-1]))
        )

        # NIEUW: Tie-breaker. Maak elke actie héél iets goedkoper naarmate het later in de tijd valt.
        # Penalty gaat van 0.0 (nu) naar -0.01 (einde horizon).
        # Voorkomt willekeurig thread-gedrag bij platte prijzen.
        time_preference = cp.sum(
            cp.multiply(self.ufh_on + self.dhw_on, np.linspace(0.0, -0.01, T))
        )

        switching = (
            (cp.sum(self.comp_start) * 0.1) + (valve_switches * 0.1) + time_preference
        )

        terminal_penalty = (
            self.s_term_room_under * self.P_val_terminal_room
            + self.s_term_dhw_under * self.P_val_terminal_dhw
        )

        self.problem = cp.Problem(
            cp.Minimize(net_cost + comfort + switching + terminal_penalty),
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

    def _get_targets(self, now, T, use_15min=False):
        local_tz = ZoneInfo(tzlocal.get_localzone_name())
        now_local = now.astimezone(local_tz)

        if use_15min:
            # Raster voor de grafiek: altijd 15 minuten
            offsets = np.arange(T) * 0.25
        else:
            # Raster voor de solver: elastisch (15m, 1u, 2u)
            offsets = np.concatenate(([0], np.cumsum(self.dt_steps[:-1])))

        future_times = [now_local + timedelta(hours=float(h)) for h in offsets]

        # Kamer targets ophalen via interpolatie
        r_t, r_l, r_h = self._get_interpolated_values(
            self.config.room_target, future_times
        )
        r_min = r_t - r_l
        r_max = r_t + r_h

        # Boiler targets ophalen via interpolatie
        d_t, d_l, d_h = self._get_interpolated_values(
            self.config.dhw_target, future_times
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
        t_air_meas = float(state["room_temp"])
        t_out_now = float(forecast_df.temp.iloc[0])

        if self.ekf.x is None:
            # Koude start: fysische inversie als initialisatie
            dT_dt = self._robust_trend(recent_df)
            q_mass = (self.ident.C_air * (dT_dt - current_solar_gain)) + (
                t_air_meas - t_out_now
            ) / self.ident.R_oa
            t_mass_init = float(
                np.clip(
                    t_air_meas + self.ident.R_im * q_mass,
                    t_air_meas - 1.0,
                    t_air_meas + 5.0,
                )
            )
            self.ekf.reset(t_air_meas, t_mass_init)
            logger.info(f"[EKF] Koude start: T_mass={t_mass_init:.2f}°C")
        else:
            # Runtime: EKF predict + update
            p_heat_prev = 0.0
            if not recent_df.empty:
                last = recent_df.sort_values("timestamp").iloc[-1]
                if last["hvac_mode"] == HvacMode.HEATING.value:
                    dt = max(0, last["supply_temp"] - last["return_temp"])
                    p_heat_prev = dt * FACTOR_UFH

            self.ekf.predict_step(p_heat_prev, t_out_now, current_solar_gain)
            _, t_mass_init = self.ekf.update(t_air_meas)

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
        DT = self.dt_steps

        # ── Toestand initialiseren ────────────────────────────────────────
        r_min, r_max, d_min, d_max, r_t, d_t = self._get_targets(state["now"], T)

        t_air_meas = float(state["room_temp"])
        t_top_measured = float(state["dhw_top"])
        t_bot_measured = float(state["dhw_bottom"])

        t_avg_measured = (t_top_measured + t_bot_measured) / 2.0
        # Stratification Gap: Hoeveel kouder is de onderkant t.o.v. het gemiddelde?
        strat_gap_measured = max(0.0, t_avg_measured - t_bot_measured)

        # 2. INITIALISEER DE START-TOESTAND (Parameters)
        t_mass_init = self.update_and_get_initial_state(
            state, recent_df, forecast_df, solar_gains[0]
        )

        self.P_t_air_init.value = t_air_meas
        self.P_t_mass_init.value = t_mass_init
        self.P_t_dhw_init.value = (
            t_avg_measured  # De solver rekent energie-balans op Average
        )

        self.P_init_ufh.value = (
            1.0 if state["hvac_mode"] == HvacMode.HEATING.value else 0.0
        )
        self.P_init_dhw.value = 1.0 if state["hvac_mode"] == HvacMode.DHW.value else 0.0
        self.P_comp_on_init.value = (
            1.0
            if state["hvac_mode"] in [HvacMode.HEATING.value, HvacMode.DHW.value]
            else 0.0
        )

        # 1. Bouw tijdas indices (Kwartier-mapping naar Solver-stappen)
        quarters_per_step = np.maximum(1, np.round(self.dt_steps / 0.25).astype(int))
        cum_q = np.concatenate(([0], np.cumsum(quarters_per_step)))
        src_len = len(forecast_df)

        def resample_mean(data):
            """Voor intensieve variabelen: Temperatuur, Prijs per kWh, Vermogen in kW"""
            out = np.zeros(T)
            for i in range(T):
                lo, hi = cum_q[i], cum_q[i + 1]
                # Pak de beschikbare data, maar blijf binnen de grenzen van de forecast
                chunk = data[min(lo, src_len - 1) : min(hi, src_len)]
                out[i] = (
                    np.mean(chunk) if len(chunk) > 0 else data[min(lo, src_len - 1)]
                )
            return out

        def resample_sum(data):
            """Voor extensieve variabelen: Zon-opwarming (Kelvin), Waterverbruik (Liters)"""
            out = np.zeros(T)
            for i in range(T):
                lo, hi = cum_q[i], cum_q[i + 1]
                chunk = data[min(lo, src_len - 1) : min(hi, src_len)]
                out[i] = np.sum(chunk) if len(chunk) > 0 else 0.0
            return out

        t_out_arr = resample_mean(forecast_df.temp.values)
        dhw_demand_raw = self.res_dhw.predict(forecast_df)

        # 2. Resample alle inputs
        self.P_temp_out.value = t_out_arr
        self.P_solar.value = resample_mean(forecast_df.power_corrected.values)
        self.P_base_load.value = resample_mean(forecast_df.load_corrected.values)
        self.P_prices.value = np.full(T, float(avg_price)) * DT
        self.P_export_prices.value = np.full(T, float(export_price)) * DT

        # Energie-pakketten (Sommeren!)
        self.P_solar_gain.value = resample_sum(solar_gains)
        self.P_dhw_demand.value = resample_sum(dhw_demand_raw)

        self.P_room_min.value = r_min
        self.P_room_max.value = r_max
        self.P_dhw_min.value = d_min
        self.P_dhw_max.value = d_max

        surplus = self.P_solar.value - self.P_base_load.value
        effective_prices = np.where(surplus > 0, float(export_price), float(avg_price))

        # Gebruik effective_prices alleen voor switching, strictness en terminal value
        effective_max_price = float(np.max(effective_prices))

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
                        hist_p_th[i] = dt * FACTOR_UFH
                    else:
                        hist_p_th[i] = 0.0

        self.P_ufh_p_th_hist.value = hist_p_th

        logger.info(
            f"[Debug] DHW demand totaal: {dhw_demand_raw.sum():.2f} K max: {dhw_demand_raw.max():.2f} K"
        )

        # 3. INITIALISEER DE 'GUESSED' CURVES (Error-Corrected Receding Horizon)
        if self._prev_t_mass is not None and len(self._prev_t_mass) == T:
            # Bereken de fouten op de vorige voorspelling
            error_mass = t_mass_init - self._prev_t_mass[1]
            error_dhw = t_avg_measured - self._prev_t_dhw[1]

            # Schuif de horizon door en pas de fout-correctie toe
            guessed_t_mass = (
                np.append(self._prev_t_mass[1:], self._prev_t_mass[-1]) + error_mass
            )
            guessed_t_dhw = (
                np.append(self._prev_t_dhw[1:], self._prev_t_dhw[-1]) + error_dhw
            )

            logger.info(
                f"[MPC] Error-Corrected Horizon. "
                f"Afwijking: Mass {error_mass:+.2f}K, DHW {error_dhw:+.2f}K | "
                f"Stratification Gap: {strat_gap_measured:.1f}K"
            )
        else:
            # Fallback bij koude start
            guessed_t_mass = np.full(T, t_mass_init)
            guessed_t_dhw = np.full(T, t_avg_measured)
            logger.info("[MPC] Start met platte fallback-state.")

        costs_set = False
        prev_obj = None
        MAX_OUTER = 5

        best_obj = np.inf
        best_state = {}

        for outer in range(MAX_OUTER):
            if state["hvac_mode"] == HvacMode.DHW.value:
                # Als de pomp draait, is de tank gemengd (Gap = 0)
                guessed_t_sink = guessed_t_dhw
            else:
                # Bij stilstand behouden we de gemeten gelaagdheid (Gap)
                guessed_t_sink = guessed_t_dhw - strat_gap_measured

            # 1. PWA op huidige schatting
            pwa_data = self.pwa.compute(t_out_arr, guessed_t_mass, guessed_t_sink)
            self.plan_t_sup_ufh = pwa_data["sup_ufh"]
            self.plan_t_sup_dhw = pwa_data["sup_dhw"]

            # 2. MILP parameters invullen (Taylor Coëfficiënten)
            self.P_pel_const_ufh.value = pwa_data["pel_const_ufh"]
            self.P_pel_slope_ufh.value = pwa_data["pel_slope_ufh"]
            self.P_pth_const_ufh.value = pwa_data["pth_const_ufh"]
            self.P_pth_slope_ufh.value = pwa_data["pth_slope_ufh"]

            self.P_pel_const_dhw.value = pwa_data["pel_const_dhw"]
            self.P_pel_slope_dhw.value = pwa_data["pel_slope_dhw"]
            self.P_pth_const_dhw.value = pwa_data["pth_const_dhw"]
            self.P_pth_slope_dhw.value = pwa_data["pth_slope_dhw"]

            # 3. Comfortboetes (gebruik gemiddelde COP van deze iteratie)
            if not costs_set:
                C_total_room = self.ident.C_air + self.ident.C_mass
                costs = ComfortCostCalculator(
                    C_room=C_total_room,
                    C_tank=self.ident.C_tank,
                    avg_cop_ufh=float(np.mean(pwa_data["avg_cop_ufh"])),
                    avg_cop_dhw=float(np.mean(pwa_data["avg_cop_dhw"])),
                    export_price=float(export_price),
                ).compute(max_price=effective_max_price)

                self.P_cost_room_under.value = costs["room_under"] * np.mean(DT)
                self.P_cost_room_over.value = costs["room_over"] * np.mean(DT)
                self.P_cost_dhw_under.value = costs["tank_under"] * np.mean(DT)
                self.P_cost_dhw_over.value = costs["tank_over"] * np.mean(DT)
                self.P_val_terminal_room.value = costs["terminal_room"]
                self.P_val_terminal_dhw.value = costs["terminal_tank"]
                t_out_end = float(t_out_arr[-1])
                self.P_term_target_mass.value = float(
                    r_t[-1] + self.ident.R_im * (r_t[-1] - t_out_end) / self.ident.R_oa
                )
                self.P_term_target_dhw.value = float(d_t[-1])
                costs_set = True

            # 4. Oplossen
            try:
                self.problem.solve(
                    solver=cp.HIGHS,
                    warm_start=(outer > 0 or self._prev_ufh_on is not None),
                    verbose=False,
                    highs_options={
                        "mip_rel_gap": 0.01,  # stop bij % van optimum
                        "mip_abs_gap": 0.01,  # absolute gap in euro
                        "time_limit": 300.0,  # maximaal 300 seconden per solve
                        "presolve": "on",
                        "parallel": "on",  # gebruik meerdere cores
                        "threads": 3,  # aantal threads
                    },
                )
            except Exception as e:
                logger.error(f"[MPC] Solver exception in outer={outer}: {e}")
                break

            status = self.problem.status
            obj = self.problem.value if self.problem.value is not None else np.inf
            logger.info(f"[MPC] Outer={outer} status={status} obj={obj:.4f}")

            if status not in ("optimal", "optimal_inaccurate"):
                logger.warning(f"[MPC] Niet-optimale status: {status}")
                break

            # Sla de beste oplossing op
            if obj < best_obj and self.t_air.value is not None:
                best_obj = obj
                best_state = {
                    "ufh_on": self.ufh_on.value.copy(),
                    "dhw_on": self.dhw_on.value.copy(),
                    "p_el_ufh": self.p_el_ufh.value.copy(),
                    "p_el_dhw": self.p_el_dhw.value.copy(),
                    "t_air": self.t_air.value.copy(),
                    "t_mass": self.t_mass.value.copy(),
                    "t_dhw": self.t_dhw.value.copy(),
                    "p_grid": self.p_grid.value.copy(),
                    "p_export": self.p_export.value.copy(),
                    "p_solar_self": self.p_solar_self.value.copy(),
                    "z_ufh": self.z_ufh.value.copy(),
                    "z_dhw": self.z_dhw.value.copy(),
                    "s_room_low": self.s_room_low.value.copy(),
                    "s_room_high": self.s_room_high.value.copy(),
                    "s_dhw_low": self.s_dhw_low.value.copy(),
                    "s_dhw_high": self.s_dhw_high.value.copy(),
                    "sup_ufh": pwa_data["sup_ufh"].copy(),
                    "sup_dhw": pwa_data["sup_dhw"].copy(),
                }
                logger.info(f"[MPC] Beste oplossing bijgewerkt: obj={best_obj:.4f}")

            # Convergentiecheck
            if (
                prev_obj is not None
                and abs(obj - prev_obj) / (abs(prev_obj) + 1e-6) < 1e-3
            ):
                logger.info(f"[MPC] Geconvergeerd na {outer + 1} iteraties")
                break
            prev_obj = obj

            # 6. Update schatting: solver heeft nu de echte trajectorie gezien
            if self.t_mass.value is not None:
                guessed_t_mass = self.t_mass.value[:-1].copy()
                guessed_t_dhw = self.t_dhw.value[:-1].copy()

        # ── Herstel de beste oplossing ────────────────────────────────────
        if not best_state:
            logger.warning("[MPC] Geen geldige oplossing gevonden")
            return

        self.ufh_on.value = best_state["ufh_on"]
        self.dhw_on.value = best_state["dhw_on"]
        self.p_el_ufh.value = best_state["p_el_ufh"]
        self.p_el_dhw.value = best_state["p_el_dhw"]
        self.t_air.value = best_state["t_air"]
        self.t_mass.value = best_state["t_mass"]
        self.t_dhw.value = best_state["t_dhw"]
        self.p_grid.value = best_state["p_grid"]
        self.p_export.value = best_state["p_export"]
        self.p_solar_self.value = best_state["p_solar_self"]
        self.z_ufh.value = best_state["z_ufh"]
        self.z_dhw.value = best_state["z_dhw"]
        self.s_room_low.value = best_state["s_room_low"]
        self.s_room_high.value = best_state["s_room_high"]
        self.s_dhw_low.value = best_state["s_dhw_low"]
        self.s_dhw_high.value = best_state["s_dhw_high"]
        self.plan_t_sup_ufh = best_state["sup_ufh"]
        self.plan_t_sup_dhw = best_state["sup_dhw"]

        logger.info(f"[MPC] Beste oplossing hersteld: obj={best_obj:.4f}")

        # ── Warm-start opslaan ────────────────────────────────────────────
        if self.t_air.value is not None:
            self._prev_ufh_on = self.ufh_on.value.copy()
            self._prev_dhw_on = self.dhw_on.value.copy()
            self._prev_t_mass = self.t_mass.value[:-1].copy()
            self._prev_t_dhw = self.t_dhw.value[:-1].copy()
            self._prev_obj = self.problem.value
            self.ekf.x[0] = float(self.t_air.value[1])
            self.ekf.x[1] = float(self.t_mass.value[1])

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
        self.mpc = ThermalMPC(
            config, self.ident, self.perf_map, self.hydraulic, self.res_dhw
        )

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

        # Herbouw MPC én herbereken PWA-grid na training
        self.mpc._build_problem()
        self.mpc.ekf = ThermalEKF(self.ident)  # reset EKF met nieuwe R/C
        self.mpc.pwa.rebuild()  # herbereken grid met nieuwe modellen

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
        tz = ZoneInfo(tzlocal.get_localzone_name())
        now_local = context.now.astimezone(tz)
        today = now_local.date()

        elapsed = np.concatenate(([0.0], np.cumsum(self.mpc.dt_steps)))
        steps_remaining = sum(
            1
            for t in range(self.mpc.horizon)
            if (now_local + timedelta(hours=float(elapsed[t]))).date() == today
        )

        pv_rem = solar_self_rem = export_rem = grid_rem = 0.0
        if steps_remaining > 0:
            sl = steps_remaining
            # We vermenigvuldigen de arrays element-wise met de tijdstappen
            pv_rem = float(np.sum(self.mpc.P_solar.value[:sl] * self.mpc.dt_steps[:sl]))
            solar_self_rem = float(
                np.sum(self.mpc.p_solar_self.value[:sl] * self.mpc.dt_steps[:sl])
            )
            export_rem = float(
                np.sum(self.mpc.p_export.value[:sl] * self.mpc.dt_steps[:sl])
            )
            grid_rem = float(
                np.sum(self.mpc.p_grid.value[:sl] * self.mpc.dt_steps[:sl])
            )

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
        total_export_revenue = float(
            np.sum(
                self.mpc.p_export.value[:steps_remaining]
                * export_price
                * self.mpc.dt_steps[:steps_remaining]
            )
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

        # 1. Grondstoffen uit de solver (allemaal arrays van lengte T of T+1)
        u_on = self.mpc.ufh_on.value
        d_on = self.mpc.dhw_on.value
        p_u = self.mpc.p_el_ufh.value
        p_d = self.mpc.p_el_dhw.value
        t_r = self.mpc.t_air.value
        t_d = self.mpc.t_dhw.value
        u_sup = self.mpc.plan_t_sup_ufh
        d_sup = self.mpc.plan_t_sup_dhw
        grid = self.mpc.p_grid.value
        export = self.mpc.p_export.value
        solar_self = self.mpc.p_solar_self.value

        # COP berekeningen (Power/Power = Dimensieloos, dus geen dt nodig)
        u_cop = np.where(
            p_u > 0.01,
            (
                self.mpc.P_pth_const_ufh.value * u_on
                + self.mpc.P_pth_slope_ufh.value * self.mpc.z_ufh.value
            )
            / np.maximum(p_u, 1e-6),
            0.0,
        )
        d_cop = np.where(
            p_d > 0.01,
            (
                self.mpc.P_pth_const_dhw.value * d_on
                + self.mpc.P_pth_slope_dhw.value * self.mpc.z_dhw.value
            )
            / np.maximum(p_d, 1e-6),
            0.0,
        )

        # Prijzen zijn per kWh
        prices = self.mpc.P_prices.value / self.mpc.dt_steps
        export_prices = self.mpc.P_export_prices.value / self.mpc.dt_steps

        # 2. Bouw de kwartier-tijdas
        quarters_per_step = np.maximum(
            1, np.round(self.mpc.dt_steps / 0.25).astype(int)
        )
        total_quarters = int(np.sum(quarters_per_step))
        s = np.repeat(
            np.arange(T), quarters_per_step
        )  # Index mapping: kwartier -> mpc_stap

        # 3. Interpolatie van temperaturen (voor een vloeiende lijn)
        solver_times = np.concatenate(([0.0], np.cumsum(self.mpc.dt_steps)))
        quarter_times = np.arange(total_quarters) * 0.25

        t_air_q = np.interp(quarter_times, solver_times[:-1], t_r[:-1])
        t_dhw_q = np.interp(quarter_times, solver_times[:-1], t_d[:-1])
        t_out_q = np.interp(quarter_times, solver_times[:-1], self.mpc.P_temp_out.value)

        # 4. Shutters (herhalen)
        shutter_src = (
            shutters[:T]
            if len(shutters) >= T
            else np.pad(shutters, (0, T - len(shutters)), constant_values=shutters[-1])
        )

        plan = []
        for q in range(total_quarters):
            idx = s[q]  # De index in de solver resultaten
            ts = start + timedelta(minutes=q * 15)

            # Modus bepaling
            if d_on[idx] > 0.5:
                mode_str, hvac_mode = "DHW", HvacMode.DHW
            elif u_on[idx] > 0.5:
                mode_str, hvac_mode = "UFH", HvacMode.HEATING
            else:
                mode_str, hvac_mode = "-", HvacMode.OFF

            # --- BELANGRIJK: kW blijft kW, Euro = kW * Prijs * 0.25 uur ---
            p_u_val = float(p_u[idx])
            p_d_val = float(p_d[idx])
            p_solar_val = float(self.mpc.P_solar.value[idx])
            p_load_val = float(self.mpc.P_base_load.value[idx])
            p_grid_val = float(grid[idx])
            p_export_val = float(export[idx])
            p_self_val = float(solar_self[idx])
            price_val = float(prices[idx])
            ex_price_val = float(export_prices[idx])

            plan.append(
                {
                    "time": ts,
                    "mode": mode_str,
                    "hvac_mode": hvac_mode.value,
                    "t_out": f"{t_out_q[q]:.1f}",
                    "p_solar": f"{p_solar_val:.2f}",
                    "p_load": f"{p_load_val:.2f}",
                    "t_room": f"{t_air_q[q]:.2f}",
                    "t_dhw": f"{t_dhw_q[q]:.2f}",
                    "p_el_ufh": f"{p_u_val:.2f}",
                    "p_el_dhw": f"{p_d_val:.2f}",
                    "cop_ufh": f"{float(u_cop[idx]):.2f}",
                    "cop_dhw": f"{float(d_cop[idx]):.2f}",
                    "supply_ufh": f"{float(u_sup[idx]):.2f}",
                    "supply_dhw": f"{float(d_sup[idx]):.2f}",
                    "shutter": f"{float(shutter_src[idx]):.0f}",
                    "price": f"{price_val:.2f}",
                    "cost_ufh": f"{p_u_val * price_val * 0.25:.3f}",
                    "cost_dhw": f"{p_d_val * price_val * 0.25:.3f}",
                    "cost_gross": f"{(p_u_val + p_d_val + p_load_val) * price_val * 0.25:.3f}",
                    "cost_net": f"{(p_grid_val * price_val - p_export_val * ex_price_val) * 0.25:.3f}",
                    "cost_saving": f"{p_self_val * price_val * 0.25:.3f}",
                }
            )

        return plan
