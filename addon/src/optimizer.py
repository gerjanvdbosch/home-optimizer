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
# THERMAL MPC
# =========================================================


class ThermalMPC:
    def __init__(self, ident, perf_map, hydraulic, res_dhw):
        self.ident = ident
        self.perf_map = perf_map
        self.hydraulic = hydraulic
        self.res_dhw = res_dhw
        self.horizon = 96
        self.dt = 0.25

        self.plan_t_sup_ufh = np.zeros(self.horizon)
        self.plan_t_sup_dhw = np.zeros(self.horizon)

        self._build_problem()

    def _build_problem(self):
        T = self.horizon
        R = self.ident.R
        C = self.ident.C
        lag = self.ident.ufh_lag_steps

        # ── Parameters ────────────────────────────────────────────────────
        self.P_t_room_init = cp.Parameter()
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

        self.P_hist_heat = cp.Parameter(lag, nonneg=True)
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

        self.t_room = cp.Variable(T + 1)
        self.t_dhw = cp.Variable(T + 1, nonneg=True)

        self.s_room_low = cp.Variable(T, nonneg=True)
        self.s_room_high = cp.Variable(T, nonneg=True)
        self.s_dhw_low = cp.Variable(T, nonneg=True)
        self.s_dhw_high = cp.Variable(T, nonneg=True)

        # ── Constraints ───────────────────────────────────────────────────
        constraints = [
            self.t_room[0] == self.P_t_room_init,
            self.t_dhw[0] == self.P_t_dhw_init,
        ]

        # Compressorstarts detecteren (niet klep-wissels)
        comp_on = self.ufh_on + self.dhw_on
        constraints += [self.comp_start[0] >= comp_on[0] - self.P_comp_on_init]
        for t in range(1, T):
            constraints += [self.comp_start[t] >= comp_on[t] - comp_on[t - 1]]

        # Exponentieel aflopende kernel voor vloertraagheid
        tau = max(1.0, lag / 2.0)
        kernel_np = np.array([np.exp(-i / tau) for i in range(lag)])
        kernel_np /= kernel_np.sum()
        kernel = cp.Constant(kernel_np[::-1])

        p_th_ufh_future = cp.multiply(self.p_el_ufh, self.P_cop_ufh)
        p_th_ufh_lagged = cp.hstack([self.P_hist_heat, p_th_ufh_future])

        for t in range(T):
            p_el_wp = self.p_el_ufh[t] + self.p_el_dhw[t]
            p_th_dhw_now = self.p_el_dhw[t] * self.P_cop_dhw[t]
            active_heat = kernel @ p_th_ufh_lagged[t : t + lag]

            constraints += [
                # Stroombalans
                p_el_wp + self.P_base_load[t] == self.p_grid[t] + self.p_solar_self[t],
                self.P_solar[t] == self.p_solar_self[t] + self.p_export[t],
                # Thermische balans kamer
                self.t_room[t + 1]
                == self.t_room[t]
                + (
                    (active_heat - (self.t_room[t] - self.P_temp_out[t]) / R)
                    * self.dt
                    / C
                )
                + (self.P_solar_gain[t] * self.dt),
                # Thermische balans boiler
                self.t_dhw[t + 1]
                == self.t_dhw[t]
                + (
                    (p_th_dhw_now * self.dt) / self.ident.C_tank
                    - (self.t_dhw[t] - self.t_room[t])
                    * (self.ident.K_loss_dhw * self.dt)
                )
                - self.P_dhw_demand[t],
                # Niet tegelijk vloer en boiler
                self.ufh_on[t] + self.dhw_on[t] <= 1,
                # Vermogen is exact het geleerde profiel als hij aan staat
                self.p_el_ufh[t] == self.ufh_on[t] * self.P_fixed_pel_ufh[t],
                self.p_el_dhw[t] == self.dhw_on[t] * self.P_fixed_pel_dhw[t],
                # Comfortlimieten met slack
                self.t_room[t + 1] + self.s_room_low[t] >= self.P_room_min[t],
                self.t_room[t + 1] - self.s_room_high[t] <= self.P_room_max[t],
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
        switching = (cp.sum(self.comp_start) * 0.05) + (valve_switches * 0.05)

        stored_heat = (
            self.t_dhw[T] * self.P_val_terminal_dhw
            + self.t_room[T] * self.P_val_terminal_room
        )

        self.problem = cp.Problem(
            cp.Minimize(net_cost + comfort + switching - stored_heat),
            constraints,
        )

    def _get_targets(self, now, T):
        r_min = np.zeros(T)
        r_max = np.zeros(T)
        d_min = np.zeros(T)
        d_max = np.zeros(T)

        local_tz = datetime.now().astimezone().tzinfo
        now_local = now.astimezone(local_tz)

        for t in range(T):
            fut = now_local + timedelta(hours=t * self.dt)
            h = fut.hour

            if 17 <= h < 22:
                r_min[t], r_max[t] = 20.0, 21.0
            elif 11 <= h < 17:
                r_min[t], r_max[t] = 19.5, 22.0
            else:
                r_min[t], r_max[t] = 19.0, 19.5

            d_min[t] = 48.0 if 20 <= h < 21 else 10.0
            d_max[t] = 55.0

        return r_min, r_max, d_min, d_max

    def solve(
        self,
        state: dict,
        forecast_df: pd.DataFrame,
        recent_history_df: pd.DataFrame,
        solar_gains: np.ndarray,
        avg_price: float,
        export_price: float,
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
        lag = self.ident.ufh_lag_steps

        # ── Toestand initialiseren ────────────────────────────────────────
        r_min, r_max, d_min, d_max = self._get_targets(state["now"], T)

        self.P_t_room_init.value = float(state["room_temp"])
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

        dhw_demand = self.res_dhw.predict(forecast_df)
        self.P_dhw_demand.value = dhw_demand[:T].astype(float)

        logger.info(
            f"[Debug] DHW demand totaal: {dhw_demand[:T].sum():.2f} K  max: {dhw_demand[:T].max():.2f} K"
        )

        # ── Comfortboetes (dimensioneel correct) ─────────────────────────
        t_out_now = float(forecast_df.temp.values[0])
        avg_cop_ufh = self.perf_map.predict_cop(
            HvacMode.HEATING.value, t_out_now, state["room_temp"]
        )
        avg_cop_dhw = self.perf_map.predict_cop(
            HvacMode.DHW.value, t_out_now, state["dhw_bottom"]
        )

        costs = ComfortCostCalculator(
            C_room=self.ident.C,
            C_tank=self.ident.C_tank,
            avg_cop_ufh=avg_cop_ufh,
            avg_cop_dhw=avg_cop_dhw,
        ).compute(avg_price)

        self.P_cost_room_under.value = costs["room_under"]
        self.P_cost_room_over.value = costs["room_over"]
        self.P_cost_dhw_under.value = costs["tank_under"]
        self.P_cost_dhw_over.value = costs["tank_over"]
        self.P_val_terminal_room.value = costs["terminal_room"]
        self.P_val_terminal_dhw.value = costs["terminal_tank"]

        # Strictness: koud buiten = hogere urgentie
        comfort_threshold = float(self.P_room_min.value[0]) + 1.0
        delta_t_env = np.maximum(0.0, comfort_threshold - forecast_df.temp.values[:T])
        self.P_strictness.value = (3.0 + 0.10 * (delta_t_env**2)) * float(avg_price)

        # ── Historische lag-buffer ────────────────────────────────────────
        hist_heat = np.zeros(lag)
        if not recent_history_df.empty and "wp_output" in recent_history_df.columns:
            vals = recent_history_df["wp_output"].tail(lag).values
            if len(vals) > 0:
                hist_heat[-len(vals) :] = vals
        self.P_hist_heat.value = hist_heat.astype(float)

        # ── SLP-iteraties met convergentiecheck ──────────────────────────
        linearizer = PhysicsLinearizer(
            perf_map=self.perf_map,
            hydraulic=self.hydraulic,
            horizon=T,
            max_iter=10,
            tol=0.05,
        )

        t_out_arr = forecast_df.temp.values[:T].astype(float)
        guessed_t_room = np.full(T, float(state["room_temp"]))
        guessed_t_dhw = np.full(T, float(state["dhw_bottom"]))

        # Sentinel: convergentiecheck slaat eerste iteratie altijd over
        p_el_ufh_prev = np.full(T, -999.0)
        p_el_dhw_prev = np.full(T, -999.0)

        for iteration in range(linearizer.max_iter):

            # 1. Bereken bevroren vermogen en COP op basis van huidige schatting
            p_el_ufh, cop_ufh, p_el_dhw, cop_dhw = linearizer.compute(
                t_out_arr, guessed_t_room, guessed_t_dhw
            )

            # 2. Supply-temp plan voor rapportage
            self.plan_t_sup_ufh = np.array(
                [
                    guessed_t_room[t]
                    + self.hydraulic.learned_lift_ufh
                    + self.hydraulic.get_ufh_slope(t_out_arr[t])
                    for t in range(T)
                ],
                dtype=float,
            )

            self.plan_t_sup_dhw = np.array(
                [
                    guessed_t_dhw[t]
                    + self.hydraulic.learned_lift_dhw
                    + self.hydraulic.dhw_delta_base
                    + self.hydraulic.dhw_delta_slope * t_out_arr[t]
                    for t in range(T)
                ],
                dtype=float,
            )

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
            logger.debug(
                f"[SLP] Iteratie {iteration}: status={status}  "
                f"P_el UFH[0]={p_el_ufh[0]:.3f} kW  DHW[0]={p_el_dhw[0]:.3f} kW"
            )

            if status not in ("optimal", "optimal_inaccurate"):
                logger.warning(
                    f"[SLP] Niet-optimale status ({status}) in iteratie {iteration}"
                )
                break

            # 6. Toestandstrajectorie bijwerken voor volgende iteratie
            if self.t_room.value is not None:
                new_t_room = self.t_room.value[:-1].copy()
                new_t_dhw = self.t_dhw.value[:-1].copy()

                if iteration == 0:
                    guessed_t_room = new_t_room
                    guessed_t_dhw = new_t_dhw
                else:
                    alpha_room = 0.35
                    alpha_dhw = (
                        0.15  # DHW heeft grotere COP-variatie, langzamer updaten
                    )

                    guessed_t_room = (
                        alpha_room * new_t_room + (1 - alpha_room) * guessed_t_room
                    )
                    guessed_t_dhw = (
                        alpha_dhw * new_t_dhw + (1 - alpha_dhw) * guessed_t_dhw
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
        """
        Bereken het optimale plan voor de komende 24 uur.

        Parameters
        ----------
        context      : huidige context (sensoren, forecast, tijdstip)
        avg_price    : importprijs [euro/kWh] — lever dit van buiten aan
        export_price : teruglevertarief [euro/kWh]
        """
        state = {
            "now": context.now,
            "hvac_mode": context.hvac_mode.value,
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        # Recente geschiedenis voor de lag-buffer
        cutoff = context.now - timedelta(hours=4)
        raw_hist = self.db.get_history(cutoff_date=cutoff)

        recent_history_df = pd.DataFrame()
        if not raw_hist.empty:
            raw_hist = raw_hist.copy()
            raw_hist.set_index("timestamp", inplace=True)
            recent_history_df = (
                raw_hist.resample("15min")
                .mean(numeric_only=True)
                .fillna(0)
                .reset_index()
            )

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
            recent_history_df=recent_history_df,
            solar_gains=solar_gains,
            avg_price=avg_price,
            export_price=export_price,
        )

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
        tz = datetime.now().astimezone().tzinfo
        now_local = context.now.astimezone(tz)
        minute = (now_local.minute // 15) * 15
        start = now_local.replace(minute=minute, second=0, microsecond=0)

        u_on = self.mpc.ufh_on.value
        d_on = self.mpc.dhw_on.value
        p_u = self.mpc.p_el_ufh.value
        p_d = self.mpc.p_el_dhw.value
        t_r = self.mpc.t_room.value
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
            elif u_on[t] > 0.5:
                mode_str = "UFH"
            else:
                mode_str = "-"

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
                    "t_out": f"{context.forecast_df.temp.iloc[t]:.1f}",
                    "p_solar": f"{context.forecast_df.power_corrected.iloc[t]:.2f}",
                    "p_load": f"{context.forecast_df.load_corrected.iloc[t]:.2f}",
                    "t_room": f"{t_r[t + 1]:.1f}",
                    "t_dhw": f"{t_d[t + 1]:.1f}",
                    "p_el_ufh": f"{p_u[t]:.2f}",
                    "p_el_dhw": f"{p_d[t]:.2f}",
                    "cop_ufh": f"{u_cop[t]:.2f}",
                    "cop_dhw": f"{d_cop[t]:.2f}",
                    "supply_ufh": f"{u_sup[t]:.2f}",
                    "supply_dhw": f"{d_sup[t]:.2f}",
                    "strictness": f"{strict[t]:.0f}",
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
