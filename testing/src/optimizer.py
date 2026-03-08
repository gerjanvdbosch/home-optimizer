import pandas as pd
import numpy as np
import cvxpy as cp
import logging

from datetime import datetime, timedelta
from context import Context, HvacMode
from thermal import (
    SystemIdentificator,
    HPPerformanceMap,
    HydraulicPredictor,
    UfhResidualPredictor,
    DhwResidualPredictor,
)
from shutter import ShutterPredictor

logger = logging.getLogger(__name__)


# =========================================================
# 4. THERMAL MPC
# =========================================================
class ThermalMPC:
    def __init__(self, ident, perf_map, hydraulic, res_dhw):
        self.ident = ident
        self.perf_map = perf_map
        self.hydraulic = hydraulic
        self.res_dhw = res_dhw
        self.horizon = 96
        self.dt = 0.25

        self._build_problem()

    def _build_problem(self):
        T = self.horizon
        R, C = self.ident.R, self.ident.C

        # --- PARAMETERS ---
        self.P_t_room_init = cp.Parameter()
        self.P_t_dhw_init = cp.Parameter()
        self.P_init_ufh = cp.Parameter(nonneg=True)
        self.P_init_dhw = cp.Parameter(nonneg=True)
        self.P_comp_on_init = cp.Parameter(nonneg=True)  # NIEUW: Voor compressor starts

        # Prijzen en Net
        self.P_prices = cp.Parameter(T, nonneg=True)
        self.P_export_prices = cp.Parameter(T, nonneg=True)
        self.P_solar = cp.Parameter(T, nonneg=True)
        self.P_base_load = cp.Parameter(T, nonneg=True)

        # Weer en Comfort
        self.P_temp_out = cp.Parameter(T)
        self.P_room_min = cp.Parameter(T, nonneg=True)
        self.P_room_max = cp.Parameter(T, nonneg=True)
        self.P_dhw_min = cp.Parameter(T, nonneg=True)
        self.P_dhw_max = cp.Parameter(T, nonneg=True)
        self.P_solar_gain = cp.Parameter(T)
        self.P_strictness = cp.Parameter(T, nonneg=True)

        # NIEUW: Dynamisch berekende comfortkosten o.b.v. stroomprijs
        self.P_cost_room_under = cp.Parameter(nonneg=True)
        self.P_cost_room_over = cp.Parameter(nonneg=True)
        self.P_cost_dhw_under = cp.Parameter(nonneg=True)
        self.P_cost_dhw_over = cp.Parameter(nonneg=True)
        self.P_val_terminal_room = cp.Parameter(nonneg=True)
        self.P_val_terminal_dhw = cp.Parameter(nonneg=True)

        # Dynamisch berekende fysica (De VASTE voorspellingen!)
        self.P_cop_ufh = cp.Parameter(T, nonneg=True)
        self.P_cop_dhw = cp.Parameter(T, nonneg=True)
        self.P_fixed_pel_ufh = cp.Parameter(T, nonneg=True)
        self.P_fixed_pel_dhw = cp.Parameter(T, nonneg=True)

        self.P_hist_heat = cp.Parameter(self.ident.ufh_lag_steps, nonneg=True)

        self.P_dhw_demand = cp.Parameter(T, nonneg=True)

        # --- VARIABELEN ---
        # Binary ON/OFF is de ENIGE keuze voor de solver
        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)
        self.comp_start = cp.Variable(T, nonneg=True)

        # Afgeleide variabelen (worden vastgezet via de booleans)
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

        # --- CONSTRAINTS ---
        constraints = [
            self.t_room[0] == self.P_t_room_init,
            self.t_dhw[0] == self.P_t_dhw_init,
        ]

        # NIEUW: Logica om een daadwerkelijke opstart te detecteren (telt niet bij switch UFH <-> DHW)
        comp_on = self.ufh_on + self.dhw_on
        constraints += [self.comp_start[0] >= comp_on[0] - self.P_comp_on_init]
        for t in range(1, T):
            constraints += [self.comp_start[t] >= comp_on[t] - comp_on[t - 1]]

        p_th_ufh_future = cp.multiply(self.p_el_ufh, self.P_cop_ufh)
        p_th_ufh_lagged = cp.hstack([self.P_hist_heat, p_th_ufh_future])

        for t in range(T):
            p_el_wp = self.p_el_ufh[t] + self.p_el_dhw[t]
            p_th_dhw_now = self.p_el_dhw[t] * self.P_cop_dhw[t]
            active_room_heat = p_th_ufh_lagged[t]

            constraints += [
                # Stroombalans
                p_el_wp + self.P_base_load[t] == self.p_grid[t] + self.p_solar_self[t],
                self.P_solar[t] == self.p_solar_self[t] + self.p_export[t],
                # Thermische Balans
                self.t_room[t + 1]
                == self.t_room[t]
                + (
                    (active_room_heat - (self.t_room[t] - self.P_temp_out[t]) / R)
                    * self.dt
                    / C
                )
                + (self.P_solar_gain[t] * self.dt),
                self.t_dhw[t + 1]
                == self.t_dhw[t]
                + (
                    (p_th_dhw_now * self.dt) / self.ident.C_tank
                    - (self.t_dhw[t] - self.t_room[t])
                    * (self.ident.K_loss_dhw * self.dt)
                )
                - self.P_dhw_demand[t],
                # Fysieke Grenzen (Niet tegelijk vloer en boiler doen)
                self.ufh_on[t] + self.dhw_on[t] <= 1,
                # JOUW ORIGINELE FIX: Vermogen is EXACT het weersafhankelijke profiel als hij aan staat!
                self.p_el_ufh[t] == self.ufh_on[t] * self.P_fixed_pel_ufh[t],
                self.p_el_dhw[t] == self.dhw_on[t] * self.P_fixed_pel_dhw[t],
                # Comfort limieten
                self.t_room[t + 1] + self.s_room_low[t] >= self.P_room_min[t],
                self.t_room[t + 1] - self.s_room_high[t] <= self.P_room_max[t],
                self.t_dhw[t + 1] + self.s_dhw_low[t] >= self.P_dhw_min[t],
                self.t_dhw[t + 1] - self.s_dhw_high[t] <= self.P_dhw_max[t],
            ]

        # --- OBJECTIVE FUNCTION ---
        net_cost = (
            cp.sum(
                cp.multiply(self.p_grid, self.P_prices)
                - cp.multiply(self.p_export, self.P_export_prices)
            )
            * self.dt
        )

        # NIEUW: Gebruik de dynamische prijs/massa scaling voor boetes!
        extra_penalty = cp.multiply(cp.pos(self.s_room_low - 0.25), self.P_strictness)
        comfort = cp.sum(
            self.s_room_low * self.P_cost_room_under
            + self.s_room_high * self.P_cost_room_over
            + self.s_dhw_low * self.P_cost_dhw_under
            + self.s_dhw_high * self.P_cost_dhw_over
            + extra_penalty
        )

        # Hoge straf op koude starts, lage straf op het wisselen van de driewegklep
        valve_switches = (
            cp.pos(self.ufh_on[0] - self.P_init_ufh)
            + cp.sum(cp.pos(self.ufh_on[1:] - self.ufh_on[:-1]))
            + cp.pos(self.dhw_on[0] - self.P_init_dhw)
            + cp.sum(cp.pos(self.dhw_on[1:] - self.dhw_on[:-1]))
        )
        switching = (cp.sum(self.comp_start) * 20.0) + (valve_switches * 2.0)

        stored_heat_value = (self.t_dhw[T] * self.P_val_terminal_dhw) + (
            self.t_room[T] * self.P_val_terminal_room
        )

        self.problem = cp.Problem(
            cp.Minimize(net_cost + comfort + switching - stored_heat_value), constraints
        )

    def _get_targets(self, now, T):
        r_min, r_max, d_min, d_max = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        local_tz = datetime.now().astimezone().tzinfo
        now_local = now.astimezone(local_tz)

        for t in range(T):
            fut_time = now_local + timedelta(hours=t * self.dt)
            h = fut_time.hour
            if 17 <= h < 22:
                r_min[t], r_max[t] = 20.0, 21.0
            elif 11 <= h < 17:
                r_min[t], r_max[t] = 19.5, 22.0
            else:
                r_min[t], r_max[t] = 19.0, 19.5

            # Alleen een harde eis vlak voor het piekmoment
            if 20 <= h < 21:
                d_min[t] = 49.0
            else:
                d_min[t] = 10.0
            d_max[t] = 55.0

        return r_min, r_max, d_min, d_max

    def solve(self, state, forecast_df, recent_history_df, solar_gains):
        T = self.horizon
        r_min, r_max, d_min, d_max = self._get_targets(state["now"], T)

        self.P_t_room_init.value = state["room_temp"]
        self.P_t_dhw_init.value = (state["dhw_top"] + state["dhw_bottom"]) / 2.0

        self.P_init_ufh.value = (
            1.0 if state["hvac_mode"] == HvacMode.HEATING.value else 0.0
        )
        self.P_init_dhw.value = 1.0 if state["hvac_mode"] == HvacMode.DHW.value else 0.0

        was_running = (
            1.0
            if state["hvac_mode"] in [HvacMode.HEATING.value, HvacMode.DHW.value]
            else 0.0
        )
        self.P_comp_on_init.value = was_running

        self.P_temp_out.value = forecast_df.temp.values[:T]

        dhw_demand = self.res_dhw.predict(forecast_df)
        self.P_dhw_demand.value = dhw_demand[: self.horizon]

        prices = np.full(T, 0.22)
        self.P_prices.value = prices
        self.P_export_prices.value = np.full(T, 0.05)
        self.P_solar.value = forecast_df.power_corrected.values[:T]
        self.P_base_load.value = forecast_df.load_corrected.values[:T]
        self.P_room_min.value, self.P_room_max.value = r_min, r_max
        self.P_dhw_min.value, self.P_dhw_max.value = d_min, d_max
        self.P_solar_gain.value = solar_gains[:T]

        # Prijs dynamica instellen
        avg_price = max(float(np.mean(prices)), 0.10)

        self.P_cost_room_under.value = 0.5 * self.ident.C * avg_price
        self.P_cost_room_over.value = 1.0 * self.ident.C * avg_price
        # De woning lekt warmte, dus een graad nu is aan het eind van de 24u horizon minder waard.
        # We geven de kamer 15% van de waarde van de energieprijs.
        # Dit is genoeg om zon te verkiezen boven export, maar te weinig om duur stroom te kopen.
        self.P_val_terminal_room.value = 0.15 * self.ident.C * avg_price

        # Boiler krijgt prioriteit bij vraag
        self.P_cost_dhw_under.value = 15.0 * self.ident.C_tank * avg_price
        self.P_cost_dhw_over.value = 5.0 * self.ident.C_tank * avg_price
        # De boiler is goed geïsoleerd en echt een batterij: die geven we 40% waarde.
        self.P_val_terminal_dhw.value = 0.4 * self.ident.C_tank * avg_price

        temps = forecast_df.temp.values[:T]

        # --- Slimme, vloeiende strictness curve ---
        # 1. Bepaal het verschil tussen 'kamer' (ongeveer 20) en buiten.
        # (Als het buiten warmer is dan 20, is het verschil 0)
        delta_t = np.maximum(0.0, 20.0 - temps)

        # 2. Kwadratische boete: hoe kouder, hoe exponentieel strenger.
        # Bijv:
        # T_out = 15 -> Delta 5 -> 5 + 0.25 * 25 = 11
        # T_out = 10 -> Delta 10 -> 5 + 0.25 * 100 = 30
        # T_out = 0 -> Delta 20 -> 5 + 0.25 * 400 = 105
        # T_out = -5 -> Delta 25 -> 5 + 0.25 * 625 = 161
        strictness_values = (5.0 + 0.25 * (delta_t**2)) * avg_price

        self.P_strictness.value = strictness_values

        # --- SLP LOOP: 2 Iteraties voor de stooklijn/boilercurve simulatie ---
        guessed_t_room = np.full(T, state["room_temp"])
        guessed_t_dhw = np.full(T, state["dhw_bottom"])

        for iteration in range(4):
            cop_u, cop_d = np.zeros(T), np.zeros(T)
            fixed_p_ufh, fixed_p_dhw = np.zeros(T), np.zeros(T)

            self.plan_t_sup_ufh = np.zeros(T)
            self.plan_t_sup_dhw = np.zeros(T)

            for t in range(T):
                t_out = forecast_df.temp.values[t]
                t_room_current = guessed_t_room[t]
                t_dhw_current = guessed_t_dhw[t]

                k_emit = self.ident.K_emit
                k_tank = self.ident.K_tank
                f_ufh = self.hydraulic.learned_factor_ufh
                f_dhw = self.hydraulic.learned_factor_dhw

                # --- UFH LOGICA (Jouw curve sturing!) ---
                ufh_slope = self.hydraulic.get_ufh_slope(t_out)
                t_sup_u = t_room_current + self.hydraulic.learned_lift_ufh + ufh_slope

                numerator_u = k_emit * (t_sup_u - t_room_current)
                denominator_u = 1 + (k_emit / (2 * f_ufh))
                p_th_ufh = max(0.0, numerator_u / denominator_u)

                dt_u = p_th_ufh / f_ufh if p_th_ufh > 0 else 0
                t_mean_u = t_sup_u - (dt_u / 2.0)

                cop_u[t] = self.perf_map.predict_cop(
                    t_out, t_mean_u, HvacMode.HEATING.value
                )
                fixed_p_ufh[t] = p_th_ufh / cop_u[t] if cop_u[t] > 0 else 0.0
                self.plan_t_sup_ufh[t] = t_sup_u

                # --- DHW LOGICA ---
                predicted_delta_dhw = self.hydraulic.dhw_delta_base + (
                    self.hydraulic.dhw_delta_slope * t_out
                )
                t_sup_d = (
                    t_dhw_current
                    + self.hydraulic.learned_lift_dhw
                    + predicted_delta_dhw
                )

                if t_sup_d >= self.ident.T_max_dhw:
                    # Resterende drijfkracht is kleiner dan de nominale delta
                    # fixed_p_dhw is al correct via de formule — maar zorg dat de SLP
                    # minimaal 3 iteraties draait zodat guessed_t_dhw convergeert
                    pass

                numerator_d = k_tank * (t_sup_d - t_dhw_current)
                denominator_d = 1 + (k_tank / (2 * f_dhw))
                p_th_dhw = max(0.0, numerator_d / denominator_d)

                dt_d = p_th_dhw / f_dhw if p_th_dhw > 0 else 0
                t_mean_d = t_sup_d - (dt_d / 2.0)

                cop_d[t] = self.perf_map.predict_cop(
                    t_out, t_mean_d, HvacMode.DHW.value
                )
                fixed_p_dhw[t] = p_th_dhw / cop_d[t] if cop_d[t] > 0 else 0.0
                self.plan_t_sup_dhw[t] = t_sup_d

            # Vul de CVXPY parameters netjes in
            self.P_cop_ufh.value = np.clip(cop_u, 1.5, 9.0)
            self.P_cop_dhw.value = np.clip(cop_d, 1.1, 5.0)
            self.P_fixed_pel_ufh.value = fixed_p_ufh
            self.P_fixed_pel_dhw.value = fixed_p_dhw

            # Historie ophalen
            lag = self.ident.ufh_lag_steps
            hist_heat = np.zeros(lag)
            if not recent_history_df.empty and "wp_output" in recent_history_df.columns:
                vals = recent_history_df["wp_output"].tail(lag).values
                if len(vals) > 0:
                    hist_heat[-len(vals) :] = vals
            self.P_hist_heat.value = hist_heat

            # --- OPLOSSEN ---
            try:
                self.problem.solve(solver=cp.HIGHS)
            except Exception as e:
                logger.error(f"Solver exception in iteratie {iteration}: {e}")
                break

            if self.problem.status not in ["optimal", "optimal_inaccurate"]:
                logger.warning(
                    f"Solver status not optimal in iteratie {iteration}: {self.problem.status}"
                )
                break

            # Update schatting voor de 2e ronde
            if iteration == 0:
                guessed_t_room = self.t_room.value[:-1]
                guessed_t_dhw = self.t_dhw.value[:-1]


# =========================================================
# 5. OPTIMIZER (Met fysieke Supply Temp bepaling)
# =========================================================
class Optimizer:
    def __init__(self, config, database):
        self.db = database
        self.perf_map = HPPerformanceMap(config.hp_model_path)
        self.ident = SystemIdentificator(config.rc_model_path)
        self.hydraulic = HydraulicPredictor(config.hydraulic_model_path)
        self.res_ufh = UfhResidualPredictor(
            config.ufh_model_path, self.ident.R, self.ident.C
        )
        self.res_dhw = DhwResidualPredictor(config.dhw_model_path)
        self.shutter = ShutterPredictor(config.shutter_model_path)

        # Geef de hydraulic predictor mee aan MPC
        self.mpc = ThermalMPC(self.ident, self.perf_map, self.hydraulic, self.res_dhw)

    def train(self, days_back: int = 730):
        cutoff = datetime.now() - timedelta(days=days_back)
        df = self.db.get_history(cutoff_date=cutoff)
        if df.empty:
            return

        self.perf_map.train(df)
        self.ident.train(df)
        self.hydraulic.train(df)
        self.res_ufh.train(df)
        self.res_dhw.train(df)
        self.shutter.train(df)

        self.mpc._build_problem()

    def resolve(self, context: Context):
        state = {
            "now": context.now,
            "hvac_mode": context.hvac_mode.value,
            "room_temp": context.room_temp,
            "dhw_top": context.dhw_top,
            "dhw_bottom": context.dhw_bottom,
        }

        cutoff = context.now - timedelta(hours=4)
        raw_hist = self.db.get_history(cutoff_date=cutoff)

        # BUGFIX: Calculate wp_output before passing to MPC so lag works properly!
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

        shutter_room = getattr(context, "shutter_room", 100.0)
        shutters = self.shutter.predict(context.forecast_df, shutter_room)
        # FIX: Voorspel de zon-opwarming vooraf via het getrainde ML model
        solar_gains = self.res_ufh.predict(context.forecast_df, shutters)

        logger.info(
            f"[Optimizer] Max solar gain in forecast: {np.max(solar_gains):.3f} K/kwartier"
        )
        logger.info(
            f"[Optimizer] Gemiddelde solar gain: {np.mean(solar_gains):.3f} K/kwartier"
        )

        # Geef de solar_gains ook mee aan de solver
        self.mpc.solve(state, context.forecast_df, recent_history_df, solar_gains)

        mode = "OFF"
        target_pel = 0.0
        target_supply_temp = 0.0
        steps_remaining = 0
        pv_remaining = 0.0
        solar_self_remaining = 0.0
        export_remaining = 0.0
        grid_remaining = 0.0

        if self.mpc.p_el_ufh.value is None:
            return {
                "mode": mode,
                "status": self.mpc.problem.status,
                "target_pel_kw": target_pel,
                "target_supply_temp": target_supply_temp,
                "steps_remaining": steps_remaining,
                "pv_remaining": pv_remaining,
                "solar_self_remaining": solar_self_remaining,
                "export_remaining": export_remaining,
                "grid_remaining": grid_remaining,
                "plan": [],
            }

        # We berekenen hoeveel stappen er nog resteren in de huidige kalenderdag
        tz = datetime.now().astimezone().tzinfo
        now_local = context.now.astimezone(tz)
        current_day = now_local.date()

        steps_remaining = 0
        for t in range(self.mpc.horizon):
            ts_local = now_local + timedelta(minutes=t * 15)

            if ts_local.date() == current_day:
                steps_remaining += 1
            else:
                break

        # Als we bijna op het einde van de dag zitten (bv 23:55), is steps_remaining laag.
        # We pakken de slices van de arrays tot aan steps_remaining.
        if steps_remaining > 0:
            # Elektrische PV opwekking
            pv_remaining = (
                np.sum(self.mpc.P_solar.value[:steps_remaining]) * self.mpc.dt
            )
            # Energiebalans
            solar_self_remaining = (
                np.sum(self.mpc.p_solar_self.value[:steps_remaining]) * self.mpc.dt
            )
            export_remaining = (
                np.sum(self.mpc.p_export.value[:steps_remaining]) * self.mpc.dt
            )
            grid_remaining = (
                np.sum(self.mpc.p_grid.value[:steps_remaining]) * self.mpc.dt
            )

        # Wat doen we NU (index 0)
        p_el_ufh_now = self.mpc.p_el_ufh.value[0]
        p_el_dhw_now = self.mpc.p_el_dhw.value[0]

        # --- BEPALEN TARGETS MET ML (GEEN VASTE FORMULES MEER) ---
        if p_el_dhw_now > 0.1:
            mode = "DHW"
            target_pel = p_el_dhw_now
            target_supply_temp = self.mpc.plan_t_sup_dhw[0]

        elif p_el_ufh_now > 0.1:
            mode = "UFH"
            target_pel = p_el_ufh_now
            target_supply_temp = self.mpc.plan_t_sup_ufh[0]

        return {
            "status": self.mpc.problem.status,
            "mode": mode,
            "target_pel_kw": round(target_pel, 2),
            "target_supply_temp": round(target_supply_temp, 1),
            "steps_remaining": steps_remaining,
            "pv_remaining": pv_remaining,
            "solar_self_remaining": solar_self_remaining,
            "export_remaining": export_remaining,
            "grid_remaining": grid_remaining,
            "plan": self.get_plan(context, shutters),
        }

    def get_plan(self, context, shutters):
        if self.mpc.p_el_ufh.value is None:
            return []

        plan = []
        T = self.mpc.horizon

        local_tz = datetime.now().astimezone().tzinfo
        now_local = context.now.astimezone(local_tz)

        minute = (now_local.minute // 15) * 15
        start_time = now_local.replace(minute=minute, second=0, microsecond=0)

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
        strictness = self.mpc.P_strictness.value

        for t in range(T):
            ts = start_time + timedelta(minutes=t * 15)
            mode_str = "-"
            if d_on[t] > 0.5:
                mode_str = "DHW"
            elif u_on[t] > 0.5:
                mode_str = "UFH"

            shutter_val = shutters[t] if t < len(shutters) else np.nan

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
                    "strictness": f"{strictness[t]:.0f}",
                    "shutter": f"{shutter_val:.0f}",
                }
            )

        return plan
