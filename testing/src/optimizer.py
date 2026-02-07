import pandas as pd
import numpy as np
import cvxpy as cp
import joblib
import logging

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# =========================================================
# 1. HEAT PUMP PERFORMANCE MAP
# =========================================================
class HPPerformanceMap:
    def __init__(self, path):
        self.path = Path(path)
        self.is_fitted = False
        self.cop_model = None
        self.power_model = None
        self.max_freq_model = None
        self.ufh_freq_ref = 35.0
        self.dhw_freq_ref = 60.0
        self.ufh_delta_t_ref = 4.0
        self.dhw_delta_t_ref = 7.0
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.cop_model = data["cop_model"]
                self.power_model = data["power_model"]
                self.max_freq_model = data.get("max_freq_model")
                self.ufh_freq_ref = data.get("ufh_freq_ref", 35.0)
                self.dhw_freq_ref = data.get("dhw_freq_ref", 60.0)
                self.ufh_delta_t_ref = data.get("ufh_delta_t_ref", 4.0)
                self.dhw_delta_t_ref = data.get("dhw_delta_t_ref", 7.0)
                self.is_fitted = True
                logger.info("[HPPerformanceMap] Model geladen")
            except:
                logger.warning("[HPPerformanceMap] Laden mislukt")

    def train(self, df: pd.DataFrame):
        df = df.copy()
        df["delta_t"] = df["supply_temp"] - df["return_temp"]

        mask = (
            (df["compressor_freq"] > 15) &
            (df["wp_actual"] > 0.2) &
            (df["cop"].between(1.0, 7.0)) &
            (df["delta_t"] > 1.0)
        )
        mask &= ~((df["temp"].between(-3, 5)) & (df["cop"] < 1.5))
        df_clean = df[mask].copy()

        if len(df_clean) < 50:
            logger.warning("[PerformanceMap] Te weinig schone data.")
            return

        features_cop = ["temp", "supply_temp", "return_temp", "delta_t", "compressor_freq", "hvac_mode"]
        self.cop_model = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42)
        self.cop_model.fit(df_clean[features_cop], df_clean["cop"])

        features_power = ["compressor_freq", "temp", "supply_temp", "hvac_mode"]
        self.power_model = LinearRegression(fit_intercept=False)
        self.power_model.fit(df_clean[features_power], df_clean["wp_actual"])

        # Max Freq Training
        df_f = df[df["compressor_freq"] > 15].copy()
        df_f["t_rounded"] = df_f["temp"].round()
        max_freq_stats = df_f.groupby("t_rounded")["compressor_freq"].quantile(0.98).reset_index()
        self.max_freq_model = LinearRegression().fit(max_freq_stats[['t_rounded']], max_freq_stats['compressor_freq'])

        # Refs Training (Freq & Delta T)
        mask_ufh = (df["hvac_mode"] == "UFH") & (df["compressor_freq"] > 15)
        if mask_ufh.any():
            self.ufh_freq_ref = float(df.loc[mask_ufh, "compressor_freq"].median())
            self.ufh_delta_t_ref = float((df.loc[mask_ufh, "supply_temp"] - df.loc[mask_ufh, "return_temp"]).clip(1, 10).median())

        mask_dhw = (df["hvac_mode"] == "DHW") & (df["compressor_freq"] > 15)
        if mask_dhw.any():
            self.dhw_freq_ref = float(df.loc[mask_dhw, "compressor_freq"].median())
            self.dhw_delta_t_ref = float((df.loc[mask_dhw, "supply_temp"] - df.loc[mask_dhw, "return_temp"]).clip(3, 12).median())

        logger.info(f"[HPPerf] Learned refs: UFH={self.ufh_freq_ref:.1f}Hz/dT={self.ufh_delta_t_ref:.1f}, DHW={self.dhw_freq_ref:.1f}Hz/dT={self.dhw_delta_t_ref:.1f}")

        self.is_fitted = True
        joblib.dump({
            "cop_model": self.cop_model,
            "power_model": self.power_model,
            "max_freq_model": self.max_freq_model,
            "ufh_freq_ref": self.ufh_freq_ref,
            "dhw_freq_ref": self.dhw_freq_ref,
            "ufh_delta_t_ref": self.ufh_delta_t_ref,
            "dhw_delta_t_ref": self.dhw_delta_t_ref
        }, self.path)

    def predict_cop(self, t_out, t_supply, t_return, freq, mode_idx):
        if not self.is_fitted: return 3.5 if mode_idx==1 else 2.5
        delta_t = max(1.0, t_supply - t_return)
        X = [[t_out, t_supply, t_return, delta_t, freq, mode_idx]]
        return float(self.cop_model.predict(X)[0])

    def predict_p_el_slope(self, freq_ref, t_out, t_sink, mode_idx):
        if not self.is_fitted:
            return 0.04

        # Voorspel P_el met alle relevante features
        X = [[freq_ref, t_out, t_sink, mode_idx]]
        p_el = self.power_model.predict(X)[0]

        # De slope is het voorspelde vermogen gedeeld door de referentiefrequentie
        return max(0.01, p_el / freq_ref)

    def predict_max_freq(self, t_out):
        if self.max_freq_model is None:
            return 70.0
        freq = self.max_freq_model.predict([[t_out]])[0]
        return float(np.clip(freq, 25.0, 80.0))

# =========================================================
# 2. SYSTEM IDENTIFICATOR
# =========================================================
class SystemIdentificator:
    def __init__(self, path):
        self.path = Path(path)
        self.R, self.C = 15.0, 30.0
        self.K_emit = 0.15
        self.K_tank = 0.25
        self.K_loss_dhw = 0.15
        self._load()

    def _load(self):
        if self.path.exists():
            data = joblib.load(self.path)
            self.R, self.C = data.get("R", 15.0), data.get("C", 30.0)
            self.K_emit = data.get("K_emit", 0.15)
            self.K_tank = data.get("K_tank", 0.25)
            self.K_loss_dhw = data.get("K_loss_dhw", 0.15)

    def train(self, df):
        df_proc = df.copy().set_index("timestamp").sort_index().resample("15min").interpolate()
        mask_rc = (df_proc["hvac_mode"] == "UFH") & (df_proc["pv_actual"] < 0.05)
        train_rc = df_proc[mask_rc].copy()

        if len(train_rc) > 50:
            train_rc["dT_1h"] = train_rc["room_temp"].shift(-4) - train_rc["room_temp"]
            train_rc["X1"] = -(train_rc["room_temp"] - train_rc["temp"])
            train_rc["X2"] = train_rc["wp_output"]
            train_rc = train_rc.dropna()
            model = LinearRegression(fit_intercept=False).fit(train_rc[["X1","X2"]], train_rc["dT_1h"])
            self.C = 1.0 / np.clip(model.coef_[1], 1/150, 1/10)
            self.R = 1.0 / (np.clip(model.coef_[0], 1/(50*self.C), 1/(5*self.C)) * self.C)

        mask_ufh = (df["hvac_mode"] == "UFH") & (df["wp_output"] > 0.5)
        if len(df[mask_ufh]) > 50:
            dT_emit = ((df.loc[mask_ufh, "supply_temp"] + df.loc[mask_ufh, "return_temp"])/2) - df.loc[mask_ufh, "room_temp"]
            self.K_emit = np.clip(float(np.median(df.loc[mask_ufh, "wp_output"] / dT_emit.clip(lower=1.0))), 0.05, 0.5)

        mask_dhw = (df["hvac_mode"] == "DHW") & (df["wp_output"] > 1.0)
        if len(df[mask_dhw]) > 50:
            t_tank = (df.loc[mask_dhw, "dhw_top"] + df.loc[mask_dhw, "dhw_bottom"]) / 2
            dT_tank = ((df.loc[mask_dhw, "supply_temp"] + df.loc[mask_dhw, "return_temp"])/2) - t_tank
            self.K_tank = np.clip(float(np.median(df.loc[mask_dhw, "wp_output"] / dT_tank.clip(lower=1.0))), 0.1, 1.0)

        # K_loss_dhw
        df_l = df.copy().sort_values("timestamp")
        df_l["t_tank"] = (df_l["dhw_top"] + df_l["dhw_bottom"]) / 2.0
        df_l["change"] = df_l["t_tank"].diff() * 4
        mask_sb = (df_l["hvac_mode"] != "DHW") & (df_l["wp_output"] < 0.1) & (df_l["change"] < 0) & (df_l["change"] > -0.8)
        if len(df_l[mask_sb]) > 50:
            self.K_loss_dhw = np.clip(float(abs(df_l.loc[mask_sb, "change"].median())), 0.02, 0.5)

        joblib.dump({"R": self.R, "C": self.C, "K_emit": self.K_emit, "K_tank": self.K_tank, "K_loss_dhw": self.K_loss_dhw}, self.path)

# =========================================================
# 3. ML RESIDUALS
# =========================================================
class MLResidualPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None
        self.features = ["temp", "solar", "wind", "hour_sin", "hour_cos", "day_sin", "day_cos", "doy_sin", "doy_cos"]

    def train(self, df, R, C, is_dhw=False):
        df = df.copy().set_index("timestamp").sort_index().resample("15min").interpolate().reset_index()
        df = add_cyclic_time_features(df, "timestamp")
        dt = 0.25

        if not is_dhw:
            df = df[df["hvac_mode"]=="UFH"]
            target = (df["room_temp"].shift(-1) - df["room_temp"]) - ((df["wp_output"] - (df["room_temp"]-df["temp"])/R)*dt/C)
        else:
            df = df[(df["hvac_mode"] == "DHW") & (df["wp_output"] > 0.5)].copy()
            dhw_avg = (df["dhw_top"] + df["dhw_bottom"]) / 2
            target = dhw_avg.shift(-1) - dhw_avg - (df["wp_output"] * dt / 0.232)

        train_df = pd.concat([df[self.features], target], axis=1).dropna()
        if len(train_df) > 50:
            self.model = RandomForestRegressor(n_estimators=100).fit(train_df[self.features], train_df.iloc[:, -1])
            joblib.dump(self.model, self.path)

    def predict(self, forecast_df):
        if self.model is None: return np.zeros(len(forecast_df))
        df = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        return self.model.predict(df[self.features])

# =========================================================
# 4. THERMAL MPC
# =========================================================
class ThermalMPC:
    def __init__(self, ident, perf_map):
        self.ident = ident
        self.perf_map = perf_map
        self.horizon, self.dt = 48, 0.25
        self._build_problem()

    def _build_problem(self):
        T = self.horizon
        self.P_t_room_init = cp.Parameter()
        self.P_t_dhw_init = cp.Parameter()
        self.P_prices = cp.Parameter(T, nonneg=True)
        self.P_temp_out = cp.Parameter(T)

        self.P_max_freq = cp.Parameter(T, nonneg=True)
        self.P_dhw_loss_per_dt = cp.Parameter(nonneg=True)

        # We combineren slope en COP tot één parameter: "Thermisch vermogen per Hz"
        # Dit lost de DPP UserWarning op.
        self.P_th_per_hz_ufh = cp.Parameter(T, nonneg=True)
        self.P_th_per_hz_dhw = cp.Parameter(T, nonneg=True)
        self.P_el_per_hz_ufh = cp.Parameter(T, nonneg=True)
        self.P_el_per_hz_dhw = cp.Parameter(T, nonneg=True)

        self.P_ufh_res = cp.Parameter(T)
        self.P_dhw_res = cp.Parameter(T)
        self.P_solar = cp.Parameter(T, nonneg=True)

        self.f_ufh = cp.Variable(T, nonneg=True)
        self.f_dhw = cp.Variable(T, nonneg=True)
        self.ufh_on = cp.Variable(T, boolean=True)
        self.dhw_on = cp.Variable(T, boolean=True)
        self.p_grid = cp.Variable(T, nonneg=True)
        self.p_solar_self = cp.Variable(T, nonneg=True)
        self.t_room = cp.Variable(T+1)
        self.t_dhw = cp.Variable(T+1)
        self.s_room_low = cp.Variable(T, nonneg=True)
        self.s_dhw_low = cp.Variable(T, nonneg=True)

        R, C = self.ident.R, self.ident.C
        constraints = [self.t_room[0] == self.P_t_room_init, self.t_dhw[0] == self.P_t_dhw_init]

        for t in range(T):
            constraints += [
                self.f_ufh[t] <= self.ufh_on[t] * self.P_max_freq[t],
                self.f_dhw[t] <= self.dhw_on[t] * self.P_max_freq[t],
                self.ufh_on[t] + self.dhw_on[t] <= 1,
                # Minimale draaitijd simuleren: forceer frequentie als de unit aan staat
                self.f_ufh[t] >= self.ufh_on[t] * 25,
                self.f_dhw[t] >= self.dhw_on[t] * 40
            ]

            p_el_t = self.f_ufh[t] * self.P_el_per_hz_ufh[t] + self.f_dhw[t] * self.P_el_per_hz_dhw[t]

            constraints += [
                self.p_grid[t] >= p_el_t - self.P_solar[t],
                self.p_solar_self[t] <= p_el_t,
                self.p_solar_self[t] <= self.P_solar[t]
            ]

            constraints += [
                self.t_room[t+1] == self.t_room[t] + ((self.f_ufh[t]*self.P_th_per_hz_ufh[t] - (self.t_room[t]-self.P_temp_out[t])/R)*self.dt/C + self.P_ufh_res[t]),
                self.t_dhw[t+1] == self.t_dhw[t] + ((self.f_dhw[t]*self.P_th_per_hz_dhw[t]*self.dt)/0.232 + self.P_dhw_res[t] - self.P_dhw_loss_per_dt),
                self.t_room[t+1] + self.s_room_low[t] >= 18.0, # Veiligheidsondergrens
                self.t_dhw[t+1] + self.s_dhw_low[t] >= 20.0, # Veiligheidsondergrens
                self.t_room[t+1] <= 22.5
            ]

        # OBJECTIVE FUNCTION
        # Kosten elektriciteit
        cost_el = cp.sum(cp.multiply(self.p_grid, self.P_prices)) * self.dt

        # 2. PV Bonus (beloning voor eigenverbruik)
        pv_bonus = cp.sum(self.p_solar_self) * 0.22

        # 3. Comfort Targets (De "Magneten")
        # Straf voor te koud (pos(Target - T))
        comfort_room_low = cp.sum(cp.pos(20.0 - self.t_room)) * 4.0
        comfort_dhw_low = cp.sum(cp.pos(50.0 - self.t_dhw)) * 2.0

        # NIEUW: Straf voor te warm (pos(T - Target)) -> Dit dwingt uitschakeling af!
        comfort_room_high = cp.sum(cp.pos(self.t_room - 21.0)) * 5.0
        comfort_dhw_high = cp.sum(cp.pos(self.t_dhw - 51.0)) * 2.0

        # 4. Schakelkosten (Switching Costs)
        # Verlaagd van 10.0 naar 0.5 om uitschakelen rendabel te maken
        ufh_switch = cp.sum(cp.abs(self.ufh_on[1:] - self.ufh_on[:-1])) * 0.5
        dhw_switch = cp.sum(cp.abs(self.dhw_on[1:] - self.dhw_on[:-1])) * 0.5

        # 5. Veiligheid (Absolute bodem)
        safety_violation = cp.sum(self.s_room_low + self.s_dhw_low) * 100

        # TOTAAL
        self.problem = cp.Problem(cp.Minimize(
            cost_el -
            pv_bonus +
            comfort_room_low +
            comfort_room_high +
            comfort_dhw_low +
            comfort_dhw_high +
            ufh_switch +
            dhw_switch +
            safety_violation
        ), constraints)

    def solve(self, state, forecast_df, res_u, res_d):
        T = self.horizon
        t_out = forecast_df.temp.values
        t_prices = [0.22] * T
        t_solar = forecast_df.solar_forecast.values

        dhw_start = (state["dhw_top"] + state["dhw_bottom"]) / 2
        current_est_room, current_est_dhw = state["room_temp"], dhw_start

        # Tijdelijke opslag voor gecombineerde waarden (DPP optimalisatie)
        th_per_hz_u, th_per_hz_d = np.zeros(T), np.zeros(T)
        el_per_hz_u, el_per_hz_d = np.zeros(T), np.zeros(T)
        v_max_freq = np.zeros(T)

        # Haal alle geleerde parameters op
        ufh_ref = self.perf_map.ufh_freq_ref
        dhw_ref = self.perf_map.dhw_freq_ref
        ufh_dt = self.perf_map.ufh_delta_t_ref
        dhw_dt = self.perf_map.dhw_delta_t_ref

        # Stel stilstandsverlies parameter in
        self.P_dhw_loss_per_dt.value = self.ident.K_loss_dhw * self.dt

        for t in range(T):
            # Schat de maximale frequentie voor dit tijdstip
            v_max_freq[t] = self.perf_map.predict_max_freq(t_out[t])

            # 1. UFH: Dynamische Stooklijn o.b.v. warmteverlies huis
            calc_room = max(current_est_room, 20.0)
            heat_loss_kw = max(0, (calc_room - t_out[t]) / self.ident.R)
            # Overtemp = Vermogen / Afgiftecoëfficiënt
            overtemp_ufh = np.clip(heat_loss_kw / self.ident.K_emit, 0, 15.0)
            t_supply_ufh = calc_room + (ufh_dt / 2.0) + overtemp_ufh

            cop_u = self.perf_map.predict_cop(t_out[t], t_supply_ufh, t_supply_ufh - ufh_dt, ufh_ref, 1)
            # Slope nu inclusief aanvoertemperatuur (t_sink) en modus
            slope_u = self.perf_map.predict_p_el_slope(ufh_ref, t_out[t], t_supply_ufh, 1)

            el_per_hz_u[t] = slope_u
            th_per_hz_u[t] = slope_u * cop_u

            # 2. DHW: Dynamische Aanvoer o.b.v. vermogen warmtepomp
            calc_dhw = max(current_est_dhw, 40.0)

            # Stap A: Maak een eerste schatting van het thermisch vermogen
            # We gebruiken hier calc_dhw + dhw_dt als tijdelijke sink voor de allereerste COP-check
            p_th_est = dhw_ref * self.perf_map.predict_p_el_slope(dhw_ref, t_out[t], calc_dhw + dhw_dt, 2) * self.perf_map.predict_cop(t_out[t], calc_dhw + dhw_dt, calc_dhw, dhw_ref, 2)

            # Stap B: Bereken de overtemp die nodig is om DIT vermogen door de spiraal te duwen
            # Overtemp = P_th / K_tank
            overtemp_dhw = np.clip(p_th_est / self.ident.K_tank, 0, 20.0)

            # Stap C: De aanvoer is de tank-temp + de helft van de water-delta + de overtemp over de spiraal
            t_supply_dhw = min(calc_dhw + (dhw_dt / 2.0) + overtemp_dhw, 58.0)

            # Stap D: De definitieve waarden voor de solver
            cop_d = self.perf_map.predict_cop(t_out[t], t_supply_dhw, t_supply_dhw - dhw_dt, dhw_ref, 2)
            slope_d = self.perf_map.predict_p_el_slope(dhw_ref, t_out[t], t_supply_dhw, 2)

            el_per_hz_d[t] = slope_d
            th_per_hz_d[t] = slope_d * cop_d

            # 3. State Update
            current_est_room = calc_room - (calc_room - t_out[t])/(self.ident.R*self.ident.C)*self.dt + res_u[t]
            current_est_dhw = calc_dhw - (self.ident.K_loss_dhw * self.dt) + res_d[t]

        # Vul parameters
        self.P_max_freq.value = v_max_freq
        self.P_th_per_hz_ufh.value = th_per_hz_u
        self.P_th_per_hz_dhw.value = th_per_hz_d
        self.P_el_per_hz_ufh.value = el_per_hz_u
        self.P_el_per_hz_dhw.value = el_per_hz_d

        self.P_t_room_init.value = state["room_temp"]
        self.P_t_dhw_init.value = dhw_start
        self.P_temp_out.value = t_out
        self.P_prices.value = t_prices
        self.P_solar.value = t_solar
        self.P_ufh_res.value = res_u
        self.P_dhw_res.value = res_d

        try:
            self.problem.solve(solver=cp.CBC)
            return self.f_ufh.value, self.f_dhw.value
        except Exception as e:
            logger.error(f"MPC Solve failed: {e}")
            return np.zeros(T), np.zeros(T)

# =========================================================
# 5. OPTIMIZER
# =========================================================
class Optimizer:
    def __init__(self, config, database):
        self.db = database
        self.perf_map = HPPerformanceMap(config.perf_map_path)
        self.ident = SystemIdentificator(config.rc_model_path)
        self.res_ufh = MLResidualPredictor(config.ufh_res_path)
        self.res_dhw = MLResidualPredictor(config.dhw_res_path)
        self.mpc = ThermalMPC(self.ident, self.perf_map)

    def train(self):
        df = self.db.get_history(days=730)
        if df.empty: return
        self.perf_map.train(df)
        self.ident.train(df)
        self.res_ufh.train(df, self.ident.R, self.ident.C, False)
        self.res_dhw.train(df, self.ident.R, self.ident.C, True)

    def resolve(self, context):
        res_u, res_d = self.res_ufh.predict(context.forecast_df), self.res_dhw.predict(context.forecast_df)
        state = {"room_temp": context.room_temp, "dhw_top": context.dhw_top, "dhw_bottom": context.dhw_bottom}
        f_ufh, f_dhw = self.mpc.solve(state, context.forecast_df, res_u, res_d)
        hz = f_ufh[0] if f_ufh[0]>5 else f_dhw[0] if f_dhw[0]>5 else 0
        mode = "UFH" if f_ufh[0]>5 else "DHW" if f_dhw[0]>5 else "OFF"
        return {"mode": mode, "freq": round(hz,1)}