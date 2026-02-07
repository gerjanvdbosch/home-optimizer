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
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                self.cop_model = data["cop_model"]
                self.power_model = data["power_model"]
                self.max_freq_model = data["max_freq_model"]
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

        self.power_model = LinearRegression(fit_intercept=False)
        self.power_model.fit(df_clean[["compressor_freq", "temp"]], df_clean["wp_actual"])

        df['t_rounded'] = df['temp'].round()
        max_freq_stats = df.groupby('t_rounded')['compressor_freq'].quantile(0.98).reset_index()

        # We fitten een simpele lineaire regressie: MaxFreq = a * Tout + b
        self.max_freq_model = LinearRegression().fit(
            max_freq_stats.index.values.reshape(-1, 1),
            max_freq_stats.values
        )

        self.is_fitted = True
        joblib.dump({"cop_model": self.cop_model, "power_model": self.power_model,  "max_freq_model": self.max_freq_model}, self.path)

    def predict_cop(self, t_out, t_supply, t_return, freq, mode_idx):
        if not self.is_fitted: return 3.5 if mode_idx==1 else 2.5
        delta_t = max(1.0, t_supply - t_return)
        X = [[t_out, t_supply, t_return, delta_t, freq, mode_idx]]
        return float(self.cop_model.predict(X)[0])

    def predict_p_el_slope(self, freq_ref, t_out):
        if not self.is_fitted: return 0.04
        p_el = self.power_model.predict([[freq_ref, t_out]])[0]
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
        self._load()

    def _load(self):
        if self.path.exists():
            data = joblib.load(self.path)
            self.R, self.C = data.get("R", 15.0), data.get("C", 30.0)
            self.K_emit = data.get("K_emit", 0.15)
            self.K_tank = data.get("K_tank", 0.25)

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

        joblib.dump({"R": self.R, "C": self.C, "K_emit": self.K_emit, "K_tank": self.K_tank}, self.path)

# =========================================================
# 3. ML RESIDUALS
# =========================================================
class MLResidualPredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.model = None

    def train(self, df, R, C, is_dhw=False):
        df = df.copy().set_index("timestamp").sort_index().resample("15min").interpolate().reset_index()
        df = add_cyclic_time_features(df, "timestamp")
        dt = 0.25

        if not is_dhw:
            df = df[df["hvac_mode"]=="UFH"]
            target = (df["room_temp"].shift(-1) - df["room_temp"]) - ((df["wp_output"] - (df["room_temp"]-df["temp"])/R)*dt/C)
        else:
            df = df[df["hvac_mode"] == "DHW"]
            dhw_avg = (df["dhw_top"] + df["dhw_bottom"]) / 2
            target = dhw_avg.shift(-1) - dhw_avg - (df["wp_output"] * dt / 0.232)

        feats = ["temp", "solar", "wind", "hour_sin", "hour_cos", "day_sin", "day_cos"]
        train_df = pd.concat([df[feats], target], axis=1).dropna()
        if len(train_df) > 50:
            self.model = RandomForestRegressor(n_estimators=100).fit(train_df[feats], train_df.iloc[:, -1])
            joblib.dump(self.model, self.path)

    def predict(self, forecast_df):
        if self.model is None: return np.zeros(len(forecast_df))
        df = add_cyclic_time_features(forecast_df.copy(), "timestamp")
        return self.model.predict(df[["temp", "solar", "wind", "hour_sin", "hour_cos", "day_sin", "day_cos"]])

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
                self.ufh_on[t] + self.dhw_on[t] <= 1
            ]

            p_el_t = self.f_ufh[t] * self.P_el_per_hz_ufh[t] + self.f_dhw[t] * self.P_el_per_hz_dhw[t]

            constraints += [
                self.p_grid[t] >= p_el_t - self.P_solar[t],
                self.p_solar_self[t] <= p_el_t,
                self.p_solar_self[t] <= self.P_solar[t]
            ]

            p_th_ufh = self.f_ufh[t] * self.P_th_per_hz_ufh[t]
            p_th_dhw = self.f_dhw[t] * self.P_th_per_hz_dhw[t]

            constraints += [
                self.t_room[t+1] == self.t_room[t] + ((p_th_ufh - (self.t_room[t]-self.P_temp_out[t])/R)*self.dt/C + self.P_ufh_res[t]),
                self.t_dhw[t+1] == self.t_dhw[t] + ((p_th_dhw*self.dt)/0.232 + self.P_dhw_res[t]),
                self.t_room[t+1] + self.s_room_low[t] >= 19.5,
                self.t_dhw[t+1] + self.s_dhw_low[t] >= 38.0,
                self.t_room[t+1] <= 23.5
            ]

        obj = cp.Minimize(
            cp.sum(cp.multiply(self.p_grid, self.P_prices))*self.dt -
            cp.sum(self.p_solar_self)*0.22 +
            cp.sum(self.s_room_low + self.s_dhw_low)*150 +
            cp.sum(cp.pos(21.0 - self.t_room))*0.1
        )
        self.problem = cp.Problem(obj, constraints)

    def solve(self, state, forecast_df, res_u, res_d):
        T = self.horizon
        t_out = forecast_df.temp.values

        dhw_start = (state["dhw_top"] + state["dhw_bottom"]) / 2
        current_est_room, current_est_dhw = state["room_temp"], dhw_start

        # Tijdelijke opslag voor gecombineerde waarden
        th_per_hz_u, th_per_hz_d = np.zeros(T), np.zeros(T)
        el_per_hz_u, el_per_hz_d = np.zeros(T), np.zeros(T)

        v_max_freq = np.zeros(T)

        for t in range(T):
            v_max_freq[t] = self.perf_map.predict_max_freq(t_out[t])

            # 1. UFH
            calc_room = max(current_est_room, 20.0)
            heat_loss_kw = max(0, (calc_room - t_out[t]) / self.ident.R)
            overtemp_ufh = np.clip(heat_loss_kw / self.ident.K_emit, 0, 15.0)
            t_supply_ufh = calc_room + 2.0 + overtemp_ufh

            cop_u = self.perf_map.predict_cop(t_out[t], t_supply_ufh, t_supply_ufh-4.0, 35, 1)
            slope_u = self.perf_map.predict_p_el_slope(35, t_out[t])

            el_per_hz_u[t] = slope_u
            th_per_hz_u[t] = slope_u * cop_u

            # 2. DHW
            calc_dhw = max(current_est_dhw, 40.0)
            p_th_dhw_est = 55 * self.perf_map.predict_p_el_slope(55, t_out[t]) * 2.5
            overtemp_dhw = np.clip(p_th_dhw_est / self.ident.K_tank, 0, 20.0)
            t_supply_dhw = min(calc_dhw + 3.5 + overtemp_dhw, 58.0)

            cop_d = self.perf_map.predict_cop(t_out[t], t_supply_dhw, t_supply_dhw-7.0, 55, 2)
            slope_d = self.perf_map.predict_p_el_slope(55, t_out[t])

            el_per_hz_d[t] = slope_d
            th_per_hz_d[t] = slope_d * cop_d

            # 3. State Update
            current_est_room = calc_room - (calc_room - t_out[t])/(self.ident.R*self.ident.C)*self.dt + res_u[t]
            current_est_dhw = calc_dhw - 0.05 + res_d[t]

        # Vul parameters
        self.P_max_freq.value = v_max_freq
        self.P_th_per_hz_ufh.value = th_per_hz_u
        self.P_th_per_hz_dhw.value = th_per_hz_d
        self.P_el_per_hz_ufh.value = el_per_hz_u
        self.P_el_per_hz_dhw.value = el_per_hz_d

        self.P_t_room_init.value = state["room_temp"]
        self.P_t_dhw_init.value = dhw_start
        self.P_temp_out.value = t_out
        self.P_prices.value = forecast_df["price"].values
        self.P_solar.value = forecast_df["solar_forecast"].values
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