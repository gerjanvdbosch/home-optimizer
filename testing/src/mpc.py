import cvxpy as cp
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, time
from context import Context
from thermal import ThermalNowCaster, ThermalModel
from pathlib import Path

logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class SystemState:
    """Snapshot van de situatie op een bepaald moment."""
    timestamp: datetime
    room_temp: float
    outside_temp: float
    solar_rad: float
    wind_speed: float
    hp_power_factor: float # 0.0 tot 1.0 (of freq / 100)
    supply_temp: float

@dataclass
class DemandScore:
    name: str
    score: float
    reason: str

# =========================================================
# ARBITER (De Beslisser)
# =========================================================
class PriorityEvaluator:
    def __init__(self, config: MPCConfig):
        self.cfg = config

    def get_boiler_score(self, current_temp: float, mpc_power_kw: float) -> DemandScore:
        score = 0.0
        reasons = []

        # 1. VERZADIGING (Is hij vol?)
        if current_temp >= self.cfg.tank_max_temp - 1.0:
            return DemandScore("Boiler", -1.0, "Vol")

        # 2. NOODGEVAL (Koud water, niet douchen)
        # Grens: 40 graden
        if current_temp < 40.0:
            score += 5000.0 # Wint ALTIJD
            reasons.append(f"Kritiek ({current_temp:.1f}°C)")
            return DemandScore("Boiler", score, ", ".join(reasons))

        # 3. URGENTIE (Lauw water, zsm bijwarmen)
        # Grens: 40 tot 48 graden.
        # Dit moet winnen van een warmtepomp die het huis op temperatuur houdt (score ~50-100),
        # maar verliezen van een ijskoud huis (score > 300).
        if current_temp < 48.0:
            score += 300.0
            reasons.append("Urgent laag")

        # 4. MPC WENS (Efficiency / Zon / Planning)
        if mpc_power_kw > 0.1:
            score += 50.0
            reasons.append("MPC schema")

        # 5. VULLINGSGRAAD (Comfort buffer)
        # Hoe leger, hoe liever. Max 20 punten.
        # Dit zorgt dat hij bij 50 graden minder 'gretig' is dan bij 49.
        deficit = max(0, 55.0 - current_temp)
        score += deficit * 2.0

        if score > 0:
            return DemandScore("Boiler", score, ", ".join(reasons))
        return DemandScore("Boiler", 0.0, "Geen vraag")

    def get_hp_score(self, current_temp: float, target_temp: float, mpc_power_kw: float) -> DemandScore:
        # 1. VERZADIGING (Is het warm?)
        # Als we OP of BOVEN target zitten, is de score 0 (tenzij MPC wil bufferen)
        if current_temp >= target_temp:
            # We geven hier GEEN negatieve score, want misschien wil MPC bufferen (zon).
            # Maar we geven ook geen comfort-punten.
            pass

        score = 0.0
        reasons = []

        # 2. NOODGEVAL (Bevriezing)
        if current_temp < 16.0:
            score += 4000.0 # Wint van Boiler Urgent, verliest van Boiler Kritiek
            reasons.append("Kritiek koud")

        # 3. COMFORT TEKORT (Lineair)
        # 0.5 graad te koud = 100 punten.
        # 1.0 graad te koud = 200 punten.
        deficit = max(0, target_temp - current_temp)

        if deficit > 0.05: # Kleine hysterese
            comfort_score = deficit * 200.0
            score += comfort_score
            reasons.append(f"Comfort vraag ({deficit:.1f}°C)")

        # 4. MPC WENS (Bufferen / Efficiency)
        # Alleen als er GEEN comfort vraag is, is dit "Luxe".
        # Als er WEL comfort vraag is, zit MPC wens daar vaak al in verwerkt.
        if mpc_power_kw > 0.1:
            # Als huis al warm is, is dit 'Solar Bufferen'. Score laag (40).
            # Als huis koud is, is dit 'Ondersteuning'. Score hoog (+50).
            base = 40.0 if deficit <= 0 else 50.0
            score += base
            reasons.append("Vloer bufferen" if deficit <= 0 else "MPC schema")

        if score > 0:
            return DemandScore("HeatPump", score, ", ".join(reasons))
        return DemandScore("HeatPump", 0.0, "Geen vraag")

@dataclass(frozen=True)
class MPCConfig:
    # Tijdstappen
    dt_minutes: int = 15
    horizon_steps: int = 48  # Kijk 12 uur vooruit (4 * 12)

    # Apparaat Limieten (kW)
    hp_max_kw: float = 1.3       # Max elektrisch vermogen WP
    boiler_max_kw: float = 2.9   # Max elektrisch vermogen Boiler
    grid_max_kw: float = 5.0     # Hoofdzekering limiet (bijv 3x25A = ~17kW, maar hier veilig 6)

    # --- FYSICA HUIS (THERMISCH) ---
    # Hoeveel kWh is nodig om huis 1 graad op te warmen?
    # Rekenvoorbeeld: Betonvloer 50m2 * 0.1m dik * C_beton
    room_capacity_kwh_per_c: float = 5.0

    # Hoeveel kW verlies je per graad verschil met buiten?
    # Slecht huis: 0.5, Passiefhuis: 0.1
    room_loss_kw_per_c: float = 0.35

    # --- FYSICA BOILER ---
    tank_capacity_kwh_per_c: float = 0.15 # 200L water (4.18kJ/kgC) -> ~0.2 kWh/C
    tank_loss_kw: float = 0.05            # Stilstandverlies (goede isolatie)

    # Doelen
#     tank_min_temp: float = 40.0   # Minimum voor legionella/comfort
    tank_max_temp: float = 50.0   # Max temperatuur
#     tank_start_temp: float = 40.0 # Koud water inlaat schatting

    # Kosten (Wegingsfactoren voor de solver)
    grid_price_eur_per_kwh: float = 0.30
    solar_buffer_bonus: float = 0.05
    discomfort_cost: float = 50.0  # Hoge straf op koud huis
    switch_cost: float = 0.05      # Kleine straf op aan/uit klapperen

    # Comfort Profiel (Graden Celsius)
    temp_night: float = 19.0    # 22:00 - 06:00
    temp_day: float = 20.0    # 06:00 - 22:00

    min_room_temp: float = 15.0
    max_room_temp: float = 22.0

    boiler_daily_energy_kwh: float = 3.0

@dataclass
class Plan:
    action: str  # "OFF", "HEAT_PUMP", "BOILER"
    reason: str
    hp_power: float = 0.0
    boiler_power: float = 0.0

class MPCPlanner:
    def __init__(self, context: Context):
        self.context = context
        self.cfg = MPCConfig()
        self.thermal_model = ThermalModel(Path("thermal_model.pkl"))
        self.nowcaster = ThermalNowCaster() # Voor live correcties

    def create_plan(self) -> Plan:
        self.update_nowcaster()

        # 1. Validatie
        df = self.context.forecast_df
        if df is None or df.empty:
            return Plan("OFF", "Geen data")

        # 2. Horizon bepalen
        now = pd.Timestamp.now(tz=timezone.utc).floor(f"{self.cfg.dt_minutes}min")
        horizon = df[df["timestamp"] >= now].iloc[: self.cfg.horizon_steps].copy()

        if len(horizon) < 4:
            return Plan("OFF", "Horizon te kort")

        # 3. Solver draaien
        u_hp, u_boiler, status = self._solve_mpc(horizon)

        # 4. Fallback
        if status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"[MPC] Solver faalde ({status}), fallback naar simpele logica.")
            return self._fallback_plan(horizon)

        # 5. Actie bepalen
        arbiter = PriorityEvaluator(self.cfg)

        room_now = getattr(self.context, "room_temp", 20.0)
        tank_now = getattr(self.context, "tank_temp", 50.0)
        room_target = self._get_target_temp(self.context.now.time())

        arbiter = PriorityEvaluator(self.cfg)

        # Laat ze bieden!
        boiler_req = arbiter.get_boiler_score(tank_now, u_boiler)
        hp_req = arbiter.get_hp_score(room_now, room_target, u_hp)

        # De Winnaar
        if boiler_req.score <= 0 and hp_req.score <= 0:
           return Plan("OFF", "Verzadigd (Geen vraag)")

        elif boiler_req.score > hp_req.score:
           return Plan(
               "BOILER",
               f"Prioriteit: {boiler_req.reason} (Score {int(boiler_req.score)})",
               u_hp, u_boiler
           )

        else:
           return Plan(
               "HEAT_PUMP",
               f"Prioriteit: {hp_req.reason} (Score {int(hp_req.score)})",
               u_hp, u_boiler
           )

    def _solve_mpc(self, df: pd.DataFrame):
        N = len(df)
        dt = self.cfg.dt_minutes / 60.0

        # --- VARIABELEN ---
        T_room = cp.Variable(N + 1)
        E_tank_added = cp.Variable(N + 1)
        P_hp = cp.Variable(N)
        P_boiler = cp.Variable(N)
        P_grid = cp.Variable(N) # Hulpvariabele voor import (>= 0)

        # --- INITIAL STATE ---
        T_room_0 = float(getattr(self.context, "room_temp", 19.0))

        thermal_residuals = self._calculate_thermal_residuals(df, T_room_0)

        # Boiler doel: Wat moet er VANDAAG nog bij?
        already_loaded = getattr(self.context, "daily_boiler_kwh", 0.0)
        target_kwh = max(0.0, self.cfg.boiler_daily_energy_kwh - already_loaded)

        constraints = [
            T_room[0] == T_room_0,
            E_tank_added[0] == 0.0
        ]

        cost = 0

        slack_min_temp = cp.Variable(N, nonneg=True) # Te koud
        slack_max_temp = cp.Variable(N, nonneg=True) # Te warm

        slack_min = cp.Variable(N, nonneg=True)
        slack_max = cp.Variable(N, nonneg=True)

        # --- LOOP OVER HORIZON ---
        for t in range(N):
            row = df.iloc[t]

            # Inputs
            T_out = float(row.get("temp", 0.0))
            PV = float(row.get("power_corrected", 0.0))
            Load = float(row.get("consumption", 0.2))

            # Doel
            ts_local = row["timestamp"].tz_convert(datetime.now().astimezone().tzinfo)
            T_target = self._get_target_temp(ts_local.time())

            # COP Curve (Rendement)
            COP = max(2.0, min(5.5, 3.0 + 0.1 * (T_out - 7.0)))

            # 1. Power Constraints
            constraints += [
                P_hp[t] <= self.cfg.hp_max_kw,
                P_boiler[t] <= self.cfg.boiler_max_kw,
                P_hp[t] + P_boiler[t] <= self.cfg.grid_max_kw,

                # Grid balans
                P_grid[t] >= P_hp[t] + P_boiler[t] + Load - PV
            ]

            # 2. Thermische Dynamica (Huis)
            heat_in = P_hp[t] * COP
            heat_loss = self.cfg.room_loss_kw_per_c * (T_room[t] - T_out)

            phys_change = (heat_in - heat_loss) * (dt / self.cfg.room_capacity_kwh_per_c)

            constraints += [
                T_room[t+1] == T_room[t] + phys_change + thermal_residuals[t],

                # Comfort bandbreedte (Harde grenzen)
                T_room[t+1] + slack_min_temp[t] >= self.cfg.min_room_temp,
                T_room[t+1] - slack_max_temp[t] <= self.cfg.max_room_temp
            ]

            # 3. Boiler Accumulatie
            constraints += [
                E_tank_added[t+1] == E_tank_added[t] + P_boiler[t] * dt
            ]

            # 4. Kostenfunctie
            # A. Euro's
            cost += P_grid[t] * self.cfg.grid_price_eur_per_kwh * dt
            cost -= P_boiler[t] * self.cfg.solar_buffer_bonus * dt

            # B. Comfort Straf (Kwadratisch: beetje koud is ok, heel koud is erg)
            # We straffen alleen als we ONDER target zitten.
            discomfort = cp.pos(T_target - T_room[t])
            cost += self.cfg.discomfort_cost * cp.square(discomfort)

            # C. Pendel Straf (Voorkom aan/uit gedrag)
            if t > 0:
                cost += self.cfg.switch_cost * (
                    cp.abs(P_hp[t] - P_hp[t-1]) +
                    cp.abs(P_boiler[t] - P_boiler[t-1])
                )

        # --- EINDSCONSTRAINTS (Soft) ---
        # We willen dat de tank vol is aan het eind, maar we dwingen het niet hard af
        # (anders faalt de solver als de horizon te kort is).
        # We geven een ENORME straf op het tekort.
        if target_kwh > 0.1:
            shortage = cp.pos(target_kwh - E_tank_added[N])
            cost += 1000 * shortage # Prioriteit #1

        # Solver starten
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            # Probeer standaard OSQP solver
            problem.solve(
                solver=cp.OSQP,
                warm_start=False,    # <--- BELANGRIJK: Reset memory
                max_iter=50000,
                adaptive_rho=True,
                eps_abs=1e-2,
                eps_rel=1e-2
            )
        except Exception:
            try:
                # Fallback naar ECOS
                problem.solve(solver=cp.ECOS)
            except:
                return 0.0, 0.0, "failed"

        # Resultaten ophalen (veilig)
        u_hp = float(P_hp.value[0]) if P_hp.value is not None else 0.0
        u_boiler = float(P_boiler.value[0]) if P_boiler.value is not None else 0.0

        return u_hp, u_boiler, problem.status

    def _get_target_temp(self, local_time: time) -> float:
        if 6 <= local_time.hour < 22:
            return self.cfg.temp_day
        return self.cfg.temp_night

    def _fallback_plan(self, df: pd.DataFrame) -> Plan:
        """Simpele thermostaat logica als wiskunde faalt."""
        t_room = getattr(self.context, "room_temp", 19.0)
        target = self._get_target_temp(datetime.now().time())

        if t_room < target - 0.5:
            return Plan("HEAT_PUMP", "Fallback: Te koud")
        return Plan("OFF", "Fallback: OK")

    def _calculate_thermal_residuals(self, df: pd.DataFrame, start_temp: float):
        """
        Gebruikt het ML-model om te voorspellen hoeveel de temperatuur
        afwijkt van de simpele natuurkunde-formule (bijv. door wind/zon).
        Geeft een lijst met correcties terug voor elk tijdstip.
        """
        residuals = []
        sim_temp = start_temp
        prev_delta = 0.0 # Aanname

        # Instellingen voor de 'basis fysica' in de MPC (moeten matchen met config!)
        C_room = self.cfg.room_capacity_kwh_per_c
        Loss_room = self.cfg.room_loss_kw_per_c
        dt_hours = self.cfg.dt_minutes / 60.0

        for i, row in df.iterrows():
            # 1. Wat zegt het ML Model? (Met verwarming UIT)
            # We simuleren de 'natuurlijke' loop van het huis
            ml_delta = self.thermal_model.predict_step(
                inside=sim_temp,
                outside=float(row.get("temp", 0.0)),
                freq=0.0, # Verwarming UIT
                supply_temp=25.0, # Kamertemp
                solar=float(row.get('pv_estimate', 0)),
                wind=float(row.get('wind', 0)),
                prev_delta=prev_delta
            )

            # Voeg de NowCaster bias toe (live correctie)
            ml_delta += self.nowcaster.bias

            # 2. Wat zegt de simpele MPC formule? (Met verwarming UIT)
            # Delta = -Verlies * (Binnen - Buiten) / Capaciteit
            phys_delta = -Loss_room * (sim_temp - float(row.get("temp", 0.0))) * (dt_hours / C_room)

            # 3. Het verschil is de 'invloed van het weer' (Zon + Wind + Tocht)
            # Als ML zegt: +0.2 (door zon) en Fysica zegt: -0.1 (door kou),
            # Dan is de 'residual' +0.3.
            res = ml_delta - phys_delta
            residuals.append(res)

            # Update simulatie voor volgende stap
            sim_temp += ml_delta
            prev_delta = ml_delta

        return np.nan_to_num(residuals, nan=0.0).tolist()

    def update_nowcaster(self):
        """
        Leert van het verleden: Vergelijk voorspelling met realiteit.
        Moet aangeroepen worden aan het begin van elke tick.
        """
        current_state = self._capture_current_state()
        last_state = self.context.last_state

        # We kunnen pas leren als we een vorig meetpunt hebben
        if last_state is None:
            self.context.last_state = current_state
            return

        # 1. Hoeveel tijd is er verstreken?
        dt_minutes = (current_state.timestamp - last_state.timestamp).total_seconds() / 60

        # Alleen updaten als er ongeveer een kwartier voorbij is (10-20 min)
        if 10 <= dt_minutes <= 20:

            # 2. Wat is er ECHT gebeurd?
            actual_delta = current_state.room_temp - last_state.room_temp

            # 3. Wat dacht het MODEL dat er zou gebeuren?
            # We vragen het model om de delta te voorspellen o.b.v. de situatie van TOEN.
            predicted_delta = self.thermal_model.predict_step(
                inside=last_state.room_temp,
                outside=last_state.outside_temp,
                power_factor=last_state.hp_power_factor,
                supply_temp=last_state.supply_temp,
                solar=last_state.solar_rad,
                wind=last_state.wind_speed,
                prev_delta=0.0 # Aanname, of ook opslaan in history
            )

            # 4. Update de NowCaster (Leer van de fout)
            self.nowcaster.update(predicted_delta, actual_delta)

            # Update context voor logging/dashboard
            self.context.current_bias = self.nowcaster.bias
            logger.info(
                f"[NowCaster] Real: {actual_delta:+.3f}, Pred: {predicted_delta:+.3f}, "
                f"Error: {actual_delta - predicted_delta:+.3f} -> New Bias: {self.nowcaster.bias:+.3f}"
            )

        # 5. Sla huidige staat op als 'vorige' voor de volgende keer
        self.context.last_state = current_state

    def _capture_current_state(self) -> SystemState:
        """Helper om de huidige sensordata in een object te vangen."""
        # Haal data uit context of direct van sensoren
        # Power factor berekenen: Freq / Max Freq, of 1 als aan / 0 als uit
        freq = getattr(self.context, "compressor_freq", 0)
        power_factor = min(freq / 100.0, 1.0)

        return SystemState(
            timestamp=datetime.now(timezone.utc),
            room_temp=getattr(self.context, "room_temp", 20.0),
            outside_temp=getattr(self.context, "outside_temp", 10.0),
            solar_rad=getattr(self.context, "current_pv", 0.0), # Of uit PV vermogen afleiden
            wind_speed=getattr(self.context, "wind_speed", 0.0),
            hp_power_factor=power_factor,
            supply_temp=getattr(self.context, "supply_temp", 20.0)
        )