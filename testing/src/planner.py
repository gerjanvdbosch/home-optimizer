import logging

from dataclasses import dataclass
from context import Context
from config import Config
from optimizer import Optimizer
from pathlib import Path
from datetime import timedelta
from thermal import ThermalNowCaster, ThermalModel, ThermalPlanner

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str = None
    reason: str = None
    dhw_start_time: str = None


class Planner:
    def __init__(self, context: Context, config: Config):
        self.optimizer = Optimizer(config.pv_max_kw)
        self.context = context

        # Setup Thermal Models
        # HVAC (Vloer)
        self.hvac_nowcaster = ThermalNowCaster()
        self.hvac_model = ThermalModel(Path("/config/models/hvac_model.joblib"))
        self.hvac_planner = ThermalPlanner(self.hvac_model, self.hvac_nowcaster)

        # DHW (Boiler)
        self.dhw_nowcaster = ThermalNowCaster()
        self.dhw_model = ThermalModel(Path("/config/models/dhw_model.joblib"))
        self.dhw_planner = ThermalPlanner(self.dhw_model, self.dhw_nowcaster)

    def create_plan(self):
        """
        Hoofdfunctie die elke 15 min wordt aangeroepen.
        Bepaalt of Boiler en/of Verwarming aan moet.
        """
        now = self.context.now
        now = now.replace(hour=12)

        # Reset acties
        self.context.action = "WAIT"
        self.context.reason = "Geen actie nodig"

        # 1. Update NowCasters (Leer van het afgelopen kwartier)
        # Dit zou je idealiter doen met echte meetdata vs voorspelling van vorig kwartier
        # self._update_nowcasters(...)

        # =================================================
        # PRIO 1: DOMESTIC HOT WATER (BOILER)
        # =================================================
        # Stel: Water moet om 17:00 op temperatuur zijn
        dhw_deadline = now.replace(hour=17, minute=0, second=0, microsecond=0)

        # Alleen plannen als we voor de deadline zitten
        if now < dhw_deadline:
            dhw_action, dhw_reason = self._plan_dhw(now, dhw_deadline)

            # Als Boiler aan moet, is dat leidend (meestal kan maar 1 ding tegelijk)
            if dhw_action == "START":
                self.context.action = "DHW_RUN"
                self.context.reason = dhw_reason
                logger.info(f"[Planner] Actie: {self.context.action} | {self.context.reason}")
                return # Stop, we gaan boileren

        # =================================================
        # PRIO 2: HVAC (VERWARMING)
        # =================================================
        # Stel: Woonkamer moet om 17:00 warm zijn
        hvac_deadline = now.replace(hour=17, minute=0, second=0, microsecond=0)

        if now < hvac_deadline:
            hvac_action, hvac_reason = self._plan_hvac(now, hvac_deadline)

            if hvac_action == "START":
                self.context.action = "HEAT_RUN"
                self.context.reason = hvac_reason

        logger.info(f"[Planner] Actie: {self.context.action} | {self.context.reason}")


    def _plan_dhw(self, now, deadline):
        """Specifieke logica voor de boiler inclusief Coasting Check."""
        df = self.context.forecast_df
        current_temp = self.context.dhw_temp
        target_temp = self.context.dhw_setpoint

        # Stap 1: Hoe lang duurt verwarmen?
        minutes_needed, dhw_profile = self.dhw_planner.calculate_run_profile(
            start_temp=current_temp,
            target_temp=target_temp,
            df_forecast=df[df['timestamp'] >= now],
            is_dhw=True
        )

        logger.debug(f"[Planner][DHW] Minutes needed to heat: {minutes_needed}")

        if minutes_needed == 0:
            return "WAIT", "Boiler is op temperatuur"

        # Stap 2: Laatste moment dat we MOGEN starten (Just-In-Time)
        jit_start_time = deadline - timedelta(minutes=minutes_needed)

        # Veiligheidsmarge: Als we al voorbij JIT zijn, direct starten!
        if now >= jit_start_time:
            return "START", "Just-in-Time (Deadline naderbij)"

        # Stap 3: Vraag Optimizer om advies (Zoek zonne-energie)
        # We maken een profiel voor de benodigde duur
        steps = int(minutes_needed / 15)

        opt_status, opt_reason, _, _ = self.optimizer.optimize(df, now, dhw_profile)

        if opt_status == "WAIT":
            return "WAIT", opt_reason

        # Stap 4: Optimizer zegt "START" (want zon!). Nu de COASTING CHECK.
        # Als we nu starten, hoe laat zijn we dan klaar?
        finish_time = now + timedelta(minutes=minutes_needed)
        cooldown_duration = (deadline - finish_time).total_seconds() / 60

        if cooldown_duration > 15:
            # We zijn vroeg klaar. Check of het warm blijft.
            # We staan toe dat het max 3 graden afkoelt (hysteresis)
            min_allowed_temp = target_temp - 3.0

            # Snijd forecast voor de cooldown periode
            df_cooldown = df[df['timestamp'] >= finish_time]

            is_warm_enough = self.dhw_planner.simulate_cooldown(
                current_temp=target_temp, # Aanname: we halen de target
                min_temp=min_allowed_temp,
                duration_minutes=cooldown_duration,
                df_forecast=df_cooldown
            )

            if not is_warm_enough:
                # OVERRULE: Zon is leuk, maar koud douchen niet.
                return "WAIT", f"Zon genegeerd: te veel afkoeling ({int(cooldown_duration)} min)"

        # Alles OK: Starten maar
        return "START", f"Optimalisatie: {opt_reason}"


    def _plan_hvac(self, now, deadline):
        """Specifieke logica voor verwarming."""
        df = self.context.forecast_df
        # current_temp = self.context.current_temp (van thermostaat)
        # Voor demo gebruiken we een vaste waarde of uit context
        current_temp = getattr(self.context, "room_temp", 19.0)
        target_temp = 20 # Hardcoded setpoint of uit context

        # Stap 1: Duur berekenen
        minutes_needed, hvac_profile = self.dhw_planner.calculate_run_profile(
            start_temp=current_temp,
            target_temp=target_temp,
            df_forecast=df[df['timestamp'] >= now],
            is_dhw=False
        )

        if minutes_needed == 0:
            return "WAIT", "Woonkamer op temperatuur"

        # Stap 2: Just-In-Time berekening
        jit_start_time = deadline - timedelta(minutes=minutes_needed)

        if now >= jit_start_time:
             return "START", "Verwarming Just-in-Time"

        # Stap 3: Solar Optimalisatie
        # Vloerverwarming gebruikt minder vermogen, bijv 1.5 kW
        steps = int(minutes_needed / 15)

        opt_status, opt_reason, _, _ = self.optimizer.optimize(df, now, hvac_profile)

        if opt_status == "WAIT":
            return "WAIT", opt_reason

        # Stap 4: Coasting Check voor Huis
        finish_time = now + timedelta(minutes=minutes_needed)
        cooldown_duration = (deadline - finish_time).total_seconds() / 60

        if cooldown_duration > 15:
            # Huis mag max 0.5 graad afkoelen
            min_allowed_temp = target_temp - 0.5
            df_cooldown = df[df['timestamp'] >= finish_time]

            is_warm_enough = self.hvac_planner.simulate_cooldown(
                current_temp=target_temp,
                min_temp=min_allowed_temp,
                duration_minutes=cooldown_duration,
                df_forecast=df_cooldown
            )

            if not is_warm_enough:
                return "WAIT", "Verwarming uitgesteld: huis koelt te snel af"

        return "START", f"Verwarming Solar: {opt_reason}"