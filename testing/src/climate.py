from enum import Enum, auto
from statemachine import StateMachine


class ClimateState(Enum):
    NIGHT = auto()
    DAY_IDLE = auto()
    HEATING = auto()     # Actief stoken
    PAUSED_DHW = auto()  # De "Interlock" pauze

class ClimateMachine(StateMachine):
    def __init__(self, context):
        super().__init__("CLIMATE", context)
        self.state = ClimateState.NIGHT
        # self.target_temp = context.temp_night

    def on_enter(self, state: ClimateState):
        """Voert de fysieke actie uit"""
        client = self.context.client
        if not client: return

        if state == ClimateState.PAUSED_DHW:
            # Interlock: Zet WP tijdelijk uit/laag
            client.set_heat_pump_mode("STANDBY") # Of setpoint verlagen

        elif state == ClimateState.SOLAR_BOOST:
            # Bufferen: Zet vloer +1 graad (of offset)
            # Zet mode terug naar HEAT voor het geval hij op STANDBY stond
            client.set_heat_pump_mode("HEAT")
            client.set_heat_pump_offset(1.0)

        elif state == ClimateState.IDLE:
            # Normaal: Volg thermostaat (offset 0)
            client.set_heat_pump_mode("HEAT")
            client.set_heat_pump_offset(0.0)

    def process(self, plan: Plan):
        pass
#         """
#         Bepaalt de verwarming, rekening houdend met de Boiler-Interlock.
#         """
#
#         # --- 1. HARDE INTERLOCK (PRIORITEIT 1) ---
#         # Als de boiler machine zegt "Ik draai", dan moeten wij direct uit.
#         if self.dhw_machine.state == DhwState.RUNNING:
#             if self.state != ClimateState.PAUSED_DHW:
#                 self.transition(ClimateState.PAUSED_DHW, "Interlock: Boiler is aan")
#             return  # Stop hier, negeer de rest
#
#         # --- 2. HERVATTEN NA INTERLOCK ---
#         if self.state == ClimateState.PAUSED_DHW and self.dhw_machine.state != DhwState.RUNNING:
#             # De boiler is klaar. We mogen weer.
#             self.transition(ClimateState.IDLE, "Interlock vrijgegeven")
#
#         # --- 3. SOLAR LOGICA (Planner Volgen) ---
#         # Alleen als we niet gepauzeerd zijn
#
#         if self.state == ClimateState.IDLE:
#             # De planner zegt: "Er is zon over, ga maar pre-heaten/post-heaten"
#             if plan.action == "HEAT_PUMP":
#                 self.transition(ClimateState.SOLAR_BOOST, plan.reason)
#
#         elif self.state == ClimateState.SOLAR_BOOST:
#             # Stoppen met boosten als:
#             # A. De planner zegt "OFF" (geen zon meer)
#             # B. De planner zegt "BOILER" (wordt volgende tick door Interlock opgevangen, maar kan ook hier)
#             if plan.action != "HEAT_PUMP":
#                 self.transition(ClimateState.IDLE, "Solar boost voorbij")
#
#         # Noot: De normale thermostaatwerking (aan/uit op basis van kamertemp)
#         # gebeurt meestal in de warmtepomp zelf (als hij op IDLE/HEAT staat).
#         # Wij sturen hier alleen de 'Offset' (de bonus).