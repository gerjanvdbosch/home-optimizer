import logging

from planner import Plan
from enum import Enum, auto
from statemachine import StateMachine

logger = logging.getLogger(__name__)


class DhwState(Enum):
    IDLE = auto()
    WAITING = auto()
    RUNNING = auto()
    DONE = auto()


class DhwMachine(StateMachine):
    def __init__(self, context):
        super().__init__("DHW", context)
        self.state = DhwState.IDLE
        # self.run_start_time = None

    def on_enter(self, state):
        if state == "RUNNING":
            logger.info("[DHW] AAN: Start verwarmen.")
            # self.ctx.ha.set_switch(self.ctx.cfg.entity_dhw, True)
        elif state in ["IDLE", "DONE"]:
            logger.info("[DHW] UIT: Stoppen.")
            # self.ctx.ha.set_switch(self.ctx.cfg.entity_dhw, False)

    def process(self, plan: Plan):
        pass
#         """
#         Verwerkt het masterplan van de Planner.
#         """
#         now = self.context.now
#
#         # 1. Midnight Reset: Als het 00:00 is en we waren DONE -> IDLE
#         if now.hour == 0 and self.state == DhwState.DONE:
#             self.transition(DhwState.IDLE, "Nieuwe dag")
#             return
#
#         # --- STATE LOGICA ---
#
#         if self.state == DhwState.IDLE:
#             # Als de planner zegt: "NU Boiler aan" (vanwege zon of dark day logic)
#             if plan.action == "BOILER":
#                 self.transition(DhwState.RUNNING, plan.reason)
#
#         elif self.state == DhwState.RUNNING:
#             # We draaien! Check of we moeten stoppen.
#
#             elapsed_min = 0
#             if self.run_start_time:
#                 elapsed_min = (now - self.run_start_time).total_seconds() / 60
#
#             # Stopconditie A: De planner zegt ineens "OFF" of "HEAT_PUMP"
#             # (Bijv. omdat het blok in het schema voorbij is)
#             if plan.action != "BOILER":
#                 # Anti-pendel: Draai minimaal 15 minuten voordat we luisteren naar "UIT"
#                 if elapsed_min > 15:
#                     self.transition(DhwState.DONE, "Volgens planning klaar")
#
#             # Stopconditie B: Harde tijdslimiet (Veiligheid)
#             if elapsed_min >= self.max_duration_minutes:
#                 self.transition(DhwState.DONE, "Max tijd bereikt")
#
#             # Stopconditie C: Temperatuur bereikt? (Als je die sensor hebt)
#             if self.context.tank_temp >= 65:
#                self.transition(DhwState.DONE, "Temp bereikt")
#
#         elif self.state == DhwState.DONE:
#             # We zijn klaar voor vandaag.
#             # Tenzij... Paniek/Boost nodig is (handmatige override of 's avonds koud water)
#             # Voor nu: doe niks tot morgen.
#             pass