from __future__ import annotations

from pydantic import Field

from home_optimizer.domain.models import DomainModel


class Dhw1R1CConfig(DomainModel):
    state_name: str = "tank_temperature"
    notes: str = Field(
        default="Placeholder for a future 1-state DHW tank model implementation."
    )


class Dhw1R1CModel(DomainModel):
    model_kind: str = "dhw_1r1c"
    notes: str = Field(
        default="Placeholder trained-model shape for a future 1-state DHW implementation."
    )
