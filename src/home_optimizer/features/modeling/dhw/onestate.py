from __future__ import annotations

from pydantic import Field

from home_optimizer.domain.models import DomainModel


class DhwOneStateConfig(DomainModel):
    state_name: str = "tank_temperature"
    notes: str = Field(
        default="Placeholder for a future 1-state DHW tank model implementation."
    )


class DhwOneStateModel(DomainModel):
    model_kind: str = "dhw_onestate"
    notes: str = Field(
        default="Placeholder trained-model shape for a future 1-state DHW implementation."
    )
