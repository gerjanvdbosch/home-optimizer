from __future__ import annotations

from pydantic import Field

from home_optimizer.domain.models import DomainModel


class Room2R2CConfig(DomainModel):
    state_names: tuple[str, str] = ("air", "mass")
    notes: str = Field(
        default="Placeholder for a future 2-state room / mass model implementation."
    )


class Room2R2CModel(DomainModel):
    model_kind: str = "room_2r2c"
    notes: str = Field(
        default="Placeholder trained-model shape for a future 2-state room implementation."
    )
