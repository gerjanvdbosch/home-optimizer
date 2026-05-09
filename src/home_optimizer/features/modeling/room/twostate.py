from __future__ import annotations

from pydantic import Field

from home_optimizer.domain.models import DomainModel


class RoomTwoStateConfig(DomainModel):
    state_names: tuple[str, str] = ("air", "mass")
    notes: str = Field(
        default="Placeholder for a future 2-state room / mass model implementation."
    )


class RoomTwoStateModel(DomainModel):
    model_kind: str = "room_twostate"
    notes: str = Field(
        default="Placeholder trained-model shape for a future 2-state room implementation."
    )
