from __future__ import annotations

from typing import Protocol

from home_optimizer.features.modeling import StoredModelVersion


class ActiveRoomModelReaderPort(Protocol):
    def get_room_model_version(self, model_id: str) -> StoredModelVersion | None: ...

    def get_active_room_model_version(self) -> StoredModelVersion | None: ...
