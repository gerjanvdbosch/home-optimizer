from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.models import DomainModel


class ImportChunkWindow(DomainModel):
    start_time: datetime
    end_time: datetime
