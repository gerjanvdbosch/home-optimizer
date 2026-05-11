from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BeforeValidator

def _parse_flexible_datetime(value: Any, *, end_of_day: bool = False) -> Any:
    if not isinstance(value, str):
        return value
    try:
        dt = datetime.strptime(value.strip(), "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=0)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return value


FlexibleDatetime = Annotated[datetime, BeforeValidator(_parse_flexible_datetime)]
FlexibleEndDatetime = Annotated[datetime, BeforeValidator(lambda v: _parse_flexible_datetime(v, end_of_day=True))]
