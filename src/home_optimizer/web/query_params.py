from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BeforeValidator

def _parse_flexible_datetime(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return value


FlexibleDatetime = Annotated[datetime, BeforeValidator(_parse_flexible_datetime)]
