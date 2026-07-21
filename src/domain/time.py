from datetime import UTC, datetime


def parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    # return dt.astimezone(datetime.now().astimezone().tzinfo)
    return dt.astimezone(UTC)
