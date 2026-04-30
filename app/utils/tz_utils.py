from datetime import datetime, timezone, timedelta

CAMBODIA_TZ = timezone(timedelta(hours=7))


def to_cambodia_tz(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(CAMBODIA_TZ)
