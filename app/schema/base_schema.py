from typing import Optional, TypeVar, Generic, List
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

from app.utils.tz_utils import to_cambodia_tz

T = TypeVar('T')

class RESPONSE_SCHEMA(BaseModel, Generic[T]):
    status      : int
    message     : str
    data        : Optional[T] = None


class BASE_SCHEMA_OUT(BaseModel):
    """Base for all OUT schemas — converts every datetime field to Cambodia time (UTC+7)."""

    @model_validator(mode="after")
    def _convert_datetimes_to_cambodia(self):
        for field_name, field_info in self.model_fields.items():
            value = getattr(self, field_name, None)
            if isinstance(value, datetime):
                object.__setattr__(self, field_name, to_cambodia_tz(value))
        return self

    class Config:
        from_attributes = True
