from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

class Fields(BaseModel):
    WHEAT: int = Field(ge=0, le=10)
    MELON: int = Field(ge=0, le=10)

    @model_validator(mode="after")
    def total_at_most_10(self):
        if self.WHEAT + self.MELON > 10:
            raise ValueError("more than 10 fields in total")
        return self

class Decision(BaseModel):
    insights: str = ""
    plans: str = ""
    fields: Fields

    @field_validator("insights", "plans")
    @classmethod
    def trim(cls, v):
        return v[:600]

import re
def parse_decision(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no_json"
    try:
        return Decision.model_validate_json(m.group(0)).model_dump(), None
    except ValidationError as e:
        return None, e.errors()[0]["type"]
