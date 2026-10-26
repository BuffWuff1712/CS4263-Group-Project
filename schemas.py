from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class CargoLoad(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shipment_id: str = Field(min_length=1)
    quantity: int = Field(gt=0)


class Voyage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vessel_id: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    depart_hour: int = Field(ge=0)
    cargo: list[CargoLoad]


class Schedule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voyages: list[Voyage]


class Violation(BaseModel):
    constraint: str
    message: str

    vessel_id: Optional[str] = None
    shipment_id: Optional[str] = None
    voyage_index: Optional[int] = None


class ValidationReport(BaseModel):
    valid: bool
    violations: list[Violation]
    delivered: dict[str, int] = Field(
        default_factory=dict
    )
