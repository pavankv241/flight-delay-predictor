from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    airline: str = Field(..., examples=["DL"])
    origin: str = Field(..., examples=["ATL"])
    dest: str = Field(..., examples=["JFK"])
    month: int = Field(..., ge=1, le=12, examples=[7])
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="0=Monday … 6=Sunday",
        examples=[4],
    )
    hour: int = Field(..., ge=0, le=23, examples=[17])
    distance: int = Field(..., ge=50, le=5000, examples=[760])


class Factor(BaseModel):
    name: str
    detail: str


class PredictResponse(BaseModel):
    delay_probability: float
    delayed: bool
    verdict: str
    top_factors: list[Factor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetaResponse(BaseModel):
    model_version: str
    algorithm: str
    features: list[str]
    metrics: dict[str, Any]
    airlines: list[str]
    airports: list[str]
