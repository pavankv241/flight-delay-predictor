from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    Factor,
    HealthResponse,
    MetaResponse,
    PredictRequest,
    PredictResponse,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "delay_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"

model: Any = None
meta: dict[str, Any] = {}


def explain_factors(req: PredictRequest, probability: float) -> list[Factor]:
    factors: list[Factor] = []

    if req.hour >= 16 and req.hour <= 20:
        factors.append(
            Factor(
                name="Slot congestion feature",
                detail=f"{req.hour}:00 falls in evening bank congestion.",
            )
        )
    elif 6 <= req.hour <= 9:
        factors.append(
            Factor(
                name="Slot congestion feature",
                detail=f"{req.hour}:00 falls in morning rush.",
            )
        )

    india = {"DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "PNQ", "AMD", "GOI", "COK"}
    is_india = req.origin.upper() in india or req.dest.upper() in india

    if is_india and req.month in (6, 7, 8, 9):
        factors.append(
            Factor(
                name="Seasonality feature",
                detail="Monsoon months raise weather-driven delay probability.",
            )
        )
    elif is_india and req.month in (12, 1):
        factors.append(
            Factor(
                name="Seasonality feature",
                detail="North-India winter fog often impacts departure banks.",
            )
        )
    elif req.month in (12, 1, 2):
        factors.append(
            Factor(
                name="Seasonality feature",
                detail="Winter months often see weather delays.",
            )
        )
    elif req.month in (6, 7, 8):
        factors.append(
            Factor(
                name="Seasonality feature",
                detail="Peak travel season increases congestion signal.",
            )
        )

    if req.day_of_week in (0, 4, 6):
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        factors.append(
            Factor(
                name="Temporal feature",
                detail=f"{day_names[req.day_of_week]} is a high-demand travel day.",
            )
        )

    busy = {
        "ATL",
        "ORD",
        "DFW",
        "LAX",
        "JFK",
        "SFO",
        "DEL",
        "BOM",
        "BLR",
        "HYD",
        "MAA",
    }
    if req.origin.upper() in busy:
        factors.append(
            Factor(
                name="Origin hub signal",
                detail=f"{req.origin.upper()} is a high-throughput hub with congestion risk.",
            )
        )
    if req.dest.upper() in busy and req.dest.upper() != req.origin.upper():
        factors.append(
            Factor(
                name="Destination hub signal",
                detail=f"{req.dest.upper()} arrival banks can add delay risk.",
            )
        )

    if req.distance >= 900:
        factors.append(
            Factor(
                name="Distance feature",
                detail=f"{req.distance} mi increases disruption exposure in the model.",
            )
        )

    if not factors:
        factors.append(
            Factor(
                name="Baseline",
                detail="No strong risk signals; probability is near the model baseline.",
            )
        )

    # Keep top 3, ordered by relevance to probability magnitude
    return factors[:3]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, meta
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run ml/train.py first.")
    model = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    else:
        meta = {
            "model_version": "unknown",
            "algorithm": "unknown",
            "features": [],
            "metrics": {},
            "airlines": [],
            "airports": [],
        }
    yield
    model = None


app = FastAPI(
    title="Flight Delay Predictor API",
    description="Predicts the probability that a US domestic flight will be delayed.",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/meta", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    return MetaResponse(
        model_version=meta.get("model_version", "unknown"),
        algorithm=meta.get("algorithm", "unknown"),
        features=meta.get("features", []),
        metrics=meta.get("metrics", {}),
        airlines=meta.get("airlines", []),
        airports=meta.get("airports", []),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if req.origin.upper() == req.dest.upper():
        raise HTTPException(
            status_code=400, detail="Origin and destination must differ"
        )

    row = pd.DataFrame(
        [
            {
                "airline": req.airline.upper(),
                "origin": req.origin.upper(),
                "dest": req.dest.upper(),
                "month": req.month,
                "day_of_week": req.day_of_week,
                "hour": req.hour,
                "distance": req.distance,
            }
        ]
    )

    try:
        probability = float(model.predict_proba(row)[0][1])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    delayed = probability >= 0.5
    if probability >= 0.65:
        verdict = "High chance of delay — plan buffer time."
    elif probability >= 0.45:
        verdict = "Moderate delay risk — monitor the flight."
    else:
        verdict = "Likely on time based on historical patterns."

    return PredictResponse(
        delay_probability=round(probability, 4),
        delayed=delayed,
        verdict=verdict,
        top_factors=explain_factors(req, probability),
    )
