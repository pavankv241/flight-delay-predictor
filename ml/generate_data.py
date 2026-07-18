"""Generate a realistic sample flight delay dataset (US + India domestic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_flights.csv"

US_AIRLINES = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
IN_AIRLINES = ["AI", "6E", "UK", "SG", "QP", "IX"]

US_AIRPORTS = [
    "ATL",
    "ORD",
    "DFW",
    "DEN",
    "LAX",
    "JFK",
    "SFO",
    "SEA",
    "MIA",
    "BOS",
    "PHX",
    "LAS",
]
IN_AIRPORTS = [
    "DEL",
    "BOM",
    "BLR",
    "HYD",
    "MAA",
    "CCU",
    "PNQ",
    "AMD",
    "GOI",
    "COK",
]

AIRLINES = US_AIRLINES + IN_AIRLINES
AIRPORTS = US_AIRPORTS + IN_AIRPORTS

HUB_DELAY_BOOST = {
    "ATL": 0.08,
    "ORD": 0.10,
    "DFW": 0.07,
    "DEN": 0.06,
    "LAX": 0.07,
    "JFK": 0.09,
    "SFO": 0.08,
    "SEA": 0.04,
    "MIA": 0.05,
    "BOS": 0.06,
    "PHX": 0.03,
    "LAS": 0.04,
    "DEL": 0.10,
    "BOM": 0.11,
    "BLR": 0.07,
    "HYD": 0.05,
    "MAA": 0.06,
    "CCU": 0.06,
    "PNQ": 0.04,
    "AMD": 0.04,
    "GOI": 0.03,
    "COK": 0.05,
}

AIRLINE_DELAY_BIAS = {
    "AA": 0.02,
    "DL": -0.01,
    "UA": 0.03,
    "WN": 0.01,
    "B6": 0.04,
    "AS": -0.02,
    "NK": 0.06,
    "F9": 0.05,
    "AI": 0.03,
    "6E": 0.02,
    "UK": 0.01,
    "SG": 0.05,
    "QP": 0.04,
    "IX": 0.03,
}

DISTANCES: dict[tuple[str, str], int] = {
    ("ATL", "JFK"): 760,
    ("ATL", "LAX"): 1946,
    ("ATL", "ORD"): 606,
    ("ATL", "MIA"): 595,
    ("ORD", "LAX"): 1744,
    ("ORD", "JFK"): 740,
    ("ORD", "DEN"): 888,
    ("DFW", "LAX"): 1235,
    ("DFW", "JFK"): 1391,
    ("DFW", "DEN"): 641,
    ("DEN", "LAX"): 862,
    ("DEN", "SEA"): 1024,
    ("LAX", "SFO"): 337,
    ("LAX", "SEA"): 954,
    ("LAX", "JFK"): 2475,
    ("JFK", "BOS"): 187,
    ("JFK", "MIA"): 1090,
    ("SFO", "SEA"): 679,
    ("SFO", "LAS"): 414,
    ("SEA", "PHX"): 1107,
    ("MIA", "BOS"): 1258,
    ("PHX", "LAS"): 256,
    ("BOS", "ORD"): 867,
    # India domestic (approx statute miles)
    ("DEL", "BOM"): 710,
    ("DEL", "BLR"): 1080,
    ("DEL", "HYD"): 780,
    ("DEL", "MAA"): 1090,
    ("DEL", "CCU"): 810,
    ("DEL", "PNQ"): 730,
    ("DEL", "AMD"): 470,
    ("DEL", "GOI"): 940,
    ("DEL", "COK"): 1280,
    ("BOM", "BLR"): 520,
    ("BOM", "HYD"): 390,
    ("BOM", "MAA"): 640,
    ("BOM", "CCU"): 1030,
    ("BOM", "PNQ"): 90,
    ("BOM", "AMD"): 275,
    ("BOM", "GOI"): 265,
    ("BOM", "COK"): 670,
    ("BLR", "HYD"): 310,
    ("BLR", "MAA"): 180,
    ("BLR", "CCU"): 970,
    ("BLR", "PNQ"): 460,
    ("BLR", "GOI"): 310,
    ("BLR", "COK"): 230,
    ("HYD", "MAA"): 320,
    ("HYD", "CCU"): 740,
    ("MAA", "CCU"): 850,
    ("MAA", "COK"): 340,
    ("PNQ", "GOI"): 230,
    ("AMD", "BOM"): 275,
}


def pair_distance(origin: str, dest: str, rng: np.random.Generator) -> int:
    key = (origin, dest)
    rev = (dest, origin)
    if key in DISTANCES:
        return DISTANCES[key]
    if rev in DISTANCES:
        return DISTANCES[rev]
    return int(rng.integers(150, 1400))


def delay_probability(
    airline: str,
    origin: str,
    dest: str,
    month: int,
    day_of_week: int,
    hour: int,
    distance: int,
    region: str,
) -> float:
    p = 0.08
    p += AIRLINE_DELAY_BIAS.get(airline, 0.0) * 2.5
    p += HUB_DELAY_BOOST.get(origin, 0.0) * 1.6
    p += HUB_DELAY_BOOST.get(dest, 0.0) * 0.9

    if region == "IN":
        # Monsoon + fog season
        if month in (6, 7, 8, 9):
            p += 0.16
        elif month in (12, 1):
            p += 0.12
        elif month in (4, 5):
            p += 0.05
    else:
        if month in (12, 1, 2):
            p += 0.18
        elif month in (6, 7, 8):
            p += 0.12
        elif month in (3, 4, 10, 11):
            p += 0.04

    if day_of_week in (0, 4, 6):
        p += 0.12
    elif day_of_week in (1, 2):
        p -= 0.04

    if 6 <= hour <= 9 or 16 <= hour <= 20:
        p += 0.16
    elif hour < 6 or hour >= 22:
        p += 0.08
    elif 10 <= hour <= 14:
        p -= 0.05

    p += min(distance / 2800.0, 0.14)
    return float(np.clip(p, 0.03, 0.92))


def generate(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for _ in range(n_rows):
        region = "IN" if rng.random() < 0.5 else "US"
        if region == "IN":
            airlines, airports = IN_AIRLINES, IN_AIRPORTS
        else:
            airlines, airports = US_AIRLINES, US_AIRPORTS

        airline = str(rng.choice(airlines))
        origin, dest = rng.choice(airports, size=2, replace=False)
        month = int(rng.integers(1, 13))
        day_of_week = int(rng.integers(0, 7))
        hour = int(rng.integers(5, 23))
        distance = pair_distance(origin, dest, rng)
        p = delay_probability(
            airline, origin, dest, month, day_of_week, hour, distance, region
        )
        delayed = int(rng.random() < p)
        rows.append(
            {
                "airline": airline,
                "origin": origin,
                "dest": dest,
                "month": month,
                "day_of_week": day_of_week,
                "hour": hour,
                "distance": distance,
                "delayed": delayed,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    rate = df["delayed"].mean()
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"Delay rate: {rate:.1%}")
    print(f"India rows: {(df['origin'].isin(IN_AIRPORTS)).mean():.1%}")


if __name__ == "__main__":
    main()
