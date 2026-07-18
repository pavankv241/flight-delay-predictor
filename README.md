# SkySignal — Flight Delay Predictor

Portfolio full-stack project that estimates **delay probability** for domestic flights in **India and the US**.

You enter carrier, origin/destination (IATA), schedule, and distance. The app returns:

- **P(delay)** — predicted delay probability  
- A short risk verdict  
- Top **contributing features** (hub congestion, seasonality / monsoon, slot time, distance)

**Live demo:** https://flight-delay-predictor-zeta.vercel.app  
**Repository:** https://github.com/pavankv241/flight-delay-predictor

---

## Why this project

Built as a **software engineering** portfolio piece (not a research notebook):

- End-to-end product: UI → model artifact → optional API → CI → free hosting  
- Clean monorepo layout and documented local + deploy flow  
- India + US route coverage with realistic IATA labels in the UI  

---

## Features

- India domestic carriers: Air India (AI), IndiGo (6E), Vistara (UK), SpiceJet (SG), Akasa (QP), Air India Express (IX)
- India airports: DEL, BOM, BLR, HYD, MAA, CCU, PNQ, AMD, GOI, COK
- US domestic carriers and major hubs (AA, DL, UA, WN, and others)
- In-browser inference on Vercel (no cold-start API required for the live demo)
- Optional FastAPI service for local demos / interviews (`/health`, `/meta`, `/predict`, OpenAPI `/docs`)
- GitHub Actions CI: regenerate/train smoke path + frontend production build

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React + TypeScript (Vite) on Vercel                    │
│  • Form collects IATA + schedule features               │
│  • Scores delay risk with exported model weights (JSON) │
└─────────────────────────────────────────────────────────┘
                          │
                          │  same training pipeline
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Python training (ml/)                                  │
│  • Synthetic India + US flight sample CSV               │
│  • GradientBoostingClassifier → backend joblib          │
│  • LogisticRegression export → frontend weights JSON    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI (backend/) — local / optional Render           │
│  • Loads joblib pipeline                                │
│  • POST /predict  GET /meta  GET /health                │
└─────────────────────────────────────────────────────────┘
```

| Layer | Technology |
|-------|------------|
| UI | React 19, TypeScript, Vite |
| Hosted inference | Exported logistic-regression weights scored in the browser |
| Training | pandas, scikit-learn (`GradientBoostingClassifier` + LR export) |
| API (optional) | FastAPI, Pydantic, Uvicorn, joblib |
| CI | GitHub Actions |
| Hosting | Vercel (frontend + live demo); Render optional for Python API |

---

## Dataset & model

Training data is a **bundled synthetic** sample in [`data/sample_flights.csv`](data/sample_flights.csv) (~10k rows, ~50% India / ~50% US corridors).

**Features**

| Feature | Description |
|---------|-------------|
| `airline` | IATA carrier code |
| `origin` / `dest` | IATA airport codes |
| `month` | 1–12 |
| `day_of_week` | 0=Monday … 6=Sunday |
| `hour` | Local scheduled departure hour |
| `distance` | Approximate great-circle miles |
| `delayed` | Binary label (0/1) |

**India-specific signals** in generation include monsoon months and winter fog effects on busy hubs (DEL, BOM, etc.).

**Artifacts**

- [`backend/models/delay_model.joblib`](backend/models/delay_model.joblib) — GBM pipeline for FastAPI  
- [`backend/models/model_meta.json`](backend/models/model_meta.json) — version, metrics, airline/airport lists  
- [`frontend/src/model_weights.json`](frontend/src/model_weights.json) — portable weights for the live UI  

**Latest training metrics** (see `model_meta.json` after retrain):

- ROC-AUC ≈ **0.65** (GBM holdout)  
- Accuracy ≈ **0.66**  

These numbers are for a synthetic demo corpus — good enough to show the product loop, not airline-ops production accuracy.

---

## Project structure

```
Delay Prediction/
├── README.md
├── LICENSE
├── render.yaml                 # Optional Render blueprint
├── .github/workflows/ci.yml
├── data/
│   └── sample_flights.csv
├── ml/
│   ├── generate_data.py        # India + US synthetic generator
│   ├── train.py                # Train GBM + export UI weights
│   └── requirements.txt
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI routes
│   │   └── schemas.py
│   ├── models/
│   │   ├── delay_model.joblib
│   │   └── model_meta.json
│   ├── Dockerfile
│   ├── requirements.txt
│   └── runtime.txt
└── frontend/
    ├── src/
    │   ├── App.tsx             # UI
    │   ├── api.ts              # Local weights or remote API
    │   ├── localModel.ts       # Browser scoring
    │   └── model_weights.json
    ├── package.json
    └── vercel.json
```

---

## Run locally

### Prerequisites

- Python 3.12+
- Node.js 20+

### 1. Train (or refresh) the model

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt -r backend/requirements.txt
python ml/generate_data.py
python ml/train.py
```

### 2. Optional: FastAPI

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

- Health: http://127.0.0.1:8000/health  
- Docs: http://127.0.0.1:8000/docs  

Example:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "airline": "6E",
    "origin": "BOM",
    "dest": "DEL",
    "month": 7,
    "day_of_week": 4,
    "hour": 18,
    "distance": 710
  }'
```

### 3. Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open http://127.0.0.1:5173

| `VITE_API_URL` | Behavior |
|----------------|----------|
| empty (default) | Score with in-browser weights (same as production) |
| `http://127.0.0.1:8000` | Call local FastAPI |

---

## Deploy

### Vercel (live demo)

1. Import the GitHub repo in Vercel  
2. Set **Root Directory** to `frontend`  
3. Leave `VITE_API_URL` **unset**  
4. Deploy  

CLI (if linked):

```bash
cd frontend
npx vercel --prod
```

### Optional: FastAPI on Render

1. Open https://render.com/deploy?repo=https://github.com/pavankv241/flight-delay-predictor  
2. Sign in with GitHub and apply [`render.yaml`](render.yaml)  
3. Set `CORS_ORIGINS` to your Vercel URL  
4. (Optional) Set frontend `VITE_API_URL` to the Render service URL and redeploy the UI  

---

## API reference (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + whether the model loaded |
| `GET` | `/meta` | Model version, metrics, airline/airport lists |
| `POST` | `/predict` | Delay probability + factors |

`POST /predict` body:

```json
{
  "airline": "6E",
  "origin": "BOM",
  "dest": "DEL",
  "month": 7,
  "day_of_week": 4,
  "hour": 18,
  "distance": 710
}
```

---

## Tech interview talking points

- Separated **training** from **serving** (joblib for API, JSON weights for static hosting)  
- Free hosting constraint: browser inference avoids Python cold starts on the live demo  
- Feature explanations are product-facing, not SHAP dumps — still tied to domain signals  
- CI validates train → API smoke → frontend build  

---

## License

MIT — see [LICENSE](LICENSE).
