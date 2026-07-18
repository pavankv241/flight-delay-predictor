import { useEffect, useState, type FormEvent } from "react";
import {
  fetchMeta,
  predictDelay,
  type MetaResponse,
  type PredictResult,
} from "./api";
import "./App.css";

const AIRLINE_LABELS: Record<string, string> = {
  AA: "American (AA)",
  DL: "Delta (DL)",
  UA: "United (UA)",
  WN: "Southwest (WN)",
  B6: "JetBlue (B6)",
  AS: "Alaska (AS)",
  NK: "Spirit (NK)",
  F9: "Frontier (F9)",
  AI: "Air India (AI)",
  "6E": "IndiGo (6E)",
  UK: "Vistara (UK)",
  SG: "SpiceJet (SG)",
  QP: "Akasa Air (QP)",
  IX: "Air India Express (IX)",
};

const AIRPORT_LABELS: Record<string, string> = {
  ATL: "Atlanta (ATL)",
  ORD: "Chicago (ORD)",
  DFW: "Dallas (DFW)",
  DEN: "Denver (DEN)",
  LAX: "Los Angeles (LAX)",
  JFK: "New York (JFK)",
  SFO: "San Francisco (SFO)",
  SEA: "Seattle (SEA)",
  MIA: "Miami (MIA)",
  BOS: "Boston (BOS)",
  PHX: "Phoenix (PHX)",
  LAS: "Las Vegas (LAS)",
  DEL: "Delhi (DEL)",
  BOM: "Mumbai (BOM)",
  BLR: "Bengaluru (BLR)",
  HYD: "Hyderabad (HYD)",
  MAA: "Chennai (MAA)",
  CCU: "Kolkata (CCU)",
  PNQ: "Pune (PNQ)",
  AMD: "Ahmedabad (AMD)",
  GOI: "Goa (GOI)",
  COK: "Kochi (COK)",
};

const FALLBACK_AIRLINES = Object.keys(AIRLINE_LABELS);
const FALLBACK_AIRPORTS = Object.keys(AIRPORT_LABELS);

const MONTHS = [
  { value: 1, label: "January" },
  { value: 2, label: "February" },
  { value: 3, label: "March" },
  { value: 4, label: "April" },
  { value: 5, label: "May" },
  { value: 6, label: "June" },
  { value: 7, label: "July" },
  { value: 8, label: "August" },
  { value: 9, label: "September" },
  { value: 10, label: "October" },
  { value: 11, label: "November" },
  { value: 12, label: "December" },
];

const DAYS = [
  { value: 0, label: "Monday" },
  { value: 1, label: "Tuesday" },
  { value: 2, label: "Wednesday" },
  { value: 3, label: "Thursday" },
  { value: 4, label: "Friday" },
  { value: 5, label: "Saturday" },
  { value: 6, label: "Sunday" },
];

const HOURS = Array.from({ length: 19 }, (_, i) => i + 5);

type FormState = {
  airline: string;
  origin: string;
  dest: string;
  month: number;
  day_of_week: number;
  hour: number;
  distance: number;
};

const initialForm: FormState = {
  airline: "6E",
  origin: "BOM",
  dest: "DEL",
  month: 7,
  day_of_week: 4,
  hour: 18,
  distance: 710,
};

function labelAirline(code: string) {
  return AIRLINE_LABELS[code] ?? code;
}

function labelAirport(code: string) {
  return AIRPORT_LABELS[code] ?? code;
}

function App() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMeta()
      .then(setMeta)
      .catch(() => {
        /* local weights still work */
      });
  }, []);

  const airlines = meta?.airlines?.length ? meta.airlines : FALLBACK_AIRLINES;
  const airports = meta?.airports?.length ? meta.airports : FALLBACK_AIRPORTS;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictDelay(form);
      setResult(prediction);
    } catch (err: unknown) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const pct = result ? Math.round(result.delay_probability * 100) : 0;
  const riskClass =
    pct >= 65 ? "high" : pct >= 45 ? "medium" : result ? "low" : "";

  return (
    <div className="shell">
      <div className="shell-bg" aria-hidden="true" />

      <header className="topbar">
        <div className="brand-row">
          <p className="brand">SkySignal</p>
          <span className="badge">Live inference</span>
        </div>
        <p className="tagline">
          Real-time delay probability from route features — India &amp; US
          domestic corridors.
        </p>
        <ul className="tech-strip" aria-label="Model capabilities">
          <li>Feature vector scoring</li>
          <li>Hub congestion signals</li>
          <li>Monsoon seasonality</li>
          <li>IATA route coverage</li>
        </ul>
      </header>

      <main className="workspace">
        <form className="flight-form" onSubmit={onSubmit}>
          <div className="panel">
            <p className="panel-title">Flight request</p>

            <label className="field">
              <span>Carrier (IATA)</span>
              <select
                value={form.airline}
                onChange={(e) =>
                  setForm((f) => ({ ...f, airline: e.target.value }))
                }
              >
                {airlines.map((a) => (
                  <option key={a} value={a}>
                    {labelAirline(a)}
                  </option>
                ))}
              </select>
            </label>

            <div className="route-row">
              <label className="field">
                <span>Origin airport</span>
                <select
                  value={form.origin}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, origin: e.target.value }))
                  }
                >
                  {airports.map((a) => (
                    <option key={a} value={a}>
                      {labelAirport(a)}
                    </option>
                  ))}
                </select>
              </label>
              <span className="route-arrow" aria-hidden="true">
                →
              </span>
              <label className="field">
                <span>Destination airport</span>
                <select
                  value={form.dest}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, dest: e.target.value }))
                  }
                >
                  {airports.map((a) => (
                    <option key={a} value={a}>
                      {labelAirport(a)}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="schedule-row">
              <label className="field">
                <span>Month</span>
                <select
                  value={form.month}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, month: Number(e.target.value) }))
                  }
                >
                  {MONTHS.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Day of week</span>
                <select
                  value={form.day_of_week}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      day_of_week: Number(e.target.value),
                    }))
                  }
                >
                  {DAYS.map((d) => (
                    <option key={d.value} value={d.value}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Local STD hour</span>
                <select
                  value={form.hour}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, hour: Number(e.target.value) }))
                  }
                >
                  {HOURS.map((h) => (
                    <option key={h} value={h}>
                      {String(h).padStart(2, "0")}:00
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <label className="field">
              <span>Great-circle distance (mi)</span>
              <input
                type="number"
                min={50}
                max={5000}
                value={form.distance}
                onChange={(e) =>
                  setForm((f) => ({ ...f, distance: Number(e.target.value) }))
                }
              />
            </label>

            <div className="actions">
              <button type="submit" disabled={loading}>
                {loading ? "Running inference…" : "Run delay inference"}
              </button>
              {error ? <p className="error">{error}</p> : null}
            </div>
          </div>
        </form>

        <section className={`outcome ${riskClass}`} aria-live="polite">
          {result ? (
            <>
              <div className="outcome-head">
                <div>
                  <p className="pct-label">Predicted P(delay)</p>
                  <p className={`pct ${riskClass}`}>{pct}%</p>
                </div>
                <p className={`risk-chip ${riskClass}`}>
                  {riskClass === "high"
                    ? "High risk"
                    : riskClass === "medium"
                      ? "Moderate risk"
                      : "Low risk"}
                </p>
              </div>
              <div className={`meter ${riskClass}`} role="presentation">
                <i style={{ width: `${pct}%` }} />
              </div>
              <p className="verdict">{result.verdict}</p>
              <p className="factors-title">Top contributing features</p>
              <ul className="factors">
                {result.top_factors.map((f) => (
                  <li key={f.name}>
                    <strong>{f.name}</strong>
                    <span>{f.detail}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="outcome-empty">
              Submit a flight request to score delay probability and inspect
              feature contributions.
            </p>
          )}
        </section>
      </main>

      {meta?.metrics ? (
        <footer className="foot">
          <span>
            Model pipeline · {meta.algorithm} · v{meta.model_version}
          </span>
          {meta.metrics.roc_auc != null ? (
            <span>ROC-AUC {meta.metrics.roc_auc}</span>
          ) : null}
          {meta.metrics.accuracy != null ? (
            <span>Holdout accuracy {meta.metrics.accuracy}</span>
          ) : null}
        </footer>
      ) : null}
    </div>
  );
}

export default App;
