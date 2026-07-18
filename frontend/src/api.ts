import {
  getLocalMeta,
  predictLocally,
  type PredictInput,
} from "./localModel";

const raw = import.meta.env.VITE_API_URL as string | undefined;
const API_URL = raw && raw.trim() ? raw.replace(/\/$/, "") : "";

export type PredictPayload = PredictInput;

export type Factor = {
  name: string;
  detail: string;
};

export type PredictResult = {
  delay_probability: number;
  delayed: boolean;
  verdict: string;
  top_factors: Factor[];
};

export type MetaResponse = {
  model_version: string;
  algorithm: string;
  features: string[];
  metrics: {
    accuracy?: number;
    roc_auc?: number;
    n_train?: number;
    n_test?: number;
    delay_rate?: number;
    vercel_lr_roc_auc?: number;
  };
  airlines: string[];
  airports: string[];
};

export async function fetchMeta(): Promise<MetaResponse> {
  if (!API_URL) {
    return getLocalMeta() as MetaResponse;
  }
  const res = await fetch(`${API_URL}/meta`);
  if (!res.ok) {
    throw new Error(`Failed to load model metadata (${res.status})`);
  }
  return res.json() as Promise<MetaResponse>;
}

export async function predictDelay(
  payload: PredictPayload,
): Promise<PredictResult> {
  if (!API_URL) {
    if (payload.origin.toUpperCase() === payload.dest.toUpperCase()) {
      throw new Error("Origin and destination must differ");
    }
    return predictLocally(payload);
  }
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = `Prediction failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<PredictResult>;
}

export { API_URL };
