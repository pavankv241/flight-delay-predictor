import weights from "./model_weights.json";

export type PredictInput = {
  airline: string;
  origin: string;
  dest: string;
  month: number;
  day_of_week: number;
  hour: number;
  distance: number;
};

export type Factor = { name: string; detail: string };

function oneHot(feature: string, value: string): number[] {
  const cats = (weights.one_hot_categories as Record<string, string[]>)[feature];
  return cats.map((c) => (c === value ? 1 : 0));
}

function scaleNumeric(values: number[]): number[] {
  return values.map((v, i) => (v - weights.scaler_mean[i]) / weights.scaler_scale[i]);
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

export function scoreLocally(input: PredictInput): number {
  const catVec = [
    ...oneHot("airline", input.airline.toUpperCase()),
    ...oneHot("origin", input.origin.toUpperCase()),
    ...oneHot("dest", input.dest.toUpperCase()),
  ];
  const numVec = scaleNumeric([
    input.month,
    input.day_of_week,
    input.hour,
    input.distance,
  ]);
  const x = [...catVec, ...numVec];
  let z = weights.intercept;
  for (let i = 0; i < x.length; i += 1) {
    z += x[i] * weights.coefficients[i];
  }
  return sigmoid(z);
}

export function explainLocally(input: PredictInput): Factor[] {
  const factors: Factor[] = [];
  if (input.hour >= 16 && input.hour <= 20) {
    factors.push({
      name: "Slot congestion feature",
      detail: `${input.hour}:00 falls in evening bank congestion.`,
    });
  } else if (input.hour >= 6 && input.hour <= 9) {
    factors.push({
      name: "Slot congestion feature",
      detail: `${input.hour}:00 falls in morning rush.`,
    });
  }
  const india = new Set([
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
  ]);
  const isIndia =
    india.has(input.origin.toUpperCase()) ||
    india.has(input.dest.toUpperCase());

  if (isIndia && [6, 7, 8, 9].includes(input.month)) {
    factors.push({
      name: "Seasonality feature",
      detail: "Monsoon months raise weather-driven delay probability.",
    });
  } else if (isIndia && [12, 1].includes(input.month)) {
    factors.push({
      name: "Seasonality feature",
      detail: "North-India winter fog often impacts departure banks.",
    });
  } else if ([12, 1, 2].includes(input.month)) {
    factors.push({
      name: "Seasonality feature",
      detail: "Winter months often see weather delays.",
    });
  } else if ([6, 7, 8].includes(input.month)) {
    factors.push({
      name: "Seasonality feature",
      detail: "Peak travel season increases congestion signal.",
    });
  }
  if ([0, 4, 6].includes(input.day_of_week)) {
    const days = [
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday",
    ];
    factors.push({
      name: "Temporal feature",
      detail: `${days[input.day_of_week]} is a high-demand travel day.`,
    });
  }
  const busy = new Set([
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
  ]);
  if (busy.has(input.origin.toUpperCase())) {
    factors.push({
      name: "Origin hub signal",
      detail: `${input.origin.toUpperCase()} is a high-throughput hub with congestion risk.`,
    });
  }
  if (
    busy.has(input.dest.toUpperCase()) &&
    input.dest.toUpperCase() !== input.origin.toUpperCase()
  ) {
    factors.push({
      name: "Destination hub signal",
      detail: `${input.dest.toUpperCase()} arrival banks can add delay risk.`,
    });
  }
  if (input.distance >= 900) {
    factors.push({
      name: "Distance feature",
      detail: `${input.distance} mi increases disruption exposure in the model.`,
    });
  }
  if (factors.length === 0) {
    factors.push({
      name: "Baseline",
      detail: "No strong risk signals; probability is near the model baseline.",
    });
  }
  return factors.slice(0, 3);
}

export function getLocalMeta() {
  return {
    model_version: weights.model_version,
    algorithm: weights.algorithm,
    features: [
      ...weights.categorical_features,
      ...weights.numeric_features,
    ],
    metrics: weights.metrics,
    airlines: weights.airlines,
    airports: weights.airports,
  };
}

export function predictLocally(input: PredictInput) {
  const probability = scoreLocally(input);
  const delayed = probability >= 0.5;
  let verdict = "Likely on time based on historical patterns.";
  if (probability >= 0.65) {
    verdict = "High chance of delay — plan buffer time.";
  } else if (probability >= 0.45) {
    verdict = "Moderate delay risk — monitor the flight.";
  }
  return {
    delay_probability: Math.round(probability * 10000) / 10000,
    delayed,
    verdict,
    top_factors: explainLocally(input),
  };
}
