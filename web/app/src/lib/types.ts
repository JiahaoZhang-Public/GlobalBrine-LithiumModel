export type PredictionValues = {
  Selectivity: number;
  Li_Crystallization_mg_m2_h: number;
  Evap_kg_m2_h: number;
};

export type ModelArtifact = {
  name: string;
  path: string;
  sha256: string;
  size_bytes: number;
  modified_at: string;
};

export type ModelMetadata = {
  version: string;
  git_commit?: string | null;
  git_tag?: string | null;
  artifacts: ModelArtifact[];
  feature_schema: string[];
  targets: string[];
  scaler_path: string;
};

export type SinglePredictResponse = {
  predictions: PredictionValues;
  imputed_input: Record<string, number | null>;
  metadata: ModelMetadata;
};

export type GeoFeature = {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: Record<string, any>;
};

export type GeoResponse = {
  type: "FeatureCollection";
  features: GeoFeature[];
  meta: Record<string, any>;
};

export type BatchJobStatus = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  submitted_at: string;
  completed_at?: string | null;
  error?: string | null;
  download_url?: string | null;
};
