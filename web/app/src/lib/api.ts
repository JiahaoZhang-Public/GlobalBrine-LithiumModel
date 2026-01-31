import type {
  BatchJobStatus,
  GeoResponse,
  ModelMetadata,
  SinglePredictResponse,
} from "./types";

const apiBase =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") || "";

function url(path: string) {
  if (path.startsWith("http")) return path;
  return `${apiBase}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: HeadersInit | undefined = init?.body
    ? { "Content-Type": "application/json", ...(init?.headers || {}) }
    : init?.headers;
  const res = await fetch(url(path), {
    ...init,
    headers,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const fetchModelMetadata = () => request<ModelMetadata>("/api/v1/model");
export const fetchGeo = () => request<GeoResponse>("/api/v1/data/points");

export async function predictSingle(
  payload: Record<string, number | null | boolean>
): Promise<SinglePredictResponse> {
  return request<SinglePredictResponse>("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function startBatchJob(
  file: File,
  impute_missing_chemistry: boolean
): Promise<BatchJobStatus> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(url(`/api/v1/predict/batch?impute_missing_chemistry=${impute_missing_chemistry}`), {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed (${res.status})`);
  return res.json();
}

export const fetchBatchStatus = (jobId: string) =>
  request<BatchJobStatus>(`/api/v1/predict/batch/${jobId}/status`);

export const batchResultUrl = (jobId: string) =>
  url(`/api/v1/predict/batch/${jobId}/result`);
