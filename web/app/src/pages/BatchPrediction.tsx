import { useEffect, useState } from "react";
import { startBatchJob, fetchBatchStatus, batchResultUrl } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { CloudUpload, Download, Loader2 } from "lucide-react";

export default function BatchPrediction() {
  const [file, setFile] = useState<File | null>(null);
  const [impute, setImpute] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!file) return;
    try {
      setError(null);
      const job = await startBatchJob(file, impute);
      setJobId(job.job_id);
      setStatus(job);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    if (!jobId) return;
    const id = setInterval(async () => {
      const st = await fetchBatchStatus(jobId);
      setStatus(st);
      if (st.status === "completed" || st.status === "failed") {
        clearInterval(id);
      }
    }, 1800);
    return () => clearInterval(id);
  }, [jobId]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Batch Prediction Jobs"
        subtitle="Upload CSV with TDS_gL, MLR, Light_kW_m2 (g/L, unitless, kW/m²). Jobs run asynchronously and return a downloadable CSV with predictions."
        actions={
          <a
            href="/examples/experimental_samples.csv"
            className="pill px-3 py-2 text-sm text-slate-100 bg-black/30"
            download
          >
            Download sample CSV
          </a>
        }
      />

      <div className="glass rounded-2xl border border-white/10 p-6 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center gap-3">
          <label className="flex-1 flex items-center gap-3 bg-white/5 border border-dashed border-white/15 rounded-xl px-4 py-3 cursor-pointer hover:border-white/30">
            <CloudUpload />
            <div className="flex-1">
              <p className="font-semibold">
                {file ? file.name : "Choose a CSV to upload"}
              </p>
              <p className="text-sm text-slate-400">
                Max ~200k rows. Required columns: TDS_gL, MLR, Light_kW_m2. Units must match.
              </p>
            </div>
            <input
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>

          <label className="flex items-center gap-2 text-sm text-slate-100 pill px-3 py-2 bg-black/30">
            <input
              type="checkbox"
              checked={impute}
              onChange={(e) => setImpute(e.target.checked)}
            />
            Impute missing chemistry
          </label>

          <button
            onClick={submit}
            disabled={!file}
            className="px-4 py-3 rounded-full bg-gradient-to-r from-sky-400 to-fuchsia-500 text-slate-900 font-semibold shadow-lg disabled:opacity-50"
          >
            Start job
          </button>
        </div>
        {error && <p className="text-red-300 text-sm">{error}</p>}
      </div>

      {status && (
        <div className="glass rounded-2xl border border-white/10 p-6 space-y-2">
          <div className="flex items-center gap-2">
            <span className="pill px-3 py-1 text-xs text-slate-200">
              Job {status.job_id}
            </span>
            <span
              className={`px-2 py-1 rounded-full text-xs ${
                status.status === "completed"
                  ? "bg-emerald-500/20 text-emerald-200"
                : status.status === "failed"
                ? "bg-rose-500/20 text-rose-200"
                : "bg-amber-500/20 text-amber-200"
              }`}
            >
              {status.status}
            </span>
          </div>
          {status.status === "running" || status.status === "queued" ? (
            <div className="flex items-center gap-2 text-slate-300">
              <Loader2 className="animate-spin" size={18} /> Processing…
            </div>
          ) : null}
          {status.status === "failed" && (
            <p className="text-rose-300">{status.error ?? "Job failed."}</p>
          )}
          {status.status === "completed" && (
            <a
              href={batchResultUrl(status.job_id)}
              className="inline-flex items-center gap-2 px-4 py-3 bg-white text-slate-900 rounded-full font-semibold"
            >
              <Download size={18} />
              Download results
            </a>
          )}
          <p className="text-xs text-slate-400">
            Submitted at {new Date(status.submitted_at).toLocaleString()}
          </p>
          <p className="text-xs text-slate-400">
            Outputs are point estimates; ensure downstream analyses note unit assumptions.
          </p>
        </div>
      )}
    </div>
  );
}
