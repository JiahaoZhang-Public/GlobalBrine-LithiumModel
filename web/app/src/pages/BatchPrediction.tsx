import { useEffect, useState } from "react";
import { startBatchJob, fetchBatchStatus, batchResultUrl } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { CloudUpload, Download, Loader2, Info, Table as TableIcon } from "lucide-react";
import { useI18n } from "../lib/i18n";

const sampleRows = [
  { TDS_gL: 220, MLR: 4.5, Light_kW_m2: 0.22 },
  { TDS_gL: 140, MLR: 2.1, Light_kW_m2: 0.18 },
  { TDS_gL: 320, MLR: 8.9, Light_kW_m2: 0.27 },
];

export default function BatchPrediction() {
  const [file, setFile] = useState<File | null>(null);
  const [impute, setImpute] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const { t } = useI18n();

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
        title={t("batch.title")}
        subtitle={t("batch.subtitle")}
        actions={
          <a
            href="/examples/experimental_samples.csv"
            className="pill px-3 py-2 text-sm text-slate-100 bg-black/30"
            download
          >
            {t("batch.sample")}
          </a>
        }
      />

      <div className="glass rounded-2xl border border-white/10 p-6 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center gap-3">
          <label className="flex-1 flex items-center gap-3 bg-white/5 border border-dashed border-white/15 rounded-xl px-4 py-3 cursor-pointer hover:border-white/30">
            <CloudUpload />
            <div className="flex-1">
              <p className="font-semibold">
                {file ? file.name : t("batch.choose")}
              </p>
              <p className="text-sm text-slate-400">
                {t("batch.hint")}
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
            {t("batch.impute")}
          </label>

          <button
            onClick={submit}
            disabled={!file}
            className="px-4 py-3 rounded-full bg-gradient-to-r from-sky-400 to-fuchsia-500 text-slate-900 font-semibold shadow-lg disabled:opacity-50"
          >
            {t("batch.start")}
          </button>
        </div>
        <div className="grid lg:grid-cols-2 gap-3">
          <div className="bg-black/25 border border-white/10 rounded-xl p-4 text-sm text-slate-200 space-y-2">
            <div className="flex items-center gap-2 text-slate-100 font-semibold">
              <TableIcon size={16} /> {t("batch.examples")}
            </div>
            <table className="w-full text-xs">
              <thead className="text-slate-300 text-left border-b border-white/10">
                <tr>
                  <th className="py-1 pr-3">TDS_gL</th>
                  <th className="py-1 pr-3">MLR</th>
                  <th className="py-1 pr-3">Light_kW_m2</th>
                </tr>
              </thead>
              <tbody>
                {sampleRows.map((r, i) => (
                  <tr key={i} className="border-b border-white/5">
                    <td className="py-1 pr-3">{r.TDS_gL}</td>
                    <td className="py-1 pr-3">{r.MLR}</td>
                    <td className="py-1 pr-3">{r.Light_kW_m2}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-slate-400">{t("batch.copyhint")}</p>
          </div>
          <div className="bg-black/25 border border-white/10 rounded-xl p-4 text-xs text-slate-300 space-y-2">
            <div className="flex items-center gap-2 text-slate-100 font-semibold text-sm">
              <Info size={14} /> {t("batch.recipe")}
            </div>
            <ol className="list-decimal list-inside space-y-1">
              <li>{t("batch.step1")}</li>
              <li>{t("batch.step2")}</li>
              <li>{t("batch.step3")}</li>
              <li>{t("batch.step4")}</li>
            </ol>
          </div>
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
              <Loader2 className="animate-spin" size={18} /> {t("batch.processing")}
            </div>
          ) : null}
          {status.status === "failed" && (
            <p className="text-rose-300">{status.error ?? t("batch.failed")}</p>
          )}
          {status.status === "completed" && (
            <a
              href={batchResultUrl(status.job_id)}
              className="inline-flex items-center gap-2 px-4 py-3 bg-white text-slate-900 rounded-full font-semibold"
            >
              <Download size={18} />
              {t("batch.download")}
            </a>
          )}
          <p className="text-xs text-slate-400">
            {t("batch.submitted")} {new Date(status.submitted_at).toLocaleString()}
          </p>
          <p className="text-xs text-slate-400">
            {t("batch.note")}
          </p>
        </div>
      )}
    </div>
  );
}
