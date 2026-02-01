import { useQuery } from "@tanstack/react-query";
import { fetchModelMetadata } from "../lib/api";
import Loading from "../components/Loading";
import PageHeader from "../components/PageHeader";
import { ShieldCheck, GitBranch, GitCommitHorizontal, AlertTriangle } from "lucide-react";

export default function ModelPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["model"],
    queryFn: fetchModelMetadata,
  });

  if (isLoading) return <Loading label="Loading model metadata…" />;
  if (error || !data) return <p className="text-red-300">Unable to load metadata.</p>;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Reproducibility & Limits"
        subtitle="What the model expects, where it works best, and how to audit a run."
      />

      <div className="grid lg:grid-cols-4 gap-4">
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400">
            Model version
          </p>
          <p className="text-2xl font-semibold mt-1">{data.version}</p>
        </div>
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400 flex items-center gap-2">
            <GitBranch size={16} />
            Git tag
          </p>
          <p className="text-lg mt-1">{data.git_tag ?? "—"}</p>
        </div>
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400 flex items-center gap-2">
            <GitCommitHorizontal size={16} />
            Commit
          </p>
          <p className="text-xs mt-1 font-mono break-all">{data.git_commit ?? "—"}</p>
        </div>
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400 flex items-center gap-2">
            <ShieldCheck size={16} />
            Scalars & schema
          </p>
          <p className="text-sm mt-1 text-slate-200">
            Inputs (g/L, kW/m²): {data.feature_schema.join(", ")}
          </p>
          <p className="text-xs text-slate-400 mt-1">Outputs: {data.targets.join(", ")}</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <div className="glass rounded-2xl border border-white/10 p-5 space-y-2">
          <p className="text-sm font-semibold">Valid range & units</p>
          <p className="text-slate-300 text-sm">
            Chemistry in grams per liter, irradiance in kW/m². Typical training domain: TDS 10–400 g/L,
            MLR 0–25, Light 0.05–1.0. Outside these ranges, interpret with caution.
          </p>
        </div>
        <div className="glass rounded-2xl border border-white/10 p-5 space-y-2">
          <p className="text-sm font-semibold">Intended use</p>
          <p className="text-slate-300 text-sm">
            Screening and ranking of brine sites; not a substitute for lab crystallization tests.
            Partial inputs allowed; missing chemistry may be imputed if enabled.
          </p>
        </div>
        <div className="glass rounded-2xl border border-white/10 p-5 space-y-2">
          <p className="text-sm font-semibold flex items-center gap-2 text-amber-200">
            <AlertTriangle size={16} /> Known limitations
          </p>
          <ul className="list-disc list-inside text-slate-300 text-sm space-y-1">
            <li>Not calibrated for extreme salars (&gt;450 g/L TDS) or geothermal fluids.</li>
            <li>No uncertainty bands yet; treat outputs as point estimates.</li>
            <li>Batch jobs assume clean CSV/GeoJSON; validate units before upload.</li>
          </ul>
        </div>
      </div>

      <div className="glass rounded-2xl border border-white/10 p-6 space-y-3">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <ShieldCheck size={18} /> Audit trail (artifacts & checksums)
        </h3>
        <details className="bg-white/5 rounded-xl border border-white/10">
          <summary className="px-4 py-3 cursor-pointer text-sm text-slate-100">
            Show artifact table
          </summary>
          <div className="overflow-auto px-4 pb-4">
            <table className="w-full text-sm mt-2">
              <thead className="text-slate-300 text-left">
                <tr className="border-b border-white/10">
                  <th className="py-2">Name</th>
                  <th className="py-2">Path</th>
                  <th className="py-2">SHA256</th>
                  <th className="py-2">Size</th>
                </tr>
              </thead>
              <tbody>
                {data.artifacts.map((a) => (
                  <tr key={a.name} className="border-b border-white/5">
                    <td className="py-2 font-semibold">{a.name}</td>
                    <td className="py-2 text-slate-300">{a.path}</td>
                    <td className="py-2 font-mono text-xs break-all">{a.sha256}</td>
                    <td className="py-2 text-slate-300">
                      {(a.size_bytes / 1_000_000).toFixed(2)} MB
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
        <p className="text-xs text-slate-400">
          Checksums tie predictions to specific artifacts. Scaler path: {data.scaler_path}
        </p>
      </div>
    </div>
  );
}
