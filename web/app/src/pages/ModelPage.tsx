import { useQuery } from "@tanstack/react-query";
import { fetchModelMetadata } from "../lib/api";
import Loading from "../components/Loading";
import PageHeader from "../components/PageHeader";
import { ShieldCheck, GitBranch, GitCommitHorizontal } from "lucide-react";

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
        title="Model & Reproducibility"
        subtitle="Versioned artifacts, git references, and feature schema for the deployed GlobalBrine model."
      />

      <div className="grid sm:grid-cols-3 gap-4">
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
      </div>

      <div className="glass rounded-2xl border border-white/10 p-6">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <ShieldCheck size={18} /> Artifacts & checksums
        </h3>
        <div className="overflow-auto">
          <table className="w-full text-sm">
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
                  <td className="py-2 font-mono text-xs">{a.sha256}</td>
                  <td className="py-2 text-slate-300">
                    {(a.size_bytes / 1_000_000).toFixed(2)} MB
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-slate-300 mt-3">
          Feature schema: {data.feature_schema.join(", ")}
        </p>
        <p className="text-sm text-slate-300">
          Targets: {data.targets.join(", ")}
        </p>
        <p className="text-xs text-slate-400 mt-2">
          Checksums allow reproducible audits. Scaler: {data.scaler_path}
        </p>
      </div>
    </div>
  );
}
