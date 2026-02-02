import { useQuery } from "@tanstack/react-query";
import { fetchModelMetadata } from "../lib/api";
import Loading from "../components/Loading";
import PageHeader from "../components/PageHeader";
import { ShieldCheck, GitBranch, GitCommitHorizontal, AlertTriangle } from "lucide-react";
import { useI18n } from "../lib/i18n";

export default function ModelPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["model"],
    queryFn: fetchModelMetadata,
  });
  const { t } = useI18n();

  if (isLoading) return <Loading label="Loading model metadata…" />;
  if (error || !data) return <p className="text-red-300">Unable to load metadata.</p>;

  return (
    <div className="space-y-6">
      <PageHeader
        title={t("model.title")}
        subtitle={t("model.subtitle")}
      />

      <div className="grid lg:grid-cols-4 gap-4">
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400">
            {t("model.version")}
          </p>
          <p className="text-2xl font-semibold mt-1">{data.version}</p>
        </div>
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400 flex items-center gap-2">
            <GitBranch size={16} />
            {t("model.tag")}
          </p>
          <p className="text-lg mt-1">{data.git_tag ?? "—"}</p>
        </div>
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400 flex items-center gap-2">
            <GitCommitHorizontal size={16} />
            {t("model.commit")}
          </p>
          <p className="text-xs mt-1 font-mono break-all">{data.git_commit ?? "—"}</p>
        </div>
        <div className="glass rounded-2xl p-4 border border-white/10">
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400 flex items-center gap-2">
            <ShieldCheck size={16} />
            {t("model.scalars")}
          </p>
          <p className="text-sm mt-1 text-slate-200">
            Inputs (g/L, kW/m²): {data.feature_schema.join(", ")}
          </p>
          <p className="text-xs text-slate-400 mt-1">Outputs: {data.targets.join(", ")}</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <div className="glass rounded-2xl border border-white/10 p-5 space-y-2">
          <p className="text-sm font-semibold">{t("model.valid")}</p>
          <p className="text-slate-300 text-sm">
            {t("model.valid.desc")}
          </p>
        </div>
        <div className="glass rounded-2xl border border-white/10 p-5 space-y-2">
          <p className="text-sm font-semibold">{t("model.use")}</p>
          <p className="text-slate-300 text-sm">
            {t("model.use.desc")}
          </p>
        </div>
        <div className="glass rounded-2xl border border-white/10 p-5 space-y-2">
          <p className="text-sm font-semibold flex items-center gap-2 text-amber-200">
            <AlertTriangle size={16} /> {t("model.limits")}
          </p>
          <ul className="list-disc list-inside text-slate-300 text-sm space-y-1">
            <li>{t("model.limit1")}</li>
            <li>{t("model.limit2")}</li>
            <li>{t("model.limit3")}</li>
          </ul>
        </div>
      </div>

      <div className="glass rounded-2xl border border-white/10 p-6 space-y-3">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <ShieldCheck size={18} /> {t("model.audit")}
        </h3>
        <details className="bg-white/5 rounded-xl border border-white/10">
          <summary className="px-4 py-3 cursor-pointer text-sm text-slate-100">
            {t("model.artifacts")}
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
          {t("model.checksums")} {data.scaler_path}
        </p>
      </div>
    </div>
  );
}
