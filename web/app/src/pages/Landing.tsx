import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchGeo, fetchModelMetadata } from "../lib/api";
import { StatCard } from "../components/StatCard";
import Loading from "../components/Loading";
import { Compass, Cpu, Globe2, Ruler } from "lucide-react";
import DatasetDownloads from "../components/DatasetDownloads";
import { useI18n } from "../lib/i18n";

export default function Landing() {
  const { data: model } = useQuery({ queryKey: ["model"], queryFn: fetchModelMetadata });
  const { data: geo, isLoading } = useQuery({ queryKey: ["geo"], queryFn: fetchGeo });
  const { t, lang } = useI18n();

  const sampleCount = geo?.features.length ?? 0;
  const mapCoverage = geo?.meta?.count ?? sampleCount;

  return (
    <div className="space-y-10">
      <section className="glass rounded-3xl border border-white/10 p-8 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-fuchsia-500/10 pointer-events-none" />
        <div className="relative grid md:grid-cols-2 gap-8 items-center">
          <div className="space-y-5">
            <p className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.24em] text-slate-300 pill px-3 py-2">
              {t("landing.pill")}
            </p>
            <h1 className="text-4xl md:text-5xl font-semibold leading-tight">
              {t("landing.title")}
            </h1>
            <p className="text-slate-300 text-lg">
              {t("landing.desc")}
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <Link
                to="/map"
                className="px-4 py-3 rounded-full bg-gradient-to-r from-sky-400 to-fuchsia-500 text-slate-900 font-semibold shadow-lg"
              >
                {t("landing.cta.map")}
              </Link>
              <Link
                to="/predict"
                className="pill px-4 py-3 text-slate-100 hover:bg-white/10 transition"
              >
                {t("landing.cta.predict")}
              </Link>
              <span className="text-slate-400 text-sm">
                Model v{model?.version ?? "0.1.x"} • inputs in g/L, kW/m²
              </span>
            </div>
          </div>
          <div className="glass rounded-2xl border border-white/10 p-6 grid gap-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-sky-400 to-fuchsia-500 flex items-center justify-center text-slate-900 text-xl font-bold">
                Lab
              </div>
              <div>
                <p className="text-lg font-semibold">Ready for field and bench</p>
                <p className="text-slate-300 text-sm">
                  Works with partial chemistry, returns selectivity, crystallization, and evaporation with units.
                </p>
              </div>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <StatCard
                label={t("landing.stat.global")}
                value={mapCoverage.toLocaleString()}
                helper={t("landing.stat.helper.global")}
                icon={<Globe2 size={20} />}
                tone="primary"
              />
              <StatCard
                label={t("landing.stat.inputs")}
                value="g/L + kW/m²"
                helper={t("landing.stat.helper.inputs")}
                icon={<Ruler size={20} />}
                tone="secondary"
              />
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <StatCard
                label={t("landing.stat.turnaround")}
                value="< 1 s"
                helper={t("landing.stat.helper.turnaround")}
                icon={<Cpu size={20} />}
                tone="success"
              />
              <StatCard
                label={t("landing.stat.provenance")}
                value={model?.git_tag ?? "tagged release"}
                helper={t("landing.stat.helper.provenance")}
                icon={<Compass size={20} />}
                tone="warning"
              />
            </div>
          </div>
        </div>
      </section>

      <section>
        <DatasetDownloads />
      </section>

      <section className="grid md:grid-cols-3 gap-6">
        <div className="glass rounded-2xl p-5 border border-white/10">
          <p className="text-sm text-slate-300">01</p>
          <h3 className="text-xl font-semibold mt-1">{t("landing.card1.title")}</h3>
          <p className="text-slate-300 mt-2">{t("landing.card1.desc")}</p>
        </div>
        <div className="glass rounded-2xl p-5 border border-white/10">
          <p className="text-sm text-slate-300">02</p>
          <h3 className="text-xl font-semibold mt-1">{t("landing.card2.title")}</h3>
          <p className="text-slate-300 mt-2">{t("landing.card2.desc")}</p>
        </div>
        <div className="glass rounded-2xl p-5 border border-white/10">
          <p className="text-sm text-slate-300">03</p>
          <h3 className="text-xl font-semibold mt-1">{t("landing.card3.title")}</h3>
          <p className="text-slate-300 mt-2">{t("landing.card3.desc")}</p>
        </div>
      </section>

      {isLoading ? (
        <Loading label={lang === "zh" ? "正在加载覆盖范围…" : "Loading sample coverage…"} />
      ) : (
        <section className="glass rounded-2xl p-6 border border-white/10">
          <h3 className="text-xl font-semibold mb-2">{t("landing.coverage.title")}</h3>
          <div className="flex flex-wrap gap-2 text-sm text-slate-300">
            <span className="pill px-3 py-1">
              {mapCoverage.toLocaleString()} {lang === "zh" ? t("landing.coverage.points") : "points"}
            </span>
            <span className="pill px-3 py-1">
              {t("landing.coverage.min")} {geo?.meta?.TDS_gL?.min?.toFixed?.(1) ?? "–"} g/L
            </span>
            <span className="pill px-3 py-1">
              {t("landing.coverage.max")} {geo?.meta?.Pred_Selectivity?.max?.toFixed?.(2) ?? "–"}
            </span>
          </div>
        </section>
      )}
    </div>
  );
}
