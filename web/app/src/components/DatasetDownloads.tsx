import { Download, Database, Sparkles } from "lucide-react";
import { useState } from "react";
import { useI18n } from "../lib/i18n";
import PreviewTable from "./PreviewTable";

type DatasetLink = {
  key: string;
  titleKey: string;
  descriptionKey: string;
  href: string;
  badgeKey: string;
  icon: typeof Database;
};

const datasets: DatasetLink[] = [
  {
    key: "raw",
    titleKey: "datasets.raw.title",
    descriptionKey: "datasets.raw.desc",
    href: "/data/brines.csv",
    badgeKey: "datasets.badge.raw",
    icon: Database,
  },
  {
    key: "predictions",
    titleKey: "datasets.pred.title",
    descriptionKey: "datasets.pred.desc",
    href: "/data/brines_with_predictions.csv",
    badgeKey: "datasets.badge.pred",
    icon: Sparkles,
  },
];

export default function DatasetDownloads() {
  const { t } = useI18n();
  const [openKey, setOpenKey] = useState<string | null>(null);
  return (
    <div className="glass rounded-3xl border border-white/10 p-6 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-[0.22em] text-slate-300 pill px-3 py-2">
            {t("datasets.heading")}
          </p>
          <h3 className="text-2xl font-semibold">{t("datasets.title")}</h3>
          <p className="text-slate-300 text-sm">
            {t("datasets.desc")}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Download size={14} /> {t("datasets.version")}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {datasets.map((item) => {
          const open = openKey === item.key;
          return (
            <div
              key={item.key}
              className={`group relative rounded-2xl border border-white/10 bg-white/5 p-5 transition hover:-translate-y-0.5 hover:border-sky-400/50 ${open ? "md:col-span-2" : ""}`}
            >
            <div className="absolute inset-0 pointer-events-none bg-gradient-to-br from-white/0 via-white/0 to-sky-500/0 group-hover:from-white/5 group-hover:to-fuchsia-500/10 transition" />
            <div className="relative flex items-start gap-3">
              <div className="h-12 w-12 shrink-0 rounded-2xl bg-gradient-to-br from-sky-400 to-fuchsia-500 flex items-center justify-center text-slate-900">
                <item.icon size={18} />
              </div>
              <div className="space-y-2 w-full min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-lg font-semibold leading-tight">{t(item.titleKey as any)}</p>
                  <span className="pill px-3 py-1 text-xs text-slate-900 bg-white/90 font-semibold">
                    {t(item.badgeKey as any)}
                  </span>
                </div>
                <p className="text-slate-200 text-sm leading-relaxed">{t(item.descriptionKey as any)}</p>
                <div className="flex flex-wrap items-center gap-3 text-sm">
                  <a
                    href={item.href}
                    download
                    className="inline-flex items-center gap-1 font-semibold text-slate-900 bg-white px-3 py-2 rounded-full shadow-lg"
                  >
                    <Download size={14} /> {t("datasets.download")}
                  </a>
                  <button
                    type="button"
                    onClick={() => setOpenKey(openKey === item.key ? null : item.key)}
                    className="text-xs text-sky-200 underline hover:text-white"
                  >
                    {open ? t("datasets.preview.hide") : t("datasets.preview.show")}
                  </button>
                </div>
                {open && (
                  <div className="border border-white/10 rounded-xl bg-black/30 p-3 overflow-auto min-w-0">
                    <PreviewTable src={item.href} limit={10} />
                  </div>
                )}
              </div>
            </div>
          </div>
          );
        })}
      </div>
    </div>
  );
}
