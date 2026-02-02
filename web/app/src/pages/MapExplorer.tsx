import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchGeo } from "../lib/api";
import BrineMap from "../components/BrineMap";
import PageHeader from "../components/PageHeader";
import Loading from "../components/Loading";
import { SlidersHorizontal } from "lucide-react";
import { useI18n } from "../lib/i18n";

export default function MapExplorer() {
  const { data, isLoading, error } = useQuery({ queryKey: ["geo"], queryFn: fetchGeo });
  const [variable, setVariable] = useState("Pred_Selectivity");
  const [tdsRange, setTdsRange] = useState<[number, number] | null>(null);
  const [mlrRange, setMlrRange] = useState<[number, number] | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    if (!data) return;
    setTdsRange([
      data.meta?.TDS_gL?.min ?? 0,
      data.meta?.TDS_gL?.max ?? 500,
    ]);
    setMlrRange([
      data.meta?.MLR?.min ?? 0,
      data.meta?.MLR?.max ?? 50,
    ]);
  }, [data]);

  const metaCards = useMemo(() => {
    if (!data) return [];
    return [
      {
        label: t("map.meta.points"),
        value: data.features.length.toLocaleString(),
      },
      {
        label: t("map.meta.selectivity"),
        value: data.meta?.Pred_Selectivity?.max?.toFixed?.(2) ?? "–",
      },
      {
        label: t("map.meta.tds"),
        value: `${data.meta?.TDS_gL?.min?.toFixed?.(1) ?? "–"} – ${
          data.meta?.TDS_gL?.max?.toFixed?.(1) ?? "–"
        } g/L`,
      },
      {
        label: t("map.meta.mlr"),
        value: `${data.meta?.MLR?.min?.toFixed?.(2) ?? "–"} – ${
          data.meta?.MLR?.max?.toFixed?.(2) ?? "–"
        }`,
      },
    ];
  }, [data]);

  if (isLoading) return <Loading label={t("map.title") + "…"} />;
  if (error || !data) return <p className="text-red-300">{t("datasets.preview.error")}</p>;

  const filtered = data.features.filter((f) => {
    const props = f.properties || {};
    const tds = Number(props["TDS_gL"]);
    const mlr = Number(props["MLR"]);
    return inRange(tds, tdsRange) && inRange(mlr, mlrRange);
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title={t("map.title")}
        subtitle={t("map.subtitle")}
        actions={
          <div className="pill px-3 py-2 text-sm text-slate-200 bg-black/30 border border-white/10">
            Updated{" "}
            {data.meta?.updated_at
              ? new Date(data.meta.updated_at).toLocaleDateString()
              : "—"}
          </div>
        }
      />

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {metaCards.map((c) => (
          <div key={c.label} className="glass rounded-2xl p-4 border border-white/10">
            <p className="text-xs uppercase tracking-[0.18em] text-slate-400">
              {c.label}
            </p>
            <p className="text-xl font-semibold mt-1">{c.value}</p>
          </div>
        ))}
      </div>

      <div className="glass rounded-2xl border border-white/10 p-4 space-y-4">
        <div className="flex flex-wrap items-center gap-3 text-sm text-slate-200">
          <span className="pill px-3 py-2 flex items-center gap-2">
            <SlidersHorizontal size={16} /> {t("map.filters")}
          </span>
          <span className="text-slate-400">{t("map.units")}</span>
          <span className="pill px-2 py-1 text-xs text-emerald-200">
            {filtered.length} shown / {data.features.length}
          </span>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-sm text-slate-100">
          <RangeInput
            label="TDS range (g/L)"
            value={tdsRange}
            onChange={setTdsRange}
            min={data.meta?.TDS_gL?.min ?? 0}
            max={data.meta?.TDS_gL?.max ?? 500}
          />
          <RangeInput
            label="MLR range"
            value={mlrRange}
            onChange={setMlrRange}
            min={data.meta?.MLR?.min ?? 0}
            max={data.meta?.MLR?.max ?? 50}
          />
          <div className="bg-black/30 border border-white/10 rounded-xl px-3 py-2 text-slate-300">
            {t("map.color.key")}
          </div>
        </div>
      </div>

      <BrineMap
        data={filtered}
        meta={data.meta}
        variable={variable}
        onVariableChange={setVariable}
        tdsRange={tdsRange}
        mlrRange={mlrRange}
      />
    </div>
  );
}

function inRange(val: number | null | undefined, range: [number, number] | null) {
  if (val === null || val === undefined || Number.isNaN(val)) return false;
  if (!range) return true;
  const [min, max] = range;
  const lowOk = min === null || min === undefined ? true : val >= min;
  const highOk = max === null || max === undefined ? true : val <= max;
  return lowOk && highOk;
}

type RangeInputProps = {
  label: string;
  value: [number, number] | null;
  onChange: (v: [number, number]) => void;
  min: number;
  max: number;
};

function RangeInput({ label, value, onChange, min, max }: RangeInputProps) {
  const [localMin, localMax] = value ?? [min, max];
  return (
    <div className="bg-black/30 border border-white/10 rounded-xl p-3 space-y-2">
      <p className="text-slate-300 text-sm">{label}</p>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step="1"
          value={localMin}
          onChange={(e) => onChange([Number(e.target.value), localMax])}
          className="flex-1"
        />
        <input
          type="number"
          value={localMin}
          onChange={(e) => onChange([Number(e.target.value), localMax])}
          className="w-20 bg-black/40 border border-white/10 rounded-md px-2 py-1"
        />
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step="1"
          value={localMax}
          onChange={(e) => onChange([localMin, Number(e.target.value)])}
          className="flex-1"
        />
        <input
          type="number"
          value={localMax}
          onChange={(e) => onChange([localMin, Number(e.target.value)])}
          className="w-20 bg-black/40 border border-white/10 rounded-md px-2 py-1"
        />
      </div>
      <p className="text-xs text-slate-400">Showing {localMin} – {localMax}</p>
    </div>
  );
}
