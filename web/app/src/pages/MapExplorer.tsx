import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchGeo } from "../lib/api";
import BrineMap from "../components/BrineMap";
import PageHeader from "../components/PageHeader";
import Loading from "../components/Loading";
import { SlidersHorizontal } from "lucide-react";

export default function MapExplorer() {
  const { data, isLoading, error } = useQuery({ queryKey: ["geo"], queryFn: fetchGeo });
  const [variable, setVariable] = useState("Pred_Selectivity");
  const [tdsRange, setTdsRange] = useState<[number, number] | null>(null);
  const [mlrRange, setMlrRange] = useState<[number, number] | null>(null);

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
        label: "Points",
        value: data.features.length.toLocaleString(),
      },
      {
        label: "Selectivity (max)",
        value: data.meta?.Pred_Selectivity?.max?.toFixed?.(2) ?? "–",
      },
      {
        label: "TDS range",
        value: `${data.meta?.TDS_gL?.min?.toFixed?.(1) ?? "–"} – ${
          data.meta?.TDS_gL?.max?.toFixed?.(1) ?? "–"
        } g/L`,
      },
      {
        label: "MLR range",
        value: `${data.meta?.MLR?.min?.toFixed?.(2) ?? "–"} – ${
          data.meta?.MLR?.max?.toFixed?.(2) ?? "–"
        }`,
      },
    ];
  }, [data]);

  if (isLoading) return <Loading label="Loading map data…" />;
  if (error || !data) return <p className="text-red-300">Unable to load map data.</p>;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Global Map Explorer"
        subtitle="Visualize predicted selectivity, crystallization, and evaporation across all known brine sites."
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

      <div className="glass rounded-2xl border border-white/10 p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-3 text-sm text-slate-200">
          <span className="pill px-3 py-2 flex items-center gap-2">
            <SlidersHorizontal size={16} />
            Filters
          </span>
          <label className="flex items-center gap-2">
            TDS ≥
            <input
              type="number"
              value={tdsRange?.[0] ?? ""}
              onChange={(e) =>
                setTdsRange([
                  Number(e.target.value),
                  tdsRange?.[1] ?? data.meta?.TDS_gL?.max ?? 0,
                ])
              }
              className="bg-black/40 border border-white/10 rounded-md px-2 py-1 w-24 text-slate-100"
            />
          </label>
          <label className="flex items-center gap-2">
            TDS ≤
            <input
              type="number"
              value={tdsRange?.[1] ?? ""}
              onChange={(e) =>
                setTdsRange([
                  tdsRange?.[0] ?? data.meta?.TDS_gL?.min ?? 0,
                  Number(e.target.value),
                ])
              }
              className="bg-black/40 border border-white/10 rounded-md px-2 py-1 w-24 text-slate-100"
            />
          </label>
          <label className="flex items-center gap-2">
            MLR ≤
            <input
              type="number"
              value={mlrRange?.[1] ?? ""}
              onChange={(e) =>
                setMlrRange([
                  mlrRange?.[0] ?? 0,
                  Number(e.target.value),
                ])
              }
              className="bg-black/40 border border-white/10 rounded-md px-2 py-1 w-24 text-slate-100"
            />
          </label>
        </div>
      </div>

      <BrineMap
        data={data.features}
        meta={data.meta}
        variable={variable}
        onVariableChange={setVariable}
        tdsRange={tdsRange}
        mlrRange={mlrRange}
      />
    </div>
  );
}
