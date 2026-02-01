import { useQuery } from "@tanstack/react-query";
import Loading from "./Loading";
import { Download, Table, BarChart3 } from "lucide-react";

type Row = Record<string, string | number>;

const CSV_URL = "/data/brines_with_predictions.csv";
const COLUMNS = [
  "Brine",
  "Location",
  "TDS_gL",
  "MLR",
  "Pred_Selectivity",
  "Pred_Li_Crystallization_mg_m2_h",
  "Pred_Evap_kg_m2_h",
];

function parseCsv(text: string) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return { total: 0, sample: [], stats: null };
  const headers = lines[0].split(",");
  const rows = lines.slice(1);

  const sample = rows.slice(0, 40).map((line) => {
    const cells = line.split(",");
    const record: Row = {};
    headers.forEach((h, i) => {
      const val = cells[i];
      const num = Number(val);
      record[h] = Number.isFinite(num) ? num : val;
    });
    return record;
  });

  const selectivityVals: number[] = [];
  const tdsVals: number[] = [];
  sample.forEach((r) => {
    const s = Number(r["Pred_Selectivity"]);
    const t = Number(r["TDS_gL"]);
    if (Number.isFinite(s)) selectivityVals.push(s);
    if (Number.isFinite(t)) tdsVals.push(t);
  });

  const stats = {
    selectivity: {
      min: Math.min(...selectivityVals),
      max: Math.max(...selectivityVals),
    },
    tds: {
      min: Math.min(...tdsVals),
      max: Math.max(...tdsVals),
    },
  };

  return { total: rows.length, sample, stats };
}

export default function DatasetPreview() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["dataset-preview"],
    queryFn: async () => {
      const res = await fetch(CSV_URL);
      if (!res.ok) throw new Error("Dataset unavailable");
      const text = await res.text();
      return parseCsv(text);
    },
  });

  if (isLoading) return <Loading label="Loading dataset…" />;
  if (error || !data) return <p className="text-red-300">Dataset unavailable.</p>;

  const { sample, total, stats } = data;

  return (
    <div className="glass rounded-2xl border border-white/10 p-6 space-y-4">
      <div className="flex flex-wrap items-center gap-3 justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-sky-400 to-fuchsia-500 flex items-center justify-center text-slate-900">
            <Table size={18} />
          </div>
          <div>
            <p className="text-lg font-semibold">Dataset preview</p>
            <p className="text-slate-300 text-sm">Brine chemistry with model outputs (sampled rows)</p>
          </div>
        </div>
        <a
          href={CSV_URL}
          download
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white text-slate-900 font-semibold shadow-lg"
        >
          <Download size={16} /> Download CSV
        </a>
      </div>

      <div className="grid sm:grid-cols-3 gap-3 text-sm">
        <div className="bg-white/5 border border-white/10 rounded-xl px-3 py-2">
          <p className="text-slate-400">Rows (total)</p>
          <p className="text-lg font-semibold">{total.toLocaleString()}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl px-3 py-2">
          <p className="text-slate-400">Selectivity span</p>
          <p className="text-lg font-semibold">
            {stats ? `${stats.selectivity.min.toFixed(2)} – ${stats.selectivity.max.toFixed(2)}` : "–"}
          </p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-xl px-3 py-2">
          <p className="text-slate-400">TDS span (g/L)</p>
          <p className="text-lg font-semibold">
            {stats ? `${stats.tds.min.toFixed(1)} – ${stats.tds.max.toFixed(1)}` : "–"}
          </p>
        </div>
      </div>

      <div className="overflow-auto">
        <table className="w-full text-sm min-w-[720px]">
          <thead className="text-slate-300 text-left">
            <tr className="border-b border-white/10">
              {COLUMNS.map((c) => (
                <th key={c} className="py-2 pr-3">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sample.map((row, idx) => (
              <tr key={idx} className="border-b border-white/5">
                {COLUMNS.map((c) => (
                  <td key={c} className="py-2 pr-3 text-slate-200">
                    {formatCell(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-2 text-xs text-slate-400">
        <BarChart3 size={14} /> Showing first {sample.length} rows; download to access full dataset.
      </div>
    </div>
  );
}

function formatCell(val: string | number | undefined) {
  if (val === undefined) return "—";
  if (typeof val === "number") {
    if (Math.abs(val) >= 1000) return val.toFixed(1);
    if (Math.abs(val) >= 1) return val.toFixed(3);
    return val.toPrecision(3);
  }
  return val;
}
