import { useEffect, useState } from "react";
import Loading from "./Loading";
import { useI18n } from "../lib/i18n";

type Props = {
  src: string;
  limit?: number;
};

type Row = Record<string, string | number>;

function parseCsv(text: string, limit: number): Row[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines
    .slice(1, limit + 1)
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const cells = line.split(",");
      const r: Row = {};
      headers.forEach((h, i) => {
        const num = Number(cells[i]);
        r[h] = Number.isFinite(num) ? num : cells[i];
      });
      return r;
    });
}

export default function PreviewTable({ src, limit = 10 }: Props) {
  const { lang } = useI18n();
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    setError(null);
    fetch(src)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((text) => {
        if (!cancelled) setRows(parseCsv(text, limit));
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [src, limit]);

  if (rows === null && !error) return <Loading label={lang === "zh" ? "正在加载预览…" : "Loading preview…"} />;
  if (error || !rows) return <p className="text-rose-300 text-sm">{lang === "zh" ? "预览不可用。" : "Preview unavailable."}</p>;
  if (rows.length === 0) return <p className="text-slate-300 text-sm">{lang === "zh" ? "无数据" : "No data"}</p>;

  const headers = Object.keys(rows[0] ?? {});

  return (
    <div className="overflow-auto max-h-56 text-xs text-slate-100">
      <table className="w-full min-w-[360px]">
        <thead className="text-slate-300 text-left border-b border-white/10">
          <tr>
            {headers.map((h) => (
              <th key={h} className="py-1 pr-3">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx} className="border-b border-white/5">
              {headers.map((h) => (
                <td key={h} className="py-1 pr-3 whitespace-nowrap">
                  {formatCell(row[h])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[11px] text-slate-400 mt-2">
        {lang === "zh" ? `仅显示前 ${limit} 行，下载可得完整数据。` : `Showing first ${limit} rows; download for full dataset.`}
      </p>
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
