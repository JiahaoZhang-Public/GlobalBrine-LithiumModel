import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { predictSingle } from "../lib/api";
import PageHeader from "../components/PageHeader";
import Loading from "../components/Loading";
import { StatCard } from "../components/StatCard";
import { FlaskConical, Sparkles, Waves, Info, Wand2 } from "lucide-react";

const fields: { key: string; label: string; placeholder?: string }[] = [
  { key: "Li_gL", label: "Li (g/L)", placeholder: "0.05" },
  { key: "Mg_gL", label: "Mg (g/L)", placeholder: "8.2" },
  { key: "Na_gL", label: "Na (g/L)", placeholder: "95" },
  { key: "K_gL", label: "K (g/L)", placeholder: "3.1" },
  { key: "Ca_gL", label: "Ca (g/L)", placeholder: "1.8" },
  { key: "SO4_gL", label: "SO₄ (g/L)", placeholder: "6.4" },
  { key: "Cl_gL", label: "Cl (g/L)", placeholder: "110" },
  { key: "TDS_gL", label: "TDS (g/L)", placeholder: "240" },
  { key: "MLR", label: "MLR", placeholder: "4.5" },
  { key: "Light_kW_m2", label: "Light (kW/m²)", placeholder: "0.22" },
];

const exampleValues: Record<string, string> = {
  Li_gL: "0.05",
  Mg_gL: "8.2",
  Na_gL: "95",
  K_gL: "3.1",
  Ca_gL: "1.8",
  SO4_gL: "6.4",
  Cl_gL: "110",
  TDS_gL: "240",
  MLR: "4.5",
  Light_kW_m2: "0.22",
};

export default function SinglePrediction() {
  const [form, setForm] = useState<Record<string, string>>({});
  const [impute, setImpute] = useState(true);

  const mutation = useMutation({
    mutationFn: (payload: Record<string, number | null | boolean>) =>
      predictSingle(payload),
  });

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    const payload: Record<string, number | null | boolean> = {
      impute_missing_chemistry: impute,
    };
    fields.forEach((f) => {
      const val = form[f.key];
      payload[f.key] = val === undefined || val === "" ? null : Number(val);
    });
    mutation.mutate(payload);
  };

  const predictions = mutation.data?.predictions;
  const imputed = useMemo(
    () => mutation.data?.imputed_input ?? {},
    [mutation.data?.imputed_input]
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Single-Point Prediction"
        subtitle="Enter one brine sample. Missing fields can be imputed. Units: g/L for chemistry, kW/m² for light."
        actions={
          <label className="flex items-center gap-2 text-sm text-slate-100 pill px-3 py-2 bg-black/30">
            <input
              type="checkbox"
              checked={impute}
              onChange={(e) => setImpute(e.target.checked)}
            />
            Impute missing chemistry
          </label>
        }
      />

      <div className="grid lg:grid-cols-2 gap-6">
        <form
          onSubmit={onSubmit}
          className="glass rounded-2xl p-6 border border-white/10 space-y-4"
        >
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-200">
            <button
              type="button"
              onClick={() => setForm(exampleValues)}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-full bg-white text-slate-900 font-semibold shadow-lg"
            >
              <Wand2 size={16} />
              Fill with example
            </button>
            <span className="text-slate-400">
              Example reflects a mid-range salar sample; adjust values to your site.
            </span>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            {fields.map((field) => (
              <label key={field.key} className="flex flex-col gap-1 text-sm">
                <span className="text-slate-200">{field.label}</span>
                <input
                  type="number"
                  step="any"
                  placeholder={field.placeholder}
                  value={form[field.key] ?? ""}
                  onChange={(e) =>
                    setForm((prev) => ({ ...prev, [field.key]: e.target.value }))
                  }
                  className="bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-slate-100"
                />
              </label>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <button
              type="submit"
              className="px-4 py-3 rounded-full bg-gradient-to-r from-sky-400 to-fuchsia-500 text-slate-900 font-semibold shadow-lg"
            >
              Run prediction
            </button>
            {mutation.isPending && <Loading label="Running inference…" />}
            {mutation.error && (
              <p className="text-red-300 text-sm">
                {(mutation.error as Error).message}
              </p>
            )}
          </div>
        </form>

        <div className="space-y-4">
          {!predictions ? (
            <div className="glass rounded-2xl border border-dashed border-white/20 p-6 text-slate-300">
              Predictions will appear here.
            </div>
          ) : (
            <div className="grid sm:grid-cols-3 gap-3">
              <StatCard
                label="Selectivity"
                value={predictions.Selectivity.toFixed(3)}
                helper="Li⁺ vs Mg²⁺"
                icon={<Sparkles size={20} />}
                tone="primary"
              />
              <StatCard
                label="Li crystallization"
                value={`${predictions.Li_Crystallization_mg_m2_h.toFixed(3)} mg/m²·h`}
                icon={<FlaskConical size={20} />}
                tone="secondary"
              />
              <StatCard
                label="Evaporation"
                value={`${predictions.Evap_kg_m2_h.toFixed(3)} kg/m²·h`}
                icon={<Waves size={20} />}
                tone="success"
              />
            </div>
          )}

          {predictions && (
            <div className="glass rounded-2xl border border-white/10 p-4">
              <h3 className="text-lg font-semibold mb-2">Imputed chemistry</h3>
              <div className="grid sm:grid-cols-2 gap-2 text-sm text-slate-200">
                {Object.entries(imputed).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2"
                  >
                    <span className="text-slate-300">{k}</span>
                    <span className="font-mono">
                      {v === null || Number.isNaN(v) ? "—" : Number(v).toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
              <div className="flex items-start gap-2 text-xs text-slate-300 mt-3">
                <Info size={14} className="text-cyan-300 mt-0.5" />
                <div>
                  Outputs are clamped to non-negative values. Predictions are point estimates; no
                  uncertainty bands yet. See Repro & Limits for applicability notes.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
