import type { ReactNode } from "react";
import clsx from "clsx";

type Props = {
  label: string;
  value: string;
  helper?: string;
  icon?: ReactNode;
  tone?: "primary" | "secondary" | "success" | "warning";
};

const toneMap: Record<NonNullable<Props["tone"]>, string> = {
  primary: "from-sky-400 to-fuchsia-500",
  secondary: "from-indigo-400 to-purple-500",
  success: "from-emerald-400 to-lime-300",
  warning: "from-amber-300 to-orange-400",
};

export function StatCard({ label, value, helper, icon, tone = "primary" }: Props) {
  return (
    <div className="glass rounded-2xl p-4 border border-white/10">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-slate-400">
            {label}
          </p>
          <p className="text-2xl font-semibold mt-1">{value}</p>
          {helper && <p className="text-sm text-slate-400 mt-1">{helper}</p>}
        </div>
        <div
          className={clsx(
            "h-12 w-12 rounded-xl bg-gradient-to-br flex items-center justify-center text-slate-900",
            toneMap[tone]
          )}
        >
          {icon}
        </div>
      </div>
    </div>
  );
}
