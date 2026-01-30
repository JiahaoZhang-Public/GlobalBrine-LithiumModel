import type { ReactNode } from "react";

type Props = {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
};

export default function PageHeader({ title, subtitle, actions }: Props) {
  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-6">
      <div>
        <h1 className="text-3xl font-semibold">{title}</h1>
        {subtitle && <p className="text-slate-300 mt-1">{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}
