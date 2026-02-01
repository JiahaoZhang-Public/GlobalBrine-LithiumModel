import type { PropsWithChildren } from "react";
import { Link, useLocation } from "react-router-dom";
import { Sparkles, Map, UploadCloud, FlaskConical, Info } from "lucide-react";
import clsx from "clsx";

const navItems = [
  { to: "/", label: "Overview", icon: Sparkles },
  { to: "/map", label: "Map Explorer", icon: Map },
  { to: "/predict", label: "Single Prediction", icon: FlaskConical },
  { to: "/batch", label: "Batch Jobs", icon: UploadCloud },
  { to: "/model", label: "Repro & Limits", icon: Info },
];

export default function Layout({ children }: PropsWithChildren) {
  const { pathname } = useLocation();
  return (
    <div className="min-h-screen text-slate-100 bg-ink bg-aurora">
      <div className="fixed inset-0 -z-10 bg-aurora opacity-80" />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <header className="flex items-center justify-between gap-4 py-3">
          <Link to="/" className="flex items-center gap-3">
            <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-sky-400 to-fuchsia-500 shadow-lg flex items-center justify-center text-xl font-bold">
              GB
            </div>
            <div>
              <p className="text-lg font-semibold">GlobalBrine</p>
              <p className="text-xs text-slate-300">Lithium selectivity for brine science</p>
            </div>
          </Link>
          <div className="hidden md:flex items-center gap-2 px-2 py-1 rounded-full border border-white/10 bg-white/5">
            {navItems.map((item) => {
              const active = pathname === item.to;
              const Icon = item.icon;
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={clsx(
                    "flex items-center gap-2 px-3 py-2 rounded-full text-sm transition-all",
                    active
                      ? "bg-white/15 text-white shadow-glass"
                      : "text-slate-300 hover:bg-white/10"
                  )}
                >
                  <Icon size={16} />
                  {item.label}
                </Link>
              );
            })}
          </div>
          <div className="flex items-center gap-2">
            <Link
              to="/model"
              className="pill px-3 py-2 text-sm text-slate-200 hover:bg-white/10 transition"
            >
              Methods & limits
            </Link>
            <Link
              to="/map"
              className="px-3 py-2 text-sm rounded-full bg-gradient-to-r from-sky-400 to-fuchsia-500 text-slate-900 font-semibold shadow-lg"
            >
              Launch explorer
            </Link>
          </div>
        </header>

        <main className="py-4">{children}</main>

        <footer className="py-8 text-sm text-slate-400 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <span>GlobalBrine-LithiumModel • Research-grade web</span>
          <span className="text-slate-500">
            Model version disclosed in-app • Reproducible by design
          </span>
        </footer>
      </div>
    </div>
  );
}
