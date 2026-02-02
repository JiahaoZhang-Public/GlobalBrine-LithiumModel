import type { PropsWithChildren } from "react";
import { Link, useLocation } from "react-router-dom";
import { Sparkles, Map, UploadCloud, FlaskConical, Info, Users, Languages } from "lucide-react";
import clsx from "clsx";
import { useI18n } from "../lib/i18n";

const navItems = [
  { to: "/", labelKey: "nav.overview", icon: Sparkles },
  { to: "/map", labelKey: "nav.map", icon: Map },
  { to: "/predict", labelKey: "nav.single", icon: FlaskConical },
  { to: "/batch", labelKey: "nav.batch", icon: UploadCloud },
  { to: "/model", labelKey: "nav.model", icon: Info },
  { to: "/team", labelKey: "nav.team", icon: Users },
];

export default function Layout({ children }: PropsWithChildren) {
  const { pathname } = useLocation();
  const { t, lang, setLang } = useI18n();
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
              <p className="text-xs text-slate-300">{t("layout.tagline")}</p>
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
                  {t(item.labelKey as any)}
                </Link>
              );
            })}
          </div>
          <div className="flex items-center gap-2">
            <a
              href="https://github.com/JiahaoZhang-Public/GlobalBrine-LithiumModel"
              target="_blank"
              rel="noreferrer"
              className="pill px-3 py-2 text-sm text-slate-200 hover:bg-white/10 transition"
            >
              {t("layout.repo")}
            </a>
            <Link
              to="/model"
              className="pill px-3 py-2 text-sm text-slate-200 hover:bg-white/10 transition"
            >
              {t("layout.methods")}
            </Link>
            <Link
              to="/map"
              className="px-3 py-2 text-sm rounded-full bg-gradient-to-r from-sky-400 to-fuchsia-500 text-slate-900 font-semibold shadow-lg"
            >
              {t("layout.launch")}
            </Link>
            <button
              onClick={() => setLang(lang === "en" ? "zh" : "en")}
              className="pill px-3 py-2 text-sm text-slate-200 hover:bg-white/10 transition inline-flex items-center gap-1"
            >
              <Languages size={16} /> {lang === "en" ? t("lang.toggle") : t("lang.toggle.zh")}
            </button>
          </div>
        </header>

        <main className="py-4">{children}</main>

        <footer className="py-8 text-sm text-slate-400 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <span>{t("layout.footer.left")}</span>
          <span className="text-slate-500">
            {t("layout.footer.right")}
          </span>
        </footer>
      </div>
    </div>
  );
}
