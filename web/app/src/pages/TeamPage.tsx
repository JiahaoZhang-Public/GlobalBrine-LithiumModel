import PageHeader from "../components/PageHeader";
import { useI18n } from "../lib/i18n";

const team = [
  {
    org: "MBZUAI — ML Department",
    roles: [
      {
        title: "Faculty mentor",
        name: "Lijie Hu",
        email: "lijie.hu@mbzuai.ac.ae",
        url: "https://lijie-hu.github.io/",
      },
      {
        title: "Implementation lead",
        name: "Jiahao Zhang",
        email: "jiahao.zhang@mbzuai.ac.ae",
        url: "https://jiahaozhang-public.github.io/",
      },
    ],
  },
  {
    org: "The Zongyao Zhou Group — Harbin Institute of Technology",
    description: "Global salt lake data collection and experimental measurements.",
    roles: [
      {
        title: "Group contact",
        name: "Zongyao Zhou",
        email: "zongyao.zhou@outlook.com",
        url: "https://homepage.hit.edu.cn/zhouzongyao",
      },
    ],
  },
];

export default function TeamPage() {
  const { t } = useI18n();
  return (
    <div className="space-y-6">
      <PageHeader
        title={t("team.title")}
        subtitle={t("team.subtitle")}
      />

      <div className="grid md:grid-cols-2 gap-4">
        {team.map((item) => (
          <div key={item.org} className="glass rounded-2xl border border-white/10 p-5 space-y-3">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-slate-400">{t("team.org")}</p>
              <h3 className="text-xl font-semibold mt-1">{item.org}</h3>
              {item.description && (
                <p className="text-slate-300 text-sm mt-1">{item.description}</p>
              )}
            </div>
            <div className="space-y-2">
              {item.roles.map((r) => (
                <div key={r.name} className="bg-white/5 border border-white/10 rounded-xl px-3 py-2">
                  <p className="text-sm text-slate-200 font-semibold">
                    {r.title === "Faculty mentor"
                      ? t("team.role.faculty")
                      : r.title === "Implementation lead"
                      ? t("team.role.lead")
                      : r.title === "Group contact"
                      ? t("team.role.contact")
                      : r.title}
                  </p>
                  <p className="text-slate-100">{r.name}</p>
                  {"url" in r && r.url && (
                    <a
                      className="text-sky-300 text-sm underline hover:text-sky-200"
                      href={r.url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {t("team.personalsite")}
                    </a>
                  )}
                  {r.email && (
                    <p className="text-xs text-slate-300 mt-1">Email: {r.email}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
