export interface TeamMember {
  title: string;
  name: string;
  email: string;
  url?: string;
}

export interface TeamOrg {
  org: string;
  description?: string;
  roles: TeamMember[];
}

export const team: TeamOrg[] = [
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
