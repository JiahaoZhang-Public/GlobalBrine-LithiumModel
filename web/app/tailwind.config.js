/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Space Grotesk"', '"Source Sans 3"', "ui-sans-serif", "system-ui"],
      },
      colors: {
        ink: "#0B1021",
        slate: "#0f172a",
        neon: "#7dd3fc",
        magenta: "#a855f7",
        lime: "#a3e635",
      },
      boxShadow: {
        glass: "0 15px 60px rgba(0,0,0,0.35)",
      },
      backgroundImage: {
        aurora:
          "radial-gradient(circle at 10% 20%, rgba(124,58,237,0.35), transparent 35%), radial-gradient(circle at 80% 0%, rgba(45,212,191,0.35), transparent 30%), radial-gradient(circle at 50% 100%, rgba(59,130,246,0.25), transparent 35%)",
      },
    },
  },
  plugins: [],
};
