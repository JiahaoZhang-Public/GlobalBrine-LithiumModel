export default function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-slate-300">
      <div className="h-3 w-3 rounded-full bg-sky-400 animate-ping" />
      <span>{label}</span>
    </div>
  );
}
