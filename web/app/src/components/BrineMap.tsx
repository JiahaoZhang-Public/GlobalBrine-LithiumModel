import { useMemo, useState } from "react";
import Map, {
  Layer,
  NavigationControl,
  Source,
  type MapLayerMouseEvent,
} from "react-map-gl/maplibre";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { GeoFeature } from "../lib/types";
import { useI18n } from "../lib/i18n";

type Props = {
  data: GeoFeature[];
  meta: Record<string, any>;
  variable: string;
  onVariableChange: (v: string) => void;
  tdsRange: [number, number] | null;
  mlrRange: [number, number] | null;
};

const mapStyle =
  (import.meta.env.VITE_MAP_STYLE_URL as string | undefined) ||
  "https://demotiles.maplibre.org/style.json";

function inRange(val: number | null | undefined, range: [number, number] | null) {
  if (val === null || val === undefined || Number.isNaN(val)) return false;
  if (!range) return true;
  const [min, max] = range;
  const lowOk = min === null || min === undefined ? true : val >= min;
  const highOk = max === null || max === undefined ? true : val <= max;
  return lowOk && highOk;
}

export default function BrineMap({
  data,
  meta,
  variable,
  onVariableChange,
  tdsRange,
  mlrRange,
}: Props) {
  const [hover, setHover] = useState<GeoFeature | null>(null);
  const { t, lang } = useI18n();

  const filtered = useMemo(() => {
    return data.filter((f) => {
      const props = f.properties || {};
      const tds = Number(props["TDS_gL"]);
      const mlr = Number(props["MLR"]);
      const tdsOk = inRange(tds, tdsRange);
      const mlrOk = inRange(mlr, mlrRange);
      return tdsOk && mlrOk;
    });
  }, [data, mlrRange, tdsRange]);

  const domain =
    meta?.[variable] && meta[variable].min !== undefined
      ? [meta[variable].min, meta[variable].max]
      : [0, 1];

  const geojson = useMemo(
    () => ({ type: "FeatureCollection", features: filtered }),
    [filtered]
  );

  const paint: any = {
    "circle-radius": [
      "interpolate",
      ["linear"],
      ["zoom"],
      1,
      3,
      4,
      5,
      8,
      12,
    ],
    "circle-color": [
      "interpolate",
      ["linear"],
      ["coalesce", ["get", variable], 0],
      domain[0] ?? 0,
      "#22d3ee",
      (domain[0] + domain[1]) / 2,
      "#a855f7",
      domain[1] ?? 1,
      "#fb7185",
    ],
    "circle-stroke-color": "rgba(255,255,255,0.5)",
    "circle-stroke-width": 0.6,
    "circle-opacity": 0.85,
  };

  return (
    <div className="relative rounded-2xl overflow-hidden border border-white/10 glass">
      <div className="absolute z-10 left-4 top-4 flex flex-col gap-2">
        <div className="pill px-3 py-2 text-xs text-slate-900 bg-white/80">
          {filtered.length.toLocaleString()} {lang === "zh" ? "个点" : "sites"}
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-100 bg-black/50 px-3 py-2 rounded-xl border border-white/10 shadow-lg">
          {t("map.color.by")}
          <select
            className="bg-black/30 border border-white/10 rounded-lg px-2 py-1 text-slate-100"
            value={variable}
            onChange={(e) => onVariableChange(e.target.value)}
          >
            <option value="Pred_Selectivity">{t("map.option.selectivity")}</option>
            <option value="Pred_Li_Crystallization_mg_m2_h">
              {t("map.option.crystallization")}
            </option>
            <option value="Pred_Evap_kg_m2_h">{t("map.option.evap")}</option>
          </select>
        </label>
        <div className="bg-black/60 border border-white/10 rounded-xl px-3 py-2 text-xs text-slate-200">
          {t("map.legend")}
        </div>
      </div>

      <Map
        mapLib={maplibregl}
        initialViewState={{ longitude: -70, latitude: -15, zoom: 2.2 }}
        style={{ width: "100%", height: 540 }}
        mapStyle={mapStyle}
        interactiveLayerIds={["brine-layer"]}
        onMouseMove={(evt: MapLayerMouseEvent) => {
          const feature = evt.features?.[0] as any;
          setHover(feature?.properties ? (feature as GeoFeature) : null);
        }}
        onMouseLeave={() => setHover(null)}
      >
        <NavigationControl position="bottom-right" />
        <Source id="brines" type="geojson" data={geojson as any}>
          <Layer id="brine-layer" type="circle" paint={paint} />
        </Source>
      </Map>

      {hover && (
        <div className="absolute bottom-4 left-4 bg-black/70 rounded-xl border border-white/10 px-3 py-2 text-sm max-w-sm">
          <p className="font-semibold">{hover.properties?.Brine || (lang === "zh" ? t("map.sample") : "Sample")}</p>
          <p className="text-slate-300">
            {hover.properties?.Location} • MLR {hover.properties?.MLR ?? "–"} • TDS{" "}
            {hover.properties?.TDS_gL ?? "–"} g/L
          </p>
          <p className="text-slate-200 mt-1">
            {variable}: {hover.properties?.[variable]?.toFixed?.(3) ?? "–"}
          </p>
        </div>
      )}
    </div>
  );
}
