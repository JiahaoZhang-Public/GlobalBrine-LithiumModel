from __future__ import annotations

import math
import tempfile
from typing import Any


def attributions_barplot(attributions: dict[str, float] | None, *, title: str, top_n: int = 10):
    import matplotlib.pyplot as plt

    if not attributions:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No attributions available", ha="center", va="center")
        ax.axis("off")
        return fig

    items = sorted(attributions.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:top_n]
    labels = [k for k, _ in items][::-1]
    values = [float(v) for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(labels) + 1)))
    ax.barh(labels, values)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Attribution (raw units impact)")
    fig.tight_layout()
    return fig


def warnings_markdown(warnings: list[str] | None) -> str:
    if not warnings:
        return ""
    lines = "\n".join([f"- {w}" for w in warnings])
    return f"### Warnings\n{lines}"


def safe_get(d: dict[str, Any], *path: str, default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-safe values.

    - Replace NaN/Inf/-Inf with None
    - Convert numpy scalar types to Python scalars
    - Convert pandas NA to None
    - Convert datetime-like objects to ISO strings when possible
    """
    if obj is None:
        return None

    if isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    # numpy scalars
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return sanitize_for_json(obj.item())
        if isinstance(obj, np.ndarray):
            return sanitize_for_json(obj.tolist())
    except Exception:
        pass

    # dict / list recursion
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    # pandas NA / timestamps
    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # datetime/date objects
    iso = getattr(obj, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass

    # Last resort: stringify to avoid hard crashes in UI.
    return str(obj)


def write_temp_csv(df_like: Any, *, suffix: str = ".csv") -> str:
    import pandas as pd

    df = df_like if isinstance(df_like, pd.DataFrame) else pd.DataFrame(df_like)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    df.to_csv(tmp.name, index=False)
    return tmp.name


def _colorize_scalar(t: float) -> list[int]:
    # Simple blue -> red ramp.
    t = min(1.0, max(0.0, float(t)))
    r = int(round(255 * t))
    b = int(round(255 * (1.0 - t)))
    return [r, 0, b, 160]


def colorize_values(values: list[float | None], *, scale: str = "linear") -> list[list[int]]:
    scale = (scale or "linear").lower().strip()
    try:
        import numpy as np
    except Exception:
        # Best-effort fallback without numpy.
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
        vmin = min(finite) if finite else 0.0
        vmax = max(finite) if finite else 1.0
        out = []
        for v in values:
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                out.append([160, 160, 160, 120])
                continue
            t = 0.0 if vmax <= vmin else (float(v) - vmin) / (vmax - vmin)
            out.append(_colorize_scalar(t))
        return out

    arr = np.asarray([float(v) if v is not None else np.nan for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [[160, 160, 160, 120] for _ in values]

    if scale == "log":
        # Shift to positive then log10.
        minv = float(np.nanmin(finite))
        shift = 0.0 if minv > 0 else (abs(minv) + 1e-6)
        arr = np.log10(np.clip(arr + shift, 1e-6, None))
        finite = arr[np.isfinite(arr)]
        scale = "linear"

    if scale == "quantile":
        s = np.sort(finite)
        n = int(s.size)

        def q(v: float) -> float:
            # Percentile rank based on sorted order.
            j = int(np.searchsorted(s, v, side="left"))
            return float(j) / float(max(1, n - 1))

        colors: list[list[int]] = []
        for v in arr.tolist():
            if not math.isfinite(float(v)):
                colors.append([160, 160, 160, 120])
            else:
                colors.append(_colorize_scalar(q(float(v))))
        return colors

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    denom = (vmax - vmin) if vmax > vmin else 1.0
    out = []
    for v in arr.tolist():
        if not math.isfinite(float(v)):
            out.append([160, 160, 160, 120])
            continue
        t = (float(v) - vmin) / denom
        out.append(_colorize_scalar(t))
    return out
def build_plotly_geo_map(
    points: list[dict[str, Any]],
    *,
    metric: str,
    heatmap_points: list[dict[str, Any]] | None = None,
    point_size: int = 6,
    heat_point_size: int = 4,
    heat_opacity: float = 0.55,
    projection_type: str = "equirectangular",
    heat_marker_symbol: str = "circle",
) -> Any:
    """
    Plotly geo projection map (no tile basemap required).

    - Sample points are shown as markers with hover tooltips.
    - Heatmap grid (if provided) is shown as many semi-transparent colored markers.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as exc:
        return f"<missing plotly: {exc}>"

    lats = [p.get("lat") for p in points if isinstance(p.get("lat"), (int, float))]
    lons = [p.get("lon") for p in points if isinstance(p.get("lon"), (int, float))]

    fig = go.Figure()

    if heatmap_points:
        hlats = [p.get("lat") for p in heatmap_points if isinstance(p.get("lat"), (int, float))]
        hlons = [p.get("lon") for p in heatmap_points if isinstance(p.get("lon"), (int, float))]
        hvals = [p.get("value") for p in heatmap_points]
        fig.add_trace(
            go.Scattergeo(
                lat=hlats,
                lon=hlons,
                mode="markers",
                marker=dict(
                    size=int(heat_point_size),
                    opacity=float(heat_opacity),
                    symbol=str(heat_marker_symbol),
                    color=hvals,
                    colorscale="Viridis",
                    colorbar=dict(title=metric),
                ),
                name=f"Heatmap ({metric})",
                hovertemplate="lon=%{lon:.2f}<br>lat=%{lat:.2f}<br>value=%{marker.color:.4g}<extra></extra>",
            )
        )

    # Sample markers.
    metric_vals = [p.get("metric_value") for p in points]
    hover = []
    for p in points:
        raw = p.get("raw") or {}
        pred = p.get("prediction") or {}
        hover.append(
            "<br>".join(
                [
                    f"sample_id={p.get('sample_id')}",
                    f"Location={raw.get('Location')}",
                    f"{metric}={p.get('metric_value')}",
                    f"TDS_gL={raw.get('TDS_gL')}",
                    f"MLR={raw.get('MLR')}",
                    f"Light_kW_m2={raw.get('Light_kW_m2')}",
                    f"Pred_Selectivity={pred.get('Selectivity')}",
                    f"Pred_Li_Crystallization_mg_m2_h={pred.get('Li_Crystallization_mg_m2_h')}",
                    f"Pred_Evap_kg_m2_h={pred.get('Evap_kg_m2_h')}",
                ]
            )
        )

    rgba = []
    has_manual_colors = True
    for p in points:
        c = p.get("color")
        if not (isinstance(c, list) and len(c) == 4 and all(isinstance(x, (int, float)) for x in c)):
            has_manual_colors = False
            break
        r, g, b, a = c
        rgba.append(f"rgba({int(r)},{int(g)},{int(b)},{float(a)/255.0:.3f})")

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=dict(
                size=int(point_size),
                opacity=0.9,
                color=rgba if has_manual_colors else metric_vals,
                colorscale=None if has_manual_colors else "Turbo",
                showscale=False,
                line=dict(width=0),
            ),
            name="Samples",
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_geos(
        projection_type=str(projection_type),
        showland=True,
        landcolor="rgb(235, 235, 235)",
        showocean=True,
        oceancolor="rgb(210, 230, 255)",
        showcountries=True,
        countrycolor="rgb(180, 180, 180)",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=680,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_plotly_geo_raster_map(
    points: list[dict[str, Any]],
    *,
    metric: str,
    grid: dict[str, Any],
    heat_opacity: float = 0.75,
    point_size: int = 5,
    projection_type: str = "equirectangular",
) -> Any:
    """
    Raster-like plot on a Plotly geo projection (no tile basemap required).

    Intended for "paper-style" figures: a continuous field + overlaid known sites.
    """
    try:
        import numpy as np
        import plotly.graph_objects as go  # type: ignore
    except Exception as exc:
        return f"<missing plotly/numpy: {exc}>"

    lons = grid.get("lons") or []
    lats = grid.get("lats") or []
    z = grid.get("values") or []
    if not lons or not lats or not z:
        return "<grid missing lons/lats/values>"

    Z = np.asarray(z, dtype=float)
    # Grid centers (flatten).
    glon, glat = np.meshgrid(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    vals = Z.reshape(-1)
    lonf = glon.reshape(-1)
    latf = glat.reshape(-1)
    m = np.isfinite(vals)
    lonf = lonf[m]
    latf = latf[m]
    vals = vals[m]

    fig = go.Figure()
    # Approx marker size based on grid step.
    step_lon = float(abs(float(lons[1]) - float(lons[0]))) if len(lons) >= 2 else 2.0
    step_lat = float(abs(float(lats[1]) - float(lats[0]))) if len(lats) >= 2 else 2.0
    step = max(0.5, min(step_lon, step_lat))
    heat_size = max(4, min(18, int(round(12.0 / step))))

    fig.add_trace(
        go.Scattergeo(
            lon=lonf,
            lat=latf,
            mode="markers",
            marker=dict(
                size=int(heat_size),
                symbol="square",
                opacity=float(heat_opacity),
                color=vals,
                colorscale="Viridis",
                colorbar=dict(title=metric),
                line=dict(width=0),
            ),
            name=f"Heatmap ({metric})",
            hovertemplate="lon=%{lon:.2f}<br>lat=%{lat:.2f}<br>value=%{marker.color:.4g}<extra></extra>",
        )
    )

    # Overlay sample points.
    xs = [p.get("lon") for p in points]
    ys = [p.get("lat") for p in points]
    text = []
    for p in points:
        raw = p.get("raw") or {}
        pred = p.get("prediction") or {}
        text.append(
            "<br>".join(
                [
                    f"sample_id={p.get('sample_id')}",
                    f"Location={raw.get('Location')}",
                    f"{metric}={p.get('metric_value')}",
                    f"TDS_gL={raw.get('TDS_gL')}",
                    f"MLR={raw.get('MLR')}",
                    f"Light_kW_m2={raw.get('Light_kW_m2')}",
                    f"Pred_Selectivity={pred.get('Selectivity')}",
                    f"Pred_Li_Crystallization_mg_m2_h={pred.get('Li_Crystallization_mg_m2_h')}",
                    f"Pred_Evap_kg_m2_h={pred.get('Evap_kg_m2_h')}",
                ]
            )
        )

    fig.add_trace(
        go.Scattergeo(
            lon=xs,
            lat=ys,
            mode="markers",
            marker=dict(size=int(point_size), color="black", opacity=0.8),
            text=text,
            hovertemplate="%{text}<extra></extra>",
            name="Known sites",
        )
    )

    fig.update_geos(
        projection_type=str(projection_type),
        showland=True,
        landcolor="rgb(235, 235, 235)",
        showocean=True,
        oceancolor="rgb(210, 230, 255)",
        showcountries=True,
        countrycolor="rgb(180, 180, 180)",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=680)
    return fig

def build_latent_scatter(
    embeddings: list[list[float]],
    metadata: list[dict[str, Any]],
    *,
    color_by: str,
    point_size: int,
    top_k_categories: int = 10,
):
    """
    Return a plot object suitable for `gr.Plot` (Plotly if available; otherwise Matplotlib).
    """
    try:
        import pandas as pd
        import plotly.express as px  # type: ignore

        df = pd.DataFrame(
            {
                "x": [float(v[0]) for v in embeddings],
                "y": [float(v[1]) for v in embeddings],
                "sample_id": [m.get("sample_id") for m in metadata],
                "color": [m.get(color_by) if color_by and color_by != "none" else None for m in metadata],
            }
        )
        if color_by and color_by != "none":
            s = df["color"]
            non_null = s.dropna()
            # Heuristic: treat as numeric if most values parse as numbers.
            num = pd.to_numeric(non_null, errors="coerce")
            is_numeric = (num.notna().sum() >= int(0.8 * max(1, len(non_null)))) if len(non_null) else False
            if not is_numeric:
                counts = non_null.astype(str).value_counts()
                keep = set(counts.head(int(top_k_categories)).index.tolist())

                def _clip(v: Any) -> str | None:
                    if v is None:
                        return None
                    vv = str(v)
                    return vv if vv in keep else "Other"

                df["color"] = df["color"].map(_clip)
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="color" if color_by and color_by != "none" else None,
            hover_name="sample_id",
            opacity=0.85,
        )
        fig.update_traces(marker={"size": int(point_size)})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=640)
        return fig
    except Exception:
        import matplotlib.pyplot as plt

        xs = [float(v[0]) for v in embeddings]
        ys = [float(v[1]) for v in embeddings]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xs, ys, s=int(point_size) ** 2, alpha=0.7)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        ax.set_title("Latent space (2D)")
        fig.tight_layout()
        return fig
