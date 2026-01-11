#!/usr/bin/env python3
"""
python scripts/py/viz/figure1_map.py

To reproduce the figure 1 map, run:

python scripts/py/viz/figure1_map.py \
  --input data/predictions/brines_with_predictions.csv \
  --out reports/figures/figure1_map.png \
  --grid-deg 2 \
  --method idw \
  --k 8 \
  --power 2 \
  --max-distance-km 1500
"""

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Missing dependency: cartopy. Install it to render the world map."
    ) from exc


def _crystallization_bins(values: np.ndarray, step: float = 0.05) -> np.ndarray:
    max_val = float(np.nanmax(values))
    upper = math.ceil(max_val / step) * step
    edges = np.arange(0.0, upper + step * 0.5, step)
    if edges[-1] < max_val:
        edges = np.append(edges, max_val)
    return edges


def _selectivity_bins(values: np.ndarray, step: float = 5.0) -> np.ndarray:
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    start = math.floor(min_val / step) * step
    end = math.ceil(max_val / step) * step
    edges = np.arange(start, end + step * 0.5, step)
    if len(edges) < 3:
        edges = np.array([start, start + step, end])
    return edges


def _grid_centers(vmin: float, vmax: float, step: float) -> np.ndarray:
    if step <= 0:
        return np.array([], dtype=float)
    start = float(vmin) + float(step) / 2
    end = float(vmax) - float(step) / 2
    if end < start:
        return np.array([], dtype=float)
    return np.arange(start, end + step * 0.5, step)


def _grid_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return centers
    if centers.size == 1:
        step = 1.0
        return np.array([centers[0] - step / 2, centers[0] + step / 2], dtype=float)
    step = float(np.median(np.diff(centers)))
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - step / 2
    edges[-1] = centers[-1] + step / 2
    return edges


def _format_range_labels(edges: np.ndarray, decimals: int = 2) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        low = f"{edges[i]:.{decimals}f}"
        high = f"{edges[i + 1]:.{decimals}f}"
        labels.append(f"{low}-{high}")
    return labels


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _interpolate_knn_idw(
    *,
    sample_lat: np.ndarray,
    sample_lon: np.ndarray,
    sample_val: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    method: str,
    k: int,
    power: float,
    max_distance_km: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.neighbors import BallTree  # type: ignore

        X = np.stack([np.deg2rad(sample_lat), np.deg2rad(sample_lon)], axis=1)
        Q = np.stack([np.deg2rad(grid_lat), np.deg2rad(grid_lon)], axis=1)
        tree = BallTree(X, metric="haversine")
        kk = min(int(k), int(X.shape[0]))
        dist_rad, ind = tree.query(Q, k=kk)
        dist_km = dist_rad * 6371.0
        nn_km = dist_km[:, 0]
        vals = sample_val[ind]
        if method == "knn":
            pred = vals.mean(axis=1)
        else:
            w = 1.0 / (np.power(dist_km, float(power)) + 1e-6)
            pred = (w * vals).sum(axis=1) / w.sum(axis=1)
    except Exception:
        lat1 = np.deg2rad(grid_lat).reshape(-1, 1)
        lon1 = np.deg2rad(grid_lon).reshape(-1, 1)
        lat2 = np.deg2rad(sample_lat).reshape(1, -1)
        lon2 = np.deg2rad(sample_lon).reshape(1, -1)
        d = _haversine_km(lat1, lon1, lat2, lon2)
        nn_km = d.min(axis=1)
        kk = min(int(k), d.shape[1])
        idx = np.argpartition(d, kth=kk - 1, axis=1)[:, :kk]
        d_k = np.take_along_axis(d, idx, axis=1)
        v_k = sample_val[idx]
        if method == "knn":
            pred = v_k.mean(axis=1)
        else:
            w = 1.0 / (np.power(d_k, float(power)) + 1e-6)
            pred = (w * v_k).sum(axis=1) / w.sum(axis=1)

    if max_distance_km is not None:
        pred = np.where(nn_km <= float(max_distance_km), pred, np.nan)
    return pred.astype(np.float32), nn_km.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 1: global Li crystallization + selectivity map."
    )
    parser.add_argument(
        "--input",
        default="data/predictions/brines_with_predictions.csv",
        help="Input CSV with predictions and lat/lon columns.",
    )
    parser.add_argument(
        "--out",
        default="reports/figures/figure1_map.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--grid-deg",
        type=float,
        default=2.0,
        help="Interpolation grid size in degrees.",
    )
    parser.add_argument(
        "--method",
        choices=["idw", "knn"],
        default="idw",
        help="Interpolation method (idw or knn).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of nearest neighbors for interpolation.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.0,
        help="IDW power (method=idw).",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=1500.0,
        help="Mask grid cells farther than this (km); <=0 disables.",
    )
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        default=[-180, 180, -60, 80],
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Map extent in degrees.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output dpi.")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[11, 6],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df.dropna(
        subset=[
            "Latitude",
            "Longitude",
            "Pred_Li_Crystallization_mg_m2_h",
            "Pred_Selectivity",
        ]
    )

    lat = df["Latitude"].to_numpy(dtype=float)
    lon = df["Longitude"].to_numpy(dtype=float)
    crystallization = df["Pred_Li_Crystallization_mg_m2_h"].to_numpy(dtype=float)
    selectivity = df["Pred_Selectivity"].to_numpy(dtype=float)

    mask = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(crystallization)
        & np.isfinite(selectivity)
    )
    lat = lat[mask]
    lon = lon[mask]
    crystallization = crystallization[mask]
    selectivity = selectivity[mask]

    cryst_bins = _crystallization_bins(crystallization, step=0.05)
    cryst_cmap = mpl.colormaps.get_cmap("RdYlBu_r")
    if hasattr(cryst_cmap, "copy"):
        cryst_cmap = cryst_cmap.copy()
    no_data_color = "#e6e6e6"
    cryst_cmap.set_bad((0, 0, 0, 0))
    cryst_norm = mpl.colors.Normalize(vmin=cryst_bins[0], vmax=cryst_bins[-1])

    grid_step = float(args.grid_deg)
    lons = _grid_centers(args.extent[0], args.extent[1], grid_step)
    lats = _grid_centers(args.extent[2], args.extent[3], grid_step)
    if lons.size == 0 or lats.size == 0:
        raise SystemExit("Empty grid (check --extent/--grid-deg).")
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_lon_f = grid_lon.reshape(-1)
    grid_lat_f = grid_lat.reshape(-1)

    sample_lat = lat.copy()
    sample_lon = lon.copy()
    sample_val = crystallization.copy()
    key = np.stack([np.round(sample_lat, 5), np.round(sample_lon, 5)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    if uniq.shape[0] != key.shape[0]:
        sums = np.zeros((uniq.shape[0],), dtype=np.float64)
        counts = np.zeros((uniq.shape[0],), dtype=np.int64)
        for i, g in enumerate(inv):
            sums[g] += float(sample_val[i])
            counts[g] += 1
        sample_lat = uniq[:, 0].astype(np.float32)
        sample_lon = uniq[:, 1].astype(np.float32)
        sample_val = (sums / np.maximum(1, counts)).astype(np.float32)

    max_dist = None if float(args.max_distance_km) <= 0 else float(args.max_distance_km)
    interp, _ = _interpolate_knn_idw(
        sample_lat=sample_lat,
        sample_lon=sample_lon,
        sample_val=sample_val,
        grid_lat=grid_lat_f,
        grid_lon=grid_lon_f,
        method=str(args.method),
        k=int(args.k),
        power=float(args.power),
        max_distance_km=max_dist,
    )
    grid = interp.reshape((len(lats), len(lons)))
    grid_masked = np.ma.masked_invalid(grid)
    lon_edges = _grid_edges(lons)
    lat_edges = _grid_edges(lats)

    sel_bins = _selectivity_bins(selectivity, step=5.0)
    sel_labels = _format_range_labels(sel_bins, decimals=1)
    sel_colors = mpl.colormaps.get_cmap("OrRd")(
        np.linspace(0.35, 0.95, len(sel_bins) - 1)
    )
    sel_sizes = np.linspace(30, 170, len(sel_bins) - 1)
    sel_codes = pd.Categorical(
        pd.cut(selectivity, bins=sel_bins, include_lowest=True),
        ordered=True,
    ).codes

    fig = plt.figure(figsize=tuple(args.figsize), dpi=args.dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(args.extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=no_data_color, edgecolor="none", zorder=0.1)

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        grid_masked,
        cmap=cryst_cmap,
        norm=cryst_norm,
        shading="auto",
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.5", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.6", zorder=3)

    for i in range(len(sel_bins) - 1):
        mask = sel_codes == i
        if not np.any(mask):
            continue
        ax.scatter(
            lon[mask],
            lat[mask],
            s=sel_sizes[i],
            color=sel_colors[i],
            edgecolor="black",
            linewidth=0.3,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="0.6", alpha=0.6, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    ax.spines["geo"].set_linewidth(1.2)
    ax.spines["geo"].set_edgecolor("black")

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="vertical",
        fraction=0.035,
        pad=0.02,
        shrink=0.82,
    )
    cryst_labels = _format_range_labels(cryst_bins, decimals=2)
    cryst_ticks = [
        (cryst_bins[i] + cryst_bins[i + 1]) / 2 for i in range(len(cryst_bins) - 1)
    ]
    cbar.set_ticks(cryst_ticks)
    cbar.set_ticklabels(cryst_labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(
        "Li+ crystallization rate (mg m$^{-2}$ h$^{-1}$)", fontsize=9
    )

    handles = []
    for i in range(len(sel_bins) - 1):
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=math.sqrt(sel_sizes[i]),
                markerfacecolor=sel_colors[i],
                markeredgecolor="black",
                markeredgewidth=0.4,
            )
        )
    legend = ax.legend(
        handles,
        sel_labels,
        title="Li+/Mg2+ selectivity",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )
    legend.get_frame().set_edgecolor("black")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
