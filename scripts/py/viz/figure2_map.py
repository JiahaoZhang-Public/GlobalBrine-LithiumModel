"""
python scripts/py/viz/figure2_map.py

To reproduce the figure 2 map, run:

python scripts/py/viz/figure2_map.py \
  --input data/predictions/brines_with_predictions.csv \
  --out reports/figures/figure2_map_evap.png \
  --no-ghi
  
"""
import argparse
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_EXTENT = [-180, 180, -60, 65]
FIGURE1_EXTENT = [-180, 180, -60, 80]
NO_DATA_COLOR = "#e6e6e6"

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Missing dependency: cartopy. Install it to render the world map."
    ) from exc

try:
    import rasterio
    from rasterio import Affine
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Missing dependency: rasterio. Install it to read the GHI GeoTIFF."
    ) from exc


def _format_range_labels(edges: np.ndarray, decimals: int = 2) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        low = f"{edges[i]:.{decimals}f}"
        high = f"{edges[i + 1]:.{decimals}f}"
        labels.append(f"{low}-{high}")
    return labels


def _compute_limits(
    values: np.ndarray, vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 0.0, 1.0
    if vmin is None or vmax is None:
        lo, hi = np.percentile(vals, [2, 98])
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if vmin >= vmax:
        vmin = float(vals.min())
        vmax = float(vals.max())
    return float(vmin), float(vmax)


def _grid_centers(vmin: float, vmax: float, step: float) -> np.ndarray:
    if step <= 0:
        return np.array([], dtype=float)
    start = float(vmin) + float(step) / 2
    end = float(vmax) - float(step) / 2
    if end < start:
        return np.array([], dtype=float)
    return np.arange(start, end + step * 0.5, step)


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


def _load_ghi_raster(
    path: str,
    extent: list[float],
    scale: float,
    subsample: int,
) -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    with rasterio.open(path) as src:
        if src.crs is not None and not src.crs.is_geographic:
            raise SystemExit(
                "GHI raster CRS is not geographic (lon/lat). "
                "Reproject to EPSG:4326 before plotting."
            )
        lon_min, lon_max, lat_min, lat_max = extent
        bounds = src.bounds
        lon_min = max(lon_min, bounds.left)
        lon_max = min(lon_max, bounds.right)
        lat_min = max(lat_min, bounds.bottom)
        lat_max = min(lat_max, bounds.top)
        if lon_min >= lon_max or lat_min >= lat_max:
            raise SystemExit("Requested extent is outside the GHI raster bounds.")
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        window = window.round_offsets().round_lengths()
        height = max(1, int(window.height))
        width = max(1, int(window.width))
        subsample = max(1, int(subsample))
        out_height = max(1, int(height / subsample))
        out_width = max(1, int(width / subsample))
        data = src.read(
            1,
            window=window,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear,
            masked=True,
        )
        transform = src.window_transform(window)
        transform = transform * Affine.scale(
            window.width / float(out_width),
            window.height / float(out_height),
        )

        x0, y0 = transform * (0, 0)
        resx = transform.a
        resy = transform.e
        lon_edges = x0 + np.arange(out_width + 1) * resx
        lat_edges = y0 + np.arange(out_height + 1) * resy

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        data = data[:, ::-1]
    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        data = data[::-1, :]

    data = np.ma.masked_invalid(data.astype(float) * float(scale))

    return lon_edges, lat_edges, data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 2: GHI background + evaporation rate field."
    )
    parser.add_argument(
        "--input",
        default="data/predictions/brines_with_predictions.csv",
        help="Input CSV with predictions and lat/lon columns.",
    )
    parser.add_argument(
        "--ghi-geotiff",
        required=False,
        help="Path to Global Solar Atlas GHI GeoTIFF.",
    )
    parser.add_argument(
        "--out",
        default="reports/figures/figure2_map.png",
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
        default=DEFAULT_EXTENT,
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
    parser.add_argument(
        "--ghi-scale",
        type=float,
        default=1.0,
        help="Multiply GHI raster by this scale (e.g., 1/24 to kW/m^2).",
    )
    parser.add_argument(
        "--ghi-subsample",
        type=int,
        default=100,
        help="Subsample GHI raster by this factor to speed plotting.",
    )
    parser.add_argument("--ghi-vmin", type=float, default=None, help="GHI color min.")
    parser.add_argument("--ghi-vmax", type=float, default=None, help="GHI color max.")
    parser.add_argument(
        "--no-ghi",
        action="store_true",
        help="Disable GHI background rendering.",
    )
    parser.add_argument(
        "--ghi-label",
        default="GHI (kWh m$^{-2}$)",
        help="GHI colorbar label.",
    )
    parser.add_argument(
        "--evap-label",
        default="Evaporation rate (kg m$^{-2}$ h$^{-1}$)",
        help="Evaporation colorbar label.",
    )
    parser.add_argument(
        "--ghi-cmap",
        default="YlOrBr",
        help="Matplotlib colormap for GHI.",
    )
    parser.add_argument(
        "--evap-cmap",
        default="custom",
        help="Matplotlib colormap for evaporation field (or 'custom').",
    )
    parser.add_argument(
        "--evap-bins",
        type=int,
        default=6,
        help="Number of evaporation ranges.",
    )
    parser.add_argument(
        "--evap-alpha",
        type=float,
        default=0.55,
        help="Alpha for evaporation overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extent = list(args.extent)
    if args.no_ghi and extent == DEFAULT_EXTENT:
        extent = FIGURE1_EXTENT
    df = pd.read_csv(args.input)
    df = df.dropna(
        subset=[
            "Latitude",
            "Longitude",
            "Pred_Evap_kg_m2_h",
        ]
    )

    lat = df["Latitude"].to_numpy(dtype=float)
    lon = df["Longitude"].to_numpy(dtype=float)
    evap = df["Pred_Evap_kg_m2_h"].to_numpy(dtype=float)

    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(evap)
    lat = lat[mask]
    lon = lon[mask]
    evap = evap[mask]

    # Collapse duplicate coordinates by mean value for stability.
    key = np.stack([np.round(lat, 5), np.round(lon, 5)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    if uniq.shape[0] != key.shape[0]:
        sums = np.zeros((uniq.shape[0],), dtype=np.float64)
        counts = np.zeros((uniq.shape[0],), dtype=np.int64)
        for i, g in enumerate(inv):
            sums[g] += float(evap[i])
            counts[g] += 1
        lat = uniq[:, 0].astype(np.float32)
        lon = uniq[:, 1].astype(np.float32)
        evap = (sums / np.maximum(1, counts)).astype(np.float32)

    grid_step = float(args.grid_deg)
    lons = _grid_centers(extent[0], extent[1], grid_step)
    lats = _grid_centers(extent[2], extent[3], grid_step)
    if lons.size == 0 or lats.size == 0:
        raise SystemExit("Empty grid (check --extent/--grid-deg).")
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_lon_f = grid_lon.reshape(-1)
    grid_lat_f = grid_lat.reshape(-1)

    max_dist = None if float(args.max_distance_km) <= 0 else float(args.max_distance_km)
    interp, _ = _interpolate_knn_idw(
        sample_lat=lat,
        sample_lon=lon,
        sample_val=evap,
        grid_lat=grid_lat_f,
        grid_lon=grid_lon_f,
        method=str(args.method),
        k=int(args.k),
        power=float(args.power),
        max_distance_km=max_dist,
    )
    evap_grid = interp.reshape((len(lats), len(lons)))
    evap_masked = np.ma.masked_invalid(evap_grid)
    evap_vmin = float(np.nanmin(evap_masked))
    evap_vmax = float(np.nanmax(evap_masked))
    if evap_vmin >= evap_vmax:
        evap_vmin, evap_vmax = _compute_limits(evap_masked.compressed(), None, None)
    bin_count = max(2, int(args.evap_bins))
    evap_bins = np.linspace(evap_vmin, evap_vmax, bin_count + 1)
    evap_labels = _format_range_labels(evap_bins, decimals=2)
    palette = ["#4575B4", "#91BFDB", "#E0F3F8", "#FEE090", "#FC8D59", "#D73027"]
    if str(args.evap_cmap).lower() == "custom":
        evap_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "evap", palette, N=256
        )
    else:
        evap_cmap = mpl.colormaps.get_cmap(args.evap_cmap)
    evap_norm = mpl.colors.Normalize(vmin=evap_bins[0], vmax=evap_bins[-1])
    evap_cmap.set_bad(NO_DATA_COLOR)

    fig = plt.figure(figsize=tuple(args.figsize), dpi=args.dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=NO_DATA_COLOR, edgecolor="none", zorder=0.1)
    ghi_mesh = None
    if not args.no_ghi:
        if not args.ghi_geotiff:
            raise SystemExit("--ghi-geotiff is required unless --no-ghi is set.")
        ghi_lon_edges, ghi_lat_edges, ghi_data = _load_ghi_raster(
            args.ghi_geotiff,
            extent,
            args.ghi_scale,
            max(1, int(args.ghi_subsample)),
        )
        ghi_vals = ghi_data.compressed()
        ghi_vmin, ghi_vmax = _compute_limits(ghi_vals, args.ghi_vmin, args.ghi_vmax)
        ghi_cmap = mpl.colormaps.get_cmap(args.ghi_cmap)
        if hasattr(ghi_cmap, "copy"):
            ghi_cmap = ghi_cmap.copy()
        ghi_cmap.set_bad(NO_DATA_COLOR)
        ghi_norm = mpl.colors.Normalize(vmin=ghi_vmin, vmax=ghi_vmax)
        ghi_mesh = ax.pcolormesh(
            ghi_lon_edges,
            ghi_lat_edges,
            ghi_data,
            cmap=ghi_cmap,
            norm=ghi_norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
            zorder=0.5,
        )

    evap_vals = evap_masked.reshape(-1)
    evap_lon = grid_lon.reshape(-1)
    evap_lat = grid_lat.reshape(-1)
    valid = np.isfinite(evap_vals)
    evap_vals = evap_vals[valid]
    evap_lon = evap_lon[valid]
    evap_lat = evap_lat[valid]

    step = max(0.5, float(args.grid_deg))
    heat_size = max(4, min(18, int(round(12.0 / step))))
    ax.scatter(
        evap_lon,
        evap_lat,
        s=int(heat_size) ** 2,
        c=evap_vals,
        cmap=evap_cmap,
        norm=evap_norm,
        marker="s",
        edgecolor="none",
        alpha=float(args.evap_alpha),
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.5", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="0.6", zorder=3)

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="0.6", alpha=0.6, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    ax.spines["geo"].set_linewidth(1.2)
    ax.spines["geo"].set_edgecolor("black")

    if ghi_mesh is not None:
        cbar_ghi = fig.colorbar(
            ghi_mesh,
            ax=ax,
            orientation="horizontal",
            fraction=0.04,
            pad=0.06,
            shrink=0.9,
        )
        cbar_ghi.ax.tick_params(labelsize=8)
        cbar_ghi.set_label(args.ghi_label, fontsize=9)

    evap_ticks = [
        (evap_bins[i] + evap_bins[i + 1]) / 2 for i in range(len(evap_bins) - 1)
    ]
    evap_sm = mpl.cm.ScalarMappable(norm=evap_norm, cmap=evap_cmap)
    evap_sm.set_array([])
    cbar_evap = fig.colorbar(
        evap_sm,
        ax=ax,
        orientation="vertical",
        fraction=0.03,
        pad=0.03,
        shrink=0.82,
    )
    cbar_evap.set_ticks(evap_ticks)
    cbar_evap.set_ticklabels(evap_labels)
    cbar_evap.ax.tick_params(labelsize=8)
    cbar_evap.set_label(args.evap_label, fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
