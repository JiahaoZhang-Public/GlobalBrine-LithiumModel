from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class GeoTiffError(RuntimeError):
    pass


def _require_rasterio():
    try:
        import rasterio  # type: ignore
        from rasterio.warp import transform  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise GeoTiffError(
            "GeoTIFF sampling requires the optional dependency 'rasterio'. "
            "Install it (e.g. `pip install rasterio`) or omit --light-geotiff."
        ) from exc
    return rasterio, transform


@dataclass
class GeoTiffSampler:
    """Sample a single-band GeoTIFF at WGS84 (lat, lon) points."""

    path: Path
    band: int = 1

    def __post_init__(self) -> None:
        rasterio, _ = _require_rasterio()
        self._ds = rasterio.open(self.path)
        self._crs = self._ds.crs

    def close(self) -> None:
        self._ds.close()

    def __enter__(self) -> GeoTiffSampler:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def sample_lat_lon(self, lat: float, lon: float) -> Optional[float]:
        rasterio, transform = _require_rasterio()

        x, y = float(lon), float(lat)
        if self._crs is not None and str(self._crs).upper() != "EPSG:4326":
            xs, ys = transform("EPSG:4326", self._crs, [x], [y])
            x, y = float(xs[0]), float(ys[0])

        bounds = self._ds.bounds
        if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
            return None

        sample_iter = self._ds.sample([(x, y)], indexes=self.band, masked=True)
        arr = next(sample_iter)
        value = float(arr[0])
        if getattr(arr, "mask", None) is not None and bool(arr.mask[0]):
            return None

        nodata = self._ds.nodata
        if nodata is not None and value == float(nodata):
            return None
        if not (value == value):  # NaN
            return None
        return value


def sample_light_kW_m2(
    geotiff_path: Path,
    *,
    lat: Optional[float],
    lon: Optional[float],
    band: int = 1,
    scale: float = 1.0,
) -> Optional[float]:
    """Convenience sampler for a single point.

    Use `scale` to convert raw GeoTIFF units into `kW/m^2` if needed.
    """
    if lat is None or lon is None:
        return None
    with GeoTiffSampler(geotiff_path, band=band) as sampler:
        value = sampler.sample_lat_lon(lat, lon)
    if value is None:
        return None
    return float(value) * float(scale)
