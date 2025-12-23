from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Point = Tuple[float, float]  # (lon, lat)


@dataclass(frozen=True)
class BBox:
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    def to_polygon(self) -> List[Point]:
        return [
            (self.lon_min, self.lat_min),
            (self.lon_max, self.lat_min),
            (self.lon_max, self.lat_max),
            (self.lon_min, self.lat_max),
            (self.lon_min, self.lat_min),
        ]


def parse_bbox_param(bbox: str) -> BBox:
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'min_lon,min_lat,max_lon,max_lat'")
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in parts)
    if lon_min > lon_max or lat_min > lat_max:
        raise ValueError("bbox min values must be <= max values")
    return BBox(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """
    Ray casting algorithm for point-in-polygon.
    polygon is a sequence of (lon, lat); may be open or closed.
    """
    x, y = point
    if len(polygon) < 3:
        return False
    # Ensure closed ring.
    pts = list(polygon)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    inside = False
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        # Check edge intersects ray.
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def filter_points_in_polygon(
    points: Iterable[Tuple[float, float, object]], polygon: Sequence[Point]
) -> List[object]:
    """
    points yields (lon, lat, payload); returns payloads inside polygon.
    """
    out: List[object] = []
    for lon, lat, payload in points:
        if point_in_polygon((float(lon), float(lat)), polygon):
            out.append(payload)
    return out

