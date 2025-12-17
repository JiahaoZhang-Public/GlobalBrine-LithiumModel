from __future__ import annotations

import re
from typing import Optional, Tuple

Manual_Points: dict[str, tuple[float, float]] = {
    "Israel": (31.5, 34.8),
    "Jordan": (31.2, 36.3),
    "USA": (39.8, -98.6),
    "AK,USA": (64.2, -149.5),
    "Argentina": (-34.6, -58.4),
    "Australia": (-25.3, 133.8),
    "Southern Puna": (-24.0, -67.0),
    "Tunisia": (34.0, 9.0),
    "Iran": (32.0, 53.0),
    "Bolivia": (-16.5, -64.8),
    "Chile": (-30.0, -71.0),
    "China": (35.0, 103.0),
    "Louisiana, USA": (31.0, -92.0),
    "Northern Chile": (-22.5, -69.3),
    "Tibet, China": (31.5, 88.0),
    "Sydney,Australia": (-33.87, 151.21),
    "Liaoning Province, P.R. China": (41.5, 123.0),
    "offshore Beihai City, Guangxi Province, China": (21.4, 109.3),
    "Germany": (51.2, 10.5),
    "Italy": (42.5, 12.5),
    "England": (52.3, -1.5),
    "France": (46.6, 2.2),
    "Tuzla, Bosnia and Herzegovina": (44.54, 18.67),
    "Poland": (52.0, 19.0),
    "west Poland": (52.0, 15.0),
    "Jianghan Basin, China": (30.5, 112.0),
    "Mississippi, USA": (32.6, -89.7),
    "Alabama, USA": (32.3, -86.9),
    "Texas, USA": (31.0, -99.0),
    "Arkansas, USA": (35.0, -92.5),
    "Alberta, Canada": (54.5, -115.0),
}


_CHAR_REPLACEMENTS = {
    "’": "'",
    "‘": "'",
    "`": "'",
    "´": "'",
    "′": "'",
    "‵": "'",
    "ʹ": "'",
    "˝": '"',
    "″": '"',
    "“": '"',
    "”": '"',
    "º": "°",
}


def normalize_location_text(text: str) -> str:
    out = text.strip()
    for src, dst in _CHAR_REPLACEMENTS.items():
        out = out.replace(src, dst)
    out = re.sub(r"\s+", " ", out)
    # normalize some common separators
    out = out.replace(" / ", ", ")
    out = out.replace(" /", ", ")
    out = out.replace("/ ", ", ")
    return out.strip()


def dms_to_decimal(deg: float, minutes: float = 0.0, seconds: float = 0.0) -> float:
    return float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0


_DMS_RE = re.compile(
    r"""
    (?P<deg>\d{1,3})\s*°\s*
    (?:(?P<min>\d{1,2})\s*'?\s*)?
    (?:(?P<sec>\d{1,2}(?:\.\d+)?)\s*"?\s*)?
    (?P<hem>
        [NSEW]|
        North|South|East|West
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DECIMAL_HEM_RE = re.compile(
    r"""
    (?P<val>\d{1,3}(?:\.\d+)?)\s*
    (?P<hem>[NSEW]|North|South|East|West)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _hemisphere_sign(hem: str) -> int:
    h = hem.strip().lower()
    if h in {"s", "south", "w", "west"}:
        return -1
    return 1


def _hemisphere_axis(hem: str) -> str:
    h = hem.strip().lower()
    if h in {"n", "north", "s", "south"}:
        return "lat"
    return "lon"


def _parse_lat_lon_from_text(text: str) -> tuple[Optional[float], Optional[float]]:
    lat: Optional[float] = None
    lon: Optional[float] = None

    for m in _DMS_RE.finditer(text):
        deg = float(m.group("deg"))
        minutes = float(m.group("min") or 0.0)
        seconds = float(m.group("sec") or 0.0)
        hem = m.group("hem")
        value = dms_to_decimal(deg, minutes, seconds) * _hemisphere_sign(hem)
        axis = _hemisphere_axis(hem)
        if axis == "lat" and lat is None:
            lat = value
        elif axis == "lon" and lon is None:
            lon = value

    # fallback: decimal degrees with hemisphere (rare)
    if lat is None or lon is None:
        for m in _DECIMAL_HEM_RE.finditer(text):
            val = float(m.group("val"))
            hem = m.group("hem")
            value = val * _hemisphere_sign(hem)
            axis = _hemisphere_axis(hem)
            if axis == "lat" and lat is None:
                lat = value
            elif axis == "lon" and lon is None:
                lon = value

    return lat, lon


def location_to_lat_lon(location: object) -> Tuple[Optional[float], Optional[float]]:
    """Convert a free-text location into (lat, lon) if possible.

    Priority:
    1) Parse explicit coordinates from the string (DMS / decimal with hemisphere).
    2) Fall back to a small manual dictionary for place names.
    """
    if location is None:
        return None, None
    if not isinstance(location, str):
        return None, None
    raw = location.strip()
    if raw == "" or raw.lower() == "nan":
        return None, None

    text = normalize_location_text(raw)
    lat, lon = _parse_lat_lon_from_text(text)
    if lat is not None and lon is not None:
        return lat, lon

    # Manual lookup (try normalized variants)
    key = text
    if key in Manual_Points:
        return Manual_Points[key]
    key = key.replace(" ,", ",").strip()
    if key in Manual_Points:
        return Manual_Points[key]
    key = key.lstrip()
    if key in Manual_Points:
        return Manual_Points[key]

    return None, None
