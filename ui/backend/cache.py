from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


def _repo_root() -> Path:
    # ui/backend/cache.py -> ui/backend -> ui -> repo root
    return Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    root = _repo_root()
    env = os.environ.get("UI_CACHE_DIR", "").strip()
    if env:
        p = Path(env)
        return p if p.is_absolute() else (root / p)
    return root / "ui" / "backend" / "cached"


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class CacheKey:
    namespace: str
    key: str

    def filename_prefix(self) -> str:
        return f"{self.namespace}__{self.key}"


def cache_key(namespace: str, *, params: dict[str, Any]) -> CacheKey:
    return CacheKey(namespace=namespace, key=stable_hash(params))


def _paths(k: CacheKey) -> Tuple[Path, Path]:
    base = cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{k.filename_prefix()}.json", base / f"{k.filename_prefix()}.npy"


def load_cached(k: CacheKey) -> Optional[Tuple[dict[str, Any], Any]]:
    meta_path, arr_path = _paths(k)
    if not meta_path.exists() or not arr_path.exists():
        return None
    try:
        import numpy as np
    except Exception:
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    arr = np.load(arr_path, allow_pickle=False)
    return meta, arr


def save_cached(k: CacheKey, *, meta: dict[str, Any], array) -> None:
    meta_path, arr_path = _paths(k)
    tmp_meta = meta_path.with_suffix(".tmp.json")
    # Important: np.save appends ".npy" unless the filename ends with ".npy".
    # So the temp path must also end with ".npy" to avoid writing to a different file.
    tmp_arr = arr_path.with_suffix(".tmp.npy")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for caching arrays") from exc
    np.save(tmp_arr, array, allow_pickle=False)
    os.replace(tmp_meta, meta_path)
    os.replace(tmp_arr, arr_path)
