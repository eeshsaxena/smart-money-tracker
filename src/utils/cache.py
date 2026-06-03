"""File-based caching for API responses to avoid redundant calls."""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATA_DIR

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_TTL = 3600


def _cache_key(prefix: str, *args) -> str:
    raw = f"{prefix}:{'|'.join(str(a) for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached(prefix: str, *args, ttl: int = DEFAULT_TTL) -> Optional[pd.DataFrame]:
    key = _cache_key(prefix, *args)
    path = CACHE_DIR / f"{key}.parquet"
    meta_path = CACHE_DIR / f"{key}.meta"

    if not path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if time.time() - meta.get("timestamp", 0) > ttl:
            return None
        return pd.read_parquet(path)
    except Exception:
        return None


def set_cached(prefix: str, *args, data: pd.DataFrame):
    key = _cache_key(prefix, *args)
    path = CACHE_DIR / f"{key}.parquet"
    meta_path = CACHE_DIR / f"{key}.meta"

    try:
        data.to_parquet(path, index=False)
        with open(meta_path, "w") as f:
            json.dump({"timestamp": time.time(), "prefix": prefix}, f)
    except Exception:
        pass


def get_cached_json(prefix: str, *args, ttl: int = DEFAULT_TTL) -> Optional[dict]:
    key = _cache_key(prefix, *args)
    path = CACHE_DIR / f"{key}.json"

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        if time.time() - data.get("_timestamp", 0) > ttl:
            return None
        data.pop("_timestamp", None)
        return data
    except Exception:
        return None


def set_cached_json(prefix: str, *args, data: dict):
    key = _cache_key(prefix, *args)
    path = CACHE_DIR / f"{key}.json"

    try:
        data["_timestamp"] = time.time()
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def clear_cache():
    for f in CACHE_DIR.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass
