"""Daily rainfall from the Open-Meteo archive API (free, no key), cached per cell."""
from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import requests

API = "https://archive-api.open-meteo.com/v1/archive"


def cell_key(lat: float, lon: float, deg: float):
    return (round(round(lat / deg) * deg, 4), round(round(lon / deg) * deg, 4))


def cell_file(cfg, key):
    return cfg.path("data", "raw", "rainfall", f"{key[0]:.4f}_{key[1]:.4f}.csv")


def fetch_cell(lat, lon, start, end, retries=8) -> pd.DataFrame:
    params = dict(
        latitude=lat, longitude=lon, start_date=start, end_date=end,
        daily="precipitation_sum", timezone="Asia/Kolkata",
    )
    wait = 5
    for attempt in range(retries):
        last = attempt == retries - 1
        try:
            r = requests.get(API, params=params, timeout=120)
        except requests.RequestException as err:      # dropped or reset connection
            if last:
                raise RuntimeError(
                    f"Rainfall download kept failing ({type(err).__name__}). Check your "
                    "connection and run the same command again; finished cells are kept.") from err
            print(f"    connection problem ({type(err).__name__}); waiting {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 300)
            continue
        if r.status_code == 200:
            d = r.json()["daily"]
            return pd.DataFrame({"date": d["time"], "rain_mm": d["precipitation_sum"]})
        if r.status_code in (429, 500, 502, 503, 504) and not last:
            print(f"    rainfall API returned {r.status_code}; waiting {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 300)
            continue
        raise RuntimeError(f"Open-Meteo error {r.status_code}: {r.text[:200]}")


def ensure_cells(cfg, keys):
    rf = cfg["rainfall"]
    keys = sorted(set(keys))
    todo = [k for k in keys if not cell_file(cfg, k).exists()]
    for i, k in enumerate(todo, 1):
        print(f"  rainfall cell {i}/{len(todo)}  lat={k[0]} lon={k[1]}")
        fetch_cell(k[0], k[1], rf["start"], rf["end"]).to_csv(cell_file(cfg, k), index=False)
        time.sleep(1.5)                   # stay under the API rate limit


def bbox_cells(cfg):
    w, s, e, n = cfg["study_area"]["bbox"]
    deg = cfg["rainfall"]["cell_deg"]
    lats = np.arange(round(s / deg), round(n / deg) + 1) * deg
    lons = np.arange(round(w / deg), round(e / deg) + 1) * deg
    return [cell_key(la, lo, deg) for la in lats for lo in lons]


class RainStore:
    """window(lat, lon, day) -> the `window_days` daily totals ending on `day`."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.deg = cfg["rainfall"]["cell_deg"]
        self.T = int(cfg["rainfall"]["window_days"])
        self._cache = {}

    def _load(self, key):
        if key not in self._cache:
            f = cell_file(self.cfg, key)
            if not f.exists():
                ensure_cells(self.cfg, [key])
            df = pd.read_csv(f)
            start = date.fromisoformat(str(df["date"].iloc[0])).toordinal()
            arr = df["rain_mm"].fillna(0.0).to_numpy(dtype=np.float32)
            self._cache[key] = (start, arr)
        return self._cache[key]

    def window(self, lat, lon, day):
        key = cell_key(float(lat), float(lon), self.deg)
        start, arr = self._load(key)
        if not isinstance(day, date):
            day = date.fromisoformat(str(day)[:10])
        end = day.toordinal() - start + 1
        if end - self.T < 0 or end > len(arr):
            return None
        return arr[end - self.T:end]
