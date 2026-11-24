"""Offline smoke test: builds a small MADE-UP world in ./demo so every step after
'download' can be run without internet. The numbers it produces mean nothing;
it only proves the code runs on your machine.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from .common import load_config, make_grid, write_tif
from .rainfall import bbox_cells, cell_file

EVENTS = [date(2018, 8, 16), date(2018, 7, 10), date(2017, 9, 18)]


def run(base_config: str):
    base = load_config(base_config)
    root = base.root / "demo"
    root.mkdir(exist_ok=True)
    cfg_d = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    cfg_d["study_area"].update(name="DEMO", bbox=[76.95, 9.95, 77.07, 10.07])
    cfg_d["samples"].update(block_km=1.5, negative_buffer_m=300, temporal_negative_years=[2010, 2016])
    cfg_d["factors"]["stream_threshold_cells"] = 300
    cfg_d["train"].update(epochs=25, patience=8)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg_d, sort_keys=False), encoding="utf-8")
    cfg = load_config(root / "config.yaml")
    grid = make_grid(cfg)
    rng = np.random.default_rng(0)
    H, W = grid.height, grid.width

    def smooth(sigma):
        a = ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma)
        return (a - a.min()) / (a.max() - a.min())

    dem = 300 + 1500 * smooth(40) + 250 * smooth(10) + 40 * smooth(3)
    interim = cfg.folder("data", "interim")
    write_tif(interim / "dem.tif", dem.astype(np.float32), grid)
    ndvi = 0.15 + 0.7 * smooth(15)
    write_tif(interim / "ndvi.tif", ndvi.astype(np.float32), grid)
    lc = np.select([ndvi > 0.6, ndvi > 0.45, ndvi > 0.3], [10, 20, 40], 50).astype(np.uint8)
    write_tif(interim / "landcover.tif", lc, grid, nodata=0)
    roads = np.zeros((H, W), np.uint8)
    for k in range(4):
        r = (H // 5) * (k + 1) + (15 * np.sin(np.arange(W) / 30 + k)).astype(int)
        roads[np.clip(r, 0, H - 1), np.arange(W)] = 1
    write_tif(interim / "roads.tif", roads, grid, nodata=255)

    # rainfall: wet monsoon, plus a very wet fortnight before each event
    days = pd.date_range(cfg["rainfall"]["start"], cfg["rainfall"]["end"], freq="D")
    cells = bbox_cells(cfg)
    strength = {k: 0.6 + 0.6 * rng.random() for k in cells}
    for k in cells:
        mm = np.where(days.month.isin([6, 7, 8, 9]), rng.gamma(0.8, 14, len(days)), rng.gamma(0.3, 4, len(days)))
        for ev in EVENTS:
            sel = (days.date <= ev) & (days.date > ev - timedelta(days=12))
            mm[sel] += rng.gamma(4, 20, sel.sum()) * strength[k]
        pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "rain_mm": mm.round(1)}).to_csv(cell_file(cfg, k), index=False)

    # landslides: steep, sparsely vegetated ground, on event dates
    d_row, d_col = np.gradient(dem, grid.res)
    slope = np.degrees(np.arctan(np.hypot(d_row, d_col)))
    weight = 1 / (1 + np.exp(-(slope - np.percentile(slope, 75)) / 2.0)) * (1.2 - ndvi)
    weight[:20], weight[-20:], weight[:, :20], weight[:, -20:] = 0, 0, 0, 0
    pick = rng.choice(H * W, size=350, replace=False, p=(weight / weight.sum()).ravel())
    rows, cols = np.divmod(pick, W)
    lon, lat = grid.rowcol_to_lonlat(rows, cols)
    inv = pd.DataFrame({"latitude": lat, "longitude": lon,
                        "date": [str(EVENTS[i]) for i in rng.integers(0, len(EVENTS), len(pick))]})
    inv.to_csv(cfg.folder("data", "raw", "inventory") / "demo_inventory.csv", index=False)
    print(f"Demo world written to {root} ({H} x {W} pixels, {len(inv)} made-up landslides).")
    return str(root / "config.yaml")
