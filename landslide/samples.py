"""Step 3: turn the inventory + factors + rainfall into a model-ready dataset.

Every sample is (terrain patch, 30-day rainfall sequence, label).

Positives   landslide location on its event date.
Negatives   two kinds, mixed by samples.temporal_negative_fraction:
  spatial   a location far from every landslide, on a landslide's event date
  temporal  a landslide location on an earlier monsoon date when it did not fail
The temporal negatives are what force the model to use rainfall: without them
every sample of a single-event inventory shares one date and the LSTM has
nothing to learn. Set the fraction to 0 for a pure susceptibility setup.

The train/val/test split is by spatial block, so nearby (and repeated)
locations never straddle the split.

Writes data/processed/dataset.npz and data/processed/norm.json
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import ndimage

from . import factors
from .common import make_grid
from .rainfall import RainStore

LAT_NAMES = ("lat", "latitude", "y", "lat_dd", "ycoord")
LON_NAMES = ("lon", "long", "lng", "longitude", "x", "lon_dd", "xcoord")
RAIN_WINDOWS = (1, 3, 7, 15, 30)


def _find(columns, wanted, explicit=None):
    if explicit:
        return explicit
    low = {c.lower(): c for c in columns}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def load_inventory(cfg) -> pd.DataFrame:
    inv = cfg["inventory"]
    folder = cfg.root / inv["dir"]
    files = [f for f in sorted(folder.glob("*")) if f.suffix.lower() in
             (".shp", ".gpkg", ".geojson", ".json", ".kml", ".csv")] if folder.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No landslide inventory found in {folder}. See README.md, 'Landslide inventory'.")
    default = pd.Timestamp(inv["default_event_date"])
    frames = []
    for f in files:
        if f.suffix.lower() == ".csv":
            df = pd.read_csv(f)
            la = _find(df.columns, LAT_NAMES, inv.get("lat_column"))
            lo = _find(df.columns, LON_NAMES, inv.get("lon_column"))
            if la is None or lo is None:
                raise ValueError(f"{f.name}: set inventory.lat_column / lon_column in config.yaml")
            out = pd.DataFrame({"lon": pd.to_numeric(df[lo], errors="coerce"),
                                "lat": pd.to_numeric(df[la], errors="coerce")})
            cols = df
        else:
            import geopandas as gpd
            gdf = gpd.read_file(f)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
            gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf
            pts = gdf.geometry.representative_point().to_crs("EPSG:4326")
            out = pd.DataFrame({"lon": pts.x.to_numpy(), "lat": pts.y.to_numpy()})
            cols = pd.DataFrame(gdf.drop(columns="geometry")).reset_index(drop=True)
        dc = inv.get("date_column") or next((c for c in cols.columns if "date" in c.lower()), None)
        if dc is not None and dc in cols.columns:
            d = pd.to_datetime(cols[dc], errors="coerce").reset_index(drop=True)
            out["date"] = d.fillna(default).to_numpy()
            note = f"dates from '{dc}' ({int(d.isna().sum())} missing -> default)"
        else:
            out["date"] = default
            note = f"no date column -> {default.date()}"
        print(f"  {f.name}: {len(out)} landslides, {note}")
        frames.append(out)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["lon", "lat"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fit_norm(layers, lc, valid, seed):
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid)
    idx = rng.choice(idx, size=min(len(idx), 500_000), replace=False)
    norm = {"continuous": {}, "landcover_classes": [int(c) for c in np.unique(lc.ravel()[idx]) if c != 0]}
    for name, arr in layers.items():
        v = arr.ravel()[idx]
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [1, 99])
        norm["continuous"][name] = [float(lo), float(hi if hi > lo else lo + 1.0)]
    return norm


def build_stack(layers, lc, norm):
    """(C, H, W) float32 in [0, 1]: Min-Max scaled factors, then one-hot land cover."""
    chans, names = [], []
    for name, (lo, hi) in norm["continuous"].items():
        a = np.clip((layers[name] - lo) / (hi - lo), 0, 1)
        chans.append(np.nan_to_num(a, nan=0.0).astype(np.float32))
        names.append(name)
    for c in norm["landcover_classes"]:
        chans.append((lc == c).astype(np.float32))
        names.append(f"landcover_{c}")
    return np.stack(chans), names


def rain_features(seq):
    return [float(seq[-w:].sum()) for w in RAIN_WINDOWS] + [float(seq.max())]


def scale_rain(seq, rain_max):
    return (np.log1p(seq) / np.log1p(rain_max)).astype(np.float32)


def run(cfg):
    sc = cfg["samples"]
    rng = np.random.default_rng(sc["seed"])
    grid = make_grid(cfg)
    P = int(sc["patch_size"])
    h = P // 2
    layers, lc, valid = factors.load(cfg, grid)
    rain = RainStore(cfg)

    # ---- positives -------------------------------------------------------
    inv = load_inventory(cfg)
    inv["row"], inv["col"] = grid.lonlat_to_rowcol(inv["lon"], inv["lat"])
    ok = grid.inside(inv["row"].to_numpy(), inv["col"].to_numpy(), margin=h)
    inv = inv[ok]
    inv = inv[valid[inv["row"], inv["col"]]]
    inv = inv.drop_duplicates(subset=["row", "col", "date"]).reset_index(drop=True)
    inv = inv[[rain.window(r.lat, r.lon, r.date) is not None for r in inv.itertuples()]]
    if sc.get("max_positives") and len(inv) > sc["max_positives"]:
        inv = inv.sample(int(sc["max_positives"]), random_state=sc["seed"])
    inv = inv.reset_index(drop=True)
    n_pos = len(inv)
    print(f"  {n_pos} landslides inside the study area with rainfall history")
    if n_pos < 50:
        raise RuntimeError("Fewer than 50 usable landslides. Check the bbox, the inventory "
                           "coordinates and that rainfall.start/end cover the event dates.")
    rows = [dict(row=r.row, col=r.col, date=r.date, y=1, kind="landslide") for r in inv.itertuples()]

    # ---- temporal negatives: same slope, a date it did not fail ----------
    n_temp = int(round(n_pos * float(sc["temporal_negative_fraction"])))
    y0, y1 = sc["temporal_negative_years"]
    months = set(sc["temporal_negative_months"])
    pool = [d for d in (date(y0, 1, 1) + timedelta(days=i)
                        for i in range((date(y1, 12, 31) - date(y0, 1, 1)).days + 1))
            if d.month in months]
    events_at = inv.groupby(["row", "col"])["date"].apply(list).to_dict()
    tries = 0
    while sum(r["kind"] == "neg_temporal" for r in rows) < n_temp and tries < n_temp * 50:
        tries += 1
        p = inv.iloc[rng.integers(n_pos)]
        d = pool[rng.integers(len(pool))]
        if any(abs((d - e).days) < 60 for e in events_at[(p.row, p.col)]):
            continue
        if rain.window(p.lat, p.lon, d) is None:
            continue
        rows.append(dict(row=p.row, col=p.col, date=d, y=0, kind="neg_temporal"))
    n_temp = sum(r["kind"] == "neg_temporal" for r in rows)

    # ---- spatial negatives: far from any landslide, on an event date -----
    n_spat = n_pos - n_temp
    ls_mask = np.zeros(valid.shape, bool)
    ls_mask[inv["row"], inv["col"]] = True
    far = ndimage.distance_transform_edt(~ls_mask) * grid.res >= float(sc["negative_buffer_m"])
    cand = valid & far
    cand[:h], cand[-h:], cand[:, :h], cand[:, -h:] = False, False, False, False
    cand_idx = np.flatnonzero(cand)
    if len(cand_idx) < n_spat:
        raise RuntimeError("Not enough room for spatial negatives; lower samples.negative_buffer_m.")
    for flat in rng.choice(cand_idx, size=n_spat, replace=False):
        r, c = divmod(int(flat), grid.width)
        rows.append(dict(row=r, col=c, date=inv["date"].iloc[rng.integers(n_pos)], y=0, kind="neg_spatial"))

    df = pd.DataFrame(rows)
    df["lon"], df["lat"] = grid.rowcol_to_lonlat(df["row"], df["col"])
    print(f"  samples: {n_pos} landslide, {n_spat} spatial negatives, {n_temp} temporal negatives")

    # ---- spatial block split ---------------------------------------------
    tr, va, _ = sc["split"]
    if sc.get("split_method", "block") == "random":
        # stratified random split, as in the base paper (optimistic: neighbours leak)
        df["split"] = 2
        for label in (0, 1):
            idx = rng.permutation(df.index[df["y"] == label].to_numpy())
            a, b = int(len(idx) * tr), int(len(idx) * (tr + va))
            df.loc[idx[:a], "split"], df.loc[idx[a:b], "split"] = 0, 1
    else:
        cells = max(1, int(round(float(sc["block_km"]) * 1000 / grid.res)))
        block = (df["row"] // cells) * 100_000 + (df["col"] // cells)
        ids = rng.permutation(block.unique())
        counts = block.value_counts()
        split_of, seen = {}, 0
        for blk in ids:
            frac = seen / len(df)
            split_of[blk] = 0 if frac < tr else (1 if frac < tr + va else 2)
            seen += counts[blk]
        df["split"] = block.map(split_of)
    if df["split"].nunique() < 3:
        raise RuntimeError("Split produced an empty set; lower samples.block_km.")

    # ---- arrays ----------------------------------------------------------
    norm = fit_norm(layers, lc, valid, sc["seed"])
    seqs = np.stack([rain.window(r.lat, r.lon, r.date) for r in df.itertuples()])
    norm["rain_max"] = float(max(seqs[df["split"].to_numpy() == 0].max(), 1.0))
    norm["patch_size"] = P
    stack, chan_names = build_stack(layers, lc, norm)
    norm["channels"] = chan_names

    X_patch = np.stack([stack[:, r - h:r + h, c - h:c + h] for r, c in zip(df["row"], df["col"])])
    X_seq = np.stack([scale_rain(s, norm["rain_max"]) for s in seqs])[..., None]

    cont = list(norm["continuous"])
    tab = [[float(np.nan_to_num(layers[n][r, c])) for n in cont]
           + [float(lc[r, c] == k) for k in norm["landcover_classes"]]
           for r, c in zip(df["row"], df["col"])]
    tab_names = cont + [f"landcover_{k}" for k in norm["landcover_classes"]]
    rain_tab = [rain_features(s) for s in seqs]
    rain_names = [f"rain_{w}d" for w in RAIN_WINDOWS] + ["rain_max_day"]

    np.savez_compressed(
        cfg.path("data", "processed", "dataset.npz"),
        X_patch=X_patch.astype(np.float32), X_seq=X_seq.astype(np.float32),
        X_tab=np.asarray(tab, np.float32), X_rain=np.asarray(rain_tab, np.float32),
        tab_names=np.array(tab_names, dtype="U"), rain_names=np.array(rain_names, dtype="U"),
        y=df["y"].to_numpy(np.int64), split=df["split"].to_numpy(np.int64),
        kind=np.array(df["kind"].tolist(), dtype="U"), lon=df["lon"].to_numpy(), lat=df["lat"].to_numpy(),
        date=np.array([str(x) for x in df["date"]], dtype="U"),
    )
    cfg.path("data", "processed", "norm.json").write_text(json.dumps(norm, indent=2))
    df.drop(columns=["row", "col"]).to_csv(cfg.path("data", "processed", "samples.csv"), index=False)
    for i, name in enumerate(["train", "val", "test"]):
        part = df[df["split"] == i]
        print(f"  {name:5s} {len(part):5d} samples  ({int(part['y'].sum())} landslide)")
    print(f"  patch tensor {X_patch.shape}, rainfall tensor {X_seq.shape}")
