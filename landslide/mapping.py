"""Step 7: wall-to-wall landslide probability map from the trained hybrid model.

The map is for one rainfall scenario: the 30 days ending on map.date.
Writes outputs/susceptibility.tif (probability), susceptibility_class.tif
(1 low, 2 medium, 3 high) and susceptibility.png.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from affine import Affine
from numpy.lib.stride_tricks import sliding_window_view

from . import factors
from .common import make_grid, write_tif
from .models import LandslideNet
from .rainfall import RainStore, cell_key
from .samples import build_stack, scale_rain


def run(cfg):
    mc = cfg["map"]
    grid = make_grid(cfg)
    norm = json.loads((cfg.root / "data" / "processed" / "norm.json").read_text())
    ck = torch.load(cfg.root / "outputs" / "models" / "hybrid.pt", map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LandslideNet(ck["in_channels"], ck["patch_size"], ck["mode"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    layers, lc, valid = factors.load(cfg, grid)
    stack, _ = build_stack(layers, lc, norm)
    del layers
    P, s = int(norm["patch_size"]), int(mc["stride"])
    h = P // 2
    windows = sliding_window_view(stack, (P, P), axis=(1, 2))   # (C, H-P+1, W-P+1, P, P), no copy

    rows = np.arange(h, grid.height - h, s)
    cols = np.arange(h, grid.width - h, s)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    keep = valid[rr, cc]

    # one rainfall sequence per rainfall cell
    rain = RainStore(cfg)
    lon, lat = grid.rowcol_to_lonlat(rr, cc)
    deg = cfg["rainfall"]["cell_deg"]
    keys = [cell_key(a, o, deg) for a, o in zip(lat, lon)]
    uniq = sorted(set(keys))
    seq_of = {}
    for k in uniq:
        w = rain.window(k[0], k[1], mc["date"])
        if w is None:
            raise RuntimeError(f"map.date {mc['date']} is outside the downloaded rainfall period.")
        seq_of[k] = scale_rain(w, norm["rain_max"])
    seq_table = torch.from_numpy(np.stack([seq_of[k] for k in uniq]))[..., None]
    lookup = {k: i for i, k in enumerate(uniq)}
    seq_idx = np.array([lookup[k] for k in keys])

    prob = np.full(len(rr), np.nan, np.float32)
    todo = np.flatnonzero(keep)
    bs = int(mc["batch_size"])
    print(f"  predicting {len(todo):,} locations (every {s * grid.res:g} m) for {mc['date']}")
    with torch.no_grad():
        for i in range(0, len(todo), bs):
            b = todo[i:i + bs]
            patch = torch.from_numpy(np.ascontiguousarray(
                windows[:, rr[b] - h, cc[b] - h].transpose(1, 0, 2, 3)))
            p = torch.sigmoid(model(patch.to(device), seq_table[seq_idx[b]].to(device)))
            prob[b] = p.cpu().numpy()
            if (i // bs) % 100 == 0:
                print(f"    {min(i + bs, len(todo)):,} / {len(todo):,}")
    prob = prob.reshape(len(rows), len(cols))

    t = grid.transform * Affine.translation(cols[0] + 0.5 - s / 2, rows[0] + 0.5 - s / 2) * Affine.scale(s)
    out = cfg.folder("outputs")
    write_tif(out / "susceptibility.tif", prob, grid, transform=t)
    lo, hi = mc["thresholds"]
    cls = np.where(np.isnan(prob), 0, 1 + (prob >= lo).astype(np.uint8) + (prob >= hi).astype(np.uint8))
    write_tif(out / "susceptibility_class.tif", cls.astype(np.uint8), grid, nodata=0, transform=t)

    w, n = t.c, t.f
    extent = [w, w + prob.shape[1] * t.a, n + prob.shape[0] * t.e, n]
    fig, ax = plt.subplots(figsize=(7, 7 * prob.shape[0] / prob.shape[1] + 0.6))
    im = ax.imshow(prob, cmap="RdYlGn_r", vmin=0, vmax=1, extent=extent)
    ax.set_title(f"Landslide probability, rainfall scenario {mc['date']}")
    ax.set_xlabel(f"Easting (m, {grid.crs})"); ax.set_ylabel("Northing (m)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Probability")
    plt.tight_layout(); plt.savefig(out / "susceptibility.png", dpi=200); plt.close()
    share = [(cls == k).sum() / max((cls > 0).sum(), 1) * 100 for k in (1, 2, 3)]
    print(f"  area share  low {share[0]:.1f}%  medium {share[1]:.1f}%  high {share[2]:.1f}%")
    print(f"  wrote susceptibility.tif, susceptibility_class.tif, susceptibility.png to {out}")
