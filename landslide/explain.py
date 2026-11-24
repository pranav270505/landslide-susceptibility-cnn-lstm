"""Step 7: what the trained hybrid model relies on (permutation importance).

For each input, its values are shuffled across the test samples and the drop in
test AUC is recorded: a large drop means the model needs that input. This is the
same kind of importance the base paper reports, but computed for the hybrid
model itself and split three ways:

  factors        each terrain / land-cover factor, and rainfall as a whole
  rainfall days  the 30 antecedent days, in 5-day blocks
  branches       all terrain inputs together versus the rainfall sequence

Writes outputs/explain_factors.png, explain_rain_days.png, explain.csv
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .models import LandslideNet
from .train import load_dataset, predict

NICE = {"elevation": "Elevation", "slope": "Slope", "northness": "Aspect (north-south)",
        "eastness": "Aspect (east-west)", "twi": "TWI", "spi": "SPI", "ndvi": "NDVI",
        "annual_rain": "Mean annual rainfall", "dist_river": "Distance to rivers",
        "dist_road": "Distance to roads", "landcover": "Land use / land cover"}


def run(cfg):
    d = load_dataset(cfg)
    norm = json.loads((cfg.root / "data" / "processed" / "norm.json").read_text())
    ck = torch.load(cfg.root / "outputs" / "models" / "hybrid.pt", map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LandslideNet(ck["in_channels"], ck["patch_size"], ck["mode"], ck["dropout"]).to(device)
    model.load_state_dict(ck["state"])

    te = np.flatnonzero(d["split"] == 2)
    Xp, Xs = torch.from_numpy(d["X_patch"][te]), torch.from_numpy(d["X_seq"][te])
    y = d["y"][te]
    idx = np.arange(len(te))
    repeats = int(cfg.get("explain", {}).get("repeats", 10))
    rng = np.random.default_rng(cfg["train"]["seed"])

    def auc(patch, seq):
        return roc_auc_score(y, predict(model, patch, seq, idx, device))

    base = auc(Xp, Xs)
    print(f"  hybrid test AUC {base:.3f}; shuffling each input {repeats} times")

    def drop(channels=None, days=None):
        vals = []
        for _ in range(repeats):
            perm = torch.from_numpy(rng.permutation(len(te)))
            p, s = Xp, Xs
            if channels is not None:
                p = Xp.clone()
                p[:, channels] = Xp[perm][:, channels]
            if days is not None:
                s = Xs.clone()
                s[:, days] = Xs[perm][:, days]
            vals.append(base - auc(p, s))
        return float(np.mean(vals)), float(np.std(vals))

    # group one-hot land-cover channels into a single factor
    groups = {}
    for i, name in enumerate(norm["channels"]):
        groups.setdefault("landcover" if name.startswith("landcover_") else name, []).append(i)
    T = Xs.shape[1]
    rows = []
    for name, chans in groups.items():
        m, sd = drop(channels=chans)
        rows.append(dict(group="factor", input=NICE.get(name, name), auc_drop=m, std=sd))
    m, sd = drop(days=list(range(T)))
    rows.append(dict(group="factor", input="Rainfall (30-day sequence)", auc_drop=m, std=sd))
    m_t, sd_t = drop(channels=list(range(Xp.shape[1])))
    rows.append(dict(group="branch", input="All terrain inputs (CNN branch)", auc_drop=m_t, std=sd_t))
    rows.append(dict(group="branch", input="Rainfall sequence (LSTM branch)", auc_drop=m, std=sd))
    # single days are too redundant to matter on their own, so shuffle 5-day blocks
    B = 5
    for start in range(T - B, -1, -B):
        m_d, sd_d = drop(days=list(range(start, start + B)))
        rows.append(dict(group="rain_window", auc_drop=m_d, std=sd_d,
                         input=f"{T - start - B}-{T - start - 1} days before"))

    df = pd.DataFrame(rows)
    for g in df["group"].unique():
        sel = df["group"] == g
        pos = df.loc[sel, "auc_drop"].clip(lower=0)
        df.loc[sel, "share_pct"] = 100 * pos / pos.sum() if pos.sum() > 0 else 0.0
    out = cfg.folder("outputs")
    df.round(4).to_csv(out / "explain.csv", index=False)

    f = df[df["group"] == "factor"].sort_values("auc_drop")
    plt.figure(figsize=(6.5, 0.38 * len(f) + 1.4))
    plt.barh(f["input"], f["auc_drop"], xerr=f["std"], color="#2a7f8a", ecolor="#555", capsize=2)
    for v, s, yv in zip(f["auc_drop"], f["share_pct"], range(len(f))):
        plt.text(max(v, 0) + f["std"].max() + 0.002, yv, f"{s:.0f}%", va="center", fontsize=8)
    plt.xlabel("Drop in test AUC when the input is shuffled")
    plt.title(f"What the hybrid model relies on (base AUC {base:.3f})")
    plt.tight_layout(); plt.savefig(out / "explain_factors.png", dpi=200); plt.close()

    r = df[df["group"] == "rain_window"]
    plt.figure(figsize=(6.5, 3.4))
    plt.bar(r["input"].str.replace(" days before", ""), r["auc_drop"], yerr=r["std"],
            color="#2a7f8a", ecolor="#555", capsize=2)
    plt.xlabel("Days before the event date (0 = the day itself)")
    plt.ylabel("Drop in test AUC when shuffled")
    plt.title("Which part of the 30-day rainfall window matters")
    plt.tight_layout(); plt.savefig(out / "explain_rain_days.png", dpi=200); plt.close()

    top = df[df["group"] == "factor"].sort_values("auc_drop", ascending=False).head(5)
    print("  top inputs:")
    for row in top.itertuples():
        print(f"    {row.input:28s} AUC drop {row.auc_drop:.3f}  ({row.share_pct:.0f}%)")
    b = df[df["group"] == "branch"]
    print("  branches: " + ";  ".join(f"{x.input} {x.auc_drop:.3f}" for x in b.itertuples()))
    print(f"  wrote explain_factors.png, explain_rain_days.png, explain.csv to {out}")
