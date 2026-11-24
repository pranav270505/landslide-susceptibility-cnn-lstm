"""Local web app for the landslide CNN-LSTM project: live model inference + every pipeline output.

Put this file next to run.py and the folder web/ (with index.html), then:
    pip install flask
    python server.py          ->  http://127.0.0.1:5000

Needs the finished pipeline: data/processed/{dataset.npz,norm.json}, outputs/models/*.pt, outputs/*.
Only the held-out TEST samples are served, so every prediction is on areas the models never trained on.
"""
import json
import math
import os
from pathlib import Path

for _v in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):      # same Windows fix as run.py
    os.environ.pop(_v, None)

import numpy as np
import pandas as pd
import torch
from flask import Flask, abort, jsonify, request, send_from_directory

from landslide.common import load_config
from landslide.models import LandslideNet
from landslide.rainfall import RainStore
from landslide.samples import scale_rain

HERE = Path(__file__).resolve().parent
cfg = load_config(str(HERE / "config.yaml"))
OUT = cfg.root / "outputs"
norm = json.loads((cfg.root / "data" / "processed" / "norm.json").read_text())

D = dict(np.load(cfg.root / "data" / "processed" / "dataset.npz", allow_pickle=False))
TE = np.flatnonzero(D["split"] == 2)
XP, XS = D["X_patch"][TE], D["X_seq"][TE]
Y, KIND, LON, LAT, DATE = D["y"][TE], D["kind"][TE], D["lon"][TE], D["lat"][TE], D["date"][TE]
del D
RAW_MM = np.expm1(XS[:, :, 0] * math.log1p(norm["rain_max"]))      # undo the log scaling for display
CH = {n: i for i, n in enumerate(norm["channels"])}
SHOW = [c for c in ("slope", "elevation", "ndvi") if c in CH]

NAMES = {"hybrid": "CNN-LSTM (hybrid)", "cnn": "CNN-only", "lstm": "LSTM-only"}
MODELS = {}
for key in NAMES:
    f = OUT / "models" / f"{key}.pt"
    if f.exists():
        ck = torch.load(f, map_location="cpu")
        net = LandslideNet(ck["in_channels"], ck["patch_size"], ck["mode"], ck["dropout"])
        net.load_state_dict(ck["state"])
        net.eval()
        MODELS[key] = net
if not MODELS:
    raise SystemExit("No trained models in outputs/models. Run 'python run.py train' first.")

STORED = {}                                   # saved test predictions, same order as TE (incl. baselines)
for f in (OUT / "preds").glob("*.npy"):
    p = np.load(f)
    if len(p) == len(TE):
        STORED[f.stem] = p

RAIN = RainStore(cfg)
app = Flask(__name__)


def table(name):
    f = OUT / name
    return pd.read_csv(f).round(4).to_dict("records") if f.exists() else []


@app.get("/")
def index():
    return send_from_directory(HERE / "web", "index.html")


@app.get("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUT, name)


@app.get("/api/meta")
def meta():
    files = sorted(p.name for p in OUT.iterdir() if p.is_file()) if OUT.exists() else []
    return jsonify(
        study=cfg["study_area"].get("name", ""), models=list(MODELS), stored=list(STORED),
        counts={k: int((KIND == k).sum()) for k in ("landslide", "neg_spatial", "neg_temporal")},
        results=table("results.csv"), explain=table("explain.csv"), files=files,
        map_date=str(cfg["map"]["date"]), rain_window=int(cfg["rainfall"]["window_days"]))


@app.get("/api/samples")
def samples():
    return jsonify(kind=KIND.tolist(), y=Y.tolist(), date=DATE.tolist(),
                   rain=np.round(RAW_MM.sum(1), 1).tolist())


@app.post("/api/predict")
def predict():
    b = request.get_json(force=True)
    i = int(b.get("i", -1))
    if not 0 <= i < len(TE):
        abort(404)
    if b.get("date"):                         # swap in the rainfall of any other date at this location
        w = RAIN.window(LAT[i], LON[i], b["date"])
        if w is None:
            return jsonify(error="No 30-day rainfall history for that date (outside the downloaded period)."), 400
        mm = np.asarray(w, np.float32)
    else:
        mm = RAW_MM[i].astype(np.float32)
    mm = mm * float(b.get("scale", 1.0))
    seq = torch.from_numpy(scale_rain(mm, norm["rain_max"]))[None, :, None]
    patch = torch.from_numpy(XP[i:i + 1])
    with torch.no_grad():
        probs = {k: float(torch.sigmoid(m(patch, seq))[0]) for k, m in MODELS.items()}
    return jsonify(
        i=i, kind=str(KIND[i]), truth=int(Y[i]), sample_date=str(DATE[i]), lon=float(LON[i]), lat=float(LAT[i]),
        rain_mm=np.round(mm, 1).tolist(), probs=probs, names=NAMES,
        stored={k: float(v[i]) for k, v in STORED.items()},
        patch={c: np.round(XP[i, CH[c]], 2).tolist() for c in SHOW})


if __name__ == "__main__":
    print(f"{len(TE)} held-out samples, models: {', '.join(MODELS)}")
    app.run(host="127.0.0.1", port=5000, debug=False)
