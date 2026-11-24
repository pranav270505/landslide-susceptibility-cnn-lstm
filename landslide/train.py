"""Steps 4 and 5: classical baselines, then the CNN-only / LSTM-only / hybrid models.

Everything is scored on the same held-out test blocks. Predictions are saved to
outputs/preds/<model>.npy so report.py can build the tables and figures.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from .models import LandslideNet


def load_dataset(cfg):
    f = cfg.root / "data" / "processed" / "dataset.npz"
    if not f.exists():
        raise FileNotFoundError("dataset.npz not found: run the 'samples' step first.")
    return dict(np.load(f, allow_pickle=False))


# ------------------------------------------------------------------ baselines
def run_baselines(cfg):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    d = load_dataset(cfg)
    names = list(d["tab_names"])
    X = d["X_tab"]
    if cfg["baselines"]["use_rainfall"]:
        X = np.hstack([X, d["X_rain"]])
        names += list(d["rain_names"])
    y, split = d["y"], d["split"]
    fit = split <= 1                      # baselines have no early stopping: use train + val
    seed = cfg["train"]["seed"]
    models = {
        "SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=1.0, kernel="rbf"), ensemble=False)),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                                             early_stopping=True, random_state=seed)),
    }
    out = cfg.folder("outputs", "preds")
    for name, m in models.items():
        m.fit(X[fit], y[fit])
        p = m.predict_proba(X[split == 2])[:, 1]
        np.save(out / f"{name}.npy", p)
        print(f"  {name:14s} test accuracy {((p >= 0.5) == y[split == 2]).mean():.3f}")
    imp = dict(zip(names, models["Random Forest"].feature_importances_.tolist()))
    cfg.path("outputs", "rf_feature_importance.json").write_text(json.dumps(imp, indent=2))


# --------------------------------------------------------------- deep models
def _batches(idx, bs, shuffle, rng):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model, Xp, Xs, idx, device, bs=256):
    model.eval()
    out = []
    for b in _batches(idx, bs, False, None):
        out.append(torch.sigmoid(model(Xp[b].to(device), Xs[b].to(device))).cpu())
    return torch.cat(out).numpy()


def train_one(cfg, d, mode):
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    rng = np.random.default_rng(tc["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xp, Xs = torch.from_numpy(d["X_patch"]), torch.from_numpy(d["X_seq"])
    y = torch.from_numpy(d["y"].astype(np.float32))
    tr, va, te = (np.flatnonzero(d["split"] == i) for i in range(3))

    model = LandslideNet(Xp.shape[1], Xp.shape[-1], mode, tc["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    l2 = float(tc.get("l2", 0.0))
    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(1, int(tc["epochs"]) + 1):
        model.train()
        tot = 0.0
        for b in _batches(tr, int(tc["batch_size"]), True, rng):
            if len(b) < 2:
                continue                   # BatchNorm needs at least 2 samples
            opt.zero_grad()
            loss = loss_fn(model(Xp[b].to(device), Xs[b].to(device)), y[b].to(device))
            if l2 > 0:                     # Keras-style L2 penalty on the dense layers
                loss = loss + l2 * sum((m.weight ** 2).sum() for m in model.head if isinstance(m, nn.Linear))
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        pv = predict(model, Xp, Xs, va, device)
        yv = d["y"][va]
        vloss = float(-np.mean(yv * np.log(pv + 1e-7) + (1 - yv) * np.log(1 - pv + 1e-7)))
        ptr = predict(model, Xp, Xs, tr, device)
        hist.append(dict(epoch=epoch, train_loss=tot / len(tr), val_loss=vloss,
                         train_acc=float(((ptr >= 0.5) == d["y"][tr]).mean()),
                         val_acc=float(((pv >= 0.5) == yv).mean())))
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0 or bad >= tc["patience"]:
            h = hist[-1]
            print(f"    epoch {epoch:3d}  train loss {h['train_loss']:.4f}  "
                  f"val loss {h['val_loss']:.4f}  val acc {h['val_acc']:.3f}")
        if bad >= tc["patience"]:
            print(f"    early stop (best val loss {best:.4f})")
            break
    model.load_state_dict(best_state)
    return model, hist, predict(model, Xp, Xs, te, device)


NAMES = {"cnn": "CNN-only", "lstm": "LSTM-only", "hybrid": "CNN-LSTM (hybrid)"}


def run_deep(cfg):
    d = load_dataset(cfg)
    preds, models = cfg.folder("outputs", "preds"), cfg.folder("outputs", "models")
    for mode in ("cnn", "lstm", "hybrid"):
        print(f"  training {NAMES[mode]}")
        model, hist, p = train_one(cfg, d, mode)
        np.save(preds / f"{NAMES[mode]}.npy", p)
        torch.save({"state": model.state_dict(), "mode": mode,
                    "in_channels": int(d["X_patch"].shape[1]),
                    "patch_size": int(d["X_patch"].shape[-1]),
                    "dropout": cfg["train"]["dropout"]}, models / f"{mode}.pt")
        (models / f"{mode}_history.json").write_text(json.dumps(hist, indent=2))
        print(f"    test accuracy {((p >= 0.5) == d['y'][d['split'] == 2]).mean():.3f}")
