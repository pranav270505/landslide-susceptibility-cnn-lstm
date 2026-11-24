"""Step 6: metrics table and figures for every model that has saved predictions.

Writes to outputs/: results.csv, results.md, roc_curves.png, confusion_matrices.png,
training_curves.png, feature_importance.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .train import load_dataset

ORDER = ["CNN-LSTM (hybrid)", "CNN-only", "LSTM-only", "Random Forest", "MLP", "SVM", "Decision Tree"]


def run(cfg):
    d = load_dataset(cfg)
    te = d["split"] == 2
    y, kind = d["y"][te], d["kind"][te]
    out = cfg.folder("outputs")
    preds = {n: np.load(out / "preds" / f"{n}.npy") for n in ORDER if (out / "preds" / f"{n}.npy").exists()}
    if not preds:
        raise FileNotFoundError("No predictions yet: run the 'baselines' and 'train' steps first.")

    rows = []
    for name, p in preds.items():
        yhat = (p >= 0.5).astype(int)
        row = dict(Model=name, Accuracy=accuracy_score(y, yhat),
                   Precision=precision_score(y, yhat, zero_division=0),
                   Recall=recall_score(y, yhat, zero_division=0),
                   F1=f1_score(y, yhat, zero_division=0), AUC=roc_auc_score(y, p))
        # how well each kind of negative is rejected: shows which branch does what
        for k, label in (("neg_spatial", "Spec. spatial neg"), ("neg_temporal", "Spec. temporal neg")):
            m = kind == k
            row[label] = float((yhat[m] == 0).mean()) if m.any() else np.nan
        rows.append(row)
    res = pd.DataFrame(rows).round(3)
    res.to_csv(out / "results.csv", index=False)
    cols = list(res.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in res.itertuples(index=False)]
    note = (f"\n\nTest set: {len(y)} samples ({int(y.sum())} landslide, "
            f"{int((kind == 'neg_spatial').sum())} spatial negatives, "
            f"{int((kind == 'neg_temporal').sum())} temporal negatives), held-out spatial blocks. "
            "'Spec.' columns = share of that kind of negative correctly rejected.\n")
    (out / "results.md").write_text("\n".join(lines) + note, encoding="utf-8")
    print(res.to_string(index=False))

    plt.figure(figsize=(6, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=2.2 if "hybrid" in name else 1.2, label=f"{name} ({roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)"); plt.legend(fontsize=8, title="Model (AUC)")
    plt.tight_layout(); plt.savefig(out / "roc_curves.png", dpi=200); plt.close()

    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["No LS", "LS"]); ax.set_yticks([0, 1], ["No LS", "LS"])
        ax.set_title(name, fontsize=9); ax.set_xlabel("Predicted")
    np.atleast_1d(axes)[0].set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(out / "confusion_matrices.png", dpi=200); plt.close()

    hists = {m: out / "models" / f"{m}_history.json" for m in ("cnn", "lstm", "hybrid")}
    hists = {m: json.loads(f.read_text()) for m, f in hists.items() if f.exists()}
    if hists:
        fig, axes = plt.subplots(1, len(hists), figsize=(4.2 * len(hists), 3.4), squeeze=False)
        for ax, (m, h) in zip(axes[0], hists.items()):
            e = [r["epoch"] for r in h]
            ax.plot(e, [r["train_loss"] for r in h], label="train loss")
            ax.plot(e, [r["val_loss"] for r in h], label="val loss")
            ax.plot(e, [r["val_acc"] for r in h], ":", label="val accuracy")
            ax.set_title(m.upper() if m != "hybrid" else "CNN-LSTM hybrid"); ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=200); plt.close()

    fi = out / "rf_feature_importance.json"
    if fi.exists():
        imp = pd.Series(json.loads(fi.read_text())).sort_values().tail(15)
        plt.figure(figsize=(6, 4.5))
        plt.barh(imp.index, imp.values * 100)
        plt.xlabel("Importance (%)"); plt.title("Random Forest feature importance")
        plt.tight_layout(); plt.savefig(out / "feature_importance.png", dpi=200); plt.close()
    print(f"  tables and figures written to {out}")
