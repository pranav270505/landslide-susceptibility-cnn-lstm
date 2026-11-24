"""Landslide CNN-LSTM pipeline.

  python run.py download     fetch DEM, land cover, NDVI, roads, rainfall (needs internet)
  python run.py factors      slope, aspect, TWI, SPI, distances
  python run.py samples      build patches + rainfall sequences + split
  python run.py baselines    SVM, Decision Tree, Random Forest, MLP
  python run.py train        CNN-only, LSTM-only, hybrid CNN-LSTM
  python run.py report       metrics table and figures
  python run.py explain      which factors and which rainfall days the hybrid relies on
  python run.py map          susceptibility map
  python run.py all          everything above, in order

  python run.py demo         offline smoke test on made-up data (no internet needed)

Add --config path/to/config.yaml to use a different config file.
"""
import argparse
import os
import sys
import time

# Other software (PostgreSQL/PostGIS, QGIS, OSGeo4W) can point these variables at its own,
# incompatible map-projection files. Clearing them makes the libraries use their bundled copies.
for _var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_var, None)

STEPS = ["download", "factors", "samples", "baselines", "train", "report", "explain", "map"]


def run_step(step, cfg):
    print(f"\n=== {step} ===")
    t = time.time()
    if step == "download":
        from landslide import download
        download.run(cfg)
    elif step == "factors":
        from landslide import factors
        factors.run(cfg)
    elif step == "samples":
        from landslide import samples
        samples.run(cfg)
    elif step == "baselines":
        from landslide import train
        train.run_baselines(cfg)
    elif step == "train":
        from landslide import train
        train.run_deep(cfg)
    elif step == "report":
        from landslide import report
        report.run(cfg)
    elif step == "explain":
        from landslide import explain
        explain.run(cfg)
    elif step == "map":
        from landslide import mapping
        mapping.run(cfg)
    print(f"--- {step} done in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all", "demo"])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    from landslide.common import load_config
    if args.step == "demo":
        from landslide import demo
        cfg = load_config(demo.run(args.config))
        steps = STEPS[1:]                  # demo data replaces the download step
    else:
        cfg = load_config(args.config)
        steps = STEPS if args.step == "all" else [args.step]
    for s in steps:
        run_step(s, cfg)


if __name__ == "__main__":
    sys.exit(main())
