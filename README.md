# Landslide prediction with a hybrid CNN-LSTM (Western Ghats)

End-to-end pipeline for Team 7's project: real terrain and rainfall data for a
Kerala hill district, four classical baselines, CNN-only and LSTM-only
ablations, the hybrid CNN-LSTM, and a susceptibility map.

## 1. Set up

Python 3.10 or newer. A GPU is not needed; the models are small and train on a laptop CPU.

```
python -m venv .venv
.venv\Scripts\activate            (Windows)   |   source .venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
```

## 2. Check the code runs (no internet needed)

```
python run.py demo
```

This builds a small made-up world in `demo/` and runs every step on it in about
a minute. The numbers mean nothing; it only proves your install works.

## 3. Landslide inventory (the one manual download)

Put one or more landslide files in `data/raw/inventory/`. Shapefile, GeoPackage,
GeoJSON, KML and CSV all work; polygons are reduced to one point each.

Recommended: the 2018 Kerala monsoon inventory (4,728 landslides) by Hao et al.,
Earth System Science Data 12, 2020. Dataset DOI: https://doi.org/10.17026/dans-x6c-y7x2
Download it from that page and unzip the shapefile(s) into the folder. It is open
access (CC BY 4.0), a point shapefile, and has no per-landslide date field, so
the default event date below is what gets used.

If a file has a column with "date" in its name it is used; otherwise every
landslide gets `inventory.default_event_date` from `config.yaml` (16 Aug 2018,
the peak of the event). For a CSV the latitude/longitude columns are
auto-detected, or name them in `config.yaml`.

## 4. Run on real data

```
python run.py download     DEM, land cover, NDVI, roads, rainfall   (internet, 20-60 min)
python run.py factors      slope, aspect, TWI, SPI, distances
python run.py samples      patches + 30-day rainfall sequences + split
python run.py baselines    SVM, Decision Tree, Random Forest, MLP
python run.py train        CNN-only, LSTM-only, hybrid CNN-LSTM
python run.py report       metrics table and figures
python run.py explain      what the hybrid relies on (factors, rainfall days)
python run.py map          susceptibility map
```

or `python run.py all`. Every step caches its output, so a failed step can be re-run alone.

The study area is Idukki by default. To use Wayanad, change `study_area.bbox`
in `config.yaml` to `[75.78, 11.45, 76.45, 11.98]` and delete `data/interim`
and `data/processed`.

## 5. What you get (`outputs/`)

| File | Use |
|---|---|
| `results.csv`, `results.md` | Accuracy, Precision, Recall, F1, AUC for all 7 models |
| `roc_curves.png`, `confusion_matrices.png` | Result slides |
| `training_curves.png` | Loss / accuracy per epoch for the three deep models |
| `feature_importance.png` | Random Forest factor importance |
| `explain_factors.png` | Permutation importance of every factor and of rainfall, for the hybrid model |
| `explain_rain_days.png` | Which part of the 30-day rainfall window the hybrid relies on (5-day blocks) |
| `explain.csv` | The numbers behind both, plus terrain branch versus rainfall branch |
| `susceptibility.tif`, `susceptibility_class.tif`, `susceptibility.png` | The map (open the .tif files in QGIS) |
| `models/*.pt` | Trained weights |

`data/processed/factors/*.tif` are the individual factor maps, useful as figures.

## 6. Data sources

The `data/` folder (about 625 MB) is not stored in this repository. `python run.py download`
fetches everything below automatically, except the landslide inventory, which you
download by hand (section 3).

| Layer | Source | Resolution | Link |
|---|---|---|---|
| Elevation, slope, aspect, TWI, SPI, distance to rivers | Copernicus GLO-30 DEM (AWS open data) | 30 m | https://registry.opendata.aws/copernicus-dem/ |
| NDVI | Landsat 8 Collection 2 Level-2, Jan-Mar 2018 (Microsoft Planetary Computer) | 30 m | https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2 |
| Land use / land cover | ESA WorldCover 2020 (AWS open data) | 10 m, resampled | https://esa-worldcover.org/en |
| Distance to roads | OpenStreetMap (Overpass API) | vector | https://overpass-api.de (bulk alternative: https://download.geofabrik.de/asia/india/southern-zone.html) |
| Daily rainfall, mean annual rainfall | Open-Meteo archive (ERA5 reanalysis) | about 10-25 km | https://open-meteo.com/en/docs/historical-weather-api |
| Landslides | Hao et al. 2020 inventory | points | https://doi.org/10.17026/dans-x6c-y7x2 |

None of these need an account or API key. If the OpenStreetMap servers refuse
the roads query, save any roads layer as `data/raw/roads.shp` (for example
`gis_osm_roads_free_1.shp` from the Geofabrik India southern-zone extract) and
re-run `download`.

## 7. Checked against the base paper (Teng et al. 2026, Scientific Reports)

Same as the paper:
- CNN branch: 3 conv blocks, 32/64/128 filters, 3x3, ReLU, 2x2 max-pool, batch norm.
- LSTM branch: 2 stacked layers, 128 and 64 units, 30-day window.
- Late fusion by concatenation, dense 128 and 64 (ReLU), sigmoid output.
- Adam, learning rate 0.001, batch 32, 50 epochs, dropout 0.2, L2 0.01,
  best-validation-loss model kept.
- Min-Max scaling, one-hot land use, 1:1 landslide to non-landslide samples, 70/15/15.
- Factors: elevation, slope, aspect, land use, TWI, SPI, mean annual rainfall,
  distance to rivers and roads.
- Baselines SVM, DT, RF, MLP; Accuracy, Precision, Recall, F1, AUC; three map classes.

Different from the paper:
- **Framework.** The paper used TensorFlow/Keras; this code is PyTorch. The
  network is the same, except PyTorch has no recurrent dropout, so the 0.2
  dropout sits between the two LSTM layers.
- **Missing factors.** Distance to faults and engineering rock group have no
  open, scriptable source. Save a fault layer (for example from GSI Bhukosh) as
  `data/raw/extra_lines/fault.shp` and re-run `factors`; it becomes `dist_fault`.
- **Extra factor.** NDVI is included (it is in your Review 1 plan); the paper
  describes NDVI but leaves it out of its final predictor list.
- **No reservoir level** in the time series; rainfall only.
- **Rainfall source.** The paper used station data; this uses Open-Meteo, which
  is coarse: slopes in the same cell share one series. To use IMD gridded data,
  write one CSV per cell (`date,rain_mm`) into `data/raw/rainfall/`.
- **Negative samples.** The paper does not say what rainfall sequence its
  non-landslide points get. Here half the negatives are other locations on the
  event date, and half are landslide slopes on earlier monsoon dates when they
  did not fail. Without the second kind a single-event inventory gives every
  sample the same date and the LSTM has nothing to learn. In `results.md` the
  "Spec." columns show the CNN rejecting spatial negatives, the LSTM rejecting
  temporal ones, and only the hybrid doing both.
- **Split.** The paper used a stratified random split and itself notes this can
  inflate scores. The default here is a 5 km spatial block split.
- **Patch size.** The paper does not state one; 32 x 32 pixels is my choice.
- **Baseline settings.** The paper gives no kernel, depth or tree count, so
  these use ordinary defaults.
- The paper reports no numbers for CNN-only and LSTM-only; yours will be new.

To replicate the paper as closely as open data allows, set in `config.yaml`:
`samples.temporal_negative_fraction: 0` and `samples.split_method: random`,
add the fault layer, then re-run from `factors`.

## 8. Things to know before presenting

- With the 2018 inventory alone, all landslides share one extreme-rainfall
  event. The temporal branch then mostly learns "was this August 2018", which is
  easy, so its scores will look strong. Adding dated landslides from other years
  (2019 Puthumala/Kavalappara, 2020 Pettimudi, NASA's Global Landslide Catalog)
  as extra files in the inventory folder makes the temporal result much more
  convincing. Extend `rainfall.end` to cover their dates.
- WorldCover (2020) and OpenStreetMap (today) post-date the 2018 event. Both
  are slow-changing, but say so if asked.
- The map is for one rainfall scenario (`map.date`). Change the date to show how
  the map responds to a dry versus a wet month.

## 9. Code map

```
run.py                 command line entry point
config.yaml            every setting
landslide/download.py  step 1   data acquisition
landslide/factors.py   step 2   terrain factors
landslide/samples.py   step 3   inventory, negatives, patches, split, normalisation
landslide/models.py    CNN branch, LSTM branch, fusion classifier
landslide/train.py     steps 4-5 baselines and deep models
landslide/report.py    step 6   metrics and figures
landslide/explain.py   step 7   permutation importance for the hybrid
landslide/mapping.py   step 8   susceptibility map
landslide/rainfall.py  rainfall download and lookup
landslide/demo.py      offline smoke test
```
