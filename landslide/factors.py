"""Step 2: derive the conditioning factors from the gridded layers.

Reads  data/interim/{dem,ndvi,landcover,roads}.tif
Writes data/processed/factors/*.tif  (open these in QGIS for figures)

Optional: any line layer saved as data/raw/extra_lines/<name>.(shp|gpkg|geojson)
(for example faults from GSI Bhukosh) becomes an extra factor "dist_<name>".
"""
from __future__ import annotations

import heapq

import numpy as np
from numba import njit
from scipy import ndimage

from . import rainfall
from .common import make_grid, read_tif, write_tif

CONTINUOUS = ["elevation", "slope", "northness", "eastness", "twi", "spi",
              "ndvi", "annual_rain", "dist_river", "dist_road"]


def slope_aspect(dem, res):
    d_row, d_col = np.gradient(dem, res)          # rows run north -> south
    dz_east, dz_north = d_col, -d_row
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_north)))
    aspect = np.arctan2(-dz_east, -dz_north)      # downslope direction, clockwise from north
    return slope.astype(np.float32), np.cos(aspect).astype(np.float32), np.sin(aspect).astype(np.float32)


@njit(cache=True)
def _fill_and_accumulate(dem, eps):
    """Priority-flood depression filling, D8 steepest descent, then flow accumulation."""
    H, W = dem.shape
    filled = dem.copy()
    closed = np.zeros((H, W), np.bool_)
    heap = [(filled[0, 0], 0, 0)]
    closed[0, 0] = True
    for r in range(H):
        for c in range(W):
            if (r == 0 or c == 0 or r == H - 1 or c == W - 1) and not closed[r, c]:
                closed[r, c] = True
                heapq.heappush(heap, (filled[r, c], r, c))
    dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    while len(heap) > 0:
        z, r, c = heapq.heappop(heap)
        for k in range(8):
            rr, cc = r + dr[k], c + dc[k]
            if rr < 0 or cc < 0 or rr >= H or cc >= W or closed[rr, cc]:
                continue
            closed[rr, cc] = True
            if filled[rr, cc] <= z:
                filled[rr, cc] = z + eps
            heapq.heappush(heap, (filled[rr, cc], rr, cc))
    # steepest-descent receiver of every cell (-1 = flows off the map)
    rec = np.full(H * W, -1, np.int64)
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            best, z = 0.0, filled[r, c]
            for k in range(8):
                rr, cc = r + dr[k], c + dc[k]
                drop = (z - filled[rr, cc]) / (1.41421356 if dr[k] != 0 and dc[k] != 0 else 1.0)
                if drop > best:
                    best = drop
                    rec[r * W + c] = rr * W + cc
    acc = np.ones(H * W, np.float64)
    order = np.argsort(filled.ravel())
    for i in range(H * W - 1, -1, -1):
        j = order[i]
        if rec[j] >= 0:
            acc[rec[j]] += acc[j]
    return acc.reshape(H, W)


def flow_accumulation(dem):
    """Number of cells draining through each cell (itself included)."""
    return _fill_and_accumulate(dem.astype(np.float64), 1e-4).astype(np.float32)


def distance_m(mask, res):
    if not mask.any():
        return np.full(mask.shape, 1e5, np.float32)
    return (ndimage.distance_transform_edt(~mask) * res).astype(np.float32)


def annual_rain(cfg, grid):
    """Mean annual rainfall (mm) from the cached daily series, interpolated to the grid."""
    import pandas as pd
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    cells = rainfall.bbox_cells(cfg)
    deg = cfg["rainfall"]["cell_deg"]
    lats = sorted({k[0] for k in cells}, reverse=True)
    lons = sorted({k[1] for k in cells})
    coarse = np.full((len(lats), len(lons)), np.nan, np.float32)
    for k in cells:
        f = rainfall.cell_file(cfg, k)
        if not f.exists():
            raise FileNotFoundError("Rainfall is missing: run 'python run.py download' first.")
        df = pd.read_csv(f)
        yearly = df.groupby(df["date"].str[:4])["rain_mm"].agg(["sum", "count"])
        full = yearly[yearly["count"] >= 365]["sum"]
        coarse[lats.index(k[0]), lons.index(k[1])] = (full if len(full) else yearly["sum"]).mean()
    out = np.full((grid.height, grid.width), np.nan, np.float32)
    reproject(coarse, out, src_transform=from_origin(lons[0] - deg / 2, lats[0] + deg / 2, deg, deg),
              src_crs="EPSG:4326", dst_transform=grid.transform, dst_crs=grid.crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    out[np.isnan(out)] = np.nanmean(coarse)
    return out


def run(cfg):
    grid = make_grid(cfg)
    interim = cfg.folder("data", "interim")
    out = cfg.folder("data", "processed", "factors")

    dem = read_tif(interim / "dem.tif", grid).astype(np.float32)
    valid = np.isfinite(dem) & (dem > -500)
    if not valid.any():
        raise RuntimeError("dem.tif has no valid pixels")
    dem_f = np.where(valid, dem, np.nanmin(dem[valid]))

    print("  slope and aspect")
    slope, north, east = slope_aspect(dem_f, grid.res)

    print("  flow accumulation (the slow part: a few minutes on a full district)")
    acc = flow_accumulation(dem_f)
    tan_b = np.maximum(np.tan(np.radians(slope)), 0.001)
    area = acc * grid.res                         # specific catchment area
    twi = np.log(area / tan_b)
    spi = np.log1p(area * tan_b)

    print("  distances to rivers and roads")
    streams = acc >= float(cfg["factors"]["stream_threshold_cells"])
    layers = {
        "elevation": dem_f, "slope": slope, "northness": north, "eastness": east,
        "twi": twi, "spi": spi,
        "ndvi": read_tif(interim / "ndvi.tif", grid).astype(np.float32),
        "annual_rain": annual_rain(cfg, grid),
        "dist_river": distance_m(streams, grid.res),
        "dist_road": distance_m(read_tif(interim / "roads.tif", grid) == 1, grid.res),
    }

    extra_dir = cfg.root / "data" / "raw" / "extra_lines"
    if extra_dir.exists():
        import geopandas as gpd
        from rasterio import features
        for f in sorted(extra_dir.iterdir()):
            if f.suffix.lower() not in (".shp", ".gpkg", ".geojson", ".kml"):
                continue
            gdf = gpd.read_file(f).to_crs(grid.crs)
            burned = features.rasterize(
                [(geom, 1) for geom in gdf.geometry if geom is not None],
                out_shape=(grid.height, grid.width), transform=grid.transform,
                fill=0, dtype="uint8", all_touched=True)
            layers[f"dist_{f.stem.lower()}"] = distance_m(burned == 1, grid.res)
            print(f"  extra factor dist_{f.stem.lower()}")

    for name, arr in layers.items():
        arr = arr.astype(np.float32)
        arr[~valid] = np.nan
        write_tif(out / f"{name}.tif", arr, grid)
    lc = read_tif(interim / "landcover.tif", grid).astype(np.uint8)
    lc[~valid] = 0
    write_tif(out / "landcover.tif", lc, grid, nodata=0)
    print(f"  wrote {len(layers) + 1} factor rasters to {out}")


def load(cfg, grid):
    """Return ({name: float32 array}, landcover uint8 array, valid mask)."""
    folder = cfg.root / "data" / "processed" / "factors"
    names = CONTINUOUS + sorted(
        p.stem for p in folder.glob("dist_*.tif") if p.stem not in CONTINUOUS)
    layers = {n: read_tif(folder / f"{n}.tif", grid) for n in names}
    lc = read_tif(folder / "landcover.tif", grid)
    valid = np.isfinite(layers["elevation"])
    return layers, lc, valid
