"""Config loading, the target raster grid, and GeoTIFF helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from affine import Affine
from pyproj import Transformer
from rasterio.warp import transform_bounds


class Config(dict):
    """Dict with a root folder; cfg.path("data/x") resolves against it."""

    root: Path

    def path(self, *parts) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def folder(self, *parts) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str) -> Config:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    cfg.root = path.parent
    return cfg


@dataclass
class Grid:
    crs: str
    transform: Affine
    height: int
    width: int
    res: float

    def lonlat_to_rowcol(self, lon, lat):
        t = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        x, y = t.transform(np.asarray(lon, float), np.asarray(lat, float))
        col = np.floor((x - self.transform.c) / self.res).astype(int)
        row = np.floor((self.transform.f - y) / self.res).astype(int)
        return row, col

    def rowcol_to_lonlat(self, row, col):
        x = self.transform.c + (np.asarray(col, float) + 0.5) * self.res
        y = self.transform.f - (np.asarray(row, float) + 0.5) * self.res
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return t.transform(x, y)

    def inside(self, row, col, margin=0):
        return (
            (row >= margin) & (row < self.height - margin)
            & (col >= margin) & (col < self.width - margin)
        )


def make_grid(cfg: Config) -> Grid:
    sa = cfg["study_area"]
    res = float(sa["resolution"])
    w, s, e, n = transform_bounds("EPSG:4326", sa["crs"], *sa["bbox"], densify_pts=21)
    w, s = math.floor(w / res) * res, math.floor(s / res) * res
    e, n = math.ceil(e / res) * res, math.ceil(n / res) * res
    return Grid(
        crs=sa["crs"],
        transform=Affine(res, 0, w, 0, -res, n),
        height=int(round((n - s) / res)),
        width=int(round((e - w) / res)),
        res=res,
    )


def write_tif(path, arr, grid: Grid, nodata=None, transform=None):
    arr = np.asarray(arr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if nodata is None and np.issubdtype(arr.dtype, np.floating):
        nodata = np.nan
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=arr.dtype, crs=grid.crs, transform=transform or grid.transform,
        nodata=nodata, compress="deflate", tiled=True,
    ) as dst:
        dst.write(arr, 1)


def read_tif(path, grid: Grid | None = None):
    with rasterio.open(path) as src:
        arr = src.read(1)
        if grid is not None and (src.height, src.width) != (grid.height, grid.width):
            raise ValueError(
                f"{path} is {src.height}x{src.width} but the study grid is "
                f"{grid.height}x{grid.width}. Delete it and rebuild, or fix the bbox."
            )
    return arr
