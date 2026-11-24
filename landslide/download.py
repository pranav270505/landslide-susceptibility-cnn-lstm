"""Step 1: fetch open data and put every layer on the study grid.

Outputs (data/interim): dem.tif, landcover.tif, ndvi.tif, roads.tif, plus rainfall CSVs.
Each layer is skipped if its file already exists, so you can drop in your own
raster (same grid) or re-run after a failed download without repeating work.

Sources, all free and without an account:
  DEM        Copernicus GLO-30 (AWS open data)
  Land cover ESA WorldCover 2020, 10 m (AWS open data)
  NDVI       Landsat 8 Collection 2 Level-2 (Microsoft Planetary Computer)
  Roads      OpenStreetMap (Overpass API)
  Rainfall   Open-Meteo archive (ERA5-based daily totals)
"""
from __future__ import annotations

import json
import math

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject

from . import rainfall
from .common import make_grid, write_tif

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
    GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
)


def warp_to_grid(src_path, grid, resampling, dtype="float32", fill=np.nan, band=1):
    """Reproject one band of any raster (local path or URL) onto the study grid."""
    dst = np.full((grid.height, grid.width), fill, dtype=dtype)
    with rasterio.Env(**GDAL_ENV), rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band), destination=dst,
            dst_transform=grid.transform, dst_crs=grid.crs,
            dst_nodata=fill, resampling=resampling,
        )
    return dst


def _tiles(bbox, step):
    w, s, e, n = bbox
    for lat in range(int(math.floor(s / step) * step), int(math.ceil(n)), step):
        for lon in range(int(math.floor(w / step) * step), int(math.ceil(e)), step):
            yield lat, lon


def _ns(lat, width=2):
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):0{width}d}"


def _ew(lon, width=3):
    return f"{'E' if lon >= 0 else 'W'}{abs(lon):0{width}d}"


def dem(cfg, grid):
    out = cfg.path("data", "interim", "dem.tif")
    if out.exists():
        return print("  dem.tif exists, skipping")
    mosaic = np.full((grid.height, grid.width), np.nan, np.float32)
    for lat, lon in _tiles(cfg["study_area"]["bbox"], 1):
        name = f"Copernicus_DSM_COG_10_{_ns(lat)}_00_{_ew(lon)}_00_DEM"
        url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
        print(f"  DEM tile {name}")
        try:
            tile = warp_to_grid(url, grid, Resampling.bilinear)
        except rasterio.errors.RasterioIOError as err:
            print(f"    not available ({err}); treating as sea / no data")
            continue
        mosaic = np.where(np.isnan(mosaic), tile, mosaic)
    if np.isnan(mosaic).all():
        raise RuntimeError("No DEM tiles could be read. Check your internet connection and bbox.")
    write_tif(out, mosaic, grid)


def landcover(cfg, grid):
    out = cfg.path("data", "interim", "landcover.tif")
    if out.exists():
        return print("  landcover.tif exists, skipping")
    mosaic = np.zeros((grid.height, grid.width), np.uint8)
    for lat, lon in _tiles(cfg["study_area"]["bbox"], 3):
        name = f"ESA_WorldCover_10m_2020_v100_{_ns(lat)}{_ew(lon)}_Map.tif"
        url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/{name}"
        print(f"  WorldCover tile {name}")
        tile = warp_to_grid(url, grid, Resampling.mode, dtype="uint8", fill=0)
        mosaic = np.where(mosaic == 0, tile, mosaic)
    write_tif(out, mosaic, grid, nodata=0)


def ndvi(cfg, grid):
    out = cfg.path("data", "interim", "ndvi.tif")
    if out.exists():
        return print("  ndvi.tif exists, skipping")
    import planetary_computer as pc
    from pystac_client import Client

    nd = cfg["ndvi"]
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(
        collections=["landsat-c2-l2"], bbox=cfg["study_area"]["bbox"], datetime=nd["date_range"],
        query={"eo:cloud_cover": {"lt": nd["max_cloud"]}, "platform": {"in": ["landsat-8"]}},
    ).items())
    if not items:
        raise RuntimeError("No Landsat scenes found. Widen ndvi.date_range or raise ndvi.max_cloud.")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    # keep the clearest scenes for every path/row so the whole area is covered
    per_tile, chosen = {}, []
    for it in items:
        k = (it.properties.get("landsat:wrs_path"), it.properties.get("landsat:wrs_row"))
        per_tile.setdefault(k, []).append(it)
    quota = max(1, int(nd["max_scenes"]) // len(per_tile))
    for group in per_tile.values():
        chosen += group[:quota]

    layers = []
    for it in chosen:
        print(f"  Landsat scene {it.id}")
        red = warp_to_grid(it.assets["red"].href, grid, Resampling.bilinear, fill=0)
        nir = warp_to_grid(it.assets["nir08"].href, grid, Resampling.bilinear, fill=0)
        qa = warp_to_grid(it.assets["qa_pixel"].href, grid, Resampling.nearest, dtype="uint16", fill=1)
        bad = (red == 0) | (nir == 0) | ((qa & 0b11010) != 0) | (qa == 1)  # fill, cloud, shadow
        red, nir = red * 2.75e-5 - 0.2, nir * 2.75e-5 - 0.2
        v = (nir - red) / np.maximum(nir + red, 1e-6)
        v[bad] = np.nan
        layers.append(np.clip(v, -1, 1).astype(np.float32))
    stack = np.stack(layers)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comp = np.nanmedian(stack, axis=0)
    missing = np.isnan(comp)
    if missing.all():
        raise RuntimeError("All Landsat pixels were masked as cloud. Change ndvi.date_range.")
    print(f"  NDVI coverage {100 * (1 - missing.mean()):.1f}% (gaps filled with the median)")
    comp[missing] = np.nanmedian(comp)
    write_tif(out, comp.astype(np.float32), grid)


def lines_to_raster(lines_lonlat, grid):
    """Burn a list of [(lon, lat), ...] lines into a 0/1 raster on the grid."""
    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    shapes = []
    for line in lines_lonlat:
        if len(line) < 2:
            continue
        xs, ys = t.transform([p[0] for p in line], [p[1] for p in line])
        shapes.append(({"type": "LineString", "coordinates": list(zip(xs, ys))}, 1))
    if not shapes:
        return np.zeros((grid.height, grid.width), np.uint8)
    return features.rasterize(
        shapes, out_shape=(grid.height, grid.width), transform=grid.transform,
        fill=0, dtype="uint8", all_touched=True,
    )


OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]


def _local_roads(cfg, grid):
    """Use data/raw/roads.(shp|gpkg|geojson) if the user supplied one."""
    for ext in (".shp", ".gpkg", ".geojson"):
        f = cfg.root / "data" / "raw" / f"roads{ext}"
        if f.exists():
            import geopandas as gpd
            gdf = gpd.read_file(f).to_crs(grid.crs)
            print(f"  using your road layer {f.name} ({len(gdf)} features)")
            return features.rasterize(
                [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
                out_shape=(grid.height, grid.width), transform=grid.transform,
                fill=0, dtype="uint8", all_touched=True)
    return None


def roads(cfg, grid):
    out = cfg.path("data", "interim", "roads.tif")
    if out.exists():
        return print("  roads.tif exists, skipping")
    burned = _local_roads(cfg, grid)
    if burned is None:
        raw = cfg.path("data", "raw", "osm_roads.json")
        if not raw.exists():
            w, s, e, n = cfg["study_area"]["bbox"]
            q = (f'[out:json][timeout:600];way["highway"~"^({cfg["roads"]["classes"]})$"]'
                 f"({s},{w},{n},{e});out geom;")
            headers = {"User-Agent": "landslide-cnn-lstm student project (python-requests)",
                       "Accept": "application/json"}
            text = None
            for url in OVERPASS:
                print(f"  querying OpenStreetMap via {url.split('/')[2]} (can take a few minutes)")
                try:
                    r = requests.post(url, data={"data": q}, headers=headers, timeout=900)
                    if r.status_code == 200 and r.text.lstrip().startswith("{"):
                        text = r.text
                        break
                    print(f"    server answered {r.status_code}; trying the next one")
                except requests.RequestException as err:
                    print(f"    failed ({type(err).__name__}); trying the next one")
            if text is None:
                raise RuntimeError(
                    "No Overpass server returned the roads. Either retry later, or download a "
                    "roads layer for Kerala yourself (for example the Geofabrik 'southern-zone' "
                    "extract, layer gis_osm_roads_free_1.shp) and save it as data/raw/roads.shp.")
            raw.write_text(text, encoding="utf-8")
        data = json.loads(raw.read_text(encoding="utf-8"))
        lines = [[(p["lon"], p["lat"]) for p in el["geometry"]]
                 for el in data["elements"] if el.get("type") == "way" and "geometry" in el]
        print(f"  {len(lines)} road segments")
        burned = lines_to_raster(lines, grid)
    write_tif(out, burned, grid, nodata=255)


def rain(cfg, grid):
    cells = rainfall.bbox_cells(cfg)
    print(f"  {len(cells)} rainfall cells cover the study area")
    rainfall.ensure_cells(cfg, cells)


def run(cfg):
    grid = make_grid(cfg)
    print(f"Study grid: {grid.height} x {grid.width} pixels at {grid.res:g} m ({grid.crs})")
    for name, fn in [("Rainfall", rain), ("DEM", dem), ("Land cover", landcover),
                     ("NDVI", ndvi), ("Roads", roads)]:
        print(f"[{name}]")
        fn(cfg, grid)
