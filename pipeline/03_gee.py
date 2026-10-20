"""Pasul 3: imaginea (Sentinel-1, Sentinel-2) și meteo (ERA5-Land) pe AOI-ul fiecărei activări EMS, din GEE.

Data de referință t0: eventTime EMS dacă există (EMSR656+), altfel activationTime.
Ferestre (config): pre = t0 + [−60, −10] zile, post = t0 + [+2, +30] zile; compozit median pe fiecare fereastră.

Geometria: uniunea AOI-urilor activării, simplificată la 100 m. Scara reduceRegion se adaptează la arie
(min. 20 m, astfel încât să rămână ~1e7 pixeli), iar scara folosită se salvează.

Trăsături (medii pe AOI, dacă nu se spune altfel):
  S2 (COPERNICUS/S2_SR_HARMONIZED, mască SCL): NDVI, NBR, MNDWI pre/post, dNDVI, dNBR, dMNDWI,
     fracția de pixeli clari pre/post, fracția și aria pixelilor cu dNBR > 0.27 (ars) și dMNDWI > 0.2 (apă nouă)
  S1 (COPERNICUS/S1_GRD, IW, VV+VH, dB): VV, VH pre/post, ΔVV, ΔVH, fracția și aria cu ΔVV < −3 dB (inundat)
  ERA5-Land (DAILY_AGGR; vântul din HOURLY): precipitații totale [t0−L, t0+2] și antecedente [t0−60, t0−10] (mm),
     temperatura maximă la 2 m și viteza maximă orară a vântului la 10 m pe [t0−L, t0+2].
     L = 5 zile când t0 = eventTime, 14 zile când t0 = activationTime (activarea vine uneori la o săptămână
     după eveniment, ex. EMSR334 / furtuna Vaia: eveniment 29.10.2018, activare 07.11.2018).
Număr de imagini pe fereastră pentru fiecare colecție (pentru raportul de acoperire).

Ieșire: data/interim/gee_features.csv (un rând = o activare); cache per activare în data/raw/gee/.
Rulare: uv run python pipeline/03_gee.py [--codes EMSR692,...] [--workers 8] [--refresh]
"""
import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import geopandas as gpd
import pandas as pd
from shapely import force_2d, make_valid
from shapely.geometry import mapping
from tqdm import tqdm

from config import GEE_PROJECT, INTERIM, POST_WINDOW, PRE_WINDOW, RAW

RAW_GEE = RAW / "gee"
MAX_PIXELS = 1e7
S2_CLEAR = [4, 5, 6, 7, 11]  # SCL: vegetație, sol, apă, neclasificat, zăpadă


def window(t0, w):
    return t0.advance(w[0], "day"), t0.advance(w[1], "day")


def s2_composite(geom, start, end):
    def prep(img):
        scl = img.select("SCL")
        clear = scl.remap(S2_CLEAR, [1] * len(S2_CLEAR), 0)
        img = img.updateMask(clear)
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
        mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        return ndvi.addBands([nbr, mndwi]).addBands(clear.rename("CLEAR"))
    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(start, end)
    return col.map(prep).median(), col.size()


def s1_composite(geom, start, end):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    return col.select(["VV", "VH"]).median(), col.size()


def era5(geom, t0, lead):
    """lead = câte zile înainte de t0 începe fereastra evenimentului (mai mare când t0 e data activării)."""
    col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    ev = col.filterDate(t0.advance(-lead, "day"), t0.advance(3, "day"))
    ante = col.filterDate(*window(t0, PRE_WINDOW))
    # viteza se calculează orar: maximul componentelor u și v (cu semn) din agregatul zilnic nu dă viteza maximă
    hourly = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(t0.advance(-lead, "day"), t0.advance(3, "day"))
    wind = hourly.map(lambda i: i.expression("sqrt(u*u + v*v)", {
        "u": i.select("u_component_of_wind_10m"), "v": i.select("v_component_of_wind_10m")}))
    img = ee.Image.cat([
        ev.select("total_precipitation_sum").sum().multiply(1000).rename("era5_tp_event_mm"),
        ante.select("total_precipitation_sum").sum().multiply(1000).rename("era5_tp_antecedent_mm"),
        ev.select("temperature_2m_max").max().subtract(273.15).rename("era5_t2m_max_c"),
        wind.max().rename("era5_wind10m_max_ms"),
    ])
    # celula ERA5-Land are ~9 km: AOI-urile mici primesc un buffer ca să atingă cel puțin o celulă
    return img.reduceRegion(ee.Reducer.mean(), geom.buffer(5000), 11132, bestEffort=True, maxPixels=1e9)


def counts(geom, t0):
    pre, post = window(t0, PRE_WINDOW), window(t0, POST_WINDOW)
    return ee.Dictionary({
        "s2_n_pre": s2_composite(geom, *pre)[1], "s2_n_post": s2_composite(geom, *post)[1],
        "s1_n_pre": s1_composite(geom, *pre)[1], "s1_n_post": s1_composite(geom, *post)[1],
    })


def features(geom, t0, scale, lead, has_s1, has_s2, tile_scale):
    """has_s1 / has_s2: dacă există scene în ambele ferestre; altfel blocul lipsește (gol de date, nu eroare)."""
    pre, post = window(t0, PRE_WINDOW), window(t0, POST_WINDOW)
    px_km2 = ee.Image.pixelArea().divide(1e6)
    bands, area_bands = [], []
    if has_s2:
        s2_pre, s2_post = s2_composite(geom, *pre)[0], s2_composite(geom, *post)[0]
        d2 = s2_post.select(["NDVI", "NBR", "MNDWI"]).subtract(s2_pre.select(["NDVI", "NBR", "MNDWI"]))
        # dNBR clasic e pre − post (pozitiv = ars); aici păstrăm post − pre pentru toți indicii și inversăm la prag
        burnt = d2.select("NBR").lt(-0.27).rename("s2_burnt")
        water = d2.select("MNDWI").gt(0.2).rename("s2_newwater")
        bands += [
            s2_pre.select(["NDVI", "NBR", "MNDWI"]).rename(["s2_ndvi_pre", "s2_nbr_pre", "s2_mndwi_pre"]),
            s2_post.select(["NDVI", "NBR", "MNDWI"]).rename(["s2_ndvi_post", "s2_nbr_post", "s2_mndwi_post"]),
            d2.rename(["s2_dndvi", "s2_dnbr", "s2_dmndwi"]),
            s2_pre.select("CLEAR").unmask(0).rename("s2_clear_pre"),
            s2_post.select("CLEAR").unmask(0).rename("s2_clear_post"),
            burnt, water,
        ]
        area_bands += [burnt.multiply(px_km2).rename("s2_burnt_km2"),
                       water.multiply(px_km2).rename("s2_newwater_km2")]
    if has_s1:
        s1_pre, s1_post = s1_composite(geom, *pre)[0], s1_composite(geom, *post)[0]
        d1 = s1_post.subtract(s1_pre)
        flooded = d1.select("VV").lt(-3).rename("s1_flooded")
        bands += [s1_pre.rename(["s1_vv_pre", "s1_vh_pre"]), s1_post.rename(["s1_vv_post", "s1_vh_post"]),
                  d1.rename(["s1_dvv", "s1_dvh"]), flooded]
        area_bands += [flooded.multiply(px_km2).rename("s1_flooded_km2")]

    out = era5(geom, t0, lead)
    if bands:
        means = ee.Image.cat(bands).reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9,
                                                 tileScale=tile_scale)
        sums = ee.Image.cat(area_bands).reduceRegion(ee.Reducer.sum(), geom, scale, maxPixels=1e9,
                                                     tileScale=tile_scale)
        out = out.combine(means).combine(sums)
    return out


def event_inputs():
    ev = pd.read_csv(INTERIM / "ems_events.csv")
    aoi = gpd.read_file(INTERIM / "ems_aoi.gpkg", layer="aoi")
    aoi = aoi[aoi.geometry.notna()]
    # unele AOI-uri vechi au coordonate 3D, pe care GEE le respinge ca GeoJSON invalid
    aoi["geometry"] = aoi.geometry.map(lambda g: make_valid(force_2d(g)))
    geom = aoi.to_crs(3035).dissolve("emsr_id").geometry.simplify(100).to_crs(4326)
    ev = ev[ev["emsr_id"].isin(geom.index)].copy()
    ev["t0"] = ev["event_time"].fillna(ev["activation_time"]).str[:10]
    ev["t0_source"] = ev["event_time"].notna().map({True: "event_time", False: "activation_time"})
    ev["geometry"] = ev["emsr_id"].map(geom)
    ev["area_km2"] = ev["emsr_id"].map(aoi.to_crs(3035).dissolve("emsr_id").area / 1e6)
    return ev


def run_one(row, refresh):
    path = RAW_GEE / f"{row.emsr_id}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    scale = max(20, math.ceil(math.sqrt(row.area_km2 * 1e6 / MAX_PIXELS)))
    geom = ee.Geometry(mapping(row.geometry))
    lead = 5 if row.t0_source == "event_time" else 14
    res, err, tile_scale = {}, None, 4
    try:
        n = counts(geom, ee.Date(row.t0)).getInfo()
        has_s1 = n["s1_n_pre"] > 0 and n["s1_n_post"] > 0
        has_s2 = n["s2_n_pre"] > 0 and n["s2_n_post"] > 0
        # la memorie / timp depășit: pixel de 2× mai mare și calcul împărțit în mai multe bucăți
        for attempt in range(3):
            try:
                res = features(geom, ee.Date(row.t0), scale, lead, has_s1, has_s2, tile_scale).getInfo()
                err = None
                break
            except ee.EEException as e:
                err = str(e)[:300]
                if "memory" not in err and "timed out" not in err:
                    break
                scale, tile_scale = scale * 2, 16
        res = {**n, **res}
    except ee.EEException as e:
        err = str(e)[:300]
    out = {"emsr_id": row.emsr_id, "t0": row.t0, "t0_source": row.t0_source,
           "aoi_area_km2": row.area_km2, "gee_scale_m": scale, "gee_tile_scale": tile_scale,
           "gee_error": err, **res}
    if err is None:
        path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ee.Initialize(project=GEE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
    RAW_GEE.mkdir(parents=True, exist_ok=True)
    ev = event_inputs()
    if args.codes:
        ev = ev[ev["emsr_id"].isin(args.codes.split(","))]

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run_one, r, args.refresh) for r in ev.itertuples()]
        for f in tqdm(as_completed(futs), total=len(futs)):
            rows.append(f.result())
    out = pd.DataFrame(rows).sort_values("emsr_id")
    # fișierul final conține toate activările din cache (și din rulările parțiale cu --codes), plus erorile curente
    cached = pd.DataFrame([json.loads(p.read_text()) for p in sorted(RAW_GEE.glob("EMSR*.json"))])
    failed = out[out["gee_error"].notna() & ~out["emsr_id"].isin(cached.get("emsr_id", []))]
    pd.concat([cached, failed], ignore_index=True).sort_values("emsr_id").to_csv(
        INTERIM / "gee_features.csv", index=False)

    ok = out["gee_error"].isna()
    print(f"activări cu AOI: {len(out)} | GEE reușit: {ok.sum()} | erori: {(~ok).sum()}")
    if (~ok).any():
        print(out.loc[~ok, ["emsr_id", "gee_error"]].head(10).to_string())


if __name__ == "__main__":
    main()
