"""Pasul 2: alertele GDACS (inundații FL, incendii WF) și potrivirea lor cu activările EMS.

Surse:
  - www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH   lista de evenimente (pagini de 100)
  - www.gdacs.org/gdacsapi/api/events/geteventdata          detaliul: severitate, episoade, impact „sendai”
  - rapidmapping.../public-activations-info/                gdacsId declarat de EMS (doar EMSR656+),
                                                            folosit ca adevăr de referință pentru potrivire

GDACS nu acoperă furtunile extratropicale din Europa (doar cicloni tropicali, TC), deci activările EMS
de tip „storm” rămân de regulă fără pereche GDACS.

Potrivire EMS ↔ GDACS (pentru activările fără gdacsId):
  - același tip (flood ↔ FL, fire ↔ WF), țară GDACS printre țările EMS;
  - activationTime EMS în [fromdate − 7 zile, todate + 7 zile];
  - distanța de la AOI-urile EMS (sau centroid, dacă nu există AOI) la zona afectată GDACS
    (poligonul Poly_Affected, altfel Poly_Global, altfel punctul) ≤ 25 km. Punctul singur nu ajunge:
    la inundațiile mari (ex. furtuna Boris, FL1102899) centroidul GDACS e la sute de km de AOI-urile EMS;
  - dintre candidați: cea mai mică distanță, apoi cel mai mic decalaj de timp.

Ieșiri (data/interim/):
  - gdacs_events.csv   alertele GDACS din Europa, 2016-12 – 2026-01
  - gdacs_match.csv    un rând = o activare EMS: gdacs_id, metoda, distanța, decalajul, trăsăturile GDACS
  - gdacs_sendai.csv   înregistrările de impact (evacuați, decedați, afectați) pentru alertele potrivite

Rulare: uv run python pipeline/02_gdacs.py [--refresh]
"""
import argparse
import json
import time

import geopandas as gpd
import pandas as pd
import requests
from shapely import wkt
from shapely.geometry import Point, shape
from tqdm import tqdm

from config import EUROPE, EUROPE_ISO3, INTERIM, RAW

API = "https://www.gdacs.org/gdacsapi/api/events"
RM_INFO = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"
RAW_G = RAW / "gdacs"
TYPE_MAP = {"flood": "FL", "fire": "WF"}
MAX_KM = 25  # distanța maximă de la AOI-ul EMS la zona afectată GDACS
DT_DAYS = 7

# numele folosite de GDACS diferă uneori de cele EMS
GDACS_NAMES = {"Czechia": ["Czechia", "Czech Republic"], "North Macedonia": ["North Macedonia", "Macedonia"],
               "Kosovo*": ["Kosovo"], "Moldova": ["Moldova", "Republic of Moldova"]}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "MEDE-U19 research pipeline (UNSTPB)"


def get_json(url, params=None):
    for attempt in range(4):
        try:
            r = SESSION.get(url, params=params, timeout=90)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            if r.status_code in (204, 404) or not r.text.strip():
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    return None


def fetch_country(name, etype):
    out, page = [], 1
    while True:
        d = get_json(f"{API}/geteventlist/SEARCH", {
            "eventlist": etype, "fromDate": "2016-12-01", "toDate": "2026-01-31",
            "alertlevel": "Green;Orange;Red", "country": name, "pagenumber": page})
        feats = (d or {}).get("features") or []
        out += feats
        if len(feats) < 100:
            return out
        page += 1


def fetch_events(refresh):
    path = RAW_G / "events_europe.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    feats = {}
    names = sorted({n for c in EUROPE for n in GDACS_NAMES.get(c, [c])})
    for name in tqdm(names, desc="GDACS pe țări"):
        for etype in TYPE_MAP.values():
            for f in fetch_country(name, etype):
                p = f["properties"]
                feats[(p["eventtype"], p["eventid"])] = f
    out = list(feats.values())
    path.write_text(json.dumps(out, ensure_ascii=False))
    return out


def events_frame(feats):
    rows = []
    for f in feats:
        p = f["properties"]
        sev = p.get("severitydata") or {}
        rows.append({
            "gdacs_id": f"{p['eventtype']}{p['eventid']}", "eventtype": p["eventtype"], "eventid": p["eventid"],
            "episodeid": p.get("episodeid"), "name": p.get("name"), "country": p.get("country"),
            "affected_iso3": "|".join(c["iso3"] for c in p.get("affectedcountries") or []),
            "fromdate": p.get("fromdate"), "todate": p.get("todate"),
            "alertlevel": p.get("alertlevel"), "alertscore": p.get("alertscore"),
            "episodealertscore": p.get("episodealertscore"),
            "severity": sev.get("severity"), "severity_text": sev.get("severitytext"),
            "severity_unit": sev.get("severityunit"), "source": p.get("source"),
            "geometry": Point(f["geometry"]["coordinates"][:2]),
        })
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    g["alertlevel"] = g["alertlevel"].str.capitalize()
    # filtrul pe țară al API-ului lasă să treacă și evenimente din afara Europei
    g = g[g["affected_iso3"].str.split("|").map(lambda l: any(c in EUROPE_ISO3 for c in l))]
    g["fromdate"] = pd.to_datetime(g["fromdate"])
    g["todate"] = pd.to_datetime(g["todate"])
    return g


def fetch_declared_ids(refresh):
    path = RAW_G / "ems_declared_gdacs.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    d = get_json(RM_INFO, {"limit": 1000})
    out = {x["code"]: x["gdacsId"] for x in d["results"] if x.get("gdacsId")}
    path.write_text(json.dumps(out))
    return out


def ems_geometries():
    ev = pd.read_csv(INTERIM / "ems_events.csv")
    aoi = gpd.read_file(INTERIM / "ems_aoi.gpkg", layer="aoi")
    aoi = aoi[aoi.geometry.notna()].dissolve("emsr_id")[["geometry"]]
    ev["geometry"] = ev["emsr_id"].map(aoi.geometry)
    ev["geom_from"] = ev["geometry"].map(lambda g: "aoi" if g is not None else "centroid")
    ev["geometry"] = [g if g is not None else wkt.loads(c) for g, c in zip(ev["geometry"], ev["centroid"])]
    ev = gpd.GeoDataFrame(ev, geometry="geometry", crs=4326)
    ev["activation_time"] = pd.to_datetime(ev["activation_time"])
    return ev


def affected_geometry(gid, fallback):
    path = RAW_G / "geometry" / f"{gid}.json"
    if path.exists():
        d = json.loads(path.read_text())
    else:
        d = get_json(f"{API.rsplit('/', 1)[0]}/polygons/getgeometry", {"eventtype": gid[:2], "eventid": gid[2:]})
        path.write_text(json.dumps(d))
    feats = {f["properties"].get("Class"): f for f in (d or {}).get("features", [])}
    for cls in ("Poly_Affected", "Poly_Global"):
        if cls in feats:
            return shape(feats[cls]["geometry"]), cls
    return fallback, "point"


def match(ev, gd, declared):
    (RAW_G / "geometry").mkdir(exist_ok=True)
    rows = []
    for _, e in tqdm(ev.iterrows(), total=len(ev), desc="potrivire"):
        etype = TYPE_MAP.get(e["category"])
        decl = declared.get(e["emsr_id"])
        cand = gd[(gd["eventtype"] == etype)] if etype else gd.iloc[0:0]
        cand = cand[(cand["fromdate"] - pd.Timedelta(days=DT_DAYS) <= e["activation_time"])
                    & (e["activation_time"] <= cand["todate"] + pd.Timedelta(days=DT_DAYS))]
        best, n = None, 0
        if len(cand):
            geoms = [affected_geometry(c.gdacs_id, c.geometry) for c in cand.itertuples()]
            cg = gpd.GeoSeries([g for g, _ in geoms], crs=4326, index=cand.index).to_crs(3035)
            ems_geom = gpd.GeoSeries([e.geometry], crs=4326).to_crs(3035).iloc[0]
            cand = cand.assign(dist_km=cg.distance(ems_geom) / 1000, geom_class=[c for _, c in geoms],
                               dt_days=(e["activation_time"] - cand["fromdate"]).dt.total_seconds() / 86400)
            cand = cand[cand["dist_km"] <= MAX_KM].sort_values(["dist_km", "dt_days"], key=abs)
            n = len(cand)
            best = cand.iloc[0] if n else None
        rows.append({
            "emsr_id": e["emsr_id"], "category": e["category"], "geom_from": e["geom_from"],
            "declared_gdacs_id": decl,
            "matched_gdacs_id": best["gdacs_id"] if best is not None else None,
            "n_candidates": n,
            "dist_km": best["dist_km"] if best is not None else None,
            "dt_days": best["dt_days"] if best is not None else None,
            "gdacs_geom_class": best["geom_class"] if best is not None else None,
        })
    m = pd.DataFrame(rows)
    # gdacsId declarat de EMS are prioritate; altfel potrivirea spațio-temporală
    m["gdacs_id"] = m["declared_gdacs_id"].where(m["declared_gdacs_id"].notna(), m["matched_gdacs_id"])
    m["match_method"] = (m["declared_gdacs_id"].notna().map({True: "declared", False: "spatiotemporal"})
                         .where(m["gdacs_id"].notna(), "none"))
    return m


def fetch_details(ids, refresh):
    (RAW_G / "detail").mkdir(exist_ok=True)
    rows, sendai = [], []
    for gid in tqdm(ids, desc="detalii GDACS"):
        etype, eid = gid[:2], gid[2:]
        path = RAW_G / "detail" / f"{gid}.json"
        if path.exists() and not refresh:
            d = json.loads(path.read_text())
        else:
            d = get_json(f"{API}/geteventdata", {"eventtype": etype, "eventid": eid})
            path.write_text(json.dumps(d, ensure_ascii=False))
        if not d:
            continue
        p = d["properties"]
        sev = p.get("severitydata") or {}
        rows.append({"gdacs_id": gid, "g_alertlevel": (p.get("alertlevel") or "").capitalize() or None, "g_alertscore": p.get("alertscore"),
                     "g_severity": sev.get("severity"), "g_severity_text": sev.get("severitytext"),
                     "g_severity_unit": sev.get("severityunit"), "g_fromdate": p.get("fromdate"),
                     "g_todate": p.get("todate"), "g_n_episodes": len(p.get("episodes") or []),
                     "g_source": p.get("source"), "g_glide": p.get("glide") or None,
                     "g_description": p.get("htmldescription")})
        for s in p.get("sendai") or []:
            sendai.append({"gdacs_id": gid, **s})
    return pd.DataFrame(rows), pd.DataFrame(sendai)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    RAW_G.mkdir(parents=True, exist_ok=True)

    gd = events_frame(fetch_events(args.refresh))
    gd.drop(columns="geometry").assign(lon=gd.geometry.x, lat=gd.geometry.y).to_csv(
        INTERIM / "gdacs_events.csv", index=False)
    print(f"alerte GDACS în Europa: {len(gd)} | {gd['eventtype'].value_counts().to_dict()}")

    declared = fetch_declared_ids(args.refresh)
    ev = ems_geometries()
    m = match(ev, gd, declared)

    # validarea potrivirii pe activările care au gdacsId declarat de EMS
    chk = m[m["declared_gdacs_id"].notna() & m["category"].isin(TYPE_MAP)]
    if len(chk):
        hit = (chk["matched_gdacs_id"] == chk["declared_gdacs_id"]).sum()
        miss = chk["matched_gdacs_id"].isna().sum()
        print(f"validare pe {len(chk)} activări cu gdacsId declarat: corecte {hit}, "
              f"greșite {len(chk) - hit - miss}, fără pereche {miss}")

    det, sendai = fetch_details(sorted(m["gdacs_id"].dropna().unique()), args.refresh)
    if len(sendai):
        sendai.to_csv(INTERIM / "gdacs_sendai.csv", index=False)
        agg = (sendai.assign(v=pd.to_numeric(sendai.get("sendaivalue"), errors="coerce"))
               .pivot_table(index="gdacs_id", columns="sendainame", values="v", aggfunc="max"))
        agg.columns = [f"g_sendai_{c}" for c in agg.columns]
        det = det.merge(agg, left_on="gdacs_id", right_index=True, how="left")
    m = m.merge(det, on="gdacs_id", how="left")
    m.to_csv(INTERIM / "gdacs_match.csv", index=False)

    print(m.groupby(["category", "match_method"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
