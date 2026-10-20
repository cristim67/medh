"""Pasul 1: activările Copernicus EMS Rapid Mapping (Europa, 2017–2025, inundații/incendii/furtuni).

Surse:
  - lista completă:  mapping.emergency.copernicus.eu/activations/api/activations/
  - EMSR656+:        rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/
                     (text complet, eventTime, poligoane AOI, statistici de impact pe produs)
  - EMSR < 656:      pagina HTML a activării (text complet, link-uri la pachetele vectoriale);
                     din fiecare pachet se citesc doar straturile area_of_interest, observed_event
                     și source, prin HTTP range (remotezip), fără a descărca arhiva întreagă.

Ieșiri (data/interim/):
  - ems_events.csv       un rând = o activare (cheia: emsr_id)
  - ems_aoi.gpkg         strat `aoi` (un rând = un AOI) și strat `observed_event` (poligoanele mapate)
  - ems_sources.csv      imaginile folosite de EMS pe fiecare AOI (senzor, dată, fază pre/post)

Rulare: uv run python pipeline/01_ems.py [--limit N] [--refresh]
"""
import argparse
import json
import re
import time
import geopandas as gpd
import pandas as pd
import requests
from bs4 import BeautifulSoup
from remotezip import RemoteZip
from shapely import wkt
from tqdm import tqdm

from config import CATEGORIES, EUROPE, INTERIM, RAW, YEARS

LIST_API = "https://mapping.emergency.copernicus.eu/activations/api/activations/"
PAGE_URL = "https://mapping.emergency.copernicus.eu/activations/{code}/"
RM_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"

RAW_EMS = RAW / "ems"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "MEDE-U19 research pipeline (UNSTPB)"


def get(url, **kw):
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=90, **kw)
            if r.status_code < 500:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET a eșuat: {url}")


def cached_json(path, fetch, refresh, cache_empty=True):
    if path.exists() and not refresh:
        data = json.loads(path.read_text())
        if data is not None or cache_empty:
            return data
    data = fetch()
    if data is not None or cache_empty:
        path.write_text(json.dumps(data, ensure_ascii=False))
    return data


# ---------- lista și filtrul ----------

def fetch_list():
    url, out = f"{LIST_API}?limit=200", []
    while url:
        d = get(url).json()
        out += d["results"]
        url = d["next"] and d["next"].replace("http://", "https://")
    return out


def select_events(items):
    df = pd.DataFrame(items)
    df["category"] = df["category"].map(lambda c: c["slug"] if c else None)
    df["countries"] = df["countries"].map(lambda l: [c["short_name"] for c in l])
    df["year"] = df["activationTime"].str[:4].astype(int)
    keep = (
        df["code"].str.startswith("EMSR")
        & df["year"].between(*YEARS)
        & df["category"].isin(CATEGORIES)
        & df["countries"].map(lambda l: len(l) > 0 and all(c in EUROPE for c in l))
    )
    return df[keep].sort_values("code").reset_index(drop=True)


# ---------- EMSR656+: API rapidmapping ----------

def fetch_rm_detail(code):
    r = get(RM_API, params={"code": code})
    if r.status_code != 200:
        return None
    res = r.json().get("results") or []
    return res[0] if res else None


def parse_rm(code, d):
    aois, stats, sources = [], [], []
    for a in d.get("aois", []):
        aois.append({"emsr_id": code, "aoi_num": a["number"], "aoi_name": a["name"],
                     "geometry": wkt.loads(a["extent"]) if a.get("extent") else None})
        for p in a.get("products", []):
            for img in p.get("images", []):
                sources.append({"emsr_id": code, "aoi_num": a["number"], "product": p["type"],
                                "sensor": img.get("sensorName"), "sensor_type": img.get("sensorType"),
                                "resolution_class": img.get("resolutionClass"),
                                "date": img.get("acquisitionTime"), "phase": "Post-event"})
            for group, items in (p.get("stats") or {}).items():
                for item, v in items.items():
                    stats.append({"emsr_id": code, "aoi_num": a["number"], "product_id": p["id"],
                                  "product": p["type"], "monitoring": p.get("monitoringNumber"),
                                  "group": group.strip(), "item": item.strip(), "unit": v.get("unit"),
                                  "total": v.get("total"), "affected": v.get("affected")})
    event = {"event_time": d.get("eventTime"), "reason": d.get("reason"),
             "sub_category": d.get("subCategory"), "activator": d.get("activator"),
             "glide": None, "detail_source": "rapidmapping_api"}
    return event, aois, stats, sources


# ---------- EMSR < 656: pagina HTML + pachete vectoriale ----------

def fix_mojibake(text):
    """Unele texte EMS au UTF-8 decodat ca MacRoman (ex. „Pedrog√£o” în loc de „Pedrógão”)."""
    if not text or "√" not in text:
        return text
    try:
        return text.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    reason = None
    if "Activation Reason" in lines:
        i = lines.index("Activation Reason") + 1
        stop = {"Areas", "Products", "Product list"}
        chunk = []
        while i < len(lines) and lines[i] not in stop:
            chunk.append(lines[i])
            i += 1
        reason = fix_mojibake(" ".join(chunk))
    zips = sorted({a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith("_vector.zip")})
    return reason, zips


def zip_rank(url):
    """Preferă delineation > grading > first estimate; apoi ultimul monitoring și ultima versiune."""
    name = url.rsplit("/", 1)[-1].upper()
    kind = 2 if "DELINEATION" in name or "_DEL" in name else 1 if "GRADING" in name or "_GRA" in name else 0
    mon = max([int(m) for m in re.findall(r"MONIT(\d+)", name)] or [0])
    ver = max([int(m) for m in re.findall(r"_V(\d+)", name)] or [0])
    return kind, mon, ver


def aoi_key(url, code):
    m = re.match(rf"{code}_(?:AOI)?(\d+)", url.rsplit("/", 1)[-1])
    return int(m.group(1)) if m else None


LAYER_KEYS = {
    # trei scheme de denumire în timp: area_of_interest (2017), area_of_interest_a (2018–2019), areaOfInterestA (2020+)
    "aoi": ("areaofinteresta", "areaofinterest"),
    "observed": ("observedeventa", "crisisinformationpoly"),
    "source": ("source",),
}


def norm_layer(member):
    stem = member.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    stem = re.sub(r"_r\d+_v\d+$", "", stem)
    return stem.replace("_", "")


def read_layer(rz, members, kind):
    for key in LAYER_KEYS[kind]:
        shp = [m for m in members if m.lower().endswith(".shp") and norm_layer(m).endswith(key)]
        dbf = [m for m in members if m.lower().endswith(".dbf") and norm_layer(m).endswith(key)]
        if shp or dbf:
            stem = (shp or dbf)[0][:-4]
            tmp = RAW_EMS / "_tmp"
            tmp.mkdir(exist_ok=True)
            for f in tmp.iterdir():
                f.unlink()
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                if stem + ext in members:
                    (tmp / f"layer{ext}").write_bytes(rz.read(stem + ext))
            if not shp:
                return pd.DataFrame(gpd.read_file(tmp / "layer.dbf"))
            g = gpd.read_file(tmp / "layer.shp")
            return g.to_crs(4326) if g.crs else g.set_crs(4326)
    return None


def read_vector_zip(url):
    with RemoteZip(url, session=SESSION) as rz:
        members = rz.namelist()
        return (read_layer(rz, members, "aoi"),
                read_layer(rz, members, "observed"),
                read_layer(rz, members, "source"))


def parse_legacy(code, html):
    reason, zips = parse_page(html)
    by_aoi = {}
    for z in zips:
        k = aoi_key(z, code)
        if k is not None and (k not in by_aoi or zip_rank(z) > zip_rank(by_aoi[k])):
            by_aoi[k] = z
    aois, observed, sources, glide = [], [], [], None
    for num, url in sorted(by_aoi.items()):
        try:
            aoi, obs, src = read_vector_zip(url)
        except Exception as e:  # pachet lipsă sau corupt: AOI-ul rămâne fără geometrie, se numără
            aois.append({"emsr_id": code, "aoi_num": num, "aoi_name": None, "geometry": None,
                         "vector_zip": url, "error": str(e)[:200]})
            continue
        geom = aoi.union_all() if aoi is not None and len(aoi) else None
        name = aoi["locality"].iloc[0] if aoi is not None and "locality" in aoi and len(aoi) else None
        if aoi is not None and "glide_no" in aoi and len(aoi):
            glide = glide or aoi["glide_no"].iloc[0]
        aois.append({"emsr_id": code, "aoi_num": num, "aoi_name": name, "geometry": geom,
                     "vector_zip": url, "error": None})
        if obs is not None and len(obs):
            obs = obs.assign(emsr_id=code, aoi_num=num, vector_zip=url)
            observed.append(obs[[c for c in ("emsr_id", "aoi_num", "event_type", "obj_desc",
                                              "det_method", "vector_zip", "geometry") if c in obs]])
        if src is not None and len(src):
            for _, s in src.iterrows():
                sources.append({"emsr_id": code, "aoi_num": num, "product": "vector",
                                "sensor": s.get("source_nam"), "sensor_type": None,
                                "resolution_class": s.get("sensor_gsd"), "date": s.get("src_date"),
                                "phase": s.get("eventphase")})
    event = {"event_time": None, "reason": reason, "sub_category": None, "activator": None,
             "glide": glide, "detail_source": "legacy_html+vector"}
    return event, aois, observed, sources


# ---------- magnitudinea mapată de EMS ----------

OBS_PREFIX = {"flood": "5", "fire": "8"}  # codurile event_type din stratul observed_event (5-Flood, 8-Wildfire)
AREA_GROUPS = {"Flooded area", "Burnt area", "Observed event"}


def ems_magnitude(ev):
    """Suprafața afectată (ha) și populația afectată, recalculate din fișierele salvate de pas.

    Legacy (< EMSR656): uniunea poligoanelor observed_event de tipul activării (AOI-urile se suprapun,
    o sumă simplă ar număra de două ori); dacă tipul nu se regăsește, toate poligoanele.
    API nou: statisticile produselor, ultimul monitoring pe fiecare AOI, sumă pe AOI-uri.
    """
    cat = ev.set_index("emsr_id")["category"]
    out = pd.DataFrame(index=ev["emsr_id"])
    gpkg = INTERIM / "ems_aoi.gpkg"
    if "observed_event" in gpd.list_layers(gpkg)["name"].tolist():
        obs = gpd.read_file(gpkg, layer="observed_event").to_crs(3035)
        obs["etype"] = obs["event_type"].astype(str).str.split("-").str[0]
        want = obs["emsr_id"].map(cat).map(OBS_PREFIX)
        typed = obs[want.isna() | (obs["etype"] == want)]
        typed = pd.concat([typed, obs[~obs["emsr_id"].isin(typed["emsr_id"])]])
        out["obs_area_ha"] = typed.dissolve("emsr_id").area / 1e4
    st_path = INTERIM / "ems_product_stats.csv"
    if st_path.exists() and st_path.stat().st_size > 1:
        st = pd.read_csv(st_path, keep_default_na=False, na_values=[""])  # „None” e o etichetă, nu lipsă
        st["group"] = st["group"].str.rstrip("*").str.strip()
        st = st[st["item"].eq("None")].sort_values("monitoring")
        last = st.groupby(["emsr_id", "aoi_num", "group"]).tail(1)
        out["api_area_ha"] = last[last["group"].isin(AREA_GROUPS)].groupby("emsr_id")["total"].sum()
        out["ems_pop_affected"] = last[last["group"] == "Estimated population"].groupby("emsr_id")["affected"].sum()
    out["ems_mapped_area_ha"] = out.get("obs_area_ha", pd.Series(dtype=float)).combine_first(
        out.get("api_area_ha", pd.Series(dtype=float)))
    cols = [c for c in ("ems_mapped_area_ha", "ems_pop_affected") if c in out]
    return out[cols].reset_index()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="doar primele N activări (test)")
    ap.add_argument("--codes", help="listă de coduri EMSR separate prin virgulă (test)")
    ap.add_argument("--refresh", action="store_true", help="ignoră cache-ul din data/raw/ems")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="recalculează doar magnitudinea din fișierele deja scrise în data/interim")
    args = ap.parse_args()

    if args.aggregate_only:
        ev = pd.read_csv(INTERIM / "ems_events.csv").drop(
            columns=["ems_mapped_area_ha", "ems_pop_affected"], errors="ignore")
        ev = ev.merge(ems_magnitude(ev), on="emsr_id", how="left")
        ev.to_csv(INTERIM / "ems_events.csv", index=False)
        print(f"cu suprafață mapată: {ev['ems_mapped_area_ha'].notna().sum()} | "
              f"cu populație afectată: {ev['ems_pop_affected'].notna().sum()}")
        return

    RAW_EMS.mkdir(parents=True, exist_ok=True)
    (RAW_EMS / "detail").mkdir(exist_ok=True)
    (RAW_EMS / "pages").mkdir(exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    items = cached_json(RAW_EMS / "activations_list.json", fetch_list, args.refresh)
    sel = select_events(items)
    print(f"activări în listă: {len(items)}; selectate (EMSR, Europa, {YEARS}, {CATEGORIES}): {len(sel)}")
    if args.codes:
        sel = sel[sel["code"].isin(args.codes.split(","))]
    if args.limit:
        sel = sel.head(args.limit)

    events, aois, observed, stats, sources = [], [], [], [], []
    for _, row in tqdm(sel.iterrows(), total=len(sel)):
        code = row["code"]
        # pentru EMSR656+ API-ul trebuie să răspundă; un răspuns gol e o eroare temporară, nu se păstrează în cache
        new_api = int(code[4:]) >= 656
        detail = cached_json(RAW_EMS / "detail" / f"{code}.json", lambda: fetch_rm_detail(code), args.refresh,
                             cache_empty=not new_api)
        if new_api and not detail:
            raise RuntimeError(f"{code}: API-ul rapidmapping nu a răspuns; rulează din nou cu --codes {code}")
        if detail:
            ev, a, st, src = parse_rm(code, detail)
            for x in a:
                x.update(vector_zip=None, error=None)
            stats += st
        else:
            page = RAW_EMS / "pages" / f"{code}.html"
            if not page.exists() or args.refresh:
                page.write_bytes(get(PAGE_URL.format(code=code)).content)
            ev, a, obs, src = parse_legacy(code, page.read_text(encoding="utf-8"))
            observed += obs
        aois += a
        sources += src
        events.append({
            "emsr_id": code, "name": row["name"], "category": row["category"],
            "countries": "|".join(row["countries"]), "activation_time": row["activationTime"],
            "centroid": row["centroid"], "n_aois_listed": row["n_aois"], "n_products_listed": row["n_products"],
            "closed": row["closed"], "list_snippet": row["search_snippet"], **ev,
        })

    ev = pd.DataFrame(events)
    ev["reason"] = ev["reason"].fillna(ev["list_snippet"])
    ev["text_len"] = ev["reason"].fillna("").str.len()
    g_aoi = gpd.GeoDataFrame(aois, geometry="geometry", crs=4326)
    g_aoi["aoi_area_km2"] = g_aoi.to_crs(3035).area / 1e6
    g_obs = None
    if observed:
        g_obs = gpd.GeoDataFrame(pd.concat(observed, ignore_index=True), geometry="geometry", crs=4326)
        g_obs = g_obs[g_obs.geometry.notna() & g_obs.geom_type.str.contains("Polygon")]
        g_obs["area_ha"] = g_obs.to_crs(3035).area / 1e4
    st, src = pd.DataFrame(stats), pd.DataFrame(sources)

    out_gpkg = INTERIM / "ems_aoi.gpkg"
    derived = ["ems_mapped_area_ha", "ems_pop_affected", "n_aois", "n_aois_with_geom", "aoi_area_km2"]
    if args.codes:  # actualizare parțială: se înlocuiesc doar rândurile activărilor date
        codes = set(sel["code"])

        def keep_others(old):
            return old[~old["emsr_id"].isin(codes)]
        ev = pd.concat([keep_others(pd.read_csv(INTERIM / "ems_events.csv")).drop(columns=derived, errors="ignore"),
                        ev], ignore_index=True).sort_values("emsr_id")
        g_aoi = pd.concat([keep_others(gpd.read_file(out_gpkg, layer="aoi")), g_aoi], ignore_index=True)
        if "observed_event" in gpd.list_layers(out_gpkg)["name"].tolist():
            g_obs = pd.concat([keep_others(gpd.read_file(out_gpkg, layer="observed_event")), g_obs],
                              ignore_index=True)
        st = pd.concat([keep_others(pd.read_csv(INTERIM / "ems_product_stats.csv", keep_default_na=False,
                                                na_values=[""])), st], ignore_index=True)
        src = pd.concat([keep_others(pd.read_csv(INTERIM / "ems_sources.csv")), src], ignore_index=True)

    g_aoi = gpd.GeoDataFrame(g_aoi, geometry="geometry", crs=4326)
    g_aoi.to_file(out_gpkg, layer="aoi", driver="GPKG", mode="w")
    if g_obs is not None:
        gpd.GeoDataFrame(g_obs, geometry="geometry", crs=4326).to_file(out_gpkg, layer="observed_event", driver="GPKG")
    st.to_csv(INTERIM / "ems_product_stats.csv", index=False)
    src.to_csv(INTERIM / "ems_sources.csv", index=False)

    ev = ev.merge(ems_magnitude(ev), on="emsr_id", how="left")
    aoi_summary = g_aoi.groupby("emsr_id").agg(n_aois=("aoi_num", "nunique"),
                                               n_aois_with_geom=("geometry", lambda s: s.notna().sum()),
                                               aoi_area_km2=("aoi_area_km2", "sum"))
    ev = ev.merge(aoi_summary, on="emsr_id", how="left")
    ev.to_csv(INTERIM / "ems_events.csv", index=False)

    has_geom = ev["n_aois_with_geom"].fillna(0) > 0
    print(f"evenimente: {len(ev)} | cu text: {(ev['text_len'] > 0).sum()} | cu AOI: {has_geom.sum()} "
          f"| cu suprafață mapată: {ev['ems_mapped_area_ha'].notna().sum()}")
    print(ev.groupby("detail_source").size().to_string())


if __name__ == "__main__":
    main()
