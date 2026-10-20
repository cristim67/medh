"""Pasul 4: EM-DAT (pagube, victime, persoane afectate) legat de activările EMS.

Sursa: exportul Excel din public.emdat.be (cont academic), pus manual în data/raw/emdat/.
Licența EM-DAT nu permite redistribuirea datelor brute: data/raw/ nu se versionează.

Potrivire EMS ↔ EM-DAT (un eveniment EM-DAT poate acoperi mai multe activări EMS):
  1. GLIDE: codul GLIDE din EMS (stratul AOI) sau din alerta GDACS asociată apare în „External IDs” EM-DAT;
  2. altfel: același tip (flood ↔ Flood, fire ↔ Wildfire, storm ↔ Storm), aceeași țară (ISO3),
     activarea EMS în [start − 7 zile, end + 14 zile]; dintre candidați se preferă cei ale căror unități
     administrative EM-DAT apar în numele sau textul activării, apoi cel mai mic decalaj de start.
EM-DAT nu are geometrie utilizabilă (coordonate la ~6% din evenimente), deci potrivirea e la nivel de țară
și rămâne ambiguă când în aceeași țară și perioadă există mai multe evenimente de același tip.

Ieșiri (data/interim/):
  - emdat_europe.csv   evenimentele EM-DAT din Europa, 2017–2025, de tipurile relevante
  - emdat_match.csv    un rând = o activare EMS: dis_no, metoda, n_candidați, magnitudinea EM-DAT
Rulare: uv run python pipeline/04_emdat.py
"""
import json

import pandas as pd

from config import INTERIM, RAW, YEARS

TYPE_MAP = {"flood": "Flood", "fire": "Wildfire", "storm": "Storm"}
ISO3 = {  # numele de țară din EMS → ISO3
    "Albania": "ALB", "Austria": "AUT", "Belgium": "BEL", "Bosnia and Herzegovina": "BIH", "Bulgaria": "BGR",
    "Croatia": "HRV", "Cyprus": "CYP", "Czechia": "CZE", "Denmark": "DNK", "Estonia": "EST", "Finland": "FIN",
    "France": "FRA", "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN", "Iceland": "ISL", "Ireland": "IRL",
    "Italy": "ITA", "Kosovo": "XKX", "Kosovo*": "XKX", "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Malta": "MLT", "Moldova": "MDA", "Montenegro": "MNE", "Netherlands": "NLD", "North Macedonia": "MKD",
    "Norway": "NOR", "Poland": "POL", "Portugal": "PRT", "Romania": "ROU", "Serbia": "SRB", "Slovakia": "SVK",
    "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE", "Switzerland": "CHE", "Ukraine": "UKR",
    "United Kingdom": "GBR",
}
DAMAGE = "Total Damage, Adjusted ('000 US$)"


def load_emdat():
    path = sorted((RAW / "emdat").glob("*.xlsx"))[-1]
    d = pd.read_excel(path)
    d = d[(d["Region"] == "Europe") & d["Start Year"].between(*YEARS)
          & d["Disaster Type"].isin(TYPE_MAP.values())].copy()

    def date(prefix, fallback_day):
        y, m, day = d[f"{prefix} Year"], d[f"{prefix} Month"].fillna(1), d[f"{prefix} Day"].fillna(fallback_day)
        return pd.to_datetime(dict(year=y, month=m, day=day), errors="coerce")

    d["start"] = date("Start", 1)
    d["end"] = date("End", 28).fillna(d["start"])
    d["glide"] = d["External IDs"].fillna("").str.findall(r"GLIDE:([A-Z]{2}-\d{4}-\d{6})")
    d["admin_names"] = d["Admin Units"].map(
        lambda s: [v for u in json.loads(s) for k, v in u.items() if k.endswith("_name")] if isinstance(s, str) else [])
    return d, path.name


def glide_core(g):
    """FL-2024-000169-AUT → FL-2024-000169 (EM-DAT nu păstrează sufixul de țară)."""
    return "-".join(str(g).split("-")[:3]) if isinstance(g, str) and g[:2].isalpha() and "-" in g else None


def match(ems, d):
    rows = []
    for e in ems.itertuples():
        isos = {ISO3.get(c) for c in str(e.countries).split("|")} - {None}
        t = pd.Timestamp(e.activation_time[:10])
        glides = {glide_core(e.glide), glide_core(e.g_glide)} - {None}
        by_glide = d[d["glide"].map(lambda l: bool(glides & set(l)))] if glides else d.iloc[0:0]
        cand = d[(d["Disaster Type"] == TYPE_MAP[e.category]) & d["ISO"].isin(isos)
                 & (d["start"] - pd.Timedelta(days=7) <= t) & (t <= d["end"] + pd.Timedelta(days=14))]
        text = f"{e.name} {e.reason}".lower()
        cand = cand.assign(admin_hit=cand["admin_names"].map(lambda l: any(n.lower() in text for n in l)),
                           dt_days=(t - cand["start"]).dt.days.abs())
        cand = cand.sort_values(["admin_hit", "dt_days"], ascending=[False, True])
        if len(by_glide):
            best, method = by_glide.iloc[0], "glide"
        elif len(cand):
            best = cand.iloc[0]
            method = "country_time_admin" if best["admin_hit"] else "country_time"
        else:
            best, method = None, "none"
        rows.append({
            "emsr_id": e.emsr_id, "emdat_dis_no": best["DisNo."] if best is not None else None,
            "emdat_method": method, "emdat_n_candidates": len(cand),
            **({} if best is None else {
                "emdat_deaths": best["Total Deaths"], "emdat_affected": best["Total Affected"],
                "emdat_damage_adj_kusd": best[DAMAGE], "emdat_start": best["start"].date(),
                "emdat_subtype": best["Disaster Subtype"]}),
        })
    return pd.DataFrame(rows)


def main():
    d, fname = load_emdat()
    d.drop(columns=["admin_names"]).to_csv(INTERIM / "emdat_europe.csv", index=False)
    print(f"EM-DAT ({fname}): {len(d)} evenimente în Europa, {YEARS} | {d['Disaster Type'].value_counts().to_dict()}")

    ems = pd.read_csv(INTERIM / "ems_events.csv")
    gd = pd.read_csv(INTERIM / "gdacs_match.csv")[["emsr_id", "g_glide"]]
    ems = ems.merge(gd, on="emsr_id", how="left")
    m = match(ems, d)
    m.to_csv(INTERIM / "emdat_match.csv", index=False)

    print(m.merge(ems[["emsr_id", "category"]]).groupby(["category", "emdat_method"]).size()
          .unstack(fill_value=0).to_string())
    amb = (m["emdat_n_candidates"] > 1).sum()
    print(f"ambigue (> 1 candidat EM-DAT): {amb} | activări pe același DisNo: "
          f"{m['emdat_dis_no'].dropna().duplicated(keep=False).sum()}")


if __name__ == "__main__":
    main()
