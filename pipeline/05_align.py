"""Pasul 5: tabelul aliniat (un rând = o activare EMS) și raportul de acoperire.

Cheia: emsr_id. Alinierea:
  - spațială: AOI-urile EMS (uniune) sunt geometria comună pentru S1, S2 și ERA5 (ERA5 cu buffer 5 km);
    GDACS se leagă prin distanța AOI ↔ zona afectată ≤ 25 km;
  - temporală: t0 = eventTime (EMSR656+) sau activationTime; ferestre pre/post din config;
  - semantică: flood ↔ FL, fire ↔ WF; storm nu are corespondent GDACS.

Flaguri de calitate (definesc „sursa e prezentă”):
  - text_ok:  textul EMS are cel puțin 100 de caractere
  - s1_ok:    cel puțin o scenă S1 IW VV+VH în fereastra pre și în cea post
  - s2_ok:    fracția de pixeli clari ≥ 0.5 în compozitul pre și în cel post
  - era5_ok:  precipitațiile și vântul ERA5-Land sunt definite
  - gdacs_ok: există o alertă GDACS asociată
  - emdat_ok: există un eveniment EM-DAT asociat (GLIDE sau țară + tip + perioadă)

Ieșiri:
  - data/interim/aligned.csv
  - reports/coverage.csv, reports/coverage.md (pâlnia de acoperire, pe total, tip și an)

Rulare: uv run python pipeline/05_align.py
"""
import json

import pandas as pd

from config import INTERIM, RAW, REPORTS

S2_CLEAR_MIN = 0.5
TEXT_MIN = 100


def load_gee():
    rows = [json.loads(p.read_text()) for p in sorted((RAW / "gee").glob("EMSR*.json"))]
    return pd.DataFrame(rows)


def build():
    ev = pd.read_csv(INTERIM / "ems_events.csv")
    ev["year"] = ev["activation_time"].str[:4].astype(int)
    gee = load_gee()
    gd = pd.read_csv(INTERIM / "gdacs_match.csv").drop(columns=["category", "geom_from"])

    # aria AOI vine deja din pasul EMS; din GEE păstrăm doar trăsăturile și t0
    df = ev.merge(gee.drop(columns=["gee_error", "aoi_area_km2"], errors="ignore"), on="emsr_id", how="left")
    df = df.merge(gd, on="emsr_id", how="left")
    emdat = INTERIM / "emdat_match.csv"
    if emdat.exists():
        df = df.merge(pd.read_csv(emdat), on="emsr_id", how="left")

    df["text_ok"] = df["text_len"].fillna(0) >= TEXT_MIN
    df["aoi_ok"] = df["n_aois_with_geom"].fillna(0) > 0
    df["gee_ok"] = df["emsr_id"].isin(gee["emsr_id"]) if len(gee) else False
    df["s1_ok"] = (df.get("s1_n_pre", 0).fillna(0) > 0) & (df.get("s1_n_post", 0).fillna(0) > 0)
    df["s2_ok"] = ((df.get("s2_clear_pre", 0).fillna(0) >= S2_CLEAR_MIN)
                   & (df.get("s2_clear_post", 0).fillna(0) >= S2_CLEAR_MIN))
    df["era5_ok"] = df.get("era5_tp_event_mm").notna() & df.get("era5_wind10m_max_ms").notna()
    df["gdacs_ok"] = df["gdacs_id"].notna()
    df["ems_area_ok"] = df["ems_mapped_area_ha"].notna()
    df["emdat_ok"] = df["emdat_dis_no"].notna() if "emdat_dis_no" in df else False
    return df


FUNNEL = [
    ("selectate (EMSR, Europa, 2017–2025, flood/fire/storm)", None),
    ("cu text ≥ 100 caractere", "text_ok"),
    ("+ AOI (poligon)", "aoi_ok"),
    ("+ extracție GEE reușită", "gee_ok"),
    ("+ S1 pre și post", "s1_ok"),
    ("+ ERA5-Land", "era5_ok"),
    ("= nucleu: text + S1 + ERA5", None),
    ("+ S2 clar (≥ 50%) pre și post", "s2_ok"),
]


def funnel(df):
    out, mask = [], pd.Series(True, index=df.index)
    for label, col in FUNNEL:
        if col:
            mask &= df[col]
        out.append((label, int(mask.sum())))
    return out


def coverage(df):
    rows = []
    groups = [("total", df)] + [(f"tip={k}", g) for k, g in df.groupby("category")] \
        + [(f"an={k}", g) for k, g in df.groupby("year")]
    for name, g in groups:
        r = {"grup": name, "n": len(g)}
        for col in ("text_ok", "aoi_ok", "gee_ok", "s1_ok", "s2_ok", "era5_ok", "gdacs_ok", "emdat_ok", "ems_area_ok"):
            r[col] = int(g[col].sum())
        r["nucleu_text_s1_era5"] = int((g["text_ok"] & g["aoi_ok"] & g["s1_ok"] & g["era5_ok"]).sum())
        r["nucleu_plus_s2"] = int((g["text_ok"] & g["aoi_ok"] & g["s1_ok"] & g["era5_ok"] & g["s2_ok"]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def write_report(df, cov):
    REPORTS.mkdir(exist_ok=True)
    cov.to_csv(REPORTS / "coverage.csv", index=False)
    n = len(df)
    lines = ["# Raport de acoperire — U19", "",
             f"Generat de `pipeline/05_align.py`. Unitatea: activarea EMS (n = {n}).", "",
             "## Pâlnia de aliniere", "", "| Pas | Activări | % din selectate |", "|---|---:|---:|"]
    for label, k in funnel(df):
        lines.append(f"| {label} | {k} | {100 * k / n:.0f}% |")
    lines += ["", "## Pe surse, tip și an", "",
              "Coloanele `*_ok` numără activările pentru care sursa e prezentă (definițiile sunt în script).", ""]
    lines.append(cov.to_markdown(index=False))
    lines += ["", "## Goluri per sursă (% din selectate)", ""]
    for col in ("text_ok", "aoi_ok", "s1_ok", "s2_ok", "era5_ok", "gdacs_ok", "emdat_ok", "ems_area_ok"):
        lines.append(f"- `{col}`: lipsă la {100 * (1 - df[col].mean()):.0f}%")
    (REPORTS / "coverage.md").write_text("\n".join(lines) + "\n")


def main():
    df = build()
    df.to_csv(INTERIM / "aligned.csv", index=False)
    cov = coverage(df)
    write_report(df, cov)
    for label, k in funnel(df):
        print(f"{k:4d}  {label}")


if __name__ == "__main__":
    main()
