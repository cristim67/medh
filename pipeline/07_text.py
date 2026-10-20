"""Pasul 7: trăsăturile de text — data evenimentului, câmpuri de magnitudine și embeddings BGE-M3.

Intrare: data/interim/aligned.csv (coloana `reason` = textul EMS al activării).

1. Curățare: artefactele `_x000D_` (CR din Excel) și spațiile multiple.
2. Data evenimentului din text (`txt_event_date`): toate datele de forma „13 August 2022”, „August 17”,
   „15 of October”, „14/08/2021” din text; se păstrează cea mai veche dată din [activare − 60 zile, activare].
   Anul lipsă = anul activării. Pentru EMSR656+ se compară cu eventTime declarat (validare).
   `lag_days` = activare − data din text: arată cât de departe e t0 = activationTime de eveniment.
3. Câmpuri structurate de magnitudine din text (regex, valoarea maximă menționată):
     txt_area_ha  — suprafață menționată („1500 hectares”, „30 km2” → ha)
     txt_people   — persoane evacuate / afectate / salvate („300 people were evacuated”)
     txt_deaths   — decese („27 people have died”, „3 fatalities”)
   Lipsa mențiunii = 0 (textul nu dă cifra), cu indicatorul `txt_has_number`.
4. Embeddings BGE-M3 (BAAI/bge-m3, dense, 1024 dimensiuni, normalizate L2) pe textul curățat.
   Modelul e pre-antrenat și fix (nu se potrivește pe date) → nu există leakage între fold-uri.

Ieșiri:
  - data/interim/text_features.csv   emsr_id, text curat, data din text, lag, câmpuri de magnitudine
  - data/interim/text_bge_m3.npy     matricea embeddings (rânduri în ordinea din text_features.csv)
  - reports/text_dates.md            acoperirea datei extrase și validarea pe EMSR656+
Rulare: uv run python pipeline/07_text.py [--no-embed]
"""
import argparse
import re

import numpy as np
import pandas as pd

from config import INTERIM, REPORTS

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?"
DATE_PATTERNS = [
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MON}(?:\s*,?\s*(\d{{4}}))?", re.I),   # 13 August 2022
    re.compile(rf"\b{MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s*(\d{{4}}))?", re.I),           # August 17, 2022
    re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b"),                                         # 14/08/2021
]
NUM = r"(\d{1,3}(?:[ ,.]\d{3})+|\d+(?:\.\d+)?)"
AREA = re.compile(rf"{NUM}\s*(hectares|hectare|ha|km2|km²|square kilometres|square kilometers|sq\.? ?km)\b", re.I)
PEOPLE = re.compile(rf"{NUM}\s+(?:\w+\s+){{0,2}}(people|persons|residents|inhabitants|citizens|families|households)", re.I)
DEATHS = re.compile(rf"{NUM}\s+(?:\w+\s+){{0,3}}(dead|deaths|died|fatalities|victims|killed|casualties|lives)", re.I)
DEATHS_REV = re.compile(rf"(death toll|killed|killing|claimed the lives of)\D{{0,20}}{NUM}", re.I)


def clean(t):
    t = str(t).replace("_x000D_", " ")
    return re.sub(r"\s+", " ", t).strip()


def to_number(s):
    s = s.strip()
    if re.fullmatch(r"\d{1,3}(?:[ ,.]\d{3})+", s):  # separator de mii (1,500 / 1.500 / 1 500)
        return float(re.sub(r"[ ,.]", "", s))
    return float(s)


def max_mention(pattern, text, unit_conv=None):
    vals = []
    for m in pattern.finditer(text):
        g = m.groups()
        num = g[0] if pattern is not DEATHS_REV else g[1]
        try:
            v = to_number(num)
        except ValueError:
            continue
        if 1900 <= v <= 2100 and float(v).is_integer() and unit_conv is None:  # ani, nu numere
            continue
        if unit_conv:
            v *= unit_conv(g[1])
        vals.append(v)
    return max(vals) if vals else 0.0


def area_factor(u):
    u = u.lower()
    return 100.0 if ("km" in u) else 1.0


def text_dates(text, activation):
    found = []
    for i, pat in enumerate(DATE_PATTERNS):
        for m in pat.finditer(text):
            g = m.groups()
            try:
                if i == 0:
                    day, mon, year = int(g[0]), MONTHS[g[1].lower().rstrip(".")[:3]], g[2]
                elif i == 1:
                    mon, day, year = MONTHS[g[0].lower().rstrip(".")[:3]], int(g[1]), g[2]
                else:
                    day, mon, year = int(g[0]), int(g[1]), g[2]
                has_year = bool(year)
                d = pd.Timestamp(year=int(year) if has_year else activation.year, month=mon, day=day)
            except (ValueError, KeyError):
                continue
            if not has_year and d > activation.normalize():
                d = d - pd.DateOffset(years=1)  # fără an și după activare → anul trecut (activare în ianuarie)
            if activation.normalize() - pd.Timedelta(days=60) <= d <= activation.normalize():
                found.append(d)
    return min(found) if found else pd.NaT


def build_text_table():
    df = pd.read_csv(INTERIM / "aligned.csv")
    out = df[["emsr_id", "year", "category", "activation_time", "event_time", "t0_source"]].copy()
    out["text"] = df["reason"].map(clean)
    act = pd.to_datetime(df["activation_time"])
    out["txt_event_date"] = [text_dates(t, a) for t, a in zip(out["text"], act)]
    out["lag_days"] = (act.dt.normalize() - out["txt_event_date"]).dt.days
    out["txt_area_ha"] = [max_mention(AREA, t, area_factor) for t in out["text"]]
    out["txt_people"] = [max_mention(PEOPLE, t) for t in out["text"]]
    out["txt_deaths"] = [max(max_mention(DEATHS, t), max_mention(DEATHS_REV, t)) for t in out["text"]]
    out["txt_has_number"] = ((out[["txt_area_ha", "txt_people", "txt_deaths"]] > 0).any(axis=1)).astype(int)
    out["txt_len_words"] = out["text"].str.split().str.len()
    return out


def date_report(t):
    ev = pd.to_datetime(t["event_time"]).dt.normalize()
    has = t["txt_event_date"].notna()
    val = has & ev.notna()
    diff = (t.loc[val, "txt_event_date"] - ev[val]).dt.days.abs()
    legacy = t["t0_source"] == "activation_time"
    lag = t.loc[legacy & has, "lag_days"]
    lines = ["# Data evenimentului extrasă din textul EMS — U19", "",
             "Generat de `pipeline/07_text.py`.", "",
             f"- Activări cu o dată în text: **{has.sum()} / {len(t)}** ({has.mean():.0%}).",
             f"- Validare pe EMSR656+ (eventTime declarat): {val.sum()} activări, data din text = eventTime "
             f"exact la {(diff == 0).mean():.0%}, la ±1 zi la {(diff <= 1).mean():.0%}.",
             f"- Activări vechi (t0 = activationTime) cu dată în text: {len(lag)} din {legacy.sum()}.",
             f"  Întârzierea activare − eveniment: mediana {lag.median():.0f} zile, "
             f"P90 {lag.quantile(.9):.0f} zile, max {lag.max():.0f} zile; "
             f"> 10 zile la {(lag > 10).sum()} activări.",
             "", "Fereastra pre a imaginilor se termină la t0 − 10 zile, deci rămâne înainte de eveniment "
             "pentru toate activările cu întârziere ≤ 10 zile. Activările cu întârziere mai mare sunt "
             "marcate (`lag_days > 10`) și se exclud într-o analiză de sensibilitate."]
    (REPORTS / "text_dates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    return model.encode(list(texts), batch_size=8, normalize_embeddings=True, show_progress_bar=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="nu recalculează embeddings BGE-M3")
    args = ap.parse_args()
    t = build_text_table()
    t.to_csv(INTERIM / "text_features.csv", index=False)
    date_report(t)
    if not args.no_embed:
        emb = embed(t["text"])
        np.save(INTERIM / "text_bge_m3.npy", emb.astype(np.float32))
        print("BGE-M3:", emb.shape)


if __name__ == "__main__":
    main()
