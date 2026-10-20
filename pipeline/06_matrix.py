"""Pasul 6: matricea X (n × d) pe blocuri și scalarea pe fold-uri LOYO.

Rândurile: activările din nucleu (text + AOI + S1 + ERA5). Blocurile:
  - text:    TF-IDF pe textul EMS (1–2-grame, min_df=3, max 2000 de trăsături, engleză). BGE-M3 vine la CP2.
  - imagine: S1 (ΔVV, ΔVH, fracția și aria inundată), S2 (dNDVI, dNBR, dMNDWI, fracții și arii), claritatea S2
  - meteo:   ERA5-Land (precipitații eveniment / antecedente, Tmax, vânt maxim)
  - ems:     magnitudinea de referință — EMS (suprafața și populația afectată, aria AOI, număr de AOI) și
             EM-DAT (decese, afectați, pagube) — nu intră în X
Valorile S2 lipsesc când S2 nu e clar; se imputează cu mediana din train și se adaugă indicatorul `s2_ok`.

Scalarea (parte din X):
  - ariile și precipitațiile: log1p (distribuții cu coadă lungă, ordine de mărime diferite);
  - restul numeric: standardizare z-score;
  - TF-IDF: normalizare L2 pe rând (implicită în TfidfVectorizer), fără standardizare;
  - imputarea, scalarea și vocabularul TF-IDF se potrivesc (fit) DOAR pe anii de train ai fiecărui fold LOYO.

Ieșiri:
  - data/interim/X_raw.csv               trăsăturile numerice nescalate + emsr_id, year, category
  - reports/feature_dictionary.csv       nume, bloc, sursă, unitate, transformare
  - reports/x_shape.md                   n × d pe blocuri și fold-urile LOYO
Rulare: uv run python pipeline/06_matrix.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import INTERIM, REPORTS

# nume, bloc, sursă, unitate, transformare
FEATURES = [
    ("s1_dvv", "imagine", "Sentinel-1 GRD", "dB", "z"),
    ("s1_dvh", "imagine", "Sentinel-1 GRD", "dB", "z"),
    ("s1_flooded", "imagine", "Sentinel-1 GRD", "fracție", "z"),
    ("s1_flooded_km2", "imagine", "Sentinel-1 GRD", "km²", "log1p+z"),
    ("s2_dndvi", "imagine", "Sentinel-2 L2A", "–", "z"),
    ("s2_dnbr", "imagine", "Sentinel-2 L2A", "–", "z"),
    ("s2_dmndwi", "imagine", "Sentinel-2 L2A", "–", "z"),
    ("s2_burnt", "imagine", "Sentinel-2 L2A", "fracție", "z"),
    ("s2_burnt_km2", "imagine", "Sentinel-2 L2A", "km²", "log1p+z"),
    ("s2_newwater", "imagine", "Sentinel-2 L2A", "fracție", "z"),
    ("s2_newwater_km2", "imagine", "Sentinel-2 L2A", "km²", "log1p+z"),
    ("s2_clear_pre", "imagine", "Sentinel-2 L2A", "fracție", "z"),
    ("s2_clear_post", "imagine", "Sentinel-2 L2A", "fracție", "z"),
    ("s2_ok", "imagine", "Sentinel-2 L2A", "0/1", "z"),
    ("era5_tp_event_mm", "meteo", "ERA5-Land", "mm", "log1p+z"),
    ("era5_tp_antecedent_mm", "meteo", "ERA5-Land", "mm", "log1p+z"),
    ("era5_t2m_max_c", "meteo", "ERA5-Land", "°C", "z"),
    ("era5_wind10m_max_ms", "meteo", "ERA5-Land", "m/s", "z"),
    ("ems_mapped_area_ha", "ems", "Copernicus EMS", "ha", "log1p+z"),
    ("ems_pop_affected", "ems", "Copernicus EMS (doar EMSR656+)", "persoane", "log1p+z"),
    ("emdat_deaths", "ems", "EM-DAT", "persoane", "log1p+z"),
    ("emdat_affected", "ems", "EM-DAT", "persoane", "log1p+z"),
    ("emdat_damage_adj_kusd", "ems", "EM-DAT", "mii USD (ajustat)", "log1p+z"),
    ("aoi_area_km2", "ems", "Copernicus EMS", "km²", "log1p+z"),
    ("n_aois", "ems", "Copernicus EMS", "număr", "log1p+z"),
]
INPUTS = [f for f in FEATURES if f[1] != "ems"]  # blocul ems e referința de magnitudine, nu intră în X
S2_COLS = [f for f, *_ in FEATURES if f.startswith("s2_") and f not in ("s2_ok", "s2_clear_pre", "s2_clear_post")]


def core_rows():
    df = pd.read_csv(INTERIM / "aligned.csv")
    core = df[df["text_ok"] & df["aoi_ok"] & df["s1_ok"] & df["era5_ok"]].copy()
    core.loc[~core["s2_ok"], S2_COLS] = np.nan  # S2 cu nori nu e informativ: se tratează ca lipsă
    core["s2_ok"] = core["s2_ok"].astype(int)
    return core.reset_index(drop=True)


def make_transformer():
    log_cols = [f for f, *_, t in INPUTS if t == "log1p+z"]
    z_cols = [f for f, *_, t in INPUTS if t == "z"]
    numeric = ColumnTransformer([
        ("log", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p),
                              StandardScaler()), log_cols),
        ("z", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), z_cols),
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=2000,
                                 stop_words="english", sublinear_tf=True), "reason"),
    ])
    return numeric


def loyo_folds(years):
    return [(y, years != y, years == y) for y in sorted(years.unique())]


def main():
    core = core_rows()
    cols = ["emsr_id", "year", "category"] + [f for f, *_ in FEATURES]
    core[cols].to_csv(INTERIM / "X_raw.csv", index=False)
    pd.DataFrame(FEATURES, columns=["trasatura", "bloc", "sursa", "unitate", "transformare"]).to_csv(
        REPORTS / "feature_dictionary.csv", index=False)

    lines = ["# Forma matricei X — U19", "", f"Rânduri (nucleu text + S1 + ERA5): **n = {len(core)}**", "",
             "| Bloc | d | Note |", "|---|---:|---|"]
    blocks = pd.Series({f: b for f, b, *_ in FEATURES})
    for b in ("imagine", "meteo"):
        lines.append(f"| {b} | {(blocks == b).sum()} | numeric, scalare în tabelul de trăsături |")

    fold_rows = []
    for year, tr, te in loyo_folds(core["year"]):
        if tr.sum() < 10 or te.sum() == 0:
            continue
        t = make_transformer().fit(core[tr])
        Xtr, Xte = t.transform(core[tr]), t.transform(core[te])
        d_text = len(t.named_transformers_["text"].vocabulary_)
        z = Xte[:, : len(INPUTS)]
        z = z.toarray() if hasattr(z, "toarray") else z
        fold_rows.append({"an_test": year, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "d_text_tfidf": d_text, "d_total": Xtr.shape[1],
                          "medie_abs_z_test": float(np.abs(z.mean(axis=0)).mean())})
    folds = pd.DataFrame(fold_rows)
    d_text = int(folds["d_text_tfidf"].median()) if len(folds) else 0
    lines.append(f"| text (TF-IDF) | ~{d_text} | vocabular potrivit pe train, variază pe fold (mediana) |")
    n_ems = (blocks == "ems").sum()
    lines += ["", f"**d total ≈ {len(INPUTS) + d_text}** (numeric {len(INPUTS)} + text ~{d_text}). "
              f"Separat: {n_ems} variabile EMS de magnitudine (referință, nu intră în X).", "",
              "## Fold-uri LOYO", "",
              "Imputarea, scalarea și vocabularul TF-IDF sunt potrivite doar pe anii de train. "
              "`medie_abs_z_test` arată cât de departe e media trăsăturilor numerice pe anul de test de 0 "
              "(0 = fără deriva între ani).", "", folds.to_markdown(index=False)]
    (REPORTS / "x_shape.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
