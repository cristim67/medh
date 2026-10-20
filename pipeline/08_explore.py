"""Pasul 8: matricea X documentată, analiza uni/bivariată și primul PCA pe protocolul LOYO (CP2).

Rânduri: nucleul din pasul 6 (text + AOI + S1 + ERA5), n = 391. Blocurile lui X (pe sursă):
  imagine     14  S1 (ΔVV, ΔVH, fracția/aria cu ΔVV < −3 dB), S2 (dNDVI, dNBR, dMNDWI, fracții/arii), claritate S2
  meteo        4  ERA5-Land: precipitații eveniment / antecedente, Tmax, vânt maxim
  text_struct  5  din textul EMS (pasul 7): suprafață, persoane, decese menționate, indicator, lungime
  text_emb     –  TF-IDF 1–2-grame (vocabular potrivit pe train) SAU BGE-M3 dense (1024, model fix)
Referința de magnitudine (NU intră în X): suprafața mapată EMS, populația EMS, EM-DAT, aria AOI.

Scalare (parte din X, potrivită DOAR pe anii de train ai fiecărui fold):
  - log1p pe arii, ploi, numere de persoane; z-score pe toate numericele; imputare cu mediana din train
    (S2 lipsește când nu e clar, indicatorul `s2_ok` rămâne în X);
  - TF-IDF: normalizare L2 pe rând; BGE-M3: normalizat L2 de model;
  - ponderare pe bloc ca în MFA: fiecare bloc se împarte la √λ1 (prima valoare proprie a blocului pe
    train), deci prima axă a fiecărui bloc are varianța 1. Fără ponderare, blocul cu mai multe coloane
    standardizate (imaginea, 14) decide singur axele PCA; cu inerție totală 1 pe bloc, blocul cu puține
    coloane corelate (meteo, 4) ar domina PC1 — ambele variante sunt raportate în pca_loyo.md.

Protocol (CP2): split LOYO (9 fold-uri, 2017–2025); PCA potrivit pe train, aplicat pe anul de test.
  k: analiza paralelă Horn pe train (componentele cu valoare proprie peste P95 din 20 de permutări).
  Metrici pe test, media ± abaterea standard pe fold-uri:
    - globală:   varianța reconstruită pe test (1 − eroarea de reconstrucție relativă)
    - locală:    trustworthiness (10 vecini) între X_test și proiecția Z_test
    - downstream: (a) acuratețea kNN (k = 5) pentru tipul evenimentului;
                  (b) ρ Spearman între predicția Ridge a log(suprafeței mapate EMS) din Z și valoarea reală
  Aceleași metrici pe reprezentările „o singură sursă” (baseline-ul pentru CP3) și pe X complet.

Ieșiri (reports/): feature_dictionary.csv, univariate.csv, spearman.csv, block_rv.csv,
  magnitude_by_source.md, pca_loyo.md, pca_metrics.csv, pca_loadings.csv, figures/*.png
Rulare: uv run python pipeline/08_explore.py   (după 07_text.py)
"""
import importlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import INTERIM, REPORTS

warnings.filterwarnings("ignore", category=RuntimeWarning)
m06 = importlib.import_module("06_matrix")
FIG = REPORTS / "figures"
SEED = 0

# trăsătură → (bloc, sursă, set / versiune, fereastră, rezoluție, unitate, transformare)
W_PRE, W_POST = "pre t0−60…−10 z", "post t0+2…+30 z"
S1 = ("imagine", "Sentinel-1", "COPERNICUS/S1_GRD (IW, VV+VH)", f"{W_PRE} / {W_POST}", "10 m (≥ 20 m la reducere)")
S2 = ("imagine", "Sentinel-2", "COPERNICUS/S2_SR_HARMONIZED (mască SCL)", f"{W_PRE} / {W_POST}", "10–20 m (≥ 20 m)")
E5 = ("meteo", "ERA5-Land", "ECMWF/ERA5_LAND/DAILY_AGGR, HOURLY (vânt)")
TX = ("text_struct", "Text EMS", "Copernicus EMS, câmpul reason (regex, pasul 7)", "la activare", "activare")
REF = ("referință", "Copernicus EMS / EM-DAT")
FEATURES = {
    "s1_dvv": (*S1, "dB", "z"), "s1_dvh": (*S1, "dB", "z"),
    "s1_flooded": (*S1, "fracție din AOI cu ΔVV < −3 dB", "z"), "s1_flooded_km2": (*S1, "km²", "log1p+z"),
    "s2_dndvi": (*S2, "– (post − pre)", "z"), "s2_dnbr": (*S2, "– (post − pre)", "z"),
    "s2_dmndwi": (*S2, "– (post − pre)", "z"),
    "s2_burnt": (*S2, "fracție cu dNBR < −0,27", "z"), "s2_burnt_km2": (*S2, "km²", "log1p+z"),
    "s2_newwater": (*S2, "fracție cu dMNDWI > 0,2", "z"), "s2_newwater_km2": (*S2, "km²", "log1p+z"),
    "s2_clear_pre": (*S2, "fracție pixeli clari", "z"), "s2_clear_post": (*S2, "fracție pixeli clari", "z"),
    "s2_ok": (*S2, "0/1 (S2 clar ≥ 50%)", "z"),
    "era5_tp_event_mm": (*E5, "t0−L…t0+2 z (L = 5 / 14)", "0,1° (~9 km), buffer 5 km", "mm", "log1p+z"),
    "era5_tp_antecedent_mm": (*E5, "t0−60…−10 z", "0,1° (~9 km)", "mm", "log1p+z"),
    "era5_t2m_max_c": (*E5, "t0−L…t0+2 z", "0,1° (~9 km)", "°C", "z"),
    "era5_wind10m_max_ms": (*E5, "t0−L…t0+2 z", "0,1°, orar", "m/s", "z"),
    "txt_area_ha": (*TX, "ha (max. menționat)", "log1p+z"),
    "txt_people": (*TX, "persoane (max. menționat)", "log1p+z"),
    "txt_deaths": (*TX, "persoane (max. menționat)", "log1p+z"),
    "txt_has_number": (*TX, "0/1", "z"),
    "txt_len_words": (*TX, "cuvinte", "log1p+z"),
    "ems_mapped_area_ha": (*REF, "poligoane observed event, uniune EPSG:3035", "–", "AOI", "ha", "log1p"),
    "ems_pop_affected": (*REF, "statistici EMS (doar EMSR656+)", "–", "AOI", "persoane", "log1p"),
    "emdat_deaths": (*REF, "EM-DAT export 2026-10-19", "eveniment", "țară", "persoane", "log1p"),
    "emdat_affected": (*REF, "EM-DAT export 2026-10-19", "eveniment", "țară", "persoane", "log1p"),
    "emdat_damage_adj_kusd": (*REF, "EM-DAT export 2026-10-19", "eveniment", "țară", "mii USD", "log1p"),
    "aoi_area_km2": (*REF, "AOI EMS, uniune", "–", "AOI", "km²", "log1p"),
}
NUM_BLOCKS = {b: [f for f, v in FEATURES.items() if v[0] == b] for b in ("imagine", "meteo", "text_struct")}
REFS = [f for f, v in FEATURES.items() if v[0] == "referință"]
LOG = {f for f, v in FEATURES.items() if v[-1].startswith("log1p")}
SOURCE_OF = {"imagine": "imagine", "meteo": "meteo", "text_struct": "text", "text_emb": "text"}
TFIDF = dict(ngram_range=(1, 2), min_df=3, max_features=2000, stop_words="english", sublinear_tf=True)
CONFIGS = {
    "toate sursele (TF-IDF)": ["imagine", "meteo", "text_struct", "tfidf"],
    "toate sursele (BGE-M3)": ["imagine", "meteo", "text_struct", "bge"],
    "doar imagine": ["imagine"],
    "doar meteo": ["meteo"],
    "doar text (TF-IDF)": ["text_struct", "tfidf"],
    "doar text (BGE-M3)": ["text_struct", "bge"],
}
K_SWEEP = [1, 2, 3, 5, 10, 20, 50]
CAT_COLORS = {"fire": "#C8702A", "flood": "#2F6690", "storm": "#7A7A7A"}


# ---------- date ----------

def load():
    core = m06.core_rows()
    txt = pd.read_csv(INTERIM / "text_features.csv")
    emb = np.load(INTERIM / "text_bge_m3.npy")
    txt_idx = pd.Series(range(len(txt)), index=txt["emsr_id"])
    core = core.merge(txt.drop(columns=["year", "category", "activation_time", "event_time", "t0_source"]),
                      on="emsr_id", how="left")
    bge = emb[txt_idx.loc[core["emsr_id"]].to_numpy()]
    return core, bge


# ---------- X pe un fold ----------

class FoldX:
    """Construiește blocurile lui X potrivite pe train și le aplică pe orice subset."""

    def __init__(self, df_tr, bge_tr, blocks):
        self.blocks = blocks
        self.fit = {}
        for b in blocks:
            if b in NUM_BLOCKS:
                cols = NUM_BLOCKS[b]
                imp = SimpleImputer(strategy="median").fit(self._log(df_tr[cols]))
                sc = StandardScaler().fit(imp.transform(self._log(df_tr[cols])))
                self.fit[b] = (cols, imp, sc)
            elif b == "tfidf":
                self.fit[b] = TfidfVectorizer(**TFIDF).fit(df_tr["text"])
        raw = {b: self._raw(b, df_tr, bge_tr) for b in blocks}
        # ponderare MFA (Escofier & Pagès): fiecare bloc împărțit la √λ1 al lui pe train → prima axă a
        # fiecărui bloc are varianța 1 și niciun bloc nu decide singur PC1 prin numărul sau scara coloanelor
        self.w = {b: 1.0 / np.sqrt(PCA(1).fit(raw[b]).explained_variance_[0]) for b in blocks}

    @staticmethod
    def _log(d):
        d = d.copy()
        for c in d.columns:
            if c in LOG:
                d[c] = np.log1p(d[c].clip(lower=0))
        return d

    @staticmethod
    def _source(b):
        return SOURCE_OF.get(b, "text")

    def _raw(self, b, df, bge):
        if b in NUM_BLOCKS:
            cols, imp, sc = self.fit[b]
            return sc.transform(imp.transform(self._log(df[cols])))
        if b == "tfidf":
            return self.fit[b].transform(df["text"]).toarray()
        return bge

    def transform(self, df, bge, weighted=True):
        parts = [self._raw(b, df, bge) * (self.w[b] if weighted else 1.0) for b in self.blocks]
        return np.hstack(parts)

    def names(self):
        out = []
        for b in self.blocks:
            if b in NUM_BLOCKS:
                out += [(b, c) for c in NUM_BLOCKS[b]]
            elif b == "tfidf":
                out += [("text_emb", f"tfidf:{t}") for t in self.fit[b].get_feature_names_out()]
            else:
                out += [("text_emb", f"bge:{i}") for i in range(1024)]
        return out


def loyo(years):
    return [(y, (years != y).to_numpy(), (years == y).to_numpy()) for y in sorted(years.unique())]


def horn_k(X, n_perm=20, seed=SEED):
    rng = np.random.default_rng(seed)
    ev = PCA().fit(X).explained_variance_
    perm = []
    for _ in range(n_perm):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        perm.append(PCA().fit(Xp).explained_variance_[: len(ev)])
    thr = np.percentile(np.vstack(perm), 95, axis=0)
    above = ev > thr
    return int(np.argmin(above)) if not above.all() else len(ev)


# ---------- univariat și bivariat ----------

def univariate(core):
    rows = []
    for f, v in FEATURES.items():
        x = core[f].astype(float)
        xs = x.dropna()
        xt = np.log1p(xs.clip(lower=0)) if f in LOG else xs
        rows.append({"trasatura": f, "bloc": v[0], "n_prezent": len(xs), "lipsa_%": round(100 * x.isna().mean(), 1),
                     "zero_%": round(100 * (xs == 0).mean(), 1), "mediana": xs.median(),
                     "q25": xs.quantile(.25), "q75": xs.quantile(.75), "min": xs.min(), "max": xs.max(),
                     "asimetrie_brut": skew(xs) if len(xs) > 2 else np.nan,
                     "asimetrie_dupa_transformare": skew(xt) if len(xs) > 2 else np.nan})
    u = pd.DataFrame(rows)
    u.to_csv(REPORTS / "univariate.csv", index=False, float_format="%.4g")
    return u


def fig_distributions(core):
    feats = [f for f in FEATURES if f not in ("s2_ok", "txt_has_number")]
    ncol = 6
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.1 * nrow))
    for ax, f in zip(axes.flat, feats):
        x = core[f].astype(float)
        x = np.log1p(x.clip(lower=0)) if f in LOG else x
        bins = np.histogram_bin_edges(x.dropna(), bins=25)
        for c, col in CAT_COLORS.items():
            ax.hist(x[core["category"] == c].dropna(), bins=bins, color=col, alpha=.6, label=c)
        ax.set_title(("log1p " if f in LOG else "") + f, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.text(.98, .95, f"lipsă {x.isna().mean():.0%}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(f"Distribuțiile trăsăturilor (n = {len(core)}), pe tipul evenimentului", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "distributii.png", dpi=150)
    plt.close(fig)


def spearman_matrix(core):
    feats = [f for f in FEATURES if f not in ("s2_ok",)]
    rho = core[feats].astype(float).corr(method="spearman", min_periods=20)
    rho.to_csv(REPORTS / "spearman.csv", float_format="%.3f")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)), feats, fontsize=7)
    bl = [FEATURES[f][0] for f in feats]
    for i in range(1, len(feats)):
        if bl[i] != bl[i - 1]:
            ax.axhline(i - .5, color="k", lw=1)
            ax.axvline(i - .5, color="k", lw=1)
    fig.colorbar(im, ax=ax, shrink=.7, label="ρ Spearman (perechi complete)")
    ax.set_title("Corelații Spearman: imagine | meteo | text structurat | referință de magnitudine")
    fig.tight_layout()
    fig.savefig(FIG / "spearman.png", dpi=150)
    plt.close(fig)
    return rho


def rv(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    sab = np.linalg.norm(a.T @ b) ** 2
    return sab / np.sqrt(np.linalg.norm(a.T @ a) ** 2 * np.linalg.norm(b.T @ b) ** 2)


def block_rv(core, bge, mask=None, tag="toate"):
    """Coeficientul RV între blocuri: brut, nul (permutări) și ajustat = (RV − RV0) / (1 − RV0)."""
    mask = np.ones(len(core), bool) if mask is None else mask
    d, e = core[mask].reset_index(drop=True), bge[mask]
    fx = FoldX(d, e, ["imagine", "meteo", "text_struct", "tfidf", "bge"])
    mats = {b: fx._raw(b, d, e) for b in fx.blocks}
    ref = np.log1p(d[["ems_mapped_area_ha", "aoi_area_km2"]].astype(float).clip(lower=0))
    mats["referință"] = StandardScaler().fit_transform(ref.fillna(ref.median()))
    names = {"imagine": "imagine", "meteo": "meteo", "text_struct": "text struct.", "tfidf": "TF-IDF",
             "bge": "BGE-M3", "referință": "ref. EMS"}
    keys = list(mats)
    m = pd.DataFrame([[rv(mats[a], mats[b]) for b in keys] for a in keys],
                     index=[names[k] for k in keys], columns=[names[k] for k in keys])
    rng = np.random.default_rng(SEED)
    null = pd.DataFrame([[np.mean([rv(mats[a], mats[b][rng.permutation(len(d))]) for _ in range(20)])
                          for b in keys] for a in keys], index=m.index, columns=m.columns)
    adj = (m - null) / (1 - null)
    for i in range(len(keys)):
        adj.iloc[i, i] = 1.0
    m.to_csv(REPORTS / f"block_rv_{tag}.csv", float_format="%.3f")
    adj.to_csv(REPORTS / f"block_rv_ajustat_{tag}.csv", float_format="%.3f")
    return m, null, adj


def magnitude_by_source(core):
    proxies = [("imagine", "s1_flooded_km2"), ("imagine", "s2_burnt_km2"), ("imagine", "s2_newwater_km2"),
               ("text", "txt_area_ha"), ("text", "txt_people"), ("meteo", "era5_tp_event_mm"),
               ("meteo", "era5_wind10m_max_ms"), ("ref.", "aoi_area_km2")]
    targets = ["ems_mapped_area_ha", "emdat_affected"]
    lines = ["# Magnitudinea după sursă — U19", "",
             "Generat de `pipeline/08_explore.py`. ρ Spearman între indicatorul de magnitudine al fiecărei "
             "surse și referință, pe tip de eveniment (perechi complete; pentru `txt_*` doar textele care dau "
             "cifra). n între paranteze; ρ cu n < 10 nu se afișează.", ""]
    rows = []
    for cat in ("fire", "flood"):
        d = core[core["category"] == cat]
        for src, p in proxies:
            r = {"tip": cat, "sursa": src, "indicator": p}
            for t in targets:
                x, y = d[p].astype(float), d[t].astype(float)
                ok = x.notna() & y.notna() & ((x > 0) if p.startswith("txt_") else True)
                r[t] = (f"{spearmanr(x[ok], y[ok]).statistic:+.2f} ({ok.sum()})" if ok.sum() >= 10
                        else f"– ({ok.sum()})")
            rows.append(r)
    t = pd.DataFrame(rows)
    lines += [t.to_markdown(index=False)]
    (REPORTS / "magnitude_by_source.md").write_text("\n".join(lines) + "\n")
    return t


# ---------- PCA pe LOYO ----------

def metrics(Xte, Ztr, Zte, pca, df_tr, df_te):
    if pca is not None:
        rec = pca.inverse_transform(Zte)
        var_te = 1 - ((Xte - rec) ** 2).sum() / ((Xte - pca.mean_) ** 2).sum()
        tw = trustworthiness(Xte, Zte, n_neighbors=10)
    else:  # X complet: nimic de reconstruit, vecinii sunt chiar cei din X
        var_te, tw = 1.0, 1.0
    knn = KNeighborsClassifier(5).fit(Ztr, df_tr["category"])
    acc = (knn.predict(Zte) == df_te["category"]).mean()
    y_tr, y_te = np.log1p(df_tr["ems_mapped_area_ha"]), np.log1p(df_te["ems_mapped_area_ha"])
    ok_tr, ok_te = y_tr.notna().to_numpy(), y_te.notna().to_numpy()
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Ztr[ok_tr], y_tr[ok_tr])
    rho = spearmanr(ridge.predict(Zte[ok_te]), y_te[ok_te]).statistic
    return {"var_reconstruita": var_te, "trustworthiness": tw, "knn_tip_acc": acc, "rho_suprafata_ems": rho}


def run_loyo(core, bge):
    folds = loyo(core["year"])
    # k din analiza paralelă pe X complet (TF-IDF), pe fiecare train
    ks, scree, scree_nw, share_nw = [], [], [], []
    for y, tr, te in folds:
        fx = FoldX(core[tr], bge[tr], CONFIGS["toate sursele (TF-IDF)"])
        Xtr = fx.transform(core[tr], bge[tr])
        ks.append(horn_k(Xtr))
        scree.append(PCA(30).fit(Xtr).explained_variance_ratio_)
        Xnw = fx.transform(core[tr], bge[tr], weighted=False)
        p = PCA(10).fit(Xnw)
        scree_nw.append(p.explained_variance_ratio_)
        blk = np.array([b for b, _ in fx.names()])
        share_nw.append({b: float((p.components_[:3, blk == b] ** 2).sum() / 3) for b in np.unique(blk)})
    k = int(np.median(ks))

    rows = []
    for name, blocks in CONFIGS.items():
        for y, tr, te in folds:
            fx = FoldX(core[tr], bge[tr], blocks)
            Xtr, Xte = fx.transform(core[tr], bge[tr]), fx.transform(core[te], bge[te])
            kk = min(k, Xtr.shape[1] - 1)
            pca = PCA(kk, random_state=SEED).fit(Xtr)
            r = metrics(Xte, pca.transform(Xtr), pca.transform(Xte), pca, core[tr], core[te])
            rows.append({"reprezentare": f"PCA k={kk} · {name}", "an_test": y, **r})
            if name.startswith("toate"):  # X complet, fără reducere (referință pentru downstream)
                r = metrics(Xte, Xtr, Xte, None, core[tr], core[te])
                rows.append({"reprezentare": f"X complet, fără PCA · {name}", "an_test": y, **r})
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "pca_metrics.csv", index=False, float_format="%.4f")
    sweep = []
    for kk in K_SWEEP:
        for y, tr, te in folds:
            fx = FoldX(core[tr], bge[tr], CONFIGS["toate sursele (BGE-M3)"])
            Xtr, Xte = fx.transform(core[tr], bge[tr]), fx.transform(core[te], bge[te])
            pca = PCA(kk, random_state=SEED).fit(Xtr)
            sweep.append({"k": kk, "an_test": y,
                          **metrics(Xte, pca.transform(Xtr), pca.transform(Xte), pca, core[tr], core[te])})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(REPORTS / "pca_k_sweep.csv", index=False, float_format="%.4f")
    return k, ks, sweep, np.array(scree), np.array(scree_nw), pd.DataFrame(share_nw).mean(), res


def summarize(res):
    cols = ["var_reconstruita", "trustworthiness", "knn_tip_acc", "rho_suprafata_ems"]
    g = res.groupby("reprezentare", sort=False)[cols]
    m, s = g.mean(), g.std()
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.2f} ± {b:.2f}" for a, b in zip(m[c], s[c])]
    return out.reset_index(), m


def full_fit_loadings(core, bge, k):
    fx = FoldX(core, bge, CONFIGS["toate sursele (TF-IDF)"])
    X = fx.transform(core, bge)
    pca = PCA(k, random_state=SEED).fit(X)
    names = fx.names()
    blk = np.array([b for b, _ in names])
    feat = np.array([f for _, f in names])
    rows, share = [], []
    for j in range(k):
        load = pca.components_[j]
        share.append({b: float((load[blk == b] ** 2).sum()) for b in ("imagine", "meteo", "text_struct", "text_emb")})
        for i in np.argsort(-np.abs(load))[:8]:
            rows.append({"pc": j + 1, "var_explicata": pca.explained_variance_ratio_[j], "trasatura": feat[i],
                         "bloc": blk[i], "loading": load[i]})
    pd.DataFrame(rows).to_csv(REPORTS / "pca_loadings.csv", index=False, float_format="%.4f")
    share = pd.DataFrame(share, index=[f"PC{j + 1}" for j in range(k)])
    Z = pca.transform(X)
    return pca, pd.DataFrame(rows), share, Z


def bge_stability(core, bge, k):
    """Stabilitatea axelor pe fold-uri: |cos| între PC_j pe train și PC_j pe tot setul (BGE-M3: spațiu fix)."""
    blocks = CONFIGS["toate sursele (BGE-M3)"]
    full = PCA(k).fit(FoldX(core, bge, blocks).transform(core, bge)).components_
    sims = []
    for y, tr, te in loyo(core["year"]):
        c = PCA(k).fit(FoldX(core[tr], bge[tr], blocks).transform(core[tr], bge[tr])).components_
        sims.append(np.abs((c * full).sum(1)))
    return np.array(sims).mean(0)


# ---------- figuri ----------

def fig_scree(scree, scree_nw, k, ks):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(1, scree.shape[1] + 1)
    ax.fill_between(x, scree.min(0) * 100, scree.max(0) * 100, color="#2F6690", alpha=.25, lw=0)
    ax.plot(x, scree.mean(0) * 100, "o-", color="#2F6690", ms=3, label="ponderat MFA (X folosit)")
    ax.plot(np.arange(1, 11), scree_nw.mean(0) * 100, "s--", color="#999", ms=3, label="fără ponderare")
    ax.axvline(k + .5, color="#C8702A", ls=":", lw=1.5)
    ax.text(k + .7, ax.get_ylim()[1] * .9, f"k = {k} (Horn, {min(ks)}–{max(ks)} pe fold-uri)",
            color="#C8702A", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, scree.mean(0).cumsum() * 100, color="#1B2A41", lw=1)
    ax2.set_ylabel("cumulat (%)", fontsize=8)
    ax2.set_ylim(0, 100)
    ax.set_xlabel("componenta principală")
    ax.set_ylabel("varianță explicată pe train (%)")
    ax.set_title("Scree pe 9 fold-uri LOYO (medie, bandă min–max)", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "scree.png", dpi=200)
    plt.close(fig)


def fig_loadings(share, load, k):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    im = ax.imshow(share.T.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(4), ["imagine", "meteo", "text struct.", "TF-IDF"])
    ax.set_xticks(range(k), share.index, fontsize=7, rotation=90)
    for i in range(share.shape[1]):
        for j in range(k):
            v = share.iloc[j, i]
            if v >= .1:
                ax.text(j, i, f"{v:.1f}"[1:], ha="center", va="center", fontsize=6, color="w" if v > .6 else "k")
    ax.set_title("Cât din fiecare PC vine din fiecare sursă\n(suma loadings²)", fontsize=9)
    ax = axes[1]
    ax.axis("off")
    txt = []
    for j in range(1, min(k, 6) + 1):
        d = load[load["pc"] == j].head(5)
        terms = ", ".join(f"{r.trasatura.replace('tfidf:', '“') + ('”' if 'tfidf:' in r.trasatura else '')}"
                          f" {'+' if r.loading > 0 else '−'}" for r in d.itertuples())
        txt.append(f"PC{j} ({d['var_explicata'].iloc[0]:.1%}): {terms}")
    ax.text(0, 1, "\n\n".join(txt), va="top", fontsize=8, wrap=True)
    ax.set_title("Loadings dominante (fit pe toate cele 391, doar pentru interpretare)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "loadings.png", dpi=180)
    plt.close(fig)


def fig_pc_scatter(Z, core):
    fig, ax = plt.subplots(figsize=(5, 4))
    for c, col in CAT_COLORS.items():
        m = (core["category"] == c).to_numpy()
        ax.scatter(Z[m, 0], Z[m, 1], s=10, color=col, alpha=.7, label=c)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)
    ax.set_title("Primele două componente (X ponderat MFA)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "pc1_pc2.png", dpi=180)
    plt.close(fig)


def fig_slide(adj_all, adj_fire, n_fire, scree, k, ks, mean_metrics):
    """Figura centrală pentru slide-ul 2: (a) RV între surse, (b) scree, (c) metrici PCA pe surse."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), gridspec_kw={"width_ratios": [1, 1.2, 1.4]})
    ax = axes[0]
    keys = ["imagine", "meteo", "text struct.", "TF-IDF", "BGE-M3", "ref. EMS"]
    a, f = adj_all.loc[keys, keys].values, adj_fire.loc[keys, keys].values
    m = np.where(np.tri(len(keys), k=-1, dtype=bool), a, f)  # sub diagonală: toate; deasupra: incendii
    np.fill_diagonal(m, np.nan)
    ax.imshow(m, cmap="Blues", vmin=0, vmax=.6)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if m[i, j] > .4 else "k")
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(keys)), keys, fontsize=7)
    ax.set_title(f"(a) RV ajustat între surse\n▼ toate (n = {int(adj_all.attrs.get('n', 0))}) · "
                 f"▲ doar incendii (n = {n_fire})", fontsize=9)

    ax = axes[1]
    x = np.arange(1, scree.shape[1] + 1)
    ax.fill_between(x, scree.min(0) * 100, scree.max(0) * 100, color="#2F6690", alpha=.25, lw=0)
    ax.plot(x, scree.mean(0) * 100, "o-", color="#2F6690", ms=3)
    ax.axvline(k + .5, color="#C8702A", ls=":", lw=1.5)
    cum = scree.mean(0).cumsum()[k - 1] * 100
    ax.text(k + 1, ax.get_ylim()[1] * .85, f"k = {k} (Horn; {min(ks)}–{max(ks)})\n{cum:.0f}% cumulat",
            color="#C8702A", fontsize=8)
    ax.set_xlabel("componenta principală", fontsize=8)
    ax.set_ylabel("varianță explicată (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(b) Scree pe train, 9 fold-uri LOYO", fontsize=9)

    ax = axes[2]
    sel = [r for r in mean_metrics.index if r.startswith("PCA")]
    labels = [r.split("· ")[1] for r in sel]
    yy = np.arange(len(sel))
    cols = [("trustworthiness", "trustworthiness (local)", "#2F6690"),
            ("knn_tip_acc", "kNN tip (downstream)", "#7A9E7E"),
            ("rho_suprafata_ems", "ρ suprafață EMS (downstream)", "#C8702A")]
    h = .26
    for i, (c, lab, col) in enumerate(cols):
        ax.barh(yy + (i - 1) * h, mean_metrics.loc[sel, c], h, color=col, label=lab)
    ax.set_yticks(yy, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=3, frameon=False)
    ax.set_title(f"(c) PCA k={k} pe anul de test: o sursă vs. toate", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cp2_slide2.png", dpi=220)
    plt.close(fig)


def fig_sweep(sweep, k):
    g = sweep.groupby("k")
    m, sd = g.mean(), g.std()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for c, lab, col in [("var_reconstruita", "varianță reconstruită (global)", "#1B2A41"),
                        ("trustworthiness", "trustworthiness (local)", "#2F6690"),
                        ("knn_tip_acc", "kNN tip (downstream)", "#7A9E7E"),
                        ("rho_suprafata_ems", "ρ suprafață EMS (downstream)", "#C8702A")]:
        ax.errorbar(m.index, m[c], yerr=sd[c], fmt="o-", ms=3, capsize=2, color=col, label=lab)
    ax.axvline(k, color="#C8702A", ls=":")
    ax.set_xscale("log")
    ax.set_xticks(m.index, [str(i) for i in m.index])
    ax.set_xlabel("k (număr de componente)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("PCA pe toate sursele (BGE-M3): metrici pe anul de test vs. k", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "k_sweep.png", dpi=180)
    plt.close(fig)


# ---------- raport ----------

def feature_dictionary(core):
    rows = [{"trasatura": f, "bloc": v[0], "sursa": v[1], "set_versiune": v[2], "fereastra": v[3],
             "rezolutie": v[4], "unitate": v[5], "transformare": v[6], "in_X": v[0] != "referință",
             "lipsa_%": round(100 * core[f].isna().mean(), 1)} for f, v in FEATURES.items()]
    rows += [{"trasatura": "tfidf:<termen>", "bloc": "text_emb", "sursa": "Text EMS",
              "set_versiune": "TF-IDF 1–2-grame, min_df=3, max 2000, stop words EN, sublinear_tf",
              "fereastra": "la activare", "rezolutie": "activare", "unitate": "pondere TF-IDF",
              "transformare": "L2 pe rând; vocabular pe train", "in_X": True, "lipsa_%": 0.0},
             {"trasatura": "bge:<0..1023>", "bloc": "text_emb", "sursa": "Text EMS",
              "set_versiune": "BAAI/bge-m3 (dense, 1024), sentence-transformers", "fereastra": "la activare",
              "rezolutie": "activare", "unitate": "–", "transformare": "L2 (model fix)", "in_X": "variantă",
              "lipsa_%": 0.0}]
    pd.DataFrame(rows).to_csv(REPORTS / "feature_dictionary.csv", index=False)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    core, bge = load()
    print("n =", len(core), "BGE:", bge.shape)
    feature_dictionary(core)
    u = univariate(core)
    fig_distributions(core)
    rho = spearman_matrix(core)
    rvm, rv_null, adj = block_rv(core, bge)
    adj.attrs["n"] = len(core)
    fire = (core["category"] == "fire").to_numpy()
    _, _, adj_fire = block_rv(core, bge, fire, "incendii")
    flood = (core["category"] == "flood").to_numpy()
    _, _, adj_flood = block_rv(core, bge, flood, "inundatii")
    mag = magnitude_by_source(core)
    k, ks, sweep, scree, scree_nw, share_nw, res = run_loyo(core, bge)
    table, mean_metrics = summarize(res)
    pca, loads, share, Z = full_fit_loadings(core, bge, k)
    stab = bge_stability(core, bge, k)
    fig_scree(scree, scree_nw, k, ks)
    fig_loadings(share, loads, k)
    fig_pc_scatter(Z, core)
    fig_slide(adj, adj_fire, int(fire.sum()), scree, k, ks, mean_metrics)
    fig_sweep(sweep, k)
    sw = sweep.groupby("k").agg(["mean", "std"])
    sw_tab = pd.DataFrame({c: [f"{a:.2f} ± {b:.2f}" for a, b in zip(sw[(c, "mean")], sw[(c, "std")])]
                           for c in ["var_reconstruita", "trustworthiness", "knn_tip_acc", "rho_suprafata_ems"]},
                          index=sw.index)

    d_tfidf = np.median([len(TfidfVectorizer(**TFIDF).fit(core.loc[tr, "text"]).vocabulary_)
                         for _, tr, _ in loyo(core["year"])])
    lines = ["# PCA pe protocolul LOYO — U19 (CP2)", "", "Generat de `pipeline/08_explore.py`.", "",
             "## Matricea X", "",
             f"n = {len(core)} activări; blocuri: imagine {len(NUM_BLOCKS['imagine'])}, meteo "
             f"{len(NUM_BLOCKS['meteo'])}, text structurat {len(NUM_BLOCKS['text_struct'])}, "
             f"TF-IDF ~{d_tfidf:.0f} (mediana pe fold-uri) sau BGE-M3 1024.",
             f"d total ≈ {23 + d_tfidf:.0f} (varianta TF-IDF) / {23 + 1024} (varianta BGE-M3).", "",
             "## Alegerea lui k", "",
             f"Analiza paralelă Horn pe train (X complet, TF-IDF): k pe fold-uri = {ks} → **k = {k}** (mediana). "
             f"Varianța cumulată la k pe train: {scree.mean(0).cumsum()[k - 1]:.1%} "
             f"(PC1 {scree.mean(0)[0]:.1%}, PC2 {scree.mean(0)[1]:.1%}, PC3 {scree.mean(0)[2]:.1%}).", "",
             "Fără ponderarea pe bloc, PC1–PC3 explică "
             f"{scree_nw.mean(0)[:3].sum():.1%} și se compun din: "
             + ", ".join(f"{b} {v:.0%}" for b, v in share_nw.sort_values(ascending=False).items())
             + " (media loadings² pe PC1–3) → axele ar măsura blocul cu varianța cea mai mare, nu structura.", "",
             "## Contribuția surselor la fiecare PC (fit pe tot setul, doar pentru interpretare)", "",
             share.round(2).to_markdown(), "",
             f"Stabilitatea axelor (varianta BGE-M3, |cos| între PC_j pe train și PC_j pe tot setul, media pe fold-uri): "
             + ", ".join(f"PC{j + 1} {s:.2f}" for j, s in enumerate(stab)), "",
             "## Metrici pe anul de test (media ± abaterea standard pe 9 fold-uri)", "",
             table.to_markdown(index=False), "",
             "## Baleiaj pe k (toate sursele, BGE-M3; media ± abaterea standard pe 9 fold-uri)", "",
             sw_tab.to_markdown(), "",
             "## Corelația între surse (coeficient RV)", "",
             "RV brut, tot setul:", "", rvm.round(2).to_markdown(), "",
             "Referința la zero (media RV pe 20 de permutări ale rândurilor; RV crește cu dimensiunea blocului):", "",
             rv_null.round(2).to_markdown(), "",
             "RV ajustat = (RV − RV0) / (1 − RV0), tot setul:", "", adj.round(2).to_markdown(), "",
             f"RV ajustat, doar incendii (n = {fire.sum()}):", "", adj_fire.round(2).to_markdown(), "",
             f"RV ajustat, doar inundații (n = {flood.sum()}):", "", adj_flood.round(2).to_markdown(), ""]
    (REPORTS / "pca_loyo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(mag.to_string())


if __name__ == "__main__":
    main()
