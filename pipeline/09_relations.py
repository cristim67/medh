"""Pasul 9: harta relațiilor dintre surse (CP3) — ce duplică o sursă și ce aduce nou, mai ales despre magnitudine.

Rânduri: nucleul (n = 391). Blocurile și scalarea sunt cele din pasul 8 (`FoldX`), potrivite pe train.
Toate analizele predictive sunt out-of-sample pe LOYO (9 fold-uri); analizele descriptive se fac pe tip
(incendii, inundații), pentru că pe tot setul tipul evenimentului domină orice corelație (CP2: RV).

1. Informație mutuală (MI, kNN, sklearn) între fiecare trăsătură și log(suprafața mapată EMS), pe tip;
   pragul de semnificație = P95 din 50 de permutări ale țintei.
2. CCA între perechi de surse (imagine, meteo, text = BGE-M3 redus cu PCA la 20 pe train), out-of-sample:
   CCA potrivit pe train, proiecțiile de test adunate pe fold-uri, corelația canonică 1 pe test;
   nul = aceeași procedură cu rândurile unei surse permutate (20 de permutări).
3. Graful kNN (10 vecini, cosinus) al fiecărei surse: suprapunerea Jaccard a vecinătăților între surse,
   raportată la suprapunerea din permutări (× peste întâmplare).
4. Magnitudinea: Ridge LOYO pentru log(suprafața EMS) din fiecare combinație de surse; ρ Spearman pe test
   (adunat pe fold-uri) cu IC 95% bootstrap (1000). „Ce aduce textul peste imagine” = Δρ(imagine+text − imagine),
   cu IC bootstrap pereche. Control pentru circularitate / mărimea AOI: imaginea doar cu fracții (fără arii)
   și modelul „doar aria AOI”.
5. Dezacordul text–imagine vs. incertitudinea textului: d = |ŷ_text − ŷ_imagine| (out-of-sample) și
   scorul de incertitudine (markeri de aproximare și modalitate la 100 de cuvinte); ρ Spearman + p din permutări.

Ieșiri (reports/): relations.md, mi_magnitude.csv, cca_pairs.csv, knn_overlap.csv, magnitude_increment.csv,
  hedging.csv, figures/relations_map.png, figures/mi_magnitude.png
Rulare: uv run python pipeline/09_relations.py   (după 07 și 08)
"""
import importlib
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors

from config import REPORTS

warnings.filterwarnings("ignore")
m08 = importlib.import_module("08_explore")
FIG = REPORTS / "figures"
SEED = 0
RNG = np.random.default_rng(SEED)
TYPES = ("fire", "flood")
TYPE_RO = {"fire": "incendii", "flood": "inundații", "all": "toate"}

IMG_FRAC = ["s1_dvv", "s1_dvh", "s1_flooded", "s2_dndvi", "s2_dnbr", "s2_dmndwi", "s2_burnt", "s2_newwater",
            "s2_clear_pre", "s2_clear_post", "s2_ok"]
HEDGES = [r"about", r"approximately", r"approx\.?", r"estimated", r"estimate", r"around", r"almost", r"nearly",
          r"some", r"over", r"more than", r"at least", r"up to", r"could", r"may", r"might", r"expected",
          r"possible", r"possibly", r"likely", r"unclear", r"reportedly", r"preliminary", r"not yet", r"potential"]
HEDGE_RE = re.compile(r"\b(" + "|".join(HEDGES) + r")\b", re.I)


# ---------- blocuri ----------

def text_pca(Xtr_bge, Xte_bge, n=20):
    p = PCA(n, random_state=SEED).fit(Xtr_bge)
    return p.transform(Xtr_bge), p.transform(Xte_bge)


def source_mats(core, bge, tr, te, img_cols=None):
    """Matricele pe surse, potrivite pe train. Textul = câmpuri structurate + BGE-M3 redus la 20 (PCA pe train)."""
    out = {}
    fx = m08.FoldX(core[tr], bge[tr], ["imagine", "meteo", "text_struct"])
    for b in ("imagine", "meteo", "text_struct"):
        out[b] = (fx._raw(b, core[tr], bge[tr]), fx._raw(b, core[te], bge[te]))
    if img_cols is not None:
        keep = [m08.NUM_BLOCKS["imagine"].index(c) for c in img_cols]
        out["imagine_frac"] = (out["imagine"][0][:, keep], out["imagine"][1][:, keep])
    ttr, tte = text_pca(bge[tr], bge[te])
    out["text_bge"] = (ttr / ttr[:, 0].std(), tte / ttr[:, 0].std())
    out["text"] = (np.hstack([out["text_struct"][0], out["text_bge"][0]]),
                   np.hstack([out["text_struct"][1], out["text_bge"][1]]))
    a = np.log1p(core["aoi_area_km2"].to_numpy())[:, None]
    mu, sd = a[tr].mean(), a[tr].std()
    out["aoi"] = ((a[tr] - mu) / sd, (a[te] - mu) / sd)
    return out


def loyo(df):
    return m08.loyo(df["year"].reset_index(drop=True))


# ---------- 1. MI ----------

def mi_magnitude(core):
    feats = [f for b in ("imagine", "meteo", "text_struct") for f in m08.NUM_BLOCKS[b]]
    rows = []
    for t in TYPES:
        d = core[(core["category"] == t) & core["ems_mapped_area_ha"].notna()]
        y = np.log1p(d["ems_mapped_area_ha"].to_numpy())
        X = d[feats].astype(float)
        X = X.fillna(X.median())
        X = np.column_stack([np.log1p(X[c].clip(lower=0)) if c in m08.LOG else X[c] for c in feats])
        mi = mutual_info_regression(X, y, n_neighbors=5, random_state=SEED)
        null = np.array([mutual_info_regression(X, RNG.permutation(y), n_neighbors=5, random_state=SEED)
                         for _ in range(50)])
        thr = np.percentile(null, 95, axis=0)
        for f, v, th in zip(feats, mi, thr):
            rows.append({"tip": t, "trasatura": f, "bloc": m08.FEATURES[f][0], "mi": v, "prag_p95": th,
                         "semnificativ": v > th, "n": len(d)})
    mi = pd.DataFrame(rows)
    mi.to_csv(REPORTS / "mi_magnitude.csv", index=False, float_format="%.4f")
    return mi


# ---------- 2. CCA out-of-sample ----------

def cca_oos(core, bge, a, b, perm=False, seed=0):
    rng = np.random.default_rng(seed)
    U, V = [], []
    for _, tr, te in loyo(core):
        s = source_mats(core, bge, tr, te)
        Atr, Ate = s[a]
        Btr, Bte = s[b]
        if perm:
            Btr, Bte = Btr[rng.permutation(len(Btr))], Bte[rng.permutation(len(Bte))]
        if len(Ate) < 2:
            continue
        c = CCA(n_components=1, max_iter=2000).fit(Atr, Btr)
        u, v = c.transform(Ate, Bte)
        U.append(u[:, 0])
        V.append(v[:, 0])
    U, V = np.concatenate(U), np.concatenate(V)
    return abs(np.corrcoef(U, V)[0, 1])


def cca_pairs(core, bge):
    pairs = [("imagine", "text"), ("imagine", "meteo"), ("meteo", "text"), ("imagine", "text_bge"),
             ("imagine", "text_struct")]
    rows = []
    for scope in ("all",) + TYPES:
        d = core if scope == "all" else core[core["category"] == scope]
        d = d.reset_index(drop=True)
        e = bge[(core["category"] == scope).to_numpy()] if scope != "all" else bge
        for a, b in pairs:
            r = cca_oos(d, e, a, b)
            null = [cca_oos(d, e, a, b, perm=True, seed=i) for i in range(20)]
            rows.append({"tip": scope, "pereche": f"{a} ↔ {b}", "r_canonic_test": r,
                         "nul_p95": np.percentile(null, 95), "n": len(d)})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "cca_pairs.csv", index=False, float_format="%.3f")
    return out


# ---------- 3. kNN ----------

def knn_sets(X, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    return [set(r) for r in idx]


def jaccard(A, B):
    return np.mean([len(a & b) / len(a | b) for a, b in zip(A, B)])


def knn_overlap(core, bge):
    rows = []
    for scope in ("all",) + TYPES:
        m = np.ones(len(core), bool) if scope == "all" else (core["category"] == scope).to_numpy()
        d, e = core[m].reset_index(drop=True), bge[m]
        idx = np.ones(len(d), bool)
        s = source_mats(d, e, idx, idx)
        ref = np.log1p(d[["ems_mapped_area_ha"]].astype(float).fillna(d["ems_mapped_area_ha"].median()))
        mats = {"imagine": s["imagine"][0], "meteo": s["meteo"][0], "text": s["text"][0],
                "ref. EMS (suprafață)": np.column_stack([ref.to_numpy().ravel(), np.ones(len(d))])}
        # pentru o singură variabilă cosinusul nu are sens → euclidian pe valoarea standardizată
        sets = {}
        for kname, X in mats.items():
            if kname.startswith("ref."):
                z = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
                nn = NearestNeighbors(n_neighbors=11).fit(z[:, None])
                sets[kname] = [set(r[1:]) for r in nn.kneighbors(z[:, None], return_distance=False)]
            else:
                sets[kname] = knn_sets(X)
        keys = list(sets)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                j = jaccard(sets[a], sets[b])
                null = []
                for _ in range(10):
                    p = RNG.permutation(len(d))
                    inv = np.argsort(p)
                    Bp = [set(p[list(sets[b][inv[i]])]) for i in range(len(d))]
                    null.append(jaccard(sets[a], Bp))
                rows.append({"tip": scope, "sursa_a": a, "sursa_b": b, "jaccard": j, "jaccard_nul": np.mean(null),
                             "x_peste_intamplare": j / np.mean(null), "n": len(d)})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "knn_overlap.csv", index=False, float_format="%.3f")
    return out


# ---------- 4. magnitudine: ce aduce fiecare sursă ----------

MODELS = {
    "doar aria AOI": ["aoi"],
    "doar meteo": ["meteo"],
    "doar text": ["text"],
    "doar imagine (fracții)": ["imagine_frac"],
    "doar imagine": ["imagine"],
    "imagine + text": ["imagine", "text"],
    "imagine (fracții) + text": ["imagine_frac", "text"],
    "imagine + meteo + text": ["imagine", "meteo", "text"],
}


def oos_predictions(core, bge, target):
    y = np.log1p(core[target].astype(float)).to_numpy()
    pred = {k: np.full(len(core), np.nan) for k in MODELS}
    for _, tr, te in loyo(core):
        s = source_mats(core, bge, tr, te, IMG_FRAC)
        ok = ~np.isnan(y[tr])
        if ok.sum() < 10:
            continue
        for name, blocks in MODELS.items():
            Xtr = np.hstack([s[b][0] for b in blocks])
            Xte = np.hstack([s[b][1] for b in blocks])
            r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr[ok], y[tr][ok])
            pred[name][te] = r.predict(Xte)
    return y, pred


def boot_rho(y, p, q=None, n=1000):
    ok = ~np.isnan(y) & ~np.isnan(p) & (np.ones_like(y, bool) if q is None else ~np.isnan(q))
    y, p = y[ok], p[ok]
    q = None if q is None else q[ok]
    base = spearmanr(p, y).statistic - (0 if q is None else spearmanr(q, y).statistic)
    bs = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        v = spearmanr(p[i], y[i]).statistic - (0 if q is None else spearmanr(q[i], y[i]).statistic)
        bs.append(v)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return base, lo, hi, len(y)


def magnitude_increment(core, bge):
    rows, preds = [], {}
    for target in ("ems_mapped_area_ha", "emdat_affected"):
        for scope in ("all",) + TYPES:
            m = np.ones(len(core), bool) if scope == "all" else (core["category"] == scope).to_numpy()
            d, e = core[m].reset_index(drop=True), bge[m]
            y, pred = oos_predictions(d, e, target)
            preds[(target, scope)] = (d, y, pred)
            for name in MODELS:
                r, lo, hi, n = boot_rho(y, pred[name])
                rows.append({"tinta": target, "tip": scope, "model": name, "rho": r, "ic_lo": lo, "ic_hi": hi, "n": n})
            for a, b in [("imagine + text", "doar imagine"), ("imagine + text", "doar text"),
                         ("imagine (fracții) + text", "doar imagine (fracții)")]:
                r, lo, hi, n = boot_rho(y, pred[a], pred[b])
                rows.append({"tinta": target, "tip": scope, "model": f"Δ {a} − {b}", "rho": r, "ic_lo": lo,
                             "ic_hi": hi, "n": n})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "magnitude_increment.csv", index=False, float_format="%.3f")
    return out, preds


# ---------- 5. dezacord vs. incertitudine ----------

def hedge_score(text):
    words = max(len(text.split()), 1)
    return 100 * len(HEDGE_RE.findall(text)) / words


def perm_p(x, y, n=2000):
    r = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, RNG.permutation(y)).statistic for _ in range(n)])
    return r, (np.abs(null) >= abs(r)).mean()


def hedging(preds):
    rows = []
    for scope in ("all",) + TYPES:
        d, y, pred = preds[("ems_mapped_area_ha", scope)]
        h = d["text"].map(hedge_score).to_numpy()
        dis = np.abs(pred["doar text"] - pred["doar imagine"])
        err_t = np.abs(pred["doar text"] - y)
        err_i = np.abs(pred["doar imagine"] - y)
        ok = ~np.isnan(dis) & ~np.isnan(y)
        for name, v in [("|ŷ text − ŷ imagine|", dis), ("|eroare text|", err_t), ("|eroare imagine|", err_i)]:
            r, p = perm_p(h[ok], v[ok])
            rows.append({"tip": scope, "variabila": name, "rho_cu_incertitudinea": r, "p_perm": p, "n": ok.sum(),
                         "texte_cu_marker_%": 100 * (h > 0).mean()})
    # incendiile care dau cifra: suprafața din text vs. suprafața arsă din S2 (ha)
    d = preds[("ems_mapped_area_ha", "fire")][0]
    m = (d["txt_area_ha"] > 0) & d["s2_burnt_km2"].notna()
    dd = d[m]
    dis = np.abs(np.log1p(dd["txt_area_ha"]) - np.log1p(dd["s2_burnt_km2"] * 100))
    r, p = perm_p(dd["text"].map(hedge_score).to_numpy(), dis.to_numpy())
    rows.append({"tip": "fire", "variabila": "|log ha text − log ha S2| (texte cu cifră)", "rho_cu_incertitudinea": r,
                 "p_perm": p, "n": int(m.sum()), "texte_cu_marker_%": 100 * (dd["text"].map(hedge_score) > 0).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "hedging.csv", index=False, float_format="%.3f")
    return out


# ---------- figuri și raport ----------

def fig_mi(mi):
    piv = mi.pivot(index="trasatura", columns="tip", values="mi")
    sig = mi.pivot(index="trasatura", columns="tip", values="semnificativ")
    order = [f for b in ("imagine", "meteo", "text_struct") for f in m08.NUM_BLOCKS[b]]
    piv, sig = piv.loc[order], sig.loc[order]
    fig, ax = plt.subplots(figsize=(6, 6.5))
    y = np.arange(len(order))
    for i, (t, col) in enumerate([("fire", "#C8702A"), ("flood", "#2F6690")]):
        ax.barh(y + (i - .5) * .4, piv[t], .4, color=col, alpha=[.35, 1][0], label=f"{TYPE_RO[t]} (n.s.)")
        ax.barh((y + (i - .5) * .4)[sig[t].to_numpy()], piv[t][sig[t]], .4, color=col,
                label=f"{TYPE_RO[t]} (> P95 permutări)")
    ax.set_yticks(y, order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("informație mutuală cu log(suprafața EMS) (nats)")
    ax.legend(fontsize=7)
    ax.set_title("Ce trăsături poartă informație despre magnitudine, pe tip", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "mi_magnitude.png", dpi=180)
    plt.close(fig)


def fig_map(inc, cca, knn):
    """Figura centrală CP3: (a) ρ pe surse cu IC, pe tip; (b) CCA test vs. nul; (c) kNN × peste întâmplare."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    sel = ["doar aria AOI", "doar meteo", "doar text", "doar imagine (fracții)", "doar imagine", "imagine + text",
           "imagine + meteo + text"]
    d = inc[inc["tinta"] == "ems_mapped_area_ha"]
    yy = np.arange(len(sel))
    for i, (t, col) in enumerate([("fire", "#C8702A"), ("flood", "#2F6690")]):
        r = d[d["tip"] == t].set_index("model").loc[sel]
        ax.errorbar(r["rho"], yy + (i - .5) * .3, xerr=[r["rho"] - r["ic_lo"], r["ic_hi"] - r["rho"]], fmt="o",
                    color=col, ms=4, capsize=2, label=f"{TYPE_RO[t]} (n = {int(r['n'].iloc[0])})")
    ax.axvline(0, color="#999", lw=.8)
    ax.set_yticks(yy, sel, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-.4, 1)
    ax.set_xlabel("ρ Spearman cu log(suprafața EMS), test LOYO, IC 95%", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) Magnitudinea: ce sursă o poartă", fontsize=9)

    ax = axes[1]
    pairs = ["imagine ↔ text", "imagine ↔ meteo", "meteo ↔ text"]
    c = cca.set_index(["tip", "pereche"])
    yy = np.arange(len(pairs))
    for i, (t, col) in enumerate([("all", "#1B2A41"), ("fire", "#C8702A"), ("flood", "#2F6690")]):
        r = c.loc[t].loc[pairs]
        ax.barh(yy + (i - 1) * .27, r["r_canonic_test"], .27, color=col, label=TYPE_RO[t])
        ax.scatter(r["nul_p95"], yy + (i - 1) * .27, marker="|", color="k", s=60, zorder=3)
    ax.set_yticks(yy, pairs, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("corelația canonică 1 pe test (| = P95 nul)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(b) CCA între surse, out-of-sample", fontsize=9)

    ax = axes[2]
    k = knn.copy()
    k["pereche"] = k["sursa_a"].str.replace(" (suprafață)", "") + " ↔ " + k["sursa_b"].str.replace(" (suprafață)", "")
    pairs = ["imagine ↔ text", "imagine ↔ meteo", "meteo ↔ text", "imagine ↔ ref. EMS", "text ↔ ref. EMS"]
    k = k.set_index(["tip", "pereche"])
    yy = np.arange(len(pairs))
    for i, (t, col) in enumerate([("all", "#1B2A41"), ("fire", "#C8702A"), ("flood", "#2F6690")]):
        r = k.loc[t].reindex(pairs)
        ax.barh(yy + (i - 1) * .27, r["x_peste_intamplare"], .27, color=col, label=TYPE_RO[t])
    ax.axvline(1, color="#999", lw=.8, ls="--")
    ax.set_yticks(yy, pairs, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("vecini kNN comuni (× peste întâmplare)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(c) Grafuri kNN: aceiași vecini?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "relations_map.png", dpi=220)
    plt.close(fig)


def fmt_inc(inc, target):
    d = inc[inc["tinta"] == target]
    t = d.assign(v=[f"{r:+.2f} [{a:+.2f}, {b:+.2f}]" for r, a, b in zip(d["rho"], d["ic_lo"], d["ic_hi"])])
    piv = t.pivot(index="model", columns="tip", values="v")
    n = t.groupby("tip")["n"].max()
    piv.columns = [f"{TYPE_RO[c]} (n ≤ {n[c]})" for c in piv.columns]
    order = list(MODELS) + [m for m in piv.index if m.startswith("Δ")]
    return piv.loc[order].to_markdown()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    core, bge = m08.load()
    mi = mi_magnitude(core)
    cca = cca_pairs(core, bge)
    knn = knn_overlap(core, bge)
    inc, preds = magnitude_increment(core, bge)
    hed = hedging(preds)
    fig_mi(mi)
    fig_map(inc, cca, knn)

    sig = mi[mi["semnificativ"]].sort_values(["tip", "mi"], ascending=[True, False])
    lines = ["# Harta relațiilor între surse — U19 (CP3)", "", "Generat de `pipeline/09_relations.py`.", "",
             "## 1. Magnitudinea: ce aduce fiecare sursă (Ridge LOYO, ρ pe test, IC 95% bootstrap)", "",
             "Ținta: log(suprafața mapată EMS).", "", fmt_inc(inc, "ems_mapped_area_ha"), "",
             "Ținta: log(EM-DAT afectați) — referință independentă de imagini, dar la nivel de țară și cu n mic.", "",
             fmt_inc(inc, "emdat_affected"), "",
             "## 2. Informație mutuală cu log(suprafața EMS), peste P95 din 50 de permutări", "",
             sig[["tip", "trasatura", "bloc", "mi", "prag_p95"]].round(3).to_markdown(index=False), "",
             "## 3. CCA între surse (corelația canonică 1 pe test, LOYO; nul = 20 de permutări)", "",
             cca.round(3).to_markdown(index=False), "",
             "## 4. Grafuri kNN (10 vecini): suprapunerea vecinătăților între surse", "",
             knn.round(3).to_markdown(index=False), "",
             "## 5. Dezacordul text–imagine vs. incertitudinea textului", "",
             "Scorul de incertitudine = markeri de aproximare și modalitate („about”, „estimated”, „may”, "
             "„expected”…) la 100 de cuvinte. p din 2000 de permutări.", "",
             hed.round(3).to_markdown(index=False), ""]
    (REPORTS / "relations.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
