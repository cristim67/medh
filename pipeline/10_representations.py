"""Pasul 10: tabelul principal CP3 — PCA (baseline) vs. metoda aleasă, pe fiecare sursă și pe toate sursele.

Cerința CP3: PCA ca baseline și metoda aleasă, aplicate pe fiecare sursă separat și pe datele combinate;
metrici: varianță explicată, trustworthiness, eroare de reconstrucție; stabilitate pe seed-uri și subperioade.

Surse (blocurile din pasul 8, scalare și ponderare MFA potrivite pe train):
  imagine (14) · meteo (4) · text (5 câmpuri structurate + BGE-M3 1024) · toate (1047)
k pe sursă: analiza paralelă Horn pe train (mediana pe fold-uri), plafonat la d − 1; „toate” = 11 (CP2).

Reprezentări:
  - PCA                 — baseline (liniar, determinist)
  - Autoencoder         — METODA ALEASĂ: MLP d → 64 → k → 64 → d, tanh, L2 1e-3, Adam, 500 de epoci;
                          singura alternativă cu eroare de reconstrucție comparabilă direct cu PCA
  - Kernel PCA (RBF)    — γ = 1 / mediana distanțelor² pe train; pre-imagine pentru reconstrucție
  - UMAP                — 15 vecini, min_dist 0,1, k dimensiuni, `transform` pe test (fără reconstrucție)
  - CCA imagine ↔ text  — multi-view: media variatelor canonice ale celor două surse (doar pe „toate”)

Metrici pe anul de test (split LOYO, 9 fold-uri):
  - varianță explicată pe test = 1 − ‖X − X̂‖² / ‖X − media_train‖²  (eroarea de reconstrucție = 1 − ea)
  - trustworthiness@10 între X_test (al sursei) și Z_test
  - ρ distanțe: Spearman între distanțele perechi din X_test și Z_test (structura globală)
  - downstream: ρ Spearman Ridge(Z) → log(suprafața EMS); acuratețea kNN (5) pentru tip
Stabilitate:
  - seed-uri (0–4): media pe seed + abaterea între seed-uri; Jaccard al vecinilor (10-NN) din Z_test între
    perechi de seed-uri (1 = aceeași vecinătate)
  - subperioade: train 2017–2021 → test 2022–2025 și train 2021–2025 → test 2017–2020, pe „toate”
Intervale: IC 95% t pe cele 9 medii de fold; Δ față de PCA pe aceeași sursă, pereche pe fold-uri.

Ieșiri: reports/representations.md, representations_runs.csv, representations_summary.csv,
        representations_stability.csv, figures/representations.png
Rulare: uv run python pipeline/10_representations.py   (după 07 și 08)
"""
import importlib
import itertools
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, t as student
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import RidgeCV
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPRegressor

from config import REPORTS

warnings.filterwarnings("ignore")
m08 = importlib.import_module("08_explore")
m09 = importlib.import_module("09_relations")
FIG = REPORTS / "figures"
K_ALL = 11
SEEDS = [0, 1, 2, 3, 4]
SOURCES = {"imagine": ["imagine"], "meteo": ["meteo"], "text": ["text_struct", "bge"],
           "toate": m08.CONFIGS["toate sursele (BGE-M3)"]}
SUBPERIODS = {"train 2017–2021 → test 2022–2025": (range(2017, 2022), range(2022, 2026)),
              "train 2021–2025 → test 2017–2020": (range(2021, 2026), range(2017, 2021))}


# ---------- reprezentări: fit → (encode, decode | None) ----------

def rep_pca(Xtr, k, seed):
    p = PCA(k, random_state=seed).fit(Xtr)
    return p.transform, p.inverse_transform


def rep_ae(Xtr, k, seed):
    ae = MLPRegressor(hidden_layer_sizes=(64, k, 64), activation="tanh", alpha=1e-3, max_iter=500,
                      random_state=seed, learning_rate_init=1e-3).fit(Xtr, Xtr)
    W, b = ae.coefs_, ae.intercepts_

    def enc(X):
        return np.tanh(np.tanh(X @ W[0] + b[0]) @ W[1] + b[1])

    def dec(Z):
        return np.tanh(Z @ W[2] + b[2]) @ W[3] + b[3]
    return enc, dec


def rep_kpca(Xtr, k, seed):
    g = 1.0 / np.median(pdist(Xtr, "sqeuclidean"))
    p = KernelPCA(k, kernel="rbf", gamma=g, fit_inverse_transform=True, alpha=1e-3, random_state=seed).fit(Xtr)
    return p.transform, p.inverse_transform


def rep_umap(Xtr, k, seed):
    import umap
    u = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=k, random_state=seed).fit(Xtr)
    return u.transform, None


REPS = {"PCA (baseline)": (rep_pca, False), "Autoencoder (aleasă)": (rep_ae, True),
        "Kernel PCA (RBF)": (rep_kpca, False), "UMAP": (rep_umap, True)}


# ---------- metrici ----------

def metrics(Xtr, Xte, Ztr, Zte, dec, d_tr, d_te):
    out = {"var_explicata": np.nan}
    if dec is not None:
        mu = Xtr.mean(0)
        out["var_explicata"] = 1 - ((Xte - dec(Zte)) ** 2).sum() / ((Xte - mu) ** 2).sum()
    out["trustworthiness"] = trustworthiness(Xte, Zte, n_neighbors=10)
    out["rho_distante"] = spearmanr(pdist(Xte), pdist(Zte)).statistic
    out["knn_tip_acc"] = (KNeighborsClassifier(5).fit(Ztr, d_tr["category"]).predict(Zte)
                          == d_te["category"]).mean()
    y_tr, y_te = np.log1p(d_tr["ems_mapped_area_ha"]).to_numpy(), np.log1p(d_te["ems_mapped_area_ha"]).to_numpy()
    a, b = ~np.isnan(y_tr), ~np.isnan(y_te)
    pred = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Ztr[a], y_tr[a]).predict(Zte[b])
    out["rho_suprafata_ems"] = spearmanr(pred, y_te[b]).statistic
    return out


METRICS = ["var_explicata", "trustworthiness", "rho_distante", "knn_tip_acc", "rho_suprafata_ems"]


def knn_sets(Z, k=10):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return [set(r) for r in idx]


# ---------- rulare ----------

def choose_k(core, bge, folds):
    ks = {}
    for s, blocks in SOURCES.items():
        if s == "toate":
            ks[s] = K_ALL
            continue
        vals = []
        for _, tr, _ in folds:
            X = m08.FoldX(core[tr], bge[tr], blocks).transform(core[tr], bge[tr])
            vals.append(m08.horn_k(X, n_perm=10))
        d = X.shape[1]
        ks[s] = int(min(max(np.median(vals), 2), d - 1))
    return ks


def run_grid(core, bge, folds, ks):
    rows, stab = [], []
    for s, blocks in SOURCES.items():
        cache = []
        for y, tr, te in folds:
            fx = m08.FoldX(core[tr], bge[tr], blocks)
            cache.append((y, tr, te, fx.transform(core[tr], bge[tr]), fx.transform(core[te], bge[te])))
        for name, (fn, stochastic) in REPS.items():
            t0 = time.time()
            neigh = {}
            for seed in (SEEDS if stochastic else [0]):
                for y, tr, te, Xtr, Xte in cache:
                    enc, dec = fn(Xtr, ks[s], seed)
                    Ztr, Zte = enc(Xtr), enc(Xte)
                    rows.append({"sursa": s, "reprezentare": name, "k": ks[s], "seed": seed, "an_test": y,
                                 **metrics(Xtr, Xte, Ztr, Zte, dec, core[tr], core[te])})
                    neigh[(seed, y)] = knn_sets(Zte)
            if stochastic:
                for y in [f[0] for f in folds]:
                    js = [np.mean([len(a & b) / len(a | b) for a, b in zip(neigh[(s1, y)], neigh[(s2, y)])])
                          for s1, s2 in itertools.combinations(SEEDS, 2)]
                    stab.append({"sursa": s, "reprezentare": name, "an_test": y, "jaccard_intre_seeduri": np.mean(js)})
            print(f"{s:8s} {name:22s} {time.time() - t0:5.1f}s")
    # CCA multi-view (imagine ↔ text), doar pe „toate”
    for y, tr, te in folds:
        src = m09.source_mats(core, bge, tr, te)
        fx = m08.FoldX(core[tr], bge[tr], SOURCES["toate"])
        Xtr, Xte = fx.transform(core[tr], bge[tr]), fx.transform(core[te], bge[te])
        c = CCA(n_components=K_ALL, max_iter=3000).fit(src["imagine"][0], src["text"][0])
        Ztr = np.mean(c.transform(src["imagine"][0], src["text"][0]), axis=0)
        Zte = np.mean(c.transform(src["imagine"][1], src["text"][1]), axis=0)
        rows.append({"sursa": "toate", "reprezentare": "CCA imagine ↔ text", "k": K_ALL, "seed": 0, "an_test": y,
                     **metrics(Xtr, Xte, Ztr, Zte, None, core[tr], core[te])})
    runs = pd.DataFrame(rows)
    runs.to_csv(REPORTS / "representations_runs.csv", index=False, float_format="%.4f")
    return runs, pd.DataFrame(stab)


def run_subperiods(core, bge):
    rows = []
    for label, (tr_years, te_years) in SUBPERIODS.items():
        tr = core["year"].isin(tr_years).to_numpy()
        te = core["year"].isin(te_years).to_numpy()
        fx = m08.FoldX(core[tr], bge[tr], SOURCES["toate"])
        Xtr, Xte = fx.transform(core[tr], bge[tr]), fx.transform(core[te], bge[te])
        for name, (fn, stochastic) in REPS.items():
            for seed in (SEEDS if stochastic else [0]):
                enc, dec = fn(Xtr, K_ALL, seed)
                rows.append({"subperioada": label, "reprezentare": name, "seed": seed,
                             "n_train": int(tr.sum()), "n_test": int(te.sum()),
                             **metrics(Xtr, Xte, enc(Xtr), enc(Xte), dec, core[tr], core[te])})
    return pd.DataFrame(rows)


# ---------- rezumat ----------

def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    h = student.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def summarize(runs):
    num, txt = [], []
    for (s, name), r in runs.groupby(["sursa", "reprezentare"], sort=False):
        per_fold = r.groupby("an_test")[METRICS].mean()
        per_seed = r.groupby("seed")[METRICS].mean()
        base = runs[(runs["sursa"] == s) & (runs["reprezentare"] == "PCA (baseline)")].groupby("an_test")[METRICS].mean()
        nrow = {"sursa": s, "reprezentare": name, "k": r["k"].iloc[0], "seeduri": r["seed"].nunique()}
        trow = dict(nrow)
        for m in METRICS:
            mu, lo, hi = ci(per_fold[m])
            sd = per_seed[m].std(ddof=1) if len(per_seed) > 1 else np.nan
            nrow.update({m: mu, m + "_lo": lo, m + "_hi": hi, m + "_sd_seed": sd})
            if name != "PCA (baseline)":
                dmu, dlo, dhi = ci(per_fold[m] - base[m])
                nrow.update({"d_" + m: dmu, "d_" + m + "_lo": dlo, "d_" + m + "_hi": dhi})
            trow[m] = "–" if np.isnan(mu) else (f"{mu:.2f} [{lo:.2f}, {hi:.2f}]" + (f" ±{sd:.2f}" if sd == sd else ""))
        num.append(nrow)
        txt.append(trow)
    return pd.DataFrame(num), pd.DataFrame(txt)


def delta_table(num):
    rows = []
    for _, r in num[num["reprezentare"] != "PCA (baseline)"].iterrows():
        row = {"sursa": r["sursa"], "reprezentare": r["reprezentare"]}
        for m in METRICS:
            if np.isnan(r.get("d_" + m, np.nan)):
                row[m] = "–"
                continue
            lo, hi = r["d_" + m + "_lo"], r["d_" + m + "_hi"]
            v = "bate PCA" if lo > 0 else ("sub PCA" if hi < 0 else "egal")
            row[m] = f"{r['d_' + m]:+.2f} [{lo:+.2f}, {hi:+.2f}] {v}"
        rows.append(row)
    return pd.DataFrame(rows)


def fig(num):
    show = ["var_explicata", "trustworthiness", "rho_distante", "rho_suprafata_ems"]
    labels = {"var_explicata": "varianță explicată pe test\n(1 − eroarea de reconstrucție)",
              "trustworthiness": "trustworthiness@10\n(local)", "rho_distante": "ρ distanțe X–Z\n(global)",
              "rho_suprafata_ems": "ρ suprafață EMS\n(downstream)"}
    reps = ["PCA (baseline)", "Autoencoder (aleasă)", "Kernel PCA (RBF)", "UMAP", "CCA imagine ↔ text"]
    colors = {"PCA (baseline)": "#C8702A", "Autoencoder (aleasă)": "#2F6690", "Kernel PCA (RBF)": "#7A9E7E",
              "UMAP": "#8C6BB1", "CCA imagine ↔ text": "#1B2A41"}
    srcs = list(SOURCES)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
    for ax, m in zip(axes, show):
        for j, rep in enumerate(reps):
            d = num[num["reprezentare"] == rep].set_index("sursa").reindex(srcs)
            y = np.arange(len(srcs)) + (j - 2) * .15
            ok = d[m].notna().to_numpy()
            ax.errorbar(d[m][ok], y[ok], xerr=[(d[m] - d[m + "_lo"])[ok], (d[m + "_hi"] - d[m])[ok]], fmt="o",
                        ms=3.5, capsize=1.5, lw=.9, color=colors[rep], label=rep)
        ax.set_title(labels[m], fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="#eee")
        for i in range(1, len(srcs)):
            ax.axhline(i - .5, color="#ddd", lw=.6)
    ks = num.drop_duplicates("sursa").set_index("sursa")["k"]
    axes[0].set_yticks(range(len(srcs)), [f"{s} (k = {ks[s]})" for s in srcs], fontsize=8)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(-1.4, -0.12), ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "representations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    core, bge = m08.load()
    folds = m08.loyo(core["year"])
    ks = choose_k(core, bge, folds)
    print("k pe sursă:", ks)
    runs, stab = run_grid(core, bge, folds, ks)
    sub = run_subperiods(core, bge)
    num, txt = summarize(runs)
    deltas = delta_table(num)
    num.to_csv(REPORTS / "representations_summary.csv", index=False, float_format="%.4f")
    st = stab.groupby(["sursa", "reprezentare"], sort=False)["jaccard_intre_seeduri"].agg(["mean", "min"]).round(2)
    st.to_csv(REPORTS / "representations_stability.csv")
    sub_t = sub.groupby(["subperioada", "reprezentare"], sort=False)[METRICS].agg(["mean", "std"])
    sub_s = pd.DataFrame({m: [("–" if np.isnan(a) else f"{a:.2f}" + (f" ±{b:.2f}" if b == b else ""))
                              for a, b in zip(sub_t[(m, "mean")], sub_t[(m, "std")])] for m in METRICS},
                         index=sub_t.index)
    fig(num)
    lines = ["# Tabelul principal CP3 — PCA vs. metoda aleasă, pe surse · U19", "",
             "Generat de `pipeline/10_representations.py`. Split LOYO (9 fold-uri), metrici pe anul de test. "
             f"k pe sursă (Horn pe train): {ks}. Metoda aleasă: autoencoder; seed-uri {SEEDS} pentru autoencoder "
             "și UMAP.", "",
             "Celulă = media [IC 95% t pe cele 9 fold-uri] ± abaterea între seed-uri; „–” = metrica nu e definită "
             "(UMAP și CCA nu au reconstrucție).", "",
             txt.to_markdown(index=False), "",
             "## Diferența față de PCA pe aceeași sursă (pereche pe fold-uri, IC 95% t)", "",
             deltas.to_markdown(index=False), "",
             "## Stabilitate la seed: vecinii (10-NN) din Z_test, Jaccard mediu între perechi de seed-uri", "",
             st.to_markdown(), "",
             "## Stabilitate pe subperioade (toate sursele, k = 11; media ± abaterea între seed-uri)", "",
             sub_s.to_markdown(), ""]
    (REPORTS / "representations.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
