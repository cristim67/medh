# MEDE · U19 — Ce spune textul și nu spune satelitul

Magnitudinea dezastrelor din rapoartele Copernicus EMS vs. imaginile Sentinel bi-temporale
(Europa, 2017–2025, inundații / incendii / furtuni). Unitatea de analiză: o activare EMS Rapid Mapping.

## Structura

```
medh/
├── CP0/                      # fișa de concept (săpt. 3)
│   ├── CP0_U19.pptx
│   ├── CP0_U19.pdf
│   └── cerinta.md
├── CP1/                      # aliniere: data card + SOA (săpt. 4)
│   ├── CP1_U19.pptx
│   ├── CP1_U19.pdf
│   ├── datacard_U19.md       # o fișă pe sursă: EMS, GDACS, S1, S2, ERA5-Land, EM-DAT
│   ├── soa_U19.md            # tabelul SOA cu 10 articole
│   └── cerinta.md
├── CP2/                      # trăsături + reprezentare liniară (săpt. 6)
│   ├── CP2_U19.pptx
│   ├── CP2_U19.pdf
│   ├── trasaturi_U19.md      # X documentată, analiza uni/bivariată, PCA pe LOYO
│   └── cerinta.md
├── pipeline/
│   ├── config.py             # perioada, tipurile, țările, ferestrele pre/post, proiectul GEE
│   ├── 01_ems.py             # activările Copernicus EMS: text, AOI, suprafața mapată
│   ├── 02_gdacs.py           # alertele GDACS + potrivirea cu EMS
│   ├── 03_gee.py             # Sentinel-1, Sentinel-2, ERA5-Land din Google Earth Engine
│   ├── 04_emdat.py           # EM-DAT: decese, afectați, pagube
│   ├── 05_align.py           # tabelul aliniat + raportul de acoperire
│   ├── 06_matrix.py          # matricea X + scalarea pe fold-uri LOYO
│   ├── 07_text.py            # text: data evenimentului, cifre de magnitudine, embeddings BGE-M3
│   └── 08_explore.py         # analiza uni/bivariată, RV între surse, PCA pe LOYO + metrici
├── reports/                  # generate de pipeline
│   ├── coverage.md           # pâlnia de aliniere, pe tip și an
│   ├── coverage.csv
│   ├── x_shape.md            # forma lui X pe blocuri și fold-uri
│   ├── feature_dictionary.csv  # proveniența fiecărei trăsături
│   ├── pca_loyo.md           # k, scree, loadings, metrici PCA pe LOYO
│   └── figures/              # distribuții, Spearman, scree, loadings, baleiaj pe k
├── data/                     # neversionat; se creează la rulare
│   ├── raw/                  # cache-ul surselor (EMS, GDACS, GEE, EM-DAT)
│   └── interim/              # tabelele intermediare și aligned.csv
├── curs/                     # materialele cursului
├── plan_de_lucru.pdf
├── sablon.pptx               # șablonul de checkpoint
├── pyproject.toml
└── uv.lock
```

## Rulare

```bash
uv sync
uv run python pipeline/01_ems.py      # activările EMS: text, AOI, suprafața mapată
uv run python pipeline/02_gdacs.py    # alertele GDACS și potrivirea cu EMS
uv run python pipeline/03_gee.py      # Sentinel-1/2 + ERA5-Land (cere `earthengine authenticate`)
uv run python pipeline/04_emdat.py    # EM-DAT (export manual din public.emdat.be)
uv run python pipeline/05_align.py    # tabelul aliniat + raportul de acoperire
uv run python pipeline/06_matrix.py   # matricea X + scalarea pe fold-uri LOYO
uv run python pipeline/07_text.py     # trăsături de text + BGE-M3 (descarcă modelul, ~2,2 GB)
uv run python pipeline/08_explore.py  # analiza uni/bivariată + PCA pe LOYO
```

Opțiuni de test: `--limit N`, `--codes EMSR207,EMSR692`, `--refresh` (ignoră cache-ul).

## Pipeline

| Pas | Script | Ieșire (`data/interim/`, `reports/`) |
|-----|--------|--------------------------|
| 1 | `pipeline/01_ems.py` | `ems_events.csv`, `ems_aoi.gpkg` (straturile `aoi`, `observed_event`), `ems_sources.csv`, `ems_product_stats.csv` |
| 2 | `pipeline/02_gdacs.py` | `gdacs_events.csv`, `gdacs_match.csv`, `gdacs_sendai.csv` |
| 3 | `pipeline/03_gee.py` | `gee_features.csv` (Sentinel-1, Sentinel-2, ERA5-Land din Google Earth Engine) |
| 4 | `pipeline/04_emdat.py` | `emdat_europe.csv`, `emdat_match.csv` (necesită exportul EM-DAT în `data/raw/emdat/`; nu se redistribuie) |
| 5 | `pipeline/05_align.py` | `aligned.csv`, `reports/coverage.md` |
| 6 | `pipeline/06_matrix.py` | `X_raw.csv`, `reports/x_shape.md` |
| 7 | `pipeline/07_text.py` | `text_features.csv`, `text_bge_m3.npy`, `reports/text_dates.md` |
| 8 | `pipeline/08_explore.py` | `reports/feature_dictionary.csv`, `univariate.csv`, `spearman.csv`, `block_rv_*.csv`, `magnitude_by_source.md`, `pca_loyo.md`, `pca_metrics.csv`, `pca_k_sweep.csv`, `pca_loadings.csv`, `figures/` |

`data/` nu e versionat: totul se regenerează din scripturi.
