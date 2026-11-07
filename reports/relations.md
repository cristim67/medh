# Harta relațiilor între surse — U19 (CP3)

Generat de `pipeline/09_relations.py`.

## 1. Magnitudinea: ce aduce fiecare sursă (Ridge LOYO, ρ pe test, IC 95% bootstrap)

Ținta: log(suprafața mapată EMS).

| model                                               | toate (n ≤ 370)      | incendii (n ≤ 230)   | inundații (n ≤ 129)   |
|:----------------------------------------------------|:---------------------|:---------------------|:----------------------|
| doar aria AOI                                       | +0.54 [+0.46, +0.61] | +0.76 [+0.67, +0.82] | +0.69 [+0.57, +0.77]  |
| doar meteo                                          | +0.09 [-0.00, +0.19] | +0.34 [+0.21, +0.45] | +0.12 [-0.05, +0.29]  |
| doar text                                           | +0.26 [+0.16, +0.36] | +0.42 [+0.31, +0.54] | +0.10 [-0.07, +0.26]  |
| doar imagine (fracții)                              | +0.20 [+0.09, +0.29] | +0.35 [+0.23, +0.45] | -0.29 [-0.45, -0.13]  |
| doar imagine                                        | +0.70 [+0.64, +0.76] | +0.80 [+0.73, +0.86] | +0.64 [+0.51, +0.73]  |
| imagine + text                                      | +0.72 [+0.66, +0.77] | +0.83 [+0.77, +0.88] | +0.63 [+0.51, +0.73]  |
| imagine (fracții) + text                            | +0.32 [+0.23, +0.41] | +0.47 [+0.36, +0.57] | -0.06 [-0.23, +0.09]  |
| imagine + meteo + text                              | +0.72 [+0.67, +0.77] | +0.82 [+0.76, +0.87] | +0.63 [+0.52, +0.73]  |
| Δ imagine (fracții) + text − doar imagine (fracții) | +0.13 [+0.04, +0.21] | +0.12 [+0.03, +0.22] | +0.23 [+0.08, +0.38]  |
| Δ imagine + text − doar imagine                     | +0.02 [-0.02, +0.06] | +0.03 [+0.00, +0.06] | -0.00 [-0.06, +0.05]  |
| Δ imagine + text − doar text                        | +0.46 [+0.37, +0.56] | +0.41 [+0.30, +0.52] | +0.53 [+0.36, +0.72]  |

Ținta: log(EM-DAT afectați) — referință independentă de imagini, dar la nivel de țară și cu n mic.

| model                                               | toate (n ≤ 122)      | incendii (n ≤ 63)    | inundații (n ≤ 56)   |
|:----------------------------------------------------|:---------------------|:---------------------|:---------------------|
| doar aria AOI                                       | -0.47 [-0.60, -0.32] | -0.42 [-0.65, -0.18] | +0.06 [-0.21, +0.33] |
| doar meteo                                          | +0.11 [-0.09, +0.29] | -0.53 [-0.73, -0.27] | -0.18 [-0.44, +0.08] |
| doar text                                           | +0.38 [+0.21, +0.53] | +0.15 [-0.09, +0.38] | +0.18 [-0.06, +0.42] |
| doar imagine (fracții)                              | -0.05 [-0.23, +0.13] | -0.54 [-0.73, -0.31] | -0.05 [-0.32, +0.20] |
| doar imagine                                        | +0.12 [-0.06, +0.28] | -0.46 [-0.65, -0.23] | +0.17 [-0.06, +0.39] |
| imagine + text                                      | +0.33 [+0.16, +0.48] | -0.24 [-0.47, +0.01] | +0.34 [+0.10, +0.56] |
| imagine (fracții) + text                            | +0.29 [+0.10, +0.45] | -0.24 [-0.45, -0.00] | +0.29 [+0.03, +0.51] |
| imagine + meteo + text                              | +0.27 [+0.08, +0.43] | -0.20 [-0.42, +0.06] | +0.34 [+0.11, +0.56] |
| Δ imagine (fracții) + text − doar imagine (fracții) | +0.33 [+0.18, +0.49] | +0.30 [+0.02, +0.55] | +0.35 [+0.07, +0.60] |
| Δ imagine + text − doar imagine                     | +0.21 [+0.06, +0.35] | +0.23 [-0.08, +0.54] | +0.17 [-0.06, +0.42] |
| Δ imagine + text − doar text                        | -0.06 [-0.14, +0.03] | -0.39 [-0.61, -0.16] | +0.16 [-0.06, +0.37] |

## 2. Informație mutuală cu log(suprafața EMS), peste P95 din 50 de permutări

| tip   | trasatura        | bloc        |    mi |   prag_p95 |
|:------|:-----------------|:------------|------:|-----------:|
| fire  | s2_burnt_km2     | imagine     | 1.014 |      0.053 |
| fire  | s1_flooded_km2   | imagine     | 0.399 |      0.045 |
| fire  | s2_burnt         | imagine     | 0.18  |      0.043 |
| fire  | s2_newwater_km2  | imagine     | 0.179 |      0.053 |
| fire  | txt_area_ha      | text_struct | 0.154 |      0.024 |
| fire  | s2_dnbr          | imagine     | 0.13  |      0.05  |
| fire  | s2_dndvi         | imagine     | 0.091 |      0.043 |
| fire  | era5_tp_event_mm | meteo       | 0.062 |      0.048 |
| fire  | era5_t2m_max_c   | meteo       | 0.061 |      0.042 |
| fire  | txt_deaths       | text_struct | 0.003 |      0.003 |
| flood | s1_flooded_km2   | imagine     | 0.289 |      0.084 |
| flood | s2_newwater_km2  | imagine     | 0.046 |      0.031 |
| flood | s2_newwater      | imagine     | 0.037 |      0.025 |
| flood | s2_dmndwi        | imagine     | 0.033 |      0.027 |
| flood | s2_burnt_km2     | imagine     | 0.022 |      0.022 |

## 3. CCA între surse (corelația canonică 1 pe test, LOYO; nul = 20 de permutări)

| tip   | pereche               |   r_canonic_test |   nul_p95 |   n |
|:------|:----------------------|-----------------:|----------:|----:|
| all   | imagine ↔ text        |            0.817 |     0.084 | 391 |
| all   | imagine ↔ meteo       |            0.813 |     0.079 | 391 |
| all   | meteo ↔ text          |            0.863 |     0.098 | 391 |
| all   | imagine ↔ text_bge    |            0.817 |     0.066 | 391 |
| all   | imagine ↔ text_struct |            0.275 |     0.121 | 391 |
| fire  | imagine ↔ text        |            0.65  |     0.124 | 235 |
| fire  | imagine ↔ meteo       |            0.541 |     0.137 | 235 |
| fire  | meteo ↔ text          |            0.467 |     0.152 | 235 |
| fire  | imagine ↔ text_bge    |            0.645 |     0.112 | 235 |
| fire  | imagine ↔ text_struct |            0.068 |     0.095 | 235 |
| flood | imagine ↔ text        |            0.164 |     0.195 | 135 |
| flood | imagine ↔ meteo       |            0.473 |     0.245 | 135 |
| flood | meteo ↔ text          |            0.446 |     0.128 | 135 |
| flood | imagine ↔ text_bge    |            0.511 |     0.108 | 135 |
| flood | imagine ↔ text_struct |            0.172 |     0.267 | 135 |

## 4. Grafuri kNN (10 vecini): suprapunerea vecinătăților între surse

| tip   | sursa_a   | sursa_b              |   jaccard |   jaccard_nul |   x_peste_intamplare |   n |
|:------|:----------|:---------------------|----------:|--------------:|---------------------:|----:|
| all   | imagine   | meteo                |     0.036 |         0.015 |                2.454 | 391 |
| all   | imagine   | text                 |     0.033 |         0.014 |                2.414 | 391 |
| all   | imagine   | ref. EMS (suprafață) |     0.032 |         0.013 |                2.356 | 391 |
| all   | meteo     | text                 |     0.044 |         0.014 |                3.126 | 391 |
| all   | meteo     | ref. EMS (suprafață) |     0.015 |         0.013 |                1.085 | 391 |
| all   | text      | ref. EMS (suprafață) |     0.015 |         0.014 |                1.066 | 391 |
| fire  | imagine   | meteo                |     0.05  |         0.023 |                2.178 | 235 |
| fire  | imagine   | text                 |     0.035 |         0.023 |                1.492 | 235 |
| fire  | imagine   | ref. EMS (suprafață) |     0.05  |         0.024 |                2.052 | 235 |
| fire  | meteo     | text                 |     0.042 |         0.022 |                1.91  | 235 |
| fire  | meteo     | ref. EMS (suprafață) |     0.026 |         0.024 |                1.118 | 235 |
| fire  | text      | ref. EMS (suprafață) |     0.026 |         0.024 |                1.05  | 235 |
| flood | imagine   | meteo                |     0.076 |         0.039 |                1.962 | 135 |
| flood | imagine   | text                 |     0.055 |         0.038 |                1.443 | 135 |
| flood | imagine   | ref. EMS (suprafață) |     0.068 |         0.04  |                1.68  | 135 |
| flood | meteo     | text                 |     0.081 |         0.04  |                2.02  | 135 |
| flood | meteo     | ref. EMS (suprafață) |     0.042 |         0.04  |                1.07  | 135 |
| flood | text      | ref. EMS (suprafață) |     0.046 |         0.042 |                1.078 | 135 |

## 5. Dezacordul text–imagine vs. incertitudinea textului

Scorul de incertitudine = markeri de aproximare și modalitate („about”, „estimated”, „may”, „expected”…) la 100 de cuvinte. p din 2000 de permutări.

| tip   | variabila                                  |   rho_cu_incertitudinea |   p_perm |   n |   texte_cu_marker_% |
|:------|:-------------------------------------------|------------------------:|---------:|----:|--------------------:|
| all   | |ŷ text − ŷ imagine|                       |                  -0.057 |    0.27  | 370 |              56.01  |
| all   | |eroare text|                              |                   0.006 |    0.903 | 370 |              56.01  |
| all   | |eroare imagine|                           |                   0.133 |    0.009 | 370 |              56.01  |
| fire  | |ŷ text − ŷ imagine|                       |                   0.026 |    0.71  | 230 |              49.362 |
| fire  | |eroare text|                              |                   0.023 |    0.714 | 230 |              49.362 |
| fire  | |eroare imagine|                           |                   0.123 |    0.062 | 230 |              49.362 |
| flood | |ŷ text − ŷ imagine|                       |                  -0.121 |    0.16  | 129 |              69.63  |
| flood | |eroare text|                              |                  -0.159 |    0.09  | 129 |              69.63  |
| flood | |eroare imagine|                           |                  -0.167 |    0.052 | 129 |              69.63  |
| fire  | |log ha text − log ha S2| (texte cu cifră) |                   0.05  |    0.794 |  33 |              93.939 |

