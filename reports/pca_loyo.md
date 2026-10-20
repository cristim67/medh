# PCA pe protocolul LOYO — U19 (CP2)

Generat de `pipeline/08_explore.py`.

## Matricea X

n = 391 activări; blocuri: imagine 14, meteo 4, text structurat 5, TF-IDF ~1300 (mediana pe fold-uri) sau BGE-M3 1024.
d total ≈ 1323 (varianta TF-IDF) / 1047 (varianta BGE-M3).

## Alegerea lui k

Analiza paralelă Horn pe train (X complet, TF-IDF): k pe fold-uri = [10, 13, 10, 11, 9, 9, 11, 11, 11] → **k = 11** (mediana). Varianța cumulată la k pe train: 29.1% (PC1 6.8%, PC2 3.7%, PC3 3.3%).

Fără ponderarea pe bloc, PC1–PC3 explică 45.0% și se compun din: imagine 80%, meteo 14%, text_struct 5%, text_emb 0% (media loadings² pe PC1–3) → axele ar măsura blocul cu varianța cea mai mare, nu structura.

## Contribuția surselor la fiecare PC (fit pe tot setul, doar pentru interpretare)

|      |   imagine |   meteo |   text_struct |   text_emb |
|:-----|----------:|--------:|--------------:|-----------:|
| PC1  |      0.32 |    0.37 |          0.07 |       0.24 |
| PC2  |      0.14 |    0.02 |          0.45 |       0.39 |
| PC3  |      0.67 |    0.03 |          0.15 |       0.14 |
| PC4  |      0.15 |    0.01 |          0.15 |       0.69 |
| PC5  |      0.71 |    0.01 |          0.1  |       0.18 |
| PC6  |      0.74 |    0.05 |          0.03 |       0.18 |
| PC7  |      0.13 |    0.04 |          0.5  |       0.33 |
| PC8  |      0.14 |    0.37 |          0.21 |       0.28 |
| PC9  |      0.19 |    0.16 |          0.41 |       0.23 |
| PC10 |      0.29 |    0.05 |          0.03 |       0.64 |
| PC11 |      0.21 |    0.11 |          0    |       0.67 |

Stabilitatea axelor (varianta BGE-M3, |cos| între PC_j pe train și PC_j pe tot setul, media pe fold-uri): PC1 1.00, PC2 0.71, PC3 0.70, PC4 0.98, PC5 0.95, PC6 0.93, PC7 0.86, PC8 0.85, PC9 0.93, PC10 0.92, PC11 0.91

## Metrici pe anul de test (media ± abaterea standard pe 9 fold-uri)

| reprezentare                                 | var_reconstruita   | trustworthiness   | knn_tip_acc   | rho_suprafata_ems   |
|:---------------------------------------------|:-------------------|:------------------|:--------------|:--------------------|
| PCA k=11 · toate sursele (TF-IDF)            | 0.24 ± 0.03        | 0.94 ± 0.01       | 0.90 ± 0.05   | 0.58 ± 0.15         |
| X complet, fără PCA · toate sursele (TF-IDF) | 1.00 ± 0.00        | 1.00 ± 0.00       | 0.93 ± 0.05   | 0.63 ± 0.13         |
| PCA k=11 · toate sursele (BGE-M3)            | 0.52 ± 0.04        | 0.97 ± 0.01       | 0.92 ± 0.05   | 0.61 ± 0.14         |
| X complet, fără PCA · toate sursele (BGE-M3) | 1.00 ± 0.00        | 1.00 ± 0.00       | 0.94 ± 0.05   | 0.71 ± 0.09         |
| PCA k=11 · doar imagine                      | 0.98 ± 0.01        | 1.00 ± 0.00       | 0.78 ± 0.06   | 0.71 ± 0.12         |
| PCA k=3 · doar meteo                         | 0.92 ± 0.03        | 0.97 ± 0.01       | 0.87 ± 0.06   | 0.12 ± 0.20         |
| PCA k=11 · doar text (TF-IDF)                | 0.17 ± 0.04        | 0.88 ± 0.03       | 0.91 ± 0.07   | 0.03 ± 0.16         |
| PCA k=11 · doar text (BGE-M3)                | 0.44 ± 0.07        | 0.95 ± 0.02       | 0.93 ± 0.06   | 0.17 ± 0.16         |

## Baleiaj pe k (toate sursele, BGE-M3; media ± abaterea standard pe 9 fold-uri)

|   k | var_reconstruita   | trustworthiness   | knn_tip_acc   | rho_suprafata_ems   |
|----:|:-------------------|:------------------|:--------------|:--------------------|
|   1 | 0.15 ± 0.04        | 0.79 ± 0.05       | 0.88 ± 0.07   | 0.13 ± 0.25         |
|   2 | 0.19 ± 0.03        | 0.85 ± 0.04       | 0.90 ± 0.07   | 0.12 ± 0.24         |
|   3 | 0.27 ± 0.03        | 0.90 ± 0.03       | 0.92 ± 0.06   | 0.18 ± 0.24         |
|   5 | 0.35 ± 0.03        | 0.94 ± 0.01       | 0.93 ± 0.05   | 0.48 ± 0.11         |
|  10 | 0.49 ± 0.03        | 0.97 ± 0.01       | 0.92 ± 0.05   | 0.58 ± 0.16         |
|  20 | 0.61 ± 0.04        | 0.98 ± 0.01       | 0.93 ± 0.05   | 0.66 ± 0.13         |
|  50 | 0.71 ± 0.03        | 0.99 ± 0.00       | 0.93 ± 0.06   | 0.70 ± 0.11         |

## Corelația între surse (coeficient RV)

RV brut, tot setul:

|              |   imagine |   meteo |   text struct. |   TF-IDF |   BGE-M3 |   ref. EMS |
|:-------------|----------:|--------:|---------------:|---------:|---------:|-----------:|
| imagine      |      1    |    0.31 |           0.04 |     0.18 |     0.27 |       0.25 |
| meteo        |      0.31 |    1    |           0.06 |     0.23 |     0.49 |       0.12 |
| text struct. |      0.04 |    0.06 |           1    |     0.15 |     0.18 |       0.02 |
| TF-IDF       |      0.18 |    0.23 |           0.15 |     1    |     0.63 |       0.1  |
| BGE-M3       |      0.27 |    0.49 |           0.18 |     0.63 |     1    |       0.16 |
| ref. EMS     |      0.25 |    0.12 |           0.02 |     0.1  |     0.16 |       1    |

Referința la zero (media RV pe 20 de permutări ale rândurilor; RV crește cu dimensiunea blocului):

|              |   imagine |   meteo |   text struct. |   TF-IDF |   BGE-M3 |   ref. EMS |
|:-------------|----------:|--------:|---------------:|---------:|---------:|-----------:|
| imagine      |      0.02 |    0.01 |           0.01 |     0.08 |     0.04 |       0.01 |
| meteo        |      0.01 |    0.01 |           0.01 |     0.05 |     0.03 |       0.01 |
| text struct. |      0.01 |    0.01 |           0.01 |     0.06 |     0.03 |       0.01 |
| TF-IDF       |      0.08 |    0.05 |           0.06 |     0.4  |     0.2  |       0.04 |
| BGE-M3       |      0.04 |    0.03 |           0.03 |     0.2  |     0.1  |       0.02 |
| ref. EMS     |      0.01 |    0.01 |           0.01 |     0.04 |     0.02 |       0    |

RV ajustat = (RV − RV0) / (1 − RV0), tot setul:

|              |   imagine |   meteo |   text struct. |   TF-IDF |   BGE-M3 |   ref. EMS |
|:-------------|----------:|--------:|---------------:|---------:|---------:|-----------:|
| imagine      |      1    |    0.31 |           0.03 |     0.11 |     0.24 |       0.24 |
| meteo        |      0.31 |    1    |           0.05 |     0.19 |     0.48 |       0.11 |
| text struct. |      0.03 |    0.05 |           1    |     0.1  |     0.16 |       0.01 |
| TF-IDF       |      0.11 |    0.19 |           0.1  |     1    |     0.54 |       0.06 |
| BGE-M3       |      0.24 |    0.48 |           0.16 |     0.54 |     1    |       0.15 |
| ref. EMS     |      0.24 |    0.11 |           0.01 |     0.06 |     0.15 |       1    |

RV ajustat, doar incendii (n = 235):

|              |   imagine |   meteo |   text struct. |   TF-IDF |   BGE-M3 |   ref. EMS |
|:-------------|----------:|--------:|---------------:|---------:|---------:|-----------:|
| imagine      |      1    |    0.13 |           0.02 |     0.04 |     0.07 |       0.33 |
| meteo        |      0.13 |    1    |           0.02 |     0.07 |     0.13 |       0.06 |
| text struct. |      0.02 |    0.01 |           1    |     0.13 |     0.23 |       0.01 |
| TF-IDF       |      0.04 |    0.07 |           0.13 |     1    |     0.56 |       0.02 |
| BGE-M3       |      0.07 |    0.13 |           0.23 |     0.56 |     1    |       0.04 |
| ref. EMS     |      0.33 |    0.06 |           0.01 |     0.02 |     0.04 |       1    |

RV ajustat, doar inundații (n = 135):

|              |   imagine |   meteo |   text struct. |   TF-IDF |   BGE-M3 |   ref. EMS |
|:-------------|----------:|--------:|---------------:|---------:|---------:|-----------:|
| imagine      |      1    |    0.16 |           0.04 |     0.04 |     0.06 |       0.13 |
| meteo        |      0.16 |    1    |           0.02 |     0.07 |     0.11 |       0.02 |
| text struct. |      0.04 |    0.02 |           1    |     0.03 |     0.08 |       0.01 |
| TF-IDF       |      0.03 |    0.07 |           0.03 |     1    |     0.61 |       0.01 |
| BGE-M3       |      0.06 |    0.11 |           0.08 |     0.61 |     1    |       0.02 |
| ref. EMS     |      0.13 |    0.03 |           0    |     0.01 |     0.02 |       1    |

