# Forma matricei X — U19

Rânduri (nucleu text + S1 + ERA5): **n = 391**

| Bloc | d | Note |
|---|---:|---|
| imagine | 14 | numeric, scalare în tabelul de trăsături |
| meteo | 4 | numeric, scalare în tabelul de trăsături |
| text (TF-IDF) | ~1303 | vocabular potrivit pe train, variază pe fold (mediana) |

**d total ≈ 1321** (numeric 18 + text ~1303). Separat: 7 variabile EMS de magnitudine (referință, nu intră în X).

## Fold-uri LOYO

Imputarea, scalarea și vocabularul TF-IDF sunt potrivite doar pe anii de train. `medie_abs_z_test` arată cât de departe e media trăsăturilor numerice pe anul de test de 0 (0 = fără deriva între ani).

|   an_test |   n_train |   n_test |   d_text_tfidf |   d_total |   medie_abs_z_test |
|----------:|----------:|---------:|---------------:|----------:|-------------------:|
|      2017 |       356 |       35 |           1348 |      1366 |           0.247377 |
|      2018 |       346 |       45 |           1310 |      1328 |           0.299102 |
|      2019 |       353 |       38 |           1303 |      1321 |           0.173653 |
|      2020 |       355 |       36 |           1318 |      1336 |           0.167605 |
|      2021 |       349 |       42 |           1271 |      1289 |           0.145822 |
|      2022 |       334 |       57 |           1219 |      1237 |           0.229813 |
|      2023 |       346 |       45 |           1284 |      1302 |           0.179655 |
|      2024 |       344 |       47 |           1300 |      1318 |           0.150842 |
|      2025 |       345 |       46 |           1318 |      1336 |           0.16827  |
