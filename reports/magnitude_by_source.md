# Magnitudinea după sursă — U19

Generat de `pipeline/08_explore.py`. ρ Spearman între indicatorul de magnitudine al fiecărei surse și referință, pe tip de eveniment (perechi complete; pentru `txt_*` doar textele care dau cifra). n între paranteze; ρ cu n < 10 nu se afișează.

| tip   | sursa   | indicator           | ems_mapped_area_ha   | emdat_affected   |
|:------|:--------|:--------------------|:---------------------|:-----------------|
| fire  | imagine | s1_flooded_km2      | +0.63 (230)          | -0.14 (63)       |
| fire  | imagine | s2_burnt_km2        | +0.97 (170)          | -0.20 (56)       |
| fire  | imagine | s2_newwater_km2     | +0.59 (170)          | -0.12 (56)       |
| fire  | text    | txt_area_ha         | +0.83 (55)           | -0.04 (13)       |
| fire  | text    | txt_people          | +0.21 (19)           | – (5)            |
| fire  | meteo   | era5_tp_event_mm    | -0.19 (230)          | -0.06 (63)       |
| fire  | meteo   | era5_wind10m_max_ms | -0.08 (230)          | -0.00 (63)       |
| fire  | ref.    | aoi_area_km2        | +0.76 (230)          | -0.21 (63)       |
| flood | imagine | s1_flooded_km2      | +0.62 (129)          | +0.23 (56)       |
| flood | imagine | s2_burnt_km2        | +0.54 (21)           | -0.36 (13)       |
| flood | imagine | s2_newwater_km2     | +0.68 (21)           | -0.09 (13)       |
| flood | text    | txt_area_ha         | – (1)                | – (0)            |
| flood | text    | txt_people          | – (6)                | – (5)            |
| flood | meteo   | era5_tp_event_mm    | +0.16 (129)          | +0.13 (56)       |
| flood | meteo   | era5_wind10m_max_ms | +0.30 (129)          | -0.24 (56)       |
| flood | ref.    | aoi_area_km2        | +0.70 (129)          | +0.26 (56)       |
