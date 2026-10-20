# Data card — U19 · Evenimentul documentat text + imagine

**Proiect:** Ce spune textul și nu spune satelitul: magnitudinea dezastrelor din rapoarte vs. Sentinel
**Autor:** Miloiu Cristi · MEDE, IISC anul II · CP1 (aliniere), octombrie 2026
**Cod:** `pipeline/01_ems.py` … `pipeline/06_matrix.py` (README în rădăcina repo-ului). Datele nu sunt versionate; se regenerează din scripturi.

## 1. Unitatea și setul aliniat

| | |
|---|---|
| Unitatea de analiză | o activare Copernicus EMS Rapid Mapping (cod EMSRxxx); static, un rând = un eveniment |
| Selecția | EMSR, Europa geografică (fără Türkiye, Georgia, teritorii de peste mări), activare în 2017–2025, tip flood / fire / storm |
| n selectat | **411** activări (fire 246, flood 140, storm 25), 35 de țări (Grecia 90, Spania 79, Italia 47, Germania 36, Franța 30) |
| n în nucleu | **391** activări au text + AOI + Sentinel-1 pre/post + ERA5-Land (fire 235, flood 135, storm 21); 196 au și Sentinel-2 clar |
| Cheia de aliniere | `emsr_id`; geometria comună = uniunea AOI-urilor EMS; data de referință t0 = eventTime EMS (135 de activări) sau activationTime (256) |
| Matricea X | n = 391 × d ≈ 1.321: 14 imagine + 4 meteo + ~1.303 TF-IDF (vocabularul depinde de fold) |
| Referința de magnitudine | 7 variabile separate de X: suprafața mapată EMS, populația afectată EMS, aria și numărul de AOI, decese / afectați / pagube EM-DAT |
| Split | LOYO (leave-one-year-out) pe anul activării: 9 fold-uri, 2017–2025 |
| Scalare | log1p + z-score pe arii și precipitații; z-score pe restul; L2 pe TF-IDF; imputarea (mediana), scalarea și vocabularul se potrivesc doar pe anii de train |

## 2. Fișe pe sursă

### 2.1 Copernicus EMS Rapid Mapping (text + magnitudine mapată)

| Câmp | Valoare |
|---|---|
| Proveniență | Comisia Europeană (JRC), Copernicus Emergency Management Service. Lista: `mapping.emergency.copernicus.eu/activations/api/activations/`; detaliu EMSR656+: `rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/`; pentru < EMSR656 pagina HTML a activării + pachetele vectoriale de pe S3 (citite parțial, prin HTTP range) |
| Perioadă | activări 18.06.2017 – 19.11.2025 |
| Rezoluție spațială | poligoane AOI (1.320 de AOI-uri; aria pe eveniment: mediana 324 km², max ~59.000 km²); poligoane de eveniment observat la scara hărților EMS (1:25.000 – 1:50.000), din imagini VHR (SPOT, Pléiades, WorldView) și Sentinel |
| Rezoluție temporală | un raport pe activare; data activării; eventTime doar pentru EMSR656+ (din martie 2023) |
| Variabile și unități | `reason` (text în engleză, mediana 381 de caractere); categorie, țări, centroid; `ems_mapped_area_ha` (ha, uniunea poligoanelor de eveniment de tipul activării sau statisticile „Flooded / Burnt area”); `ems_pop_affected` (persoane) |
| n | 411 activări; 411 cu text, 402 cu AOI, 382 cu suprafață mapată, 92 cu populație afectată |
| Goluri | 9 activări fără produse publicate (fără AOI); populația afectată doar pentru API-ul nou; lista publică are 873 de coduri EMSR, deși numerotarea ajunge la EMSR935 (coduri anulate sau nepublice) |
| Licență | date Copernicus: acces liber și gratuit, cu atribuire „© European Union, Copernicus Emergency Management Service” |
| Limitări | trei scheme de denumire a straturilor vectoriale (2017, 2018–2019, 2020+); la < EMSR656 data activării poate întârzia față de eveniment (ex. Vaia: eveniment 29.10.2018, activare 07.11.2018); unele texte au UTF-8 stricat la sursă (reparat automat); suprafața mapată acoperă doar AOI-urile cerute, nu tot evenimentul |

### 2.2 GDACS (alertă, severitate, impact)

| Câmp | Valoare |
|---|---|
| Proveniență | GDACS (ONU + Comisia Europeană), `gdacs.org/gdacsapi/api/events/` (`geteventlist`, `geteventdata`, `polygons/getgeometry`) |
| Perioadă | alerte 12.2016 – 01.2026; inundații (FL) din 2017, incendii (WF) **abia din 2021** |
| Rezoluție spațială | punct (centroid) + poligon „zonă afectată” (la inundații, la scară de bazin GloFAS) |
| Rezoluție temporală | episoade, cu dată de început și de sfârșit |
| Variabile și unități | nivel de alertă (Green / Orange / Red), scor de alertă, severitate (unitate dependentă de tip), impact „sendai” (decese, evacuați, afectați, persoane), cod GLIDE |
| n | 951 de alerte în Europa (FL 763, WF 188); ~136 de activări EMS asociate |
| Goluri | 67% dintre activări fără GDACS; storm practic absent (GDACS nu acoperă furtunile extratropicale); 2017–2020 slab acoperit |
| Potrivire | `gdacsId` declarat de EMS (doar EMSR656+) sau tip + ±7 zile + distanța AOI ↔ zona afectată ≤ 25 km; validare pe 26 de cazuri declarate: 14 corecte, 4 greșite, 8 fără pereche |
| Licență | termenii gdacs.org (date „as is”, cu citarea sursei); fluxul GDACS e publicat pe HDX sub CC BY |
| Limitări | aproape toate alertele sunt Green (varianță mică); zona afectată GDACS e adesea la 130–600 km de AOI-urile EMS → GDACS descrie episodul meteo, nu zona mapată |

### 2.3 Sentinel-1 GRD (imagine radar, principală)

| Câmp | Valoare |
|---|---|
| Proveniență | ESA / Copernicus, Google Earth Engine `COPERNICUS/S1_GRD` (IW, VV + VH, σ⁰ în dB, corectat de GEE: calibrare, terrain correction) |
| Perioadă | ferestre pe eveniment: pre = t0 − 60 … t0 − 10 zile, post = t0 + 2 … t0 + 30 zile |
| Rezoluție spațială | 10 m nativ; agregare pe AOI la scara de 20 m (până la 77 m pentru AOI-urile foarte mari: ≤ 10⁷ pixeli, iar la depășirea memoriei GEE se reîncearcă la pixel dublu) |
| Rezoluție temporală | revizitare 6 zile cu S1A + S1B (până în 12.2021), ~12 zile cu S1A singur (2022–2024); mediana scenelor pe eveniment: 24–40 pre, 14–23 post |
| Variabile și unități | VV, VH pre/post (dB), ΔVV, ΔVH (dB), fracția și aria (km²) cu ΔVV < −3 dB (proxy pentru inundat) |
| n | 394 de activări din nucleu cu scene în ambele ferestre |
| Goluri | 3 activări fără nicio scenă S1 într-una din ferestre (EMSR218, EMSR378, EMSR785); 1 activare cu extracția GEE expirată (EMSR698); lipsa S1B în 2022–2024 |
| Licență | Copernicus Sentinel data: acces liber, complet și gratuit; atribuire „Contains modified Copernicus Sentinel data [an]” |
| Limitări | compozitul median pe 4–7 săptămâni diluează semnalul inundațiilor scurte; pragul −3 dB e generic, nu calibrat pe tip de eveniment |

### 2.4 Sentinel-2 L2A (imagine optică, secundară)

| Câmp | Valoare |
|---|---|
| Proveniență | ESA / Copernicus, GEE `COPERNICUS/S2_SR_HARMONIZED`, mască de nori din SCL (păstrate clasele 4, 5, 6, 7, 11) |
| Perioadă | aceleași ferestre pre/post ca la S1 |
| Rezoluție spațială | 10–20 m; agregare pe AOI la 20 m |
| Rezoluție temporală | revizitare ~5 zile; mediana scenelor pe eveniment: 23–60 pre, 12–33 post |
| Variabile și unități | NDVI, NBR, MNDWI pre/post (adimensional), dNDVI, dNBR, dMNDWI (post − pre), fracția de pixeli clari pre/post, fracția și aria (km²) cu dNBR < −0,27 (ars) și dMNDWI > 0,2 (apă nouă) |
| n | 200 de activări cu ≥ 50% pixeli clari în ambele compozite (196 în nucleu) |
| Goluri | **51% fără S2 clar**; la inundații doar 22 din 140 (mediana pixelilor clari post: 20%), la furtuni 2 din 25, la incendii 176 din 246 |
| Licență | ca la Sentinel-1 |
| Limitări | norii coincid cu inundațiile și furtunile (lipsă neîntâmplătoare); schimbări sezoniere între ferestre (ex. EMSR334, furtuna Vaia: 30% „ars” fals); în X, S2 cu nori se tratează ca lipsă + indicatorul `s2_ok` |

### 2.5 ERA5-Land (meteo)

| Câmp | Valoare |
|---|---|
| Proveniență | ECMWF / Copernicus Climate Change Service, GEE `ECMWF/ERA5_LAND/DAILY_AGGR` (precipitații, temperatură) și `ECMWF/ERA5_LAND/HOURLY` (vânt) |
| Perioadă | eveniment: t0 − L … t0 + 2 zile (L = 5 zile dacă t0 = eventTime, 14 zile dacă t0 = activationTime); antecedent: t0 − 60 … t0 − 10 zile |
| Rezoluție spațială | 0,1° (~9 km); media pe AOI + buffer 5 km |
| Rezoluție temporală | orară / zilnică |
| Variabile și unități | precipitații eveniment și antecedente (mm), temperatura maximă la 2 m (°C), viteza maximă orară a vântului la 10 m (m/s) |
| n | 391 de activări din nucleu |
| Goluri | 3 activări fără valoare ERA5-Land validă pe AOI (EMSR234, EMSR236, EMSR626) |
| Licență | CC BY 4.0 (din 02.07.2025 pentru produsele Copernicus); citare Muñoz-Sabater et al., ESSD 2021 |
| Limitări | 9 km e grosier pentru AOI-urile mici; reanaliza subestimează rafalele (vântul e media orară la 10 m, nu rafala) |

### 2.6 EM-DAT (pagube, victime — referință)

| Câmp | Valoare |
|---|---|
| Proveniență | CRED / UCLouvain, public.emdat.be, export Excel din 19.10.2026 (cont academic) |
| Perioadă | 2000–2026 descărcat; folosit 2017–2025, Europa |
| Rezoluție spațială | țară + unități administrative (GAUL, ~59% dintre evenimente); coordonate la ~6% |
| Rezoluție temporală | dată de început / sfârșit (zi, lună, an) |
| Variabile și unități | decese, persoane afectate (persoane), pagube totale ajustate (mii USD), subtip, coduri externe (GLIDE, HANZE) |
| n | 328 de evenimente (Storm 170, Flood 119, Wildfire 39); 137 de activări EMS asociate (8 prin GLIDE, 19 prin unitate administrativă, 110 prin țară + perioadă) |
| Goluri | pagube raportate la 26% dintre evenimente; incendiile mici lipsesc (doar ~74 din 246 de activări de incendiu au pereche); 76 de activări EMS împart același eveniment EM-DAT |
| Licență | utilizare academică necomercială; **datele brute nu se redistribuie** (nu sunt în repo) |
| Limitări | potrivirea e la nivel de țară, deci ambiguă când în aceeași perioadă există mai multe evenimente de același tip; EM-DAT agregă la nivel de eveniment național, EMS la nivel de AOI |

## 3. Raportul de acoperire

| Pas | Activări | % |
|---|---:|---:|
| selectate | 411 | 100% |
| + text ≥ 100 caractere | 406 | 99% |
| + AOI | 397 | 97% |
| + extracție GEE | 396 | 96% |
| + S1 pre și post | 394 | 96% |
| + ERA5-Land | 391 | 95% |
| **= nucleu text + S1 + ERA5** | **391** | **95%** |
| + S2 clar pre și post | 196 | 48% |

Tabelul complet (pe tip și an) e în `reports/coverage.md`.

## 4. Riscuri pentru analiză (de urmărit la CP2–CP3)

- **Lipsa S2 nu e întâmplătoare:** norii însoțesc inundațiile și furtunile. Comparația S1 vs. S2 se face doar pe subsetul cu ambele.
- **t0 incert pentru < EMSR656:** data evenimentului se poate extrage din text („On 29 October…”) pentru o aliniere temporală mai bună.
- **Leakage text ↔ imagine:** raportul EMS e scris uneori după ce s-au văzut imaginile post-eveniment; suprafața mapată de EMS e derivată din imagini, deci e referință, nu intrare.
- **Furtunile** (n = 25, 21 în nucleu) sunt prea puține pentru o analiză separată; rămân în set, dar se raportează separat.
