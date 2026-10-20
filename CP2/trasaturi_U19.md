# CP2 — Trăsături și reprezentare liniară · U19

Miloiu Cristi · U19 · Q19.2 — *Ce spune textul și nu spune satelitul: magnitudinea dezastrelor din rapoarte vs. Sentinel*

Toate cifrele de mai jos vin din pipeline (`pipeline/07_text.py`, `pipeline/08_explore.py`) și din rapoartele generate în `reports/`.
Rulare: `uv run python pipeline/07_text.py && uv run python pipeline/08_explore.py` (după pașii 01–06).

## 1. Cum am adus sursele pe aceeași unitate și același pas de timp

| | Alegere | Motiv |
|---|---|---|
| Unitatea | o activare Copernicus EMS (un rând = un eveniment, static) | întrebarea e despre evenimente, nu despre serii |
| Cheia spațială | uniunea AOI-urilor EMS ale activării (EPSG:3035 pentru arii) | geometria comună pentru S1, S2 și ERA5 |
| Grid | S1/S2 reduse pe AOI la ≥ 20 m (scara crește la AOI-uri mari, ~1e7 pixeli); ERA5-Land 0,1° cu buffer 5 km | AOI-uri mici sub o celulă ERA5 |
| Timp | t0 = eventTime (EMSR656+) sau data activării; pre = t0 − 60…−10 z, post = t0 + 2…+30 z, compozit median | ferestre decise la CP0 |
| Meteo | ploaie eveniment [t0 − L, t0 + 2], L = 5 / 14 zile; antecedentă [t0 − 60, t0 − 10]; Tmax, vânt maxim orar | activarea poate veni la câteva zile după eveniment |
| Data din text | extrasă pentru 295/411 activări; pe EMSR656+ coincide cu eventTime la 93% (±1 zi: 98%) | `reports/text_dates.md` |
| Întârziere activare − eveniment | mediana 1 zi, P90 7 zile; > 10 zile la 11 activări | fereastra pre se termină la t0 − 10 → nu cere reluarea GEE; cele 11 se marchează pentru sensibilitate |
| Goluri | S2 clar lipsește la 50% din nucleu → mediana din train + indicatorul `s2_ok`; câmpurile din text: lipsa mențiunii = 0 + `txt_has_number` | imputare doar cu statistici de train |

Nucleul: **n = 391** activări cu text + AOI + S1 + ERA5 (incendii 235, inundații 135, furtuni 21); pâlnia completă e în `reports/coverage.md`.

## 2. Matricea X (n × d, pe surse)

| Bloc | d | Trăsături | Sursă / set | Transformare |
|---|---:|---|---|---|
| imagine | 14 | ΔVV, ΔVH, fracția/aria cu ΔVV < −3 dB; dNDVI, dNBR, dMNDWI, fracția/aria arsă (dNBR < −0,27) și apă nouă (dMNDWI > 0,2); claritate S2 pre/post; `s2_ok` | `COPERNICUS/S1_GRD` (IW), `COPERNICUS/S2_SR_HARMONIZED` (mască SCL) | log1p pe arii, apoi z |
| meteo | 4 | ploaie eveniment, ploaie antecedentă, Tmax 2 m, vânt maxim 10 m | `ECMWF/ERA5_LAND/DAILY_AGGR`, `HOURLY` | log1p pe ploi, apoi z |
| text structurat | 5 | suprafață, persoane, decese menționate (maxim), indicator, lungime | textul EMS, regex (pasul 7) | log1p, apoi z |
| text — embeddings | 1.024 | BGE-M3 dense | `BAAI/bge-m3`, model fix | L2 (model) |
| text — alternativă | ~1.300 | TF-IDF 1–2-grame, min_df 3 | vocabular potrivit pe train | L2 pe rând |

**X = 391 × 1.047** (varianta BGE-M3) sau **391 × ~1.323** (varianta TF-IDF). Proveniența fiecărei trăsături (set, fereastră, rezoluție, unitate, transformare, % lipsă): `reports/feature_dictionary.csv`.

Referința de magnitudine **nu intră în X**: suprafața mapată EMS (370/391), populația EMS (107), EM-DAT decese / afectați / pagube (82 / 122 / 51), aria AOI.

**Scalarea face parte din X.** Imputarea, log1p, z-score și vocabularul TF-IDF se potrivesc doar pe anii de train. Blocurile se ponderează ca în MFA: fiecare bloc se împarte la √λ₁ (prima valoare proprie a blocului pe train). De ce:
- fără ponderare, PC1–PC3 vin 80% din imagine (14 coloane standardizate), iar textul TF-IDF are 0%;
- cu inerție totală 1 pe bloc, meteo (4 coloane corelate) făcea 73% din PC1.

## 3. Analiza univariată (`reports/univariate.csv`, `reports/figures/distributii.png`)

- Ariile (S1 inundat, S2 ars, suprafața EMS, AOI) au asimetrie 6–10 → după log1p 0,3–0,7. Ploile la fel (1,6–2,8 → −0,9…0).
- Fracțiile (s1_flooded, s2_newwater, s2_dmndwi) rămân asimetrice (2,6–3,6): sunt aproape zero la majoritatea evenimentelor. Se păstrează z, iar transformarea logit/rang e de testat la CP3.
- Textul dă rar cifre: suprafață la 57 din 235 de incendii și la 1 din 135 de inundații; persoane la 7%, decese la 2%. **Câmpurile structurate sunt 80% zero.**
- Tipul evenimentului separă clar meteo (Tmax și ploaia: incendii calde și uscate, inundații ude) → orice structură globală va fi întâi tipul.

## 4. Analiza bivariată (`reports/spearman.csv`, `reports/block_rv_*.csv`, `reports/magnitude_by_source.md`)

**Între surse (coeficient RV ajustat față de permutări):**

| | toate (391) | doar incendii (235) | doar inundații (135) |
|---|---:|---:|---:|
| imagine – meteo | 0,31 | 0,13 | 0,16 |
| imagine – BGE-M3 | 0,24 | 0,07 | 0,06 |
| meteo – BGE-M3 | 0,48 | 0,13 | 0,11 |
| TF-IDF – BGE-M3 | 0,54 | 0,56 | 0,61 |
| imagine – ref. EMS | 0,24 | 0,33 | 0,13 |
| BGE-M3 – ref. EMS | 0,15 | 0,04 | 0,02 |

→ Legătura text–imagine–meteo din setul complet e **în mare parte tipul evenimentului**. În interiorul unui tip, textul și imaginea sunt aproape independente (RV ≤ 0,07). Asta e exact spațiul în care Q19.2 poate găsi informație complementară.

**Magnitudinea după sursă (ρ Spearman cu suprafața mapată EMS):**

| Tip | Imagine | Text | Meteo |
|---|---|---|---|
| incendii | S2 suprafață arsă **+0,97** (n = 170) | suprafață menționată **+0,83** (n = 55) | ploaie −0,19 |
| inundații | S1 suprafață inundată **+0,62** (n = 129) | – (1 text cu cifră) | ploaie +0,16 |

Cu EM-DAT „afectați”, toate corelațiile sunt slabe (|ρ| ≤ 0,36, n = 13–63): EM-DAT e la nivel de țară și măsoară altceva (oameni, nu hectare).

**Perechi de reținut (Spearman, între blocuri):**
- ariile din imagine cresc cu aria AOI (ρ 0,74–0,86): o parte din „magnitudinea din imagine” e mărimea AOI-ului desenat de EMS → la CP3 testez și fracțiile (independente de AOI);
- claritatea S2 scade cu ploaia (ρ −0,53…−0,69) → golurile S2 nu sunt aleatoare (MNAR): S2 lipsește tocmai la evenimentele ude.

## 5. Primul PCA pe protocolul LOYO (`reports/pca_loyo.md`, `reports/figures/`)

**Protocol.** Split leave-one-year-out (9 fold-uri, 2017–2025). Pe fiecare fold: scalarea, ponderarea și PCA se potrivesc pe anii de train și se aplică pe anul de test. Metricile se calculează pe test și se raportează ca medie ± abatere standard pe fold-uri. PCA e determinist, deci nu are nevoie de seed-uri.

**k.** Analiza paralelă Horn pe train (20 de permutări, P95) dă k = 9–13 pe fold-uri, deci **k = 11** (mediana). Scree-ul nu are un cot clar: PC1 6,8%, PC2 3,7%, PC3 3,3%; la k = 11 cumulatul e 29% (TF-IDF) / 52% (BGE-M3).

**Metrici pe anul de test (k = 11):**

| Reprezentare | var. reconstruită (global) | trustworthiness@10 (local) | kNN tip (downstream) | ρ suprafață EMS (downstream, Ridge) |
|---|---|---|---|---|
| PCA · toate sursele (BGE-M3) | 0,52 ± 0,04 | 0,97 ± 0,01 | 0,92 ± 0,05 | 0,61 ± 0,14 |
| PCA · toate sursele (TF-IDF) | 0,24 ± 0,03 | 0,94 ± 0,01 | 0,90 ± 0,05 | 0,58 ± 0,15 |
| X complet, fără PCA (BGE-M3) | – | – | 0,94 ± 0,05 | 0,71 ± 0,09 |
| PCA · doar imagine | 0,98 ± 0,01 | 1,00 ± 0,00 | 0,78 ± 0,06 | **0,71 ± 0,12** |
| PCA · doar meteo (k = 3) | 0,92 ± 0,03 | 0,97 ± 0,01 | 0,87 ± 0,06 | 0,12 ± 0,20 |
| PCA · doar text (BGE-M3) | 0,44 ± 0,07 | 0,95 ± 0,02 | **0,93 ± 0,06** | 0,17 ± 0,16 |
| PCA · doar text (TF-IDF) | 0,17 ± 0,04 | 0,88 ± 0,03 | 0,91 ± 0,07 | 0,03 ± 0,16 |

**Baleiaj pe k** (toate sursele, BGE-M3, `reports/figures/k_sweep.png`): ρ cu suprafața EMS e 0,12 la k = 2, 0,48 la k = 5, 0,61 la k = 11, 0,66 la k = 20 și 0,70 la k = 50. Primele axe descriu tipul și vremea; magnitudinea stă pe componente mici.

**Loadings** (`reports/figures/loadings.png`, `reports/pca_loadings.csv`; fit pe toate cele 391, doar pentru interpretare):
- PC1 (6,6%) = axa vremii și a tipului: ploaie +, Tmax −, S2 clar −;
- PC2 = „textul dă cifre” (txt_has_number, suprafață, persoane, decese);
- PC3, PC5, PC6 = imagine (dNBR/dNDVI; arii S1/S2; ΔVV/ΔVH);
- PC4, PC10, PC11 = vocabularul textului.

Sursele ocupă axe diferite, ceea ce confirmă RV-ul mic din interiorul unui tip. Stabilitatea pe fold-uri (|cos| cu axele de pe tot setul, BGE-M3) e 0,85–1,00, cu excepția PC2/PC3 (~0,70), care își schimbă ordinea între fold-uri.

**Ce înseamnă pentru Q19.2.** Textul știe *ce* s-a întâmplat (tipul: 0,93 doar din text), imaginea știe *cât de mare* (ρ 0,71 doar din imagine, față de 0,17 din text). PCA pe toate sursele (k = 11) păstrează vecinii (0,97), dar pierde o parte din semnalul de magnitudine (0,61 față de 0,71 în X complet).

**Limitare (circularitate).** Suprafața mapată de EMS e delimitată tot pe imagini satelitare, iar ariile din imagine cresc cu aria AOI. Avantajul imaginii e deci parțial construit. Se verifică la CP3 cu suprafața menționată în text (ρ 0,83 la incendii), cu EM-DAT și cu fracții independente de AOI.

## 6. Alternativa declarată (CP3)

- **CCA text ↔ imagine** (direcțiile comune ale celor două modalități). Este comparația obligatorie U19: text vs. imagine vs. ambele.
- **UMAP** pe același X și același protocol, pe 5 seed-uri.
- Ambele se raportează față de baseline-ul PCA (k = 11), pe aceleași fold-uri LOYO și cu aceleași patru metrici. Comparațiile pregătite: TF-IDF vs. BGE-M3 (deja în tabel) și S1 vs. S2 (pe subsetul cu S2 clar, n = 196).
