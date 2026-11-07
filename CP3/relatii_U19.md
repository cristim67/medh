# CP3 — Relații între surse și reprezentări neliniare · U19

Miloiu Cristi · U19 · Q19.2 — *Ce spune textul și nu spune satelitul: magnitudinea dezastrelor din rapoarte vs. Sentinel*

Toate cifrele vin din `pipeline/09_relations.py` și `pipeline/10_representations.py`. Rapoartele complete sunt în `reports/relations.md` și `reports/representations.md`.
Rulare (după pașii 01–08): `uv run python pipeline/09_relations.py && uv run python pipeline/10_representations.py`.

## 1. Protocolul

| | Alegere |
|---|---|
| Date | nucleul n = 391 activări EMS (incendii 235, inundații 135, furtuni 21), X din CP2: imagine 14 · meteo 4 · text 5 + BGE-M3 1.024 |
| Split | LOYO, 9 fold-uri (2017–2025); scalarea, ponderarea MFA și reprezentarea se potrivesc doar pe anii de train |
| Ce compar | **PCA (baseline)** vs. **autoencoder (metoda aleasă)**; în plus kernel PCA (RBF), UMAP și CCA imagine ↔ text (multi-view) |
| Pe ce | fiecare sursă separat (imagine, meteo, text) și toate sursele combinate |
| k | analiza paralelă Horn pe train, pe sursă: imagine 5, meteo 2, text 45, toate 11 (CP2) |
| Metrici (pe anul de test) | varianță explicată = 1 − eroarea de reconstrucție relativă; trustworthiness@10 (locală); ρ Spearman între distanțele din X și din Z (globală); downstream: kNN pentru tip și ρ Ridge(Z) → log(suprafața EMS) |
| Intervale | IC 95% t pe cele 9 fold-uri; Δ față de PCA pe aceeași sursă, pereche pe fold-uri |
| Stabilitate | 5 seed-uri (0–4) pentru autoencoder și UMAP: abaterea între seed-uri + Jaccard al vecinilor din Z_test; 2 subperioade: train 2017–2021 → test 2022–2025 și invers |

De ce autoencoderul e metoda aleasă: e singura alternativă care are eroare de reconstrucție comparabilă direct cu PCA și care se aplică la fel pe fiecare sursă. UMAP nu are reconstrucție. CCA cere două vederi, deci rulează doar pe „toate”. t-SNE nu proiectează puncte noi (anul de test), deci e exclus din protocol.

Arhitectura autoencoderului: MLP d → 64 → k → 64 → d, tanh, L2 1e-3, Adam, 500 de epoci.

## 2. Tabelul principal (`reports/figures/representations.png`)

Media [IC 95%] pe anul de test; ± = abaterea între seed-uri.

| Sursă | Reprezentare | Var. explicată | Trustworthiness | ρ distanțe | kNN tip | ρ suprafață EMS |
|---|---|---|---|---|---|---|
| imagine (k = 5) | PCA | **0,82** [0,78; 0,86] | **0,98** | **0,96** | 0,79 | **0,64** [0,55; 0,74] |
| | Autoencoder | 0,77 ±0,00 | 0,96 | 0,85 | 0,79 | 0,60 ±0,03 |
| | Kernel PCA | 0,74 | 0,90 | 0,68 | 0,77 | 0,64 |
| | UMAP | – | 0,89 | 0,44 | 0,78 | 0,49 |
| meteo (k = 2) | PCA | **0,80** | 0,94 | **0,94** | 0,86 | 0,13 |
| | Autoencoder | 0,78 | 0,94 | 0,92 | 0,85 | 0,13 |
| | Kernel PCA | 0,73 | 0,91 | 0,79 | 0,85 | 0,16 |
| | UMAP | – | 0,92 | 0,80 | 0,88 | 0,13 |
| text (k = 45) | PCA | **0,59** | **0,97** | **0,95** | 0,94 | **0,29** [0,17; 0,42] |
| | Autoencoder | 0,18 ±0,02 | 0,91 | 0,80 | 0,94 | 0,15 ±0,05 |
| | Kernel PCA | 0,57 | 0,94 | 0,87 | 0,94 | 0,29 |
| | UMAP | – | 0,87 | 0,73 | 0,93 | 0,13 |
| toate (k = 11) | PCA | **0,52** [0,49; 0,55] | **0,97** | **0,96** | 0,92 | **0,61** [0,51; 0,72] |
| | Autoencoder | 0,22 ±0,03 | 0,90 | 0,73 | 0,94 | 0,42 ±0,06 |
| | Kernel PCA | 0,49 | 0,93 | 0,80 | 0,92 | 0,49 |
| | UMAP | – | 0,87 | 0,56 | 0,93 | 0,35 |
| | CCA imagine ↔ text | – | 0,86 | 0,67 | 0,91 | 0,55 |

**Rezultatul.** Metoda aleasă nu bate PCA pe nicio sursă:
- la varianță, vecini, distanțe și magnitudine, IC-ul diferenței e sub 0 sau conține 0;
- singura excepție e acuratețea kNN pe tip pe „toate” (+0,01 [+0,00; +0,03]), un câștig neglijabil.

Cea mai mare pierdere a autoencoderului e pe text și pe toate sursele, unde d ≈ 1.000 și n_train ≈ 350: rețeaua supra-învață, iar varianța explicată pe test scade la 0,18–0,22. Kernel PCA e aproape egal cu PCA (structura e aproape liniară). UMAP păstrează tipul, dar pierde structura globală (ρ distanțe 0,44–0,80) și magnitudinea.

**Baseline „o singură sursă” vs. toate:**
- imaginea singură dă ρ cu suprafața EMS 0,64 la k = 5 (0,71 la k = 11, CP2), mai mult decât toate sursele (0,61);
- textul singur dă cel mai bun kNN pe tip (0,94), dar o magnitudine slabă (0,29);
- combinarea surselor nu adaugă magnitudine peste imagine. O adaugă doar peste fracțiile din imagine (secțiunea 3).

**Stabilitate:**
- Seed-uri: abaterea între seed-uri e ≤ 0,06 la toate metricile. Vecinii din Z_test se păstrează mai bine la UMAP (Jaccard 0,85) decât la autoencoder (0,58 pe toate, 0,60 pe text), deci autoencoderul pe date de dimensiune mare e instabil.
- Subperioade (PCA, toate): trustworthiness 0,97 în ambele direcții. ρ cu suprafața EMS e însă 0,69 la train 2017–2021 → test 2022–2025 și doar 0,35 invers. Explicația probabilă: suprafața de referință are altă proveniență înainte de 2023 (pachete vectoriale) și după (statistici API). Relația cu magnitudinea nu e stabilă în timp, chiar dacă reprezentarea este.

## 3. Harta relațiilor între surse (`reports/figures/relations_map.png`, `reports/relations.md`)

Toate analizele se fac **pe tip de eveniment**: la CP2, pe tot setul, tipul domina orice corelație.

**Ce aduce fiecare sursă la magnitudine** (Ridge LOYO, ρ pe test, IC 95% bootstrap):

| Model | Incendii (n = 230) | Inundații (n = 129) |
|---|---|---|
| doar aria AOI (control) | 0,76 [0,67; 0,82] | 0,69 [0,57; 0,77] |
| doar meteo | 0,34 | 0,12 |
| doar text | 0,42 [0,31; 0,54] | 0,10 [−0,07; 0,26] |
| doar imagine (fracții, fără arii) | 0,35 | −0,29 |
| doar imagine | 0,80 [0,73; 0,86] | 0,64 [0,51; 0,73] |
| imagine + text | 0,83 | 0,63 |
| Δ text peste imagine | +0,03 [+0,00; +0,06] | −0,00 |
| **Δ text peste fracțiile din imagine** | **+0,12 [+0,03; +0,22]** | **+0,23 [+0,08; +0,38]** |

Cu referința independentă de imagini (EM-DAT afectați, n = 122, toate tipurile):
- textul singur dă ρ 0,38 [0,21; 0,53], imaginea doar 0,12;
- textul adaugă +0,21 [+0,06; +0,35] peste imagine.

→ **Textul spune despre oameni, imaginea despre hectare.** Mare parte din „magnitudinea din imagine” e mărimea AOI-ului desenat de EMS.

**Informație mutuală cu log(suprafața EMS)**, peste P95 din 50 de permutări:
- incendii: S2 aria arsă (1,01 nats), S1 aria schimbată (0,40), fracția arsă, dNBR, suprafața menționată în text (0,15), ploaia și Tmax;
- inundații: doar trăsăturile din imagine (S1 aria inundată 0,29, apa nouă S2).

**CCA out-of-sample** (corelația canonică 1 pe test, nul = P95 din 20 de permutări):
- imagine ↔ text: 0,82 pe tot setul (nul 0,08), 0,65 la incendii (nul 0,12);
- la inundații: 0,16 cu tot textul (sub nul), dar 0,51 doar cu BGE-M3 (câmpurile structurate sunt aproape toate zero la inundații).

**Grafuri kNN** (10 vecini): vecinătățile imaginii și ale textului se suprapun de 1,4–1,5× peste întâmplare în interiorul unui tip (2,4× pe tot setul). Vecinătățile textului nu seamănă cu cele după suprafața EMS (1,05–1,08×).

**Dezacordul text–imagine vs. incertitudinea textului** (markeri „about”, „estimated”, „may”, „expected”… la 100 de cuvinte):
- |ŷ_text − ŷ_imagine| nu crește cu incertitudinea: ρ −0,06 (p = 0,27) pe tot setul, 0,03 la incendii, −0,12 la inundații;
- la incendiile care dau o cifră: ρ 0,05 (n = 33).

→ **Rezultat negativ** pentru a doua parte a întrebării: incertitudinea lexicală a textului nu anticipează dezacordul.

## 4. Ce înseamnă pentru Q19.2 și ce urmează

- Structura datelor e aproape liniară: PCA rămâne baseline-ul de bătut la CP4 (clustering, anomalii).
- Textul și imaginea poartă magnitudini diferite: oameni (EM-DAT) vs. hectare (suprafața EMS). Peste ce vede imaginea independent de AOI, textul aduce informație măsurabilă.
- Decizia de pivot (CP4): a doua parte a întrebării („dezacordul e anticipat de incertitudinea textului”) se înlocuiește cu „textul descrie impactul uman, imaginea extinderea fizică”.
- Riscuri: aria AOI (raportez și varianta normalizată pe AOI), referința EMS care se schimbă în 2023, EM-DAT cu n mic.
