# Data evenimentului extrasă din textul EMS — U19

Generat de `pipeline/07_text.py`.

- Activări cu o dată în text: **295 / 411** (72%).
- Validare pe EMSR656+ (eventTime declarat): 123 activări, data din text = eventTime exact la 93%, la ±1 zi la 98%.
- Activări vechi (t0 = activationTime) cu dată în text: 169 din 265.
  Întârzierea activare − eveniment: mediana 1 zile, P90 7 zile, max 37 zile; > 10 zile la 11 activări.

Fereastra pre a imaginilor se termină la t0 − 10 zile, deci rămâne înainte de eveniment pentru toate activările cu întârziere ≤ 10 zile. Activările cu întârziere mai mare sunt marcate (`lag_days > 10`) și se exclud într-o analiză de sensibilitate.
