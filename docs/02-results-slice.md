# 02 — Resultados del slice runnable (Fases 1–3, corpus sintético)

Resultados del experimento estrella sobre el **corpus sintético** (sin GPU), generado por
`scripts/run_experiment.py`. Es la validación de la METODOLOGÍA end-to-end; los números a
escala (Food-101 + difusión + fine-tune) se obtienen reemplazando el generador y el modelo
por los caminos `--extra gen` / `--extra dl`.

> **Qué es el corpus sintético:** `genuine-undamaged` (limpio), `genuine-damaged` (daño +
> compresión) y `fake-damaged` (el MISMO daño + compresión **más una huella espectral del
> generador**). El único rasgo que separa fake de daño real es la huella → el detector debe
> aprenderla. 3 generadores; el held-out (`instruct-edit`) es de otra familia espectral y
> solo aparece en test.

## Setup
- `n_sources=300` → 1245 imágenes (train 840 / val 180 / test 225), `image_size=96`.
- Costos: `c_fn=10`, `c_fp=3`, `c_review=1` (FN = fraude pagado; FP = cliente honesto; review = humano).
- Calibración isotónica sobre validación. Métricas de ranking sobre score crudo; calibración/costo sobre prob calibrada.

## Resultados (test)

| modelo | PR-AUC | ROC-AUC | TPR@FPR5% | ECE | costo D0 | costo D1 | costo D2 |
|---|---|---|---|---|---|---|---|
| zero-shot-aigc (genérico) | 0.466 | 0.302 | 0.000 | 0.115 | 2.111 | 1.373 | 1.169 |
| **forensic-gbm** | **0.954** | **0.936** | **0.652** | **0.043** | **0.876** | **0.484** | **0.378** |

Breakdown por generador / clase (al umbral de costo):

| modelo | TPR sd-inpaint | TPR img2img | TPR instruct-edit (held-out) | FPR genuine-damaged | FPR genuine-undamaged |
|---|---|---|---|---|---|
| zero-shot-aigc | 0.956 | 0.978 | 0.933 | 0.844 | 1.000 |
| forensic-gbm | 0.978 | 1.000 | 1.000 | 0.733 | 0.000 |

**Mejora de costo D0→D2 (forensic):** 0.492 por caso, **IC95 bootstrap [0.231, 0.796]** (excluye 0).

## Lectura
1. **Tesis 1 — el detector genérico falla.** zero-shot tiene PR-AUC 0.47 / ROC 0.30 (ranking
   casi inverso) y, al umbral de costo, marca como fraude el **84% del daño real** y el **100%
   del intacto**: inunda de falsos positivos. No sirve para verificar reembolsos.
2. **Tesis 2 — mejor accuracy ≠ mejor decisión.** El umbral ingenuo (D0) cuesta 0.876/caso;
   el umbral por costo (D1) 0.484; y la política de 3 zonas con **revisión humana** (D2)
   0.378. La mejora D0→D2 es significativa. La clave: en vez de denegar a clientes honestos
   inciertos (la FPR alta sobre genuine-damaged), D2 los **manda a revisión**.
3. **Calibración** importa: ECE forensic 0.043 (calibrado) habilita los umbrales por costo.

## Caveat honesto — generalización cross-generator
En este slice el `forensic-gbm` **generaliza bien** al generador held-out (TPR 1.0): las
huellas sinusoidales simples son fáciles y transfieren. **Esto NO contradice la literatura** —
FraudBench y la línea de AIGC-detection muestran que con generadores de **difusión reales** la
transferencia cae fuerte (es el problema abierto #1). El toy no reproduce esa dificultad; la
**metodología** de held-out cross-generator queda montada para medir el gap real al escalar
con `--extra gen`.

## Reproducir
```bash
uv run --extra ml python scripts/run_experiment.py --config configs/default.yaml --rebuild
# artefactos: results/experiment.json y results/experiment.csv (gitignored)
```

*Generado el 2026-06-19. Números del run con seed=42, n_sources=300.*
