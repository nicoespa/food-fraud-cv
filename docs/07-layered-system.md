# 07 — Sistema en capas IMPLEMENTADO (píxel + procedencia + comportamiento → fusión)

Implementa el roadmap de `docs/05`: el modelo ya no es solo el CNN, es la **fusión** de
varias señales. Corre dentro de `scripts/run_cooked.py` (al final, automático).

## Capas implementadas
- **L1 Píxel** (`detection/baseline.py`, `detection/finetune.py`): CNN/forense → P(AI-editada).
- **L0 Procedencia** (`detection/provenance.py`, **REAL**): perceptual-hash (dhash) +
  detección de **reuso** contra una base de reclamos previos; flag de EXIF ausente.
- **L3 Comportamiento** (`detection/behavioral.py`, **SIMULADO** con ruido, etiquetado):
  refund_count, account_age, device_risk. En producción salen de datos reales.
- **L4 Fusión** (`detection/fusion.py`): GBM sobre [pixel_prob, reuse, exif, comportamiento]
  → probabilidad de fraude calibrada.
- **L5 Decisión** (`decision/policy.py`): cost-sensitive aprobar/rechazar/revisión humana.

## El demo clave: fraude por REUSO
Se inyecta un tipo de fraude que el CNN **no puede** ver: una foto **real** duplicada de un
reclamo histórico (píxeles genuinos, pero ya usada). El píxel la pasa como genuina; la
procedencia (hash) la detecta; la fusión combina y gana.

**Validación local (CPU, corpus sintético):**

| modelo | PR-AUC | TPR ai-fakes | **TPR reuso** | FPR genuino |
|---|---|---|---|---|
| pixel-only (CNN/forense) | 0.78 | 0.89 | **0.00** | 0.33 |
| **fusion-layered** | **0.86** | 0.81 | **1.00** | 0.33 |

→ El CNN solo **no atrapa el reuso** (TPR 0); la **fusión sí** (TPR 1.0). Es la prueba de que
el sistema en capas cubre puntos ciegos del modelo de visión.

## Honestidad
- Procedencia (perceptual-hash de reuso, EXIF) = **código real** que corre sobre cualquier foto.
- Comportamiento = **simulado** con correlación moderada + ruido (no hay datos públicos de
  cuentas); claramente etiquetado. En Rappi saldría de datos reales.
- En el corpus sintético los JPEG no llevan EXIF → ese feature es poco informativo acá, pero
  el código sirve sobre fotos reales de producción.

*Generado 2026-06-21. Corre como parte de `run_cooked.py`.*
