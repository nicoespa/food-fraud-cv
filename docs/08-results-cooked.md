# 08 — Resultados REALES comida cocida (Colab GPU, n=150, balanceado 1:1)

Dominio delivery real (Food-101: pizza/hamburguesa/...) + fakes con difusión multi-generador.
Autenticidad binaria (`genuine` vs `fake-damaged`). Test: 25 genuine + 25 fake (+ reuso en capas).

## Autenticidad
| modelo | PR-AUC | ROC-AUC | TPR@FPR5% | ECE | costo D0 | D1 | D2 | FPR genuine |
|---|---|---|---|---|---|---|---|---|
| zero-shot (genérico) | 0.347 | 0.162 | 0.00 | — | 1.50 | 1.50 | 1.00 | **1.00** |
| forensic-gbm | 0.855 | 0.792 | 0.60 | 0.126 | 2.46 | 1.84 | 1.48 | 0.56 |
| **finetune-resnet50** | **0.946** | **0.928** | **0.76** | 0.085 | 0.98 | 1.06 | **0.74** | 0.44 |

## Leave-one-generator-out (forense)
| held-out | TPR | | held-out | TPR |
|---|---|---|---|---|
| sd-mold | 0.900 | | ip2p-rot | 0.927 |
| sd-rot | 0.867 | | sd-fungus | 1.00 (n=3) |
| **classic-splice (no-AI)** | **0.250** | | | |

## Sistema en capas (con fraude por reuso)
| modelo | PR-AUC | TPR ai | **TPR reuso** | FPR genuine |
|---|---|---|---|---|
| pixel-only (CNN) | 0.838 | 0.720 | **0.364** | 0.160 |
| fusion-layered | 0.798 | 0.760 | **0.909** | 0.360 |

## Lectura (más rica que frutas — cocido es más difícil)
1. **Tesis 1 ✅ fuerte:** el zero-shot es peor que azar (ROC 0.16) y marca TODO como fraude
   (FPR genuine 1.0) → inservible para verificar reembolsos.
2. **resnet50 es el mejor pero NO satura a 1.0** (PR-AUC 0.946): cocido es un problema real
   y difícil (a diferencia de frutas, donde todo daba 1.0). Resultado informativo.
3. **Cross-generator — el hallazgo más fuerte:** el forense generaliza bien a generadores de
   **difusión** no vistos (ip2p 0.93, sd-mold 0.90, sd-rot 0.87) pero **colapsa a 0.25 en
   `classic-splice`** (edición NO-AI). Un detector entrenado solo con fakes de difusión
   **no atrapa manipulación clásica** → el gap cross-FAMILIA es el riesgo real (más que entre
   variantes de difusión). Justifica el ensemble multi-señal.
4. **Capa de fusión:** atrapa el fraude por **reuso** que el píxel no ve (TPR 0.36 → **0.91**).
   Cuesta algo de precisión global (PR-AUC 0.84→0.80, más FPR) por el ruido de la señal de
   comportamiento *simulada*; con datos reales de cuenta el trade-off mejoraría.
5. **Costo (Tesis 2):** la revisión humana (D2) es la política más barata para todos los
   modelos; aporta más cuanto más incierto el modelo (forense 2.46→1.48).

## Caveats
- FPR genuine alto (resnet 0.44) al umbral de costo agresivo (c_fp≪c_fn → denegar de más):
  en producción se ajusta a costos reales y la zona de revisión absorbe los inciertos.
- Comportamiento = **simulado** (etiquetado). Adversarial: ver curva dirigida (FGSM targeted)
  en la próxima corrida; el ataque chico ya muestra vulnerabilidad real del CNN.

*Generado 2026-06-21. Datos 100% públicos (Food-101 + difusión). Cero datos Rappi.*
