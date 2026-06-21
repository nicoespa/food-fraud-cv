# 04 — Resultados REALES (Colab GPU, escala informativa)

Run: `scripts/run_real.py` en Colab GPU. Datos reales (`Densu341/Fresh-rotten-fruit`) +
fake-damaged por **difusión real** (SD 1.5 inpainting, 25 pasos = fakes sutiles) +
fine-tune real. n_per_group=150, split 60/20/20, costos c_fn=10 / c_fp=3 / c_review=1.

## Tabla

| modelo | PR-AUC | ROC-AUC | TPR@FPR5% | ECE | costo D0 | costo D1 | costo D2 | FPR daño-real | FPR intacto |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot-aigc (genérico) | 0.791 | 0.592 | 0.250 | 0.225 | **3.230** | 1.061 | 0.927 | **0.905** | **1.000** |
| forensic-gbm | 0.988 | 0.978 | 0.880 | 0.103 | 1.067 | 1.067 | 1.000 | 0.000 | 0.056 |
| finetune-resnet50 | **1.000** | **1.000** | **1.000** | 0.023 | **0.182** | **0.121** | 0.139 | 0.000 | 0.000 |

Cross-generator (TPR por generador; `sd-fungus` = held-out, no visto en train):

| modelo | sd-mold (visto) | sd-rot (visto) | **sd-fungus (held-out)** |
|---|---|---|---|
| forensic-gbm | 0.944 | 0.972 | **0.611** |
| finetune-resnet50 | 0.944 | 1.000 | **1.000** |

Mejora de costo D0→D2 (resnet50): +0.041 por caso, IC95 [-0.024, 0.158] (incluye 0).

## Lectura (lo que demuestra)

1. **El detector AIGC genérico es inusable para verificar reembolsos (Tesis 1 ✅).**
   PR-AUC 0.79 / ROC 0.59 (apenas mejor que azar). Y al umbral de costo marca como fraude
   el **90% del daño REAL** y el **100% de la comida intacta** → inunda de falsos positivos.
   Costo 3.23/caso (≈18× el del resnet50). Es la prueba empírica de la tesis de FraudBench:
   *detección de imágenes-AI ≠ verificación de evidencia condicionada al reclamo*.

2. **Un modelo aprendido y claim-conditioned sí funciona.** El resnet50 fine-tuneado separa
   las 3 clases casi perfecto (PR-AUC 1.0, ECE 0.023 = bien calibrado) y con **FPR 0 sobre
   las clases genuinas** → no castiga a clientes honestos. El forense (0.99) también, muy por
   encima del genérico.

3. **El gap cross-generator es real y depende del modelo (✅).** El forense cae a **0.611 TPR
   en el generador no visto** (vs 0.94–0.97 en los vistos): reproduce el hallazgo de FraudBench
   —los detectores generalizan peor a generadores nuevos—. El deep (resnet50) transfiere mejor
   (1.0): las features semánticas generalizan donde los artefactos de bajo nivel sobreajustan.

4. **El valor de la capa de decisión cost-sensitive escala INVERSO a la calidad del modelo
   (Tesis 2, matizada ✅).** Para el zero-shot débil, la política con revisión humana baja el
   costo de **3.23 → 0.93**. Para el resnet50 casi perfecto, el umbral ingenuo ya es bueno
   (D0 0.18, D1 0.12) y la mejora no es significativa (IC incluye 0). Conclusión honesta: la
   capa de decisión es una **red de seguridad que importa más cuanto más incierto es el detector**.

## Caveats (para el informe)
- El resnet50 llega a 1.0 → la tarea sigue siendo algo "fácil" (los fakes de difusión, aun a
  25 pasos, son detectables por un deep). Para tensionar: ediciones más sutiles, más familias
  de generador, fakes adversariales. El gap cross-generator del forense y el fracaso del
  genérico son, igual, hallazgos sólidos.
- Datos 100% públicos (Densu341 + SD1.5), cero datos Rappi.

*Generado el 2026-06-21.*
