# 01 — Pipeline de datos sintéticos + selección de modelos

Diseño técnico profundo de las dos piezas más difíciles del proyecto. Acompaña al
brief (`CLAUDE.md`). Todo con datos/modelos **públicos**; cero datos Rappi.

> ⚠️ **IDs de modelos/datasets:** los nombres acá son familias y candidatos. Antes de
> depender de cualquier repo de Hugging Face hay que **verificar que existe, su licencia
> y su tarea**. No asumir IDs de memoria.

---

## Parte A — Generación del set `fake-damaged`

### A.0 Por qué generamos nosotros
FraudBench (2026) es el trabajo más cercano pero **no liberó su dataset**, y no existe
un corpus público de fraude real de reembolsos (privacidad). Generar el `fake-damaged`
nosotros es: (a) la única vía honesta con datos públicos, (b) lo que hizo FraudBench, y
(c) un aporte en sí mismo. La clave metodológica: **controlar el generador** para poder
medir generalización cross-generator (ver A.5).

### A.1 Las 3 clases y de dónde sale cada una
```
genuine-undamaged  ← Food-101 (comida sana)                      [fuente pública]
genuine-damaged    ← Fresh&Rotten / FruitVision / Freshness44     [fuente pública]
fake-damaged       ← editar un genuine-undamaged con un generador [GENERADO POR NOSOTROS]
```
El par que importa es `genuine-damaged` vs `fake-damaged`: ambos "se ven mal". El
modelo no puede ganar mirando solo "¿hay daño?" — tiene que detectar la **huella de
edición**. Por eso `fake-damaged` se construye editando una imagen *sana*, no una ya dañada.

### A.2 Tipos de edición (los "ataques" que simulamos)
| Edit | Cómo se hace | Método |
|---|---|---|
| `mold` (moho) | máscara sobre la comida → inpainting "moho verde/blanco" | inpaint |
| `foreign-object` | máscara → inpainting "pelo / insecto / plástico" | inpaint |
| `raw` (crudo) | transformación global de color/textura a "crudo/rosado" | img2img / instruct |
| `spoiled` (podrido) | global a "descompuesto, descolorido" | img2img / instruct |

`mold` y `foreign-object` son **ediciones locales** → inpainting con máscara (más
realista y más difícil de detectar; coincide con cómo edita un atacante con AI). `raw`
y `spoiled` pueden ser globales.

### A.3 Los generadores (≥3, con uno held-out)
Para medir generalización necesitamos **varios generadores de familias distintas**:
1. **Inpainting de difusión** (Stable Diffusion inpaint / SDXL inpaint): edita la región
   enmascarada. Es el caballito de batalla para `mold`/`foreign-object`.
2. **img2img de difusión** (strength alto): re-imagina la foto entera → `raw`/`spoiled`.
3. **Instruct-edit** (instrucción en lenguaje natural, p.ej. modelos tipo InstructPix2Pix
   / edit por instrucción): **reservado como held-out** (solo test) → mide transferencia
   a una familia nunca vista.

Cada imagen generada se etiqueta con `{generator, edit}` para el breakdown.

### A.4 Pipeline (lo que implementa `scripts/run_generate.py` + `src/generation`)
```
genuine-undamaged
   │  (para inpaint) segmentar la comida → máscara binaria
   │      opción simple: máscara central / por color; mejor: segmentador off-the-shelf
   ▼
edición por generador × edit  →  imagen fake-damaged
   ▼
post-proceso "como lo subiría el atacante":
   - strip EXIF (la mayoría de las apps de edición lo borran)
   - re-compresión JPEG (calidad ~70-90) — introduce artefactos realistas
   - (opcional) simular screenshot / re-foto
   ▼
data/generated/<generator>/<edit>/xxx.jpg  +  manifest.csv
   (columns: path, label, generator, edit, source_image, split)
```
**Validaciones en primera corrida:** chequear que la edición es visible pero no
caricaturesca; revisar a ojo un sample por generador (no automatizar a ciegas).

### A.5 Splits — la decisión metodológica clave
- **Train/val:** generadores 1 y 2 (inpaint, img2img) + genuine-*.
- **Test:** incluye un slice con el generador **held-out** (instruct-edit) que el modelo
  **nunca vio**. La métrica sobre ese slice es la prueba real de generalización
  cross-generator (el punto donde FraudBench muestra que los detectores se caen).
- Cuidado con **leakage**: una misma `source_image` no puede estar en train y test (ni su
  versión sana ni su versión editada). Split por `source_image`, no por imagen.

### A.6 Balance, tamaño y sesgos
- Arrancar chico (p.ej. 500–1000 por clase) para validar el pipeline; escalar después.
- Vigilar **shortcuts**: si todas las `fake-damaged` salen del mismo pipeline, el modelo
  puede aprender el artefacto de compresión en vez del fraude. Mitigación: variar
  calidad JPEG, resoluciones, y mezclar post-procesos también en las clases genuinas.
- Documentar todo en un **dataset card** (origen, licencias, conteos, generadores).

---

## Parte B — Selección de modelos (Capa 2, detección)

Escalera de menor a mayor complejidad. Cada peldaño es una fila del experimento estrella.

### B.1 Baseline zero-shot — detector AIGC off-the-shelf
- **Qué:** un detector pre-entrenado de "imagen generada por AI" (familias: CLIP-based
  como UniversalFakeDetect/Cozzolino-style; o frequency-based). Se corre **sin
  fine-tune** sobre nuestro test.
- **Para qué:** es la evidencia de la **Tesis 1**. Esperamos que: (a) marque como "fake"
  también comida *genuinamente* dañada o fotos comprimidas (falsos positivos), y (b)
  falle en `fake-damaged` de inpainting local. Mostrar ese gap **es** medio trabajo.
- **Riesgo:** verificar ID/licencia en HF; si no hay uno confiable, usar 2 y promediar.

### B.2 Fine-tune supervisado (el caballo de batalla del MVP)
- **Backbones** vía `timm`/`torchvision`, transfer learning desde ImageNet:
  - `resnet50` — baseline sólido, rápido en MPS.
  - `efficientnet_b0/b3` — mejor accuracy/cómputo.
  - `vit_base` / backbone CLIP (`open_clip`) — suele generalizar mejor a "fake vs real".
- **Tarea:** clasificación 3-clases (o binaria fraude-vs-resto + cabeza auxiliar de daño).
- **Recetas:** congelar backbone → entrenar cabeza → descongelar últimas capas; data aug
  cuidada (NO usar augmentations que borren la huella de edición, p.ej. blur fuerte).
- **Salida:** logits → softmax → **prob de `fake-damaged`** (a calibrar en B.4).

### B.3 Señales forensics (lo que sube el techo)
Features que NO son el contenido sino la **huella de manipulación**:
- **ELA** (Error Level Analysis): re-comprimir y mirar el residuo — delata regiones editadas.
- **Ruido / SRM**: filtros de alto paso; las regiones inpainted tienen estadística de
  ruido distinta.
- **Frecuencia (FFT/DCT)**: los modelos de difusión dejan artefactos espectrales.
- **(Stretch) TruFor-style**: fingerprint de cámara aprendido + localización pixel-level
  con mapa de confiabilidad. Da el **heatmap** de "dónde se editó".

### B.4 Fusión + calibración
- **Fusión:** late fusion (concatenar features de B.2 + B.3 + metadata de A.4) → un
  clasificador liviano (LogReg/GBM) que produce el score final. Alternativa: stacking.
- **Calibración (obligatoria para la Capa 3):** *temperature scaling* (1 parámetro,
  barato y efectivo para redes), o *isotonic*/*Platt* sobre validación. Se mide con ECE
  y reliability diagram (`src/evaluation/metrics.py`). **Sin calibrar, los umbrales por
  costo de la política de decisión no tienen sentido.**

### B.5 MLLM claim-conditioned (stretch, el más alineado con FraudBench)
- Pasar **imagen + texto del reclamo** a un modelo multimodal y pedir veredicto **con
  explicación** ("¿la foto es consistente con el daño reclamado? ¿hay señales de edición?").
- Ventaja: razona la consistencia imagen↔reclamo (el framing exacto del problema) y es
  interpretable. Limitación conocida: hoy los MLLMs son buenos en daño real y flojos en
  daño falso → úsalo como capa de triage/explicación, no como único juez.

### B.6 Qué reportar por modelo
Para cada peldaño: PR-AUC, ROC-AUC, TPR@FPR=5%, ECE, Brier, **costo esperado bajo D0/D1/D2**,
y todo **desglosado por generador** (con el held-out aparte). Significancia con bootstrap
CIs + McNemar entre modelos.

---

## Parte C — Cómputo y orden de ataque
- **Generación (lo más pesado):** difusión en MPS con lotes chicos, o Colab/HF GPU.
  Cachear TODO en `data/generated/` (gitignored). No regenerar en cada corrida.
- **Fine-tune:** transfer learning corre bien en MPS (Apple Silicon); usar batch
  moderado y mixed precision si está disponible.
- **Orden recomendado (MVP-first):**
  1. Wirear 1 dataset público + 1 generador (inpaint) → corpus 3-clases mínimo.
  2. Baseline zero-shot evaluado → **mostrar el gap** (Tesis 1) ya con esto.
  3. Fine-tune `resnet50` + calibración → primera curva de costo (Tesis 2) con D0/D1/D2.
  4. Recién ahí: más generadores, forensics, fusión, held-out, MLLM, localización.

*Generado como parte del setup del proyecto el 2026-06-19.*
