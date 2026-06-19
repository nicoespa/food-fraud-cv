# CLAUDE.md — Proyecto Final Ciencia de Datos

**Cómo usar este archivo:** Claude Code lo lee automáticamente como contexto
persistente en cada sesión dentro de este repo. Es autocontenido.

## 0. Identidad del proyecto
- **Nombre de trabajo:** `food-fraud-cv`
- **Una línea:** Un sistema de verificación de evidencia de reembolso para apps de
  delivery que decide, condicionado al reclamo del cliente, si una foto es
  *genuine-undamaged* / *genuine-damaged* / *fake-damaged* (daño fabricado con AI),
  demostrando que un detector genérico de "¿es AI?" **falla** en este problema, y
  que una **probabilidad calibrada** alimentando una **política de decisión
  sensible al costo** (aprobar / rechazar / revisión humana) le gana a los modelos
  que solo maximizan accuracy.
- **Reemplaza a** `meal-dispatch-uncertainty` como trabajo final del curso
  (decisión 2026-06-19). Comparte su espina intelectual: **decisión bajo
  incertidumbre con probabilidad calibrada y costo asimétrico.**

## 1. Contexto académico y la vara
Trabajo final de **Ciencia de Datos** (Lic. en Negocios Digitales, UdeSA).
Entregables: (a) código funcional y reproducible, (b) informe escrito, (c)
presentación. La vara es un **10** con impacto real. La materia premia alta
complejidad: tantos modelos y análisis como haga falta. No recortar ambición;
ordenarla (ver plan por fases, MVP vs stretch).

**Doble objetivo (confirmado 2026-06-19):** además de TP académico, debe servir
como **propuesta presentable en Rappi** (caso de uso real del área). La capa de
decisión + revisión humana enchufa con el `refunds-backoffice` de Rappi
(maker-checker / human-in-the-loop).

**Restricción dura (integridad académica):** nada de datos internos de Rappi ni de
ninguna empresa. Todo con datasets **100% públicos** o **generados por nosotros a
partir de datos públicos**. Heredada de meal-dispatch.

## 2. El problema y la tesis central
Clientes de delivery manipulan o **editan con AI** la foto de su pedido (agregan
moho, lo hacen ver crudo / desarmado) para gatillar un reembolso fraudulento. La
industria ya lo sufre (DoorDash, Uber Eats) y responde con metadata/attestation,
device-risk y ML de forgery. El problema **no** es "¿esta imagen es generada por
AI?" en abstracto — es **verificación condicionada al reclamo** (*claim-conditioned*):
¿esta foto realmente prueba el daño que el cliente dice tener?

**Tesis:**
1. **El detector AIGC genérico no alcanza.** Da falsos positivos sobre daño
   *genuino* y se le escapa el daño *falso*; además transfiere mal entre
   generadores. (Replicamos empíricamente el hallazgo de **FraudBench**, 2026,
   con datos públicos.) → este es el resultado que justifica todo el sistema.
2. **Mejor accuracy ≠ mejor decisión de negocio.** El modelo de mejor accuracy/AUC
   **no** es el que minimiza el costo esperado del reembolso (FN = plata pagada a un
   fraude; FP = cliente honesto rechazado). La probabilidad **calibrada** + política
   **cost-sensitive** con zona de revisión humana domina. Es el insight que separa
   el 7 del 10 (igual que en meal-dispatch).

## 3. Arquitectura: capas + un puente
- **Capa 1 — Datos** (`src/data`, `src/generation`): construir el corpus de 3
  clases. *genuine-undamaged* y *genuine-damaged* salen de datasets públicos de
  comida (Food-101, fresh/rotten). *fake-damaged* lo **generamos nosotros**:
  partimos de un *genuine-undamaged* y editamos con modelos de difusión/inpainting
  para simular moho / crudo / desarmado. Usamos **varios generadores** y dejamos
  ≥1 **held-out** (no visto en train) para medir generalización cross-generator.
- **Capa 2 — Detección/Predicción** (`src/detection`, corazón ML): escalera de
  modelos que producen una **probabilidad calibrada de fraude** (fake-damaged)
  condicionada al reclamo. Salida: score + intervalo/confianza + (stretch) heatmap
  de localización.
- **Puente:** la probabilidad calibrada de la Capa 2 alimenta la decisión.
- **Capa 3 — Decisión** (`src/decision`): política **sensible al costo** sobre el
  score: low-risk → auto-aprobar; high-risk → auto-rechazar; zona gris → **revisión
  humana**. Selección de umbrales por costo esperado, no por accuracy.

> **Honestidad metodológica (incluir en el informe):** el set *fake-damaged* es
> **sintético generado por nosotros**, no fraude real capturado en producción — es
> una aproximación controlada (misma estrategia que FraudBench, que tampoco liberó
> su dataset). Limitación declarada: los generadores que usamos definen el dominio;
> un atacante real podría usar otros. Por eso medimos **generalización
> cross-generator** (test sobre generador no visto) y lo declaramos explícitamente.

## 4. Datos

### 4.1 Las 3 clases (claim-conditioned)
| Clase | Qué es | Decisión esperada |
|---|---|---|
| `genuine-undamaged` | Comida OK, sin daño | Rechazar reembolso (no hay daño) |
| `genuine-damaged` | Comida realmente dañada/podrida | Aprobar reembolso |
| `fake-damaged` | Foto editada con AI para *simular* daño | Rechazar + flag de fraude |

El par crítico es `genuine-damaged` vs `fake-damaged`: **los dos "se ven mal"**. Ahí
es donde el detector genérico falla y donde está el aporte.

### 4.2 Fuentes públicas (capa genuina) — VERIFICAR en primera carga
- **Food-101** (101 clases, 101k imágenes) — base de comida sana → `genuine-undamaged`.
- **Fresh & Rotten / FruitVision / Freshness44** (Kaggle/HF) — comida realmente
  dañada → `genuine-damaged`. (FruitVision incluye incluso "formalin-mixed".)
- Validar licencias y columnas/estructura en la primera carga (registrar en un
  *dataset card* en `data/`).

### 4.3 Set sintético `fake-damaged` (lo generamos nosotros)
Pipeline detallado en `docs/01-data-generation-and-models.md`. Resumen: tomar
imágenes `genuine-undamaged` → editar con varios generadores (inpainting con máscara
para "agregar moho/objeto extraño"; img2img para "crudo/podrido"; instruct-edit) →
post-proceso opcional que imita la captura real del atacante (strip EXIF,
re-compresión JPEG, screenshot). **Splits con generador disjunto:** el test incluye
≥1 generador nunca visto en train.

### 4.4 Provenance / metadata (señal no-pixel, stretch realista)
EXIF, ausencia de C2PA, hash perceptual (reuso de fotos), re-compresión. Se modela
como features adicionales en la fusión, no como veredicto único.

## 5. Capa 2 — Detección (corazón ML)
Escalera de modelos (detalle y selección en `docs/01-...`):
- **Baseline zero-shot:** detector AIGC off-the-shelf (CLIP-based / frequency).
  Sirve para *mostrar el gap* claim-conditioned (Tesis 1). **Verificar el ID exacto
  del modelo en HF antes de usarlo — no asumir.**
- **Fine-tune supervisado:** backbones vía `timm`/`transformers` — ResNet50,
  EfficientNet, ViT/CLIP — clasificación 3-clases.
- **Señales forensics:** ELA, ruido/SRM, frecuencia (FFT/DCT), (stretch) TruFor-style.
- **Fusión + calibración:** late fusion / stacking; calibrar con temperature /
  isotonic / Platt.
- **MLLM claim-conditioned (stretch):** modelo multimodal que recibe imagen + texto
  del reclamo y razona consistencia, con explicación.
**Evaluación de la capa:** no quedarse en accuracy (ver §7).

## 6. Capa 3 — Decisión (política cost-sensitive)
- **D0 — Umbral por accuracy:** baseline ingenuo (maximiza accuracy). A propósito
  malo, para contrastar.
- **D1 — Umbral por costo esperado:** define costos `c_FN` (fraude pagado) y `c_FP`
  (cliente honesto rechazado) y elige el umbral que minimiza el costo esperado.
- **D2 — Tres zonas (LA contribución de decisión):** dos umbrales → auto-aprobar /
  auto-rechazar / **revisión humana** (con costo de revisión `c_review`). Optimiza
  el costo total incluyendo el presupuesto de revisión. Conecta con el
  refunds-backoffice (maker-checker).
- **D3 — Selección de modelo por negocio (stretch):** mostrar que el modelo elegido
  por costo esperado ≠ el de mejor AUC (Tesis 2).

## 7. Evaluación
**No usar accuracy como métrica principal** (clases desbalanceadas, costo asimétrico):
- **Discriminación:** PR-AUC, ROC-AUC, **TPR @ FPR fijo**, matriz de confusión 3×3.
- **Costo:** costo esperado de reembolso bajo cada política (la métrica de negocio).
- **Calibración:** reliability diagram, **ECE**, Brier. Necesaria para D1/D2.
- **Generalización:** breakdown **por generador**, con test sobre generador held-out.
- **Interpretabilidad:** Grad-CAM / heatmaps; (stretch) localización pixel-level.
- **Significancia:** bootstrap CIs en las diferencias; McNemar sobre predicciones
  pareadas entre modelos.

**EXPERIMENTO ESTRELLA:** matriz `modelo × régimen-de-info × generador`. Mostrar:
1. El detector AIGC genérico **falla** en claim-conditioned (gap empírico, Tesis 1).
2. El modelo de mejor accuracy/AUC **no** minimiza el costo de reembolso (Tesis 2).
3. La política de 3 zonas (D2) recupera la mayor parte del costo vs un oráculo, con
   un presupuesto de revisión humana acotado.

## 8. Convenciones
Python 3.12, seeds fijas, config-driven (yaml en `configs/`), tests para data +
decisión, deps pinneadas (`uv`), resultados versionados como artefactos en
`results/`, "cómo correr de cero" documentado en el README. Datasets y modelos
pesados **fuera de git** (ver `.gitignore`).

## 9. Plan por fases
- **Fase 0 (scaffold):** estructura, loaders de 1 dataset público, pipeline de
  generación con **1 generador**, corpus 3-clases mínimo, **1 baseline zero-shot**
  evaluado end-to-end. ← *vertical slice*.
- **Fase 1 (datos + ML):** corpus completo multi-generador + dataset card; EDA;
  fine-tune CNN/ViT; calibración.
- **Fase 2 (forensics + costo):** señales forensics + fusión; evaluación
  cost-sensitive; **holdout cross-generator**.
- **Fase 3 (decisión + estrella):** políticas D0–D2 + experimento estrella + tests
  estadísticos. MLLM claim-conditioned (stretch).
- **Fase 4 (entrega):** informe + presentación (+ dashboard stretch).
- **MVP (mantener verde antes de cualquier stretch):** Fase 0 + corpus
  multi-generador + un CNN/ViT fine-tuneado y **calibrado** + evaluación
  cost-sensitive + política D1/D2 + experimento estrella. Stretch: MLLM,
  localización pixel-level, provenance, dashboard, RL de umbrales.

## 10. Compute
Mac (Apple Silicon). **Fine-tune** de backbones con transfer learning corre en
**MPS** (o Colab/HF si hace falta GPU). **Generación de difusión** es lo más
pesado: usar Colab/HF Inference o `diffusers` en MPS con lotes chicos; cachear todo
lo generado en `data/generated/` (gitignored). Empezar con pocas imágenes por clase
para validar el pipeline antes de escalar.

## 11. Guardrails
Datos públicos / generados por nosotros únicamente — **cero datos Rappi**.
Reproducibilidad (seeds, deps pinneadas, cómo correr de cero). Honestidad
metodológica (declarar que el fake-damaged es sintético y la dependencia del
generador → medir cross-generator). No fabricar métricas. **No inventar IDs de
modelos/datasets de HF — verificar disponibilidad real antes de depender de ellos.**
Validar supuestos en la primera carga. MVP primero.
