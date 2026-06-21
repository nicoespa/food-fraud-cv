# 05 — De prototipo a producción: roadmap

El CNN fine-tuneado resuelve **una** pieza (¿la foto fue editada con AI?). Producción
requiere un **sistema en capas** que cubra los puntos ciegos. Cada capa ataca una
limitación concreta del modelo actual.

## Mapa: punto ciego → solución

| Punto ciego del CNN actual | Cómo se cubre en producción |
|---|---|
| Solo frutas/verduras (no comida real de delivery) | Datos de dominio real + reentrenamiento continuo (abajo) |
| Solo generadores SD 1.5 | Multi-generador + leave-one-generator-out + continual learning |
| Solo mira píxeles (ciego a fotos reusadas, screenshots, fraude físico) | **Capa de procedencia/metadata + señales de comportamiento** |
| Fakes "fáciles" | Fakes sutiles + **entrenamiento adversarial** |
| Escala/robustez chica | Dataset grande, splits temporales, CIs, slice analysis |
| Costos asumidos | Calibrar a costos reales del negocio + A/B |

## La arquitectura objetivo (sistema en capas)

```
Reclamo (foto + texto + cuenta + contexto del pedido)
  │
  ├─ L0  PROCEDENCIA / NO-PÍXEL   ← cubre el mayor punto ciego
  │      · C2PA / Content Credentials (firma criptográfica de origen)
  │      · EXIF / metadata forensics
  │      · capture-time attestation: foto del stream de cámara EN VIVO vs galería/screenshot
  │        (SDK de cámara in-app que firma la foto al capturarla → no se puede subir de galería)
  │      · device risk (jailbreak/root, anomalía GPS/geofence)
  │      · perceptual hash contra histórico de reclamos → detecta FOTOS REUSADAS/robadas
  │      · reverse image search
  │
  ├─ L1  FORENSICS DE PÍXELES (ensemble, no un solo CNN)
  │      · detector AIGC (CLIP-based, frequency) + el CNN fine-tuneado + forense
  │      · localización (TruFor-style) → heatmap "dónde se editó" (explicabilidad)
  │
  ├─ L2  CONSISTENCIA SEMÁNTICA (claim-conditioned)
  │      · MLLM: ¿la foto es consistente con lo que el cliente RECLAMA?
  │      · clasificador de estado (fresh/spoiled) como feature
  │
  ├─ L3  SEÑALES DE COMPORTAMIENTO/CUENTA   ← enorme en fraude real
  │      · frecuencia de reembolsos, antigüedad de cuenta, patrones de pedido
  │      · (estilo "Risk Entity Watch" de Uber)
  │
  ├─ L4  FUSIÓN → probabilidad de fraude CALIBRADA (combina L0–L3)
  │
  ├─ L5  DECISIÓN cost-sensitive → aprobar / rechazar / REVISIÓN HUMANA
  │      · zona gris → backoffice (maker-checker) → y esas decisiones humanas
  │        VUELVEN como labels nuevos (feedback loop)
  │
  └─ L6  MLOps: drift monitoring · reentrenamiento continuo · shadow mode · A/B ·
         robustez adversarial · audit log · fairness/privacidad
```

## Detalle por capa

### L0 — Procedencia / no-píxel (lo que el CNN NO puede)
El CNN es ciego a fraude que no edita píxeles: foto **real** de comida realmente dañada
**reusada/robada**, **screenshot**, **re-foto de pantalla**, comida arruinada a propósito.
La defensa NO es visión, es procedencia:
- **Capture-time attestation:** que la app obligue a sacar la foto en el momento (cámara
  en vivo firmada), no subir de galería. Mata la mayoría del fraude de reuso.
- **C2PA + EXIF:** verificar origen/ediciones; ausencia de metadata = señal.
- **Perceptual hash** contra todas las fotos de reclamos previos → detecta reuso.
- **Device & GPS risk:** root/jailbreak, ubicación incoherente con la entrega.

### L1 — Forensics de píxeles (ensemble)
El CNN actual + un detector AIGC generalista + el forense, combinados. Sumar
**localización** (heatmap) para que la decisión sea **explicable** (clave para disputas).

### L2 — Consistencia con el reclamo (MLLM)
Un modelo multimodal que lee foto **+ texto del reclamo** ("llegó con moho") y juzga si
son consistentes. Es el corazón "claim-conditioned" y aporta explicación.

### L3 — Comportamiento (no-CV pero decisivo)
La mayoría del fraude real se cae con señales de cuenta: tasa de reembolsos, antigüedad,
device sharing, patrones. Un gradient-boosting sobre estas features suele aportar tanto
como la visión.

### L4–L5 — Fusión + decisión
Combinar todas las señales en **una probabilidad calibrada**, y decidir por **costo
esperado** con zona de **revisión humana** (enchufa con el refunds-backoffice). Las
decisiones de los revisores generan **labels nuevos** → mejora continua.

### L6 — MLOps (lo que lo mantiene vivo)
- **Drift monitoring:** distribución de fotos (data drift) y **generadores nuevos**
  (concept drift = la amenaza #1, es carrera armamentista).
- **Reentrenamiento** programado + gatillado por drift; champion/challenger; **shadow
  mode** antes de actuar; **A/B**.
- **Robustez adversarial:** atacar el propio modelo (PGD/FGSM) y entrenar contra eso.
- **Gobernanza:** audit log de cada decisión (disputas/compliance), explicabilidad,
  fairness (no rechazar de más a ciertos usuarios/regiones), retención/privacidad de fotos.

## Datos a escala (el combustible)
- **Multi-generador:** fabricar fakes con muchas familias (SD 1.5/2/XL, FLUX, Midjourney
  vía API, InstructPix2Pix, apps móviles, ediciones clásicas copy-move) → evaluar
  **leave-one-generator-out** como estándar.
- **Dominio real:** comida cocida / platos (Food-101 y similares) además de frutas; en
  Rappi, fotos reales de reclamos (con consentimiento/privacidad) etiquetadas por el
  equipo de revisión del backoffice → **el backoffice genera el dataset**.
- **Split temporal:** entrenar con pasado, testear con futuro (imita el deploy).
- **Adversarial set:** fakes sutiles + ejemplos adversariales.

## Plan por fases (cómo llegar)
- **Fase A — académico-fuerte (datos públicos, factible ya):** multi-generador +
  leave-one-out + track de **features de procedencia/metadata** + **fusión** + entrenamiento
  adversarial + n grande + CIs. Convierte el TP en algo cercano a paper.
- **Fase B — piloto Rappi:** fotos reales de reclamos (consentidas), labels del backoffice,
  señales de comportamiento, **shadow mode** (predice sin actuar, se mide).
- **Fase C — producción:** serving con presupuesto de latencia, monitoreo de drift,
  reentrenamiento, gobernanza, robustez adversarial, A/B.

## Métrica de éxito en producción
No es accuracy: es **$ de fraude evitado − $ de fricción a clientes honestos − costo de
revisión**, monitoreado en el tiempo y por segmento, con la tasa de revisión bajo
presupuesto.

*Relacionado: el sistema en capas viene del deep-dive de investigación; la capa de decisión
+ revisión humana es el refunds-backoffice. Generado 2026-06-21.*
