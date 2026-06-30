# 09 — Generalización: ¿es confiable o sobreajusta?

Responde la pregunta "¿testeaste contra lo mismo que entrenaste?". **No:** hay split
train/val/test por `source_id` (sin leakage) y las métricas son sobre el test no visto.
Pero el test held-out es de la **misma distribución**. Estas pruebas miden generalización
a datos genuinamente NUEVOS (`scripts/run_generalization.py`).

## Pruebas (fuera de distribución)
1. **Cross-domain:** entrenar en UN dominio y testear en el OTRO.
   - `train frutas → test cocido` y `train cocido → test frutas` (binario: fake vs resto).
   - Comida totalmente distinta, posiblemente generadores distintos. Forense (CPU) + CNN.
   - Se compara contra el **in-domain** (train/test del mismo dominio): si `cross << in-domain`,
     el modelo NO generaliza a comida nueva.
2. **CNN leave-one-generator-out:** entrenar el CNN SIN la familia `classic-splice` y testear
   sobre ella (edición no vista). ¿El CNN también colapsa como el forense (que cayó a 0.25)?

## Cómo correr (Colab GPU, tras frutas + cocido)
```bash
uv run --extra real python scripts/run_generalization.py        # forense + CNN
uv run --extra real python scripts/run_generalization.py --skip-cnn   # solo forense (CPU)
```
Salida: `results/generalization.json` + tablas en consola.

## Cómo leer el resultado (qué significaría cada caso)
- **Cross-domain alto (~in-domain):** el modelo capta señales de manipulación **generales**
  → confiable, generaliza a comida nueva. Fuerte para el informe.
- **Cross-domain bajo:** sobreajusta al dominio/dataset → cuantifica la limitación (esperable
  en parte) y refuerza por qué producción necesita datos del dominio real (Fase B).
- **CNN-LOGO classic alto:** el CNN generaliza mejor que el forense a familias nuevas.
  **Bajo:** confirma que ninguna familia única basta → ensemble multi-señal (lo que ya mostramos).

> Honesto: lo que NO se hizo es testear contra un dataset **externo de fraude real "del mundo"**
> (no es público). Esa es la limitación de fondo que separa prototipo de producción.

*Generado 2026-06-21.*
