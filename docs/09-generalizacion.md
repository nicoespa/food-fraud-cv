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

## RESULTADOS (Colab GPU · frutas 936 imgs · cocido 300 imgs)

**Forense (PR-AUC / ROC-AUC):**
| setup | PR-AUC | ROC-AUC |
|---|---|---|
| in-domain · frutas | 0.993 | 0.980 |
| in-domain · cocido | 0.913 | 0.877 |
| **cross · frutas→cocido** | **0.535** | 0.561 |
| **cross · cocido→frutas** | **0.713** | 0.641 |

**CNN (ResNet) cross-domain:** train frutas→test cocido **0.685** · train cocido→test frutas **0.876**.

**CNN leave-one-generator-out** (held-out `classic-splice`): TPR **0.222** (PR-AUC 0.637) — el CNN **también** colapsa ante una familia no vista (forense fue 0.25).

### Lectura (la respuesta honesta)
- **Hay un gap de generalización real.** In-domain ~0.91–0.99 → **cross-domain 0.54–0.88**. O sea: los números altos NO eran testear-sobre-entrenamiento (el test es held-out), pero **sí eran misma distribución**. Sobre comida genuinamente distinta, **cae** → el modelo aprendió señales del dominio, no un "detector universal de fraude".
- **El CNN generaliza MEJOR que el forense** (0.685–0.876 vs 0.535–0.713): las features semánticas transfieren más que los artefactos de bajo nivel.
- **Entrenar en lo difícil (cocido) → testear en lo fácil (frutas) = 0.876**; al revés (frutas→cocido) = 0.685. El dominio de entrenamiento más rico da features más robustas.
- **Ni el CNN generaliza a una familia de edición no vista** (0.22). → Justifica el ensemble multi-señal + reentrenamiento continuo (drift) en producción.
- **Conclusión:** funciona in-distribution, degrada out-of-distribution. Honesto y esperable. Producción necesita datos del dominio real + monitoreo de drift (Fase B).

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
