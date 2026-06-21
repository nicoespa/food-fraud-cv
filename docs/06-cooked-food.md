# 06 — Comida cocida (dominio delivery real) + multi-generador + LOGO + adversarial

Extiende el proyecto del dominio frutas/verduras al **dominio real de delivery**: comida
cocida (pizza, hamburguesa, hot dog, tacos, papas, etc.).

## Por qué autenticidad binaria
Food-101 (`ethz/food101`) da comida cocida real y abundante para `genuine`, y de ahí
fabricamos `fake-damaged` con difusión. Pero **no existe un dataset público de comida
cocida realmente dañada a escala** (pizza con moho real, etc.) — verificado. Por eso el
experimento cocido se encara como **autenticidad binaria**:

- `genuine` = Food-101 real (categorías delivery: pizza, hamburger, hot_dog, tacos, ...).
- `fake-damaged` = esas fotos editadas con difusión para simular daño.
- **Pregunta:** ¿la foto del pedido es auténtica o manipulada por AI?

(En producción / Rappi, la clase "dañada real" la genera el backoffice con fotos reales de
reclamos — ver `docs/05`.)

## Multi-generador (familias distintas)
`src/generation/diffusion.py` ahora soporta 3 familias:
- **inpaint** (SD 1.5 inpainting) — edita región enmascarada.
- **instruct** (InstructPix2Pix) — edita por instrucción en lenguaje natural.
- **classic** (PIL copy-move + overlay) — edición NO-AI tipo app/Photoshop.

Generadores: `sd-mold`, `sd-rot` (inpaint), `ip2p-rot` (instruct), `classic-splice`
(classic), `sd-fungus` (inpaint, **held-out** solo en test).

## Evaluaciones nuevas
- **Leave-One-Generator-Out (LOGO)** (`src/evaluation/logo.py`): para cada generador, se
  entrena sin él y se mide TPR sobre sus fakes (familia no vista). Es la medida rigurosa de
  generalización cross-generator (el problema #1 de FraudBench). Validado en el slice
  sintético: el held-out cae (p.ej. sd-inpaint 0.63 TPR vs 0.95 en otros).
- **Robustez adversarial (FGSM)** (`src/detection/adversarial.py`): perturba los fakes para
  evadir el CNN y mide cuánto cae la TPR a distintos epsilon.

## Correr (Colab GPU)
```bash
# prueba chica (~5-8 min): valida Food-101 + difusión + fine-tune + LOGO + adversarial
python scripts/run_cooked.py --config configs/cooked.yaml --n 30 --steps 8 --adversarial
# informativa (~25-35 min en T4): n=150, steps=25, resnet50
python scripts/run_cooked.py --config configs/cooked.yaml --adversarial
```
Artefactos: `results/experiment_cooked.json` / `.csv`. Notebook: celdas de "Comida cocida".

*Generado 2026-06-21. Datos 100% públicos (Food-101 + difusión). Cero datos Rappi.*
