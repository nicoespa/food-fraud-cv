# 03 — Pipeline REAL (full local: datos reales + difusión + fine-tune)

A diferencia del slice sintético (`docs/02`), esto usa **fotos reales** y **difusión real**.
No es un validador: es el sistema funcionando sobre datos reales.

## Datos (100% públicos, cero Rappi)
- **Fuente:** [`Densu341/Fresh-rotten-fruit`](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit)
  (HF, 30.4K imágenes reales de frutas/verduras, labels `fresh*` / `rotten*`).
- `fresh*` → **genuine-undamaged** (comida real OK).
- `rotten*` → **genuine-damaged** (comida realmente podrida).
- **fake-damaged** = imágenes `fresh` editadas con **difusión real** para *simular* podredumbre
  (el fraude). Modelo: [`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting).
- Descarga por **streaming**: solo bajamos las imágenes que muestreamos, no los 3 GB.

## Generadores de difusión (cross-generator real)
`src/generation/diffusion.py` define variantes con prompt/máscara/guidance distintos:
- `sd-mold` (manchas de moho), `sd-rot` (podredumbre marrón) → vistos en train.
- `sd-fungus` → **held-out**, solo en test (prompt y máscara de otra "familia") → mide
  generalización a una forma de edición no vista. Para un held-out aún más fuerte, usar un
  modelo distinto (p.ej. `kandinsky-community/kandinsky-2-2-decoder-inpaint`).

## Modelos comparados
1. **zero-shot-aigc** — detector AIGC genérico (Tesis 1: debería inundar de FPs).
2. **forensic-gbm** — features forensics + GBM calibrado.
3. **finetune-<backbone>** — **fine-tune real** de ResNet/ViT (timm) sobre las 3 clases.

Evaluación idéntica al slice: PR/ROC-AUC, TPR@FPR, ECE, costo D0/D1/D2, breakdown por
generador (con held-out), bootstrap CI. (Reutiliza `src/evaluation` y `src/decision`.)

## Requisitos
- `uv sync --extra real` (~6-8 GB con los pesos de SD 1.5).
- **Disco:** dejar ≥25 GB libres recomendado. **GPU/MPS** (Apple Silicon ok).
- La **difusión es el cuello de botella**: ~3-6 s/imagen en MPS → cientos de imágenes =
  decenas de minutos. Conviene correr en background.

## Correr
```bash
uv sync --extra real
# corpus real + difusión + fine-tune + evaluación (todo end-to-end):
uv run --extra real python scripts/run_real.py --config configs/real.yaml
# prueba chica primero (rápida) para validar el flujo:
uv run --extra real python scripts/run_real.py --n-per-group 30
# re-evaluar sin regenerar el corpus:
uv run --extra real python scripts/run_real.py --reuse-corpus
```
Artefactos: `results/experiment_real.json` y `.csv`. Corpus en `data/generated_real/` (gitignored).

## Escalar
Subir `real.n_per_group`, agregar generadores (más variantes / un 2º modelo de difusión
como held-out), probar `vit_small_patch16_224` como backbone, y reportar el gap
cross-generator (que con difusión real SÍ debería aparecer, a diferencia del slice sintético).

*Documentado el 2026-06-19.*
