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

## Correr — recomendado: Colab GPU (`notebooks/food_fraud_cv_colab.ipynb`)
Localmente en MPS la difusión es lenta y a escala chica el resultado es degenerado
(clases triviales de separar → todo ~1.0). Para números informativos hace falta **escala**
(n grande + más pasos de difusión = fakes sutiles), y eso corre rápido en **Colab GPU**.

```bash
# Validación rápida del flujo (~5 min): n chico + pocos pasos de difusión
python scripts/run_real.py --config configs/real.yaml --n-per-group 30 --steps 8

# CORRIDA INFORMATIVA (escala real; ~20-30 min en una T4):
#   usa configs/real.yaml → n_per_group=150, steps=25 (sutiles), backbone resnet50
python scripts/run_real.py --config configs/real.yaml
```
Artefactos: `results/experiment_real.json` y `.csv`. Corpus en `data/generated_real/`.

> **Por qué la escala importa:** con pocos pasos de difusión los fakes tienen artefactos
> obvios y hasta el zero-shot los detecta (todo 1.0). Con 25 pasos los fakes son sutiles →
> recién ahí se ve la diferencia entre el detector genérico, el forense y el fine-tune, y
> aparece (o no) el gap cross-generator. Test ≥ ~60 imgs (split 60/20/20) para métricas estables.
Artefactos: `results/experiment_real.json` y `.csv`. Corpus en `data/generated_real/` (gitignored).

## Escalar
Subir `real.n_per_group`, agregar generadores (más variantes / un 2º modelo de difusión
como held-out), probar `vit_small_patch16_224` como backbone, y reportar el gap
cross-generator (que con difusión real SÍ debería aparecer, a diferencia del slice sintético).

*Documentado el 2026-06-19.*
