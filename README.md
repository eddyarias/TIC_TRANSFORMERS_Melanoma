# Melanoma Binary Classification Pipeline

End-to-end training and evaluation for melanoma classification (0=benign, 1=malignant) using timm backbones (ViT, Swin/Swin V2, DINOv2), Albumentations, class weighting, two-phase training, TensorBoard logging, and checkpointing.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
```

If using WSL paths in lists (e.g., `/mnt/c/...`) but running on Windows Python, ensure `dataset.path_mode: windows` in `configs/config.yml`.

## Train

```bash
python -m src.train.train --config configs/config.yml
```

## Evaluate

```bash
python -m src.eval.evaluate --config configs/config.yml --checkpoint checkpoints/phase2_epoch_19.ckpt
```

## Data Lists

Each line in `lists/train.txt`, `lists/validation.txt`, `lists/test.txt` must be:

```
/path/to/image.jpg <label>
```

Where `<label>` in `{0,1}`. Labels mapping can be provided via `lists/label_mapping.json`.