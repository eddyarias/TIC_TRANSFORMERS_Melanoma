import argparse
import os
import re
import sys
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running as a script from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.utils.config import load_config, get_mean_std
from src.data.transforms import get_val_transforms
from src.data.dataset import ListImageDataset
from src.models.build_model import build_model


def _list_pth_files(model_dir: str) -> List[str]:
    pths = []
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith('.pth'):
                pths.append(os.path.join(root, f))
    return pths


def _parse_epoch_number(name: str) -> Optional[int]:
    # Try patterns: epoch_XX, _XX.pth, phaseX_epoch_YY
    m = re.search(r'epoch[_-](\d+)', name)
    if m:
        return int(m.group(1))
    m2 = re.search(r'_(\d+)\.pth$', name)
    if m2:
        return int(m2.group(1))
    return None


def _pick_checkpoint(model_dir: str, checkpoint: Optional[str]) -> str:
    if checkpoint:
        # If absolute or relative path provided
        cand = checkpoint
        if not os.path.isabs(cand):
            cand = os.path.join(model_dir, cand)
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"Checkpoint not found: {cand}")
    pths = _list_pth_files(model_dir)
    if not pths:
        raise FileNotFoundError(f"No .pth files found under: {model_dir}")
    # Prefer highest epoch number; fallback to most recent mtime
    with_epochs = [(p, _parse_epoch_number(os.path.basename(p))) for p in pths]
    epoch_pths = [pe for pe in with_epochs if pe[1] is not None]
    if epoch_pths:
        epoch_pths.sort(key=lambda x: x[1], reverse=True)
        return epoch_pths[0][0]
    # Fallback: latest modified
    pths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pths[0]


def _build_loader(cfg) -> Tuple[DataLoader, int, bool]:
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = cfg['dataset']['image_size']
    mean, std = get_mean_std(cfg)
    path_mode = cfg['dataset'].get('path_mode', 'auto')
    val_tf = get_val_transforms(image_size, mean, std, cfg['augmentations']['val'])
    test_ds = ListImageDataset(cfg['dataset']['test_list'], val_tf, path_mode, cfg['dataset'].get('label_mapping'))
    loader = DataLoader(
        test_ds,
        batch_size=cfg['loader'].get('batch_size', 16),
        shuffle=False,
        num_workers=cfg['loader'].get('num_workers', 4),
        pin_memory=cfg['loader'].get('pin_memory', True),
    )
    num_classes = int(cfg['dataset'].get('num_classes', 2))
    binary_sigmoid = (num_classes == 1)
    return loader, num_classes, binary_sigmoid


def evaluate_and_save(model_dir: str, checkpoint: Optional[str] = None, out_path: Optional[str] = None):
    # 1) Load config
    cfg_path = os.path.join(model_dir, 'config.yml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yml not found at: {cfg_path}")
    cfg = load_config(cfg_path)

    # 2) Build data loader
    loader, num_classes, binary_sigmoid = _build_loader(cfg)

    # 3) Build model
    model = build_model(
        cfg['model']['backbone'],
        cfg['model'].get('pretrained', True),
        cfg['model']['head'],
        num_classes=num_classes,
    )

    # 4) Load checkpoint
    ckpt_path = _pick_checkpoint(model_dir, checkpoint)
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise RuntimeError('Unsupported checkpoint format: expected dict or dict with key "model"')

    # 5) Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            if num_classes == 1:
                probs = torch.sigmoid(logits.float()).view(-1, 1)
            elif logits.dim() == 2 and logits.size(1) == 2:
                probs = torch.softmax(logits.float(), dim=1)[:, 1].unsqueeze(1)  # store positive-class prob in one column
            else:
                probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    probs_cat = torch.cat(all_probs).numpy()
    labels_cat = torch.cat(all_labels).numpy().astype(np.int64)

    # 6) Prepare outputs
    scores = probs_cat  # use probabilities as scores for evaluation convenience
    norm_scores = scores  # keep same shape; downstream notebook handles this nicely
    effective_classes = 1 if (num_classes == 1) else scores.shape[1]

    # 7) Save npz
    if out_path is None:
        out_path = os.path.join(model_dir, 'scores.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        scores=scores,
        labels=labels_cat,
        norm_scores=norm_scores,
        binary_sigmoid=binary_sigmoid,
        effective_classes=np.array(effective_classes, dtype=np.int64),
        checkpoint=os.path.relpath(ckpt_path, start=model_dir)
    )
    print(f"Saved scores to: {out_path}")
    print(f"Scores shape: {scores.shape}; Labels: {labels_cat.shape}; binary_sigmoid={binary_sigmoid}; effective_classes={effective_classes}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint and save scores.npz for the test set.')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing config.yml and a subfolder with .pth checkpoints.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint file or relative path under model-dir. If omitted, picks the highest epoch or latest.')
    parser.add_argument('--out', type=str, default=None, help='Optional output path for scores.npz. Defaults to <model-dir>/scores.npz')
    args = parser.parse_args()

    evaluate_and_save(args.model_dir, args.checkpoint, args.out)


if __name__ == '__main__':
    main()
