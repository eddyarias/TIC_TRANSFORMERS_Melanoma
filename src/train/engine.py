import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import compute_metrics
from tqdm import tqdm


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.float64)
    for _, labels in tqdm(train_loader, desc='ClassWeights', leave=False):
        for c in [0, 1]:
            counts[c] += (labels == c).sum().item()
    total = counts.sum()
    weights = total / (2 * counts)
    print(f'Class counts: {counts.tolist()}, Weights: {weights.tolist()}')
    return weights.float()


def make_loss(weights: torch.Tensor, use_bce: bool = False):
    if use_bce:
        pos_weight = weights[1] / weights[0]
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return nn.CrossEntropyLoss(weight=weights)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, loss_fn: nn.Module, device: torch.device, desc: str = 'Train', accum_steps: int = 1) -> Tuple[float, Dict[str, float]]:
    model.train()
    running_loss = 0.0
    all_logits = []
    all_targets = []
    accum_steps = max(1, int(accum_steps))
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        images = images.to(device)
        labels = labels.to(device)
        # Use new torch.amp autocast API
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(images)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                loss = loss_fn(logits.squeeze(), labels.float())
            else:
                loss = loss_fn(logits, labels)
            # Scale loss for gradient accumulation
            loss = loss / accum_steps
        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    # Flush remaining grads if loop ended mid-accumulation
    if (len(loader) % accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    avg_loss = running_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    metrics = compute_metrics(logits_cat, targets_cat)
    return avg_loss, metrics


def validate_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, desc: str = 'Val') -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []
    # Use inference_mode for faster evaluation (no grad state allocations)
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Mirror train's autocast for consistency/perf on GPU
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    loss = loss_fn(logits.squeeze(), labels.float())
                else:
                    loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(labels.detach().cpu())
    avg_loss = running_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    metrics = compute_metrics(logits_cat, targets_cat)
    return avg_loss, metrics


def save_weights(model: nn.Module, path: str):
    """Save only the model weights to a .pth file using legacy serialization.

    This avoids PyTorch's zip writer and is more stable on WSL/NTFS.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict_cpu, path, _use_new_zipfile_serialization=False)


def load_weights(model: nn.Module, path: str, strict: bool = True):
    """Load model weights from a .pth file."""
    # Prefer safe loading of weights only (PyTorch 2.5+),
    # and fall back for older versions that don't support the flag.
    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict, strict=strict)