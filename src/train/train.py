import os
from datetime import datetime
import argparse
import re
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from ..utils.config import load_config, get_mean_std
from ..data.transforms import get_train_transforms, get_val_transforms
from ..data.dataset import ListImageDataset
from ..models.build_model import build_model, freeze_backbone, unfreeze_last_blocks
from .engine import compute_class_weights, make_loss, train_one_epoch, validate_one_epoch, save_weights, load_weights


def main(config_path: str):
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = cfg['dataset']['image_size']
    mean, std = get_mean_std(cfg)
    path_mode = cfg['dataset'].get('path_mode', 'auto')

    train_tf = get_train_transforms(image_size, mean, std, cfg['augmentations']['train'])
    val_tf = get_val_transforms(image_size, mean, std, cfg['augmentations']['val'])

    train_ds = ListImageDataset(cfg['dataset']['train_list'], train_tf, path_mode, cfg['dataset'].get('label_mapping'))
    val_ds = ListImageDataset(cfg['dataset']['val_list'], val_tf, path_mode, cfg['dataset'].get('label_mapping'))
    test_ds = ListImageDataset(cfg['dataset']['test_list'], val_tf, path_mode, cfg['dataset'].get('label_mapping'))

    train_loader = DataLoader(train_ds, batch_size=cfg['loader']['batch_size'], shuffle=cfg['loader']['shuffle_train'], num_workers=cfg['loader']['num_workers'], pin_memory=cfg['loader'].get('pin_memory', True))
    val_loader = DataLoader(val_ds, batch_size=cfg['loader']['batch_size'], shuffle=False, num_workers=cfg['loader']['num_workers'], pin_memory=cfg['loader'].get('pin_memory', True))
    test_loader = DataLoader(test_ds, batch_size=cfg['loader']['batch_size'], shuffle=False, num_workers=cfg['loader']['num_workers'], pin_memory=cfg['loader'].get('pin_memory', True))

    model = build_model(cfg['model']['backbone'], cfg['model']['pretrained'], cfg['model']['head'], num_classes=cfg['dataset']['num_classes'])
    model.to(device)

    freeze_backbone(model)

    if cfg['training'].get('use_precomputed_class_weights') and cfg['training'].get('precomputed_class_weights'):
        class_weights = torch.tensor(cfg['training']['precomputed_class_weights'], dtype=torch.float).to(device)
    else:
        class_weights = compute_class_weights(train_loader).to(device)
    loss_fn = make_loss(class_weights, use_bce=False)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg['optimizer']['lr_head']),
        weight_decay=float(cfg['optimizer']['weight_decay'])
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(cfg['training']['epochs_phase1']))) if cfg['scheduler']['type'] == 'cosine' else None
    # Use new torch.amp GradScaler API
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and cfg['training'].get('mixed_precision', True)))

    # Create a single run directory with timestamp for logs and checkpoints
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_out = cfg['logging'].get('output_dir', 'outputs')
    run_dir = os.path.join(base_out, run_id)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Optional: load initial weights if provided and configure resume behavior
    skip_phase1 = False
    resume_global_epoch = None  # global epoch index used in filenames
    if cfg['training'].get('resume_checkpoint'):
        resume_path = cfg['training']['resume_checkpoint']
        if os.path.exists(resume_path):
            print(f"Info: Loading checkpoint weights from {resume_path}")
            load_weights(model, resume_path)
            base_name = os.path.basename(resume_path)
            # Parse global epoch number from filename pattern: phase1_epoch_{N}.pth or phase2_epoch_{N}.pth
            m = re.search(r"phase(\d+)_epoch_(\d+)\.pth", base_name)
            if m:
                phase_id = int(m.group(1))
                resume_global_epoch = int(m.group(2))
                print(f"Info: Resuming from phase {phase_id}, global epoch {resume_global_epoch}.")
                if phase_id == 2:
                    skip_phase1 = True
                    print('Info: Detected phase2 checkpoint — skipping Phase 1 head training.')
            else:
                # Fallback: if name contains 'phase2' but no epoch number extracted
                if 'phase2' in base_name:
                    skip_phase1 = True
                    print('Info: Detected phase2 checkpoint — skipping Phase 1 head training.')
        else:
            print(f"Warning: resume_checkpoint path not found: {resume_path}")

    # Phase 1: train head only (optional skip when resuming from phase2)
    if not skip_phase1:
        for epoch in range(0, int(cfg['training']['epochs_phase1'])):
            print(f"\n=== Phase 1 (Head training) — Epoch {epoch+1}/{int(cfg['training']['epochs_phase1'])} ===")
            train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, desc=f"Phase1-Train E{epoch+1}", accum_steps=int(cfg['training'].get('gradient_accumulation_steps', 1)))
            val_loss, val_metrics = validate_one_epoch(model, val_loader, loss_fn, device, desc=f"Phase1-Val E{epoch+1}")
            # Console summary per epoch (requested: loss and acc for train/val)
            print(f"Phase 1 — E{epoch+1}: Train loss {train_loss:.4f}, acc {train_metrics.get('accuracy', float('nan')):.4f} | "
                  f"Val loss {val_loss:.4f}, acc {val_metrics.get('accuracy', float('nan')):.4f}")
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'val/{k}', v, epoch)
            if scheduler:
                scheduler.step()
            if cfg['logging'].get('save_every_epoch', True):
                save_weights(model, os.path.join(run_dir, f'phase1_epoch_{epoch}.pth'))
    else:
        print('Info: Skipping Phase 1 loop due to resume settings.')

    # Phase 2: unfreeze last blocks and fine-tune with lower LR
    unfreeze_last_blocks(model, cfg['training'].get('unfrozen_blocks', 4))
    # Enable grad checkpointing to reduce memory on ViT/DINOv2 if supported
    if hasattr(model.backbone, 'set_grad_checkpointing'):
        try:
            model.backbone.set_grad_checkpointing(True)
            print('Info: Enabled gradient checkpointing on backbone to reduce memory usage.')
        except Exception:
            pass
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg['optimizer']['lr_finetune']),
        weight_decay=float(cfg['optimizer']['weight_decay'])
    )
    # Determine phase 2 starting global epoch when resuming
    phase1_epochs = int(cfg['training']['epochs_phase1'])
    phase2_epochs = int(cfg['training']['epochs_phase2'])
    if resume_global_epoch is not None and skip_phase1:
        # Start from the next epoch after the checkpoint
        phase2_start_global_epoch = resume_global_epoch + 1
        # Clamp to at least the first phase2 global epoch
        phase2_start_global_epoch = max(phase1_epochs, phase2_start_global_epoch)
        remaining_phase2 = (phase1_epochs + phase2_epochs) - phase2_start_global_epoch
        print(f"Info: Phase 2 will resume at global epoch {phase2_start_global_epoch} with {remaining_phase2} epochs remaining.")
    else:
        phase2_start_global_epoch = phase1_epochs
        remaining_phase2 = phase2_epochs

    # Adjust scheduler length to remaining fine-tune epochs if resuming
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(remaining_phase2))) if cfg['scheduler']['type'] == 'cosine' else None

    for epoch in range(phase2_start_global_epoch, phase1_epochs + phase2_epochs):
        phase2_epoch = epoch - int(cfg['training']['epochs_phase1']) + 1
        print(f"\n=== Phase 2 (Fine-tuning) — Epoch {phase2_epoch}/{int(cfg['training']['epochs_phase2'])} ===")
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, desc=f"Phase2-Train E{phase2_epoch}", accum_steps=int(cfg['training'].get('gradient_accumulation_steps', 1)))
        val_loss, val_metrics = validate_one_epoch(model, val_loader, loss_fn, device, desc=f"Phase2-Val E{phase2_epoch}")
        # Console summary per epoch (requested: loss and acc for train/val)
        print(f"Phase 2 — E{phase2_epoch}: Train loss {train_loss:.4f}, acc {train_metrics.get('accuracy', float('nan')):.4f} | "
              f"Val loss {val_loss:.4f}, acc {val_metrics.get('accuracy', float('nan')):.4f}")
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'val/{k}', v, epoch)
        if scheduler:
            scheduler.step()
        if cfg['logging'].get('save_every_epoch', True):
            save_weights(model, os.path.join(run_dir, f'phase2_epoch_{epoch}.pth'))

    # Final evaluation on test
    test_loss, test_metrics = validate_one_epoch(model, test_loader, loss_fn, device)
    print('Test metrics:', test_metrics)
    writer.add_hparams({'backbone': cfg['model']['backbone']}, {'hparam/test_accuracy': test_metrics['accuracy'], 'hparam/test_auc': test_metrics['roc_auc']})
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yml')
    args = parser.parse_args()
    main(args.config)