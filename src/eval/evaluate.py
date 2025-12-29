import argparse
import torch
from torch.utils.data import DataLoader

from ..utils.config import load_config, get_mean_std
from ..data.transforms import get_val_transforms
from ..data.dataset import ListImageDataset
from ..models.build_model import build_model
from ..train.metrics import compute_metrics
from tqdm import tqdm


def main(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = cfg['dataset']['image_size']
    mean, std = get_mean_std(cfg)
    path_mode = cfg['dataset'].get('path_mode', 'auto')

    val_tf = get_val_transforms(image_size, mean, std, cfg['augmentations']['val'])
    test_ds = ListImageDataset(cfg['dataset']['test_list'], val_tf, path_mode, cfg['dataset'].get('label_mapping'))
    test_loader = DataLoader(test_ds, batch_size=cfg['loader']['batch_size'], shuffle=False, num_workers=cfg['loader']['num_workers'], pin_memory=cfg['loader'].get('pin_memory', True))

    model = build_model(cfg['model']['backbone'], cfg['model']['pretrained'], cfg['model']['head'], num_classes=cfg['dataset']['num_classes'])
    # Safely load checkpoints: support pure state_dict files and dict checkpoints
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif isinstance(ckpt, dict):
        # Assume this is a raw state_dict
        model.load_state_dict(ckpt)
    else:
        raise RuntimeError('Unsupported checkpoint format: expected dict or dict with key "model"')
    model.to(device)
    model.eval()

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Test', leave=False):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.detach().cpu())
            all_targets.append(labels.detach().cpu())
    logits_cat = torch.cat(all_logits)
    targets_cat = torch.cat(all_targets)
    metrics = compute_metrics(logits_cat, targets_cat, threshold=cfg['evaluation'].get('threshold', 0.5))
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)