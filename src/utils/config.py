import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_mean_std(cfg: Dict[str, Any]):
    ds = cfg.get('dataset', {})
    if ds.get('use_imagenet_norm', True):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return ds.get('mean', [0.485, 0.456, 0.406]), ds.get('std', [0.229, 0.224, 0.225])