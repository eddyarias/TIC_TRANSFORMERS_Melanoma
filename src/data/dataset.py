import json
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.paths import normalize_path


class ListImageDataset(Dataset):
    def __init__(self, list_path: str, transform, path_mode: str = 'auto', label_mapping_path: str = None):
        self.samples: List[Tuple[str, int]] = []
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.rsplit(' ', 1)
                self.samples.append((normalize_path(path, path_mode), int(label)))
        self.transform = transform
        self.label_map = None
        if label_mapping_path:
            try:
                with open(label_mapping_path, 'r', encoding='utf-8') as lf:
                    self.label_map = json.load(lf)
            except Exception:
                self.label_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Create a black image fallback to avoid crashing; user should fix paths.
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img)
        img_t = augmented['image']
        img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).float()
        label_t = torch.tensor(label, dtype=torch.long)
        return img_t, label_t