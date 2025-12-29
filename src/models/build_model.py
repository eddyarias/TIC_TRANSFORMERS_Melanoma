from typing import List
import torch.nn as nn
import timm


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], dropout: float = 0.5, activation: str = 'gelu', num_classes: int = 2):
        super().__init__()
        act = nn.GELU() if activation.lower() == 'gelu' else nn.ReLU(inplace=True)
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), act, nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(backbone_name: str, pretrained: bool, head_cfg: dict, num_classes: int = 2):
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
    in_features = model.num_features
    head = ClassificationHead(in_features, head_cfg.get('hidden_dims', [512, 128]), dropout=head_cfg.get('dropout', 0.5), activation=head_cfg.get('activation', 'gelu'), num_classes=num_classes)

    class Wrapper(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            feats = self.backbone(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            return self.head(feats)

    return Wrapper(model, head)


def freeze_backbone(model: nn.Module):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, n_blocks: int):
    """Try to unfreeze last N blocks for common timm backbones (ViT/Swin/DINOv2)."""
    # Attempt common attributes
    backbone = model.backbone
    candidates = []
    for attr in ['blocks', 'stages', 'layers']:
        if hasattr(backbone, attr):
            candidates = getattr(backbone, attr)
            break
    if isinstance(candidates, nn.ModuleList):
        to_unfreeze = candidates[-n_blocks:]
        for block in to_unfreeze:
            for p in block.parameters():
                p.requires_grad = True
    else:
        # Fallback: unfreeze entire backbone
        for p in backbone.parameters():
            p.requires_grad = True