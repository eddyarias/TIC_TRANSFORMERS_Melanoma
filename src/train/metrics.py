from typing import Dict
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    # Support both binary logits ([N]) and 2-class logits ([N, 2])
    if logits.dim() == 2 and logits.size(1) == 2:
        probs_t = torch.softmax(logits.float(), dim=1)[:, 1]
    else:
        probs_t = torch.sigmoid(logits.float()).squeeze()
    probs = probs_t.detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int).squeeze()
    y_pred = (probs >= threshold).astype(int)
    acc = (y_pred == y_true).mean()
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': float(acc),
        'roc_auc': float(auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }