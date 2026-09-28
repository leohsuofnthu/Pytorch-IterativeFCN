import numpy as np
import torch


def Segloss(pred, target, weight):
    """Soft FP / FN counts (paper): Σ ω·(1-t)·p  and  Σ ω·t·(1-p)."""
    FP = torch.sum(weight * (1 - target) * pred)
    FN = torch.sum(weight * (1 - pred) * target)
    return FP, FN


def _dc(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def DiceCoeff(pred, gt):
    pred = pred.to('cpu').numpy()
    gt = gt.to('cpu').numpy()

    # if gt is all zero (use inverse to count)
    if np.count_nonzero(gt) == 0:
        gt = gt + 1
        pred = 1 - pred

    return _dc(pred, gt)
