import numpy as np
import torch
from medpy.metric.binary import assd, dc


def Segloss(pred, target, weight):
    FP = torch.sum(weight * (1 - target) * pred)
    FN = torch.sum(weight * (1 - pred) * target)
    return FP, FN


def DiceCoeff(pred, gt):
    pred = pred.to('cpu').numpy()
    gt = gt.to('cpu').numpy()

    # if gt is all zero (use inverse to count)
    if np.count_nonzero(gt) == 0:
        gt = gt + 1
        pred = 1 - pred

    return dc(pred, gt)