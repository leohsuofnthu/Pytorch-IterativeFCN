import numpy as np
from medpy.metric.binary import assd, dc


def DiceCoeff(pred, gt):
    pred = pred.to('cpu').numpy()
    gt = gt.to('cpu').numpy()

    # if gt is all zero (use inverse to count)
    if np.count_nonzero(gt) == 0:
        gt = gt + 1
        pred = 1 - pred

    return dc(pred, gt)


def ASSD(pred, gt):
    return assd(pred.to('cpu').numpy(), gt.to('cpu').numpy())
