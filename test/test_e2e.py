"""End-to-end checks for IterativeFCN (no CSI dataset required)."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from iterativeFCN import IterativeFCN
from utils.metrics import DiceCoeff, Segloss

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / 'weights' / 'IterativeFCN_pretrained.pth'


@pytest.fixture(scope='module')
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_forward_shapes(device):
    model = IterativeFCN(num_channels=64).to(device)
    x = torch.randn(1, 2, 128, 128, 128, device=device)
    seg, cls = model(x)
    assert seg.shape == (1, 1, 128, 128, 128)
    assert cls.shape == (1,)
    assert torch.all((seg >= 0) & (seg <= 1))
    assert torch.all((cls >= 0) & (cls <= 1))


def test_sigmoid_binary_segmentation_not_softmax(device):
    """Paper predicts a binary current-instance mask (sigmoid), not multi-class softmax."""
    model = IterativeFCN(num_channels=64).to(device)
    x = torch.randn(2, 2, 128, 128, 128, device=device)
    seg, cls = model(x)
    assert seg.shape[1] == 1
    assert cls.shape[0] == 2


def test_pretrained_weights_load(device):
    if not PRETRAINED.is_file():
        pytest.skip('pretrained weights missing')
    model = IterativeFCN(num_channels=64).to(device)
    state = torch.load(PRETRAINED, map_location=device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        seg, cls = model(torch.zeros(1, 2, 128, 128, 128, device=device))
    assert seg.shape == (1, 1, 128, 128, 128)
    assert cls.shape == (1,)


def test_segloss_fp_fn_weighting(device):
    """λ·FP + FN with λ=0.1 — FN penalized more (paper)."""
    pred = torch.tensor([0.9, 0.1, 0.8], device=device)
    target = torch.tensor([0.0, 1.0, 1.0], device=device)
    weight = torch.ones_like(pred)
    FP, FN = Segloss(pred, target, weight)
    # FP: weight*(1-t)*p at idx0 -> 0.9; FN: weight*(1-p)*t at idx1 -> 0.9, idx2 -> 0.2
    assert torch.isclose(FP, torch.tensor(0.9, device=device))
    assert torch.isclose(FN, torch.tensor(1.1, device=device))
    loss = 0.1 * FP + FN
    assert loss > FN * 0.5


def test_dice_empty_gt():
    pred = torch.zeros(1, 1, 8, 8, 8)
    gt = torch.zeros_like(pred)
    # should not raise; returns a finite score
    score = DiceCoeff(pred, gt)
    assert np.isfinite(score)


def test_force_inside_img_axes():
    from utils.utils import force_inside_img
    # previously all axes incorrectly used shape[2]
    z_low, z_up = force_inside_img(10, 128, 200)
    assert z_low == 0 and z_up == 128
    x_low, x_up = force_inside_img(190, 128, 200)
    assert x_up == 200 and x_low == 72
