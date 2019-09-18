# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:31:22 2019

@author: Gabriel Hsu
"""

import torch
from torch.utils.data import Dataset, DataLoader
from dataset import CSI_Dataset

from medpy.metric.binary import assd, dc

"""
code from https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
"""

def DiceCoeff(pred, gt):
    return dc(pred.to('cpu').numpy(), gt.to('cpu').numpy())
    
def ASSD(pred, gt):
    return assd(pred.to('cpu').numpy(), gt.to('cpu').numpy())


#%% Test purpose

train_dataset = CSI_Dataset('D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/isotropic_dataset')

dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True)

img_patch, ins_patch, gt_patch, c_label = next(iter(dataloader_train))


print(DiceCoeff(gt_patch, gt_patch))
print(ASSD(gt_patch, gt_patch))
    