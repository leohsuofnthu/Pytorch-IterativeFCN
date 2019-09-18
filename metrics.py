# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:31:22 2019

@author: Gabriel Hsu
"""

from torch.utils.data import Dataset, DataLoader
from dataset import CSI_Dataset

from medpy.metric.binary import assd, dc
from sklearn.metrics import confusion_matrix


def DiceCoeff(pred, gt):
    return dc(pred.to('cpu').numpy(), gt.to('cpu').numpy())
    
def ASSD(pred, gt):
    return assd(pred.to('cpu').numpy(), gt.to('cpu').numpy())

def iterativeFCNLoss(weight, pred, gt, ):
    l = 0.1
    FP = weight*(1-gt)*pred
    FN = weight*gt*(1-pred)

#%% Test purpose

train_dataset = CSI_Dataset('D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/isotropic_dataset')

dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True)

img_patch, ins_patch, gt_patch, c_label = next(iter(dataloader_train))


print(DiceCoeff(gt_patch, gt_patch))
print(ASSD(gt_patch, gt_patch))
    