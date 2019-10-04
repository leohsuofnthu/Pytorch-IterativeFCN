# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:31:22 2019

@author: Gabriel Hsu
"""


from medpy.metric.binary import assd, dc
import torch
import numpy as np

def DiceCoeff(pred, gt):
    return dc(pred.to('cpu').numpy(), gt.to('cpu').numpy())
    
def ASSD(pred, gt):
    return assd(pred.to('cpu').numpy(), gt.to('cpu').numpy())



#%%Test 
a= torch.zeros((5,5))
b= torch.zeros((5,5))

print(DiceCoeff(a,b))