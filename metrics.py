# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:31:22 2019

@author: Gabriel Hsu
"""


from medpy.metric.binary import assd, dc


def DiceCoeff(pred, gt):
    return dc(pred.to('cpu').numpy(), gt.to('cpu').numpy())
    
def ASSD(pred, gt):
    return assd(pred.to('cpu').numpy(), gt.to('cpu').numpy())



    