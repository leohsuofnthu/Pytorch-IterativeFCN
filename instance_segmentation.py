# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:59:48 2019

@author: Gabriel Hsu
"""
import os

import numpy as np
from skimage.util import view_as_blocks

import SimpleITK as sitk

import torch
from model import  iterativeFCN

test_img = 'D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/isotropic_dataset/test/img'


f = [f for f in os.listdir(test_img) if f.endswith('.mhd')]

#read image
test = f[0]
test_numpy = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(test_img, test)))

#convert to HU
test_numpy = test_numpy*1.0 - 1000.0
test_ins = np.zeros_like(test_numpy) 
test_gt = np.zeros_like(test_numpy)


def extract(img, x, y, z, patch_size):
    offset = int(patch_size/2)
    return img[x-offset:x+offset, y-offset:y+offset, z-offset:z+offset]


#slide window with initial center coord
patch_size = 128
x = int(patch_size/2)
y = int(patch_size/2)
z = int(patch_size/2)

last_x = int(test_numpy.shape[1]-x)  
last_y = int(test_numpy.shape[2]-y) 
last_z = int(test_numpy.shape[0]-z) 

step_size = 64

while True:
    
    print('new center:',(z,x,y))
    
    patch = extract(test_numpy, x, y, z, 128)

    #to tensor
    
    #import to model
    
    #get result S, C
    
    #calculate contained vertebrae volume
    
    #move to center if > 1000
    
    #calculate center
    
    
    
    

    #window slide
    if x + step_size <= last_x:
        x = x + step_size
    else:
        x =  int(patch_size/2)
        if y + step_size <= last_y:
            y = y + step_size
        else:
            y =  int(patch_size/2)
            if z + step_size <= last_z:
                z = z + step_size
            else:
                break
        
        
#convert to tensor


# Create FCN
model = iterativeFCN().to('cuda')
#model.load_state_dict(torch.load('./drive/My Drive/IterativeFCN_best.pth'))

#instance segmentation
