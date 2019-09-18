# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:28:51 2019

@author: Gabriel Hsu

ref: https://github.com/SimpleITK/SimpleITK/issues/561

"""

"""
    Load whole dataset and resample to 1mm*1mm*1mm through Nearest Neighbor
    
"""

import os

import numpy as np
import SimpleITK as sitk


root_path = "D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/CSI_dataset"
output_path = "D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/isotropic_dataset"

files = [x for x in os.listdir(os.path.join(root_path)) if 'raw' not in x]


#%% Resample train/test image

def isotropic_resampler(input_path, output_path):
    
    raw_img = sitk.ReadImage(input_path)
    new_spacing = [1,1,1]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator = sitk.sitkNearestNeighbor
    resampler.SetOutputDirection = raw_img.GetDirection()
    resampler.SetOutputOrigin = raw_img.GetOrigin()
    resampler.SetOutputSpacing(new_spacing)
    
    orig_size = np.array(raw_img.GetSize(), dtype=np.int)
    orig_spacing = raw_img.GetSpacing()
    new_size = np.array([x*(y/z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resampler.SetSize(new_size)
    
    isotropic_img = resampler.Execute(raw_img)
    sitk.WriteImage(isotropic_img, output_path, True)


#%%
       
for f in files:
    print('Resampling ' + f + '...')
    isotropic_resampler(os.path.join(root_path, f), os.path.join(output_path, f))
    
#%%

