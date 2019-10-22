# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:28:51 2019

@author: Gabriel Hsu

ref: https://github.com/SimpleITK/SimpleITK/issues/561
     Spiral CT of the Abdomen

"""

"""
    Load whole dataset and resample to 1mm*1mm*1mm through Nearest Neighbor
    
"""

import os

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

#%% Functions for isotorpic resampling
def isotropic_resampler(input_path, output_path):
    
    raw_img = sitk.ReadImage(input_path)
    new_spacing = [1,1,1]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(raw_img.GetDirection())
    resampler.SetOutputOrigin(raw_img.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    
    
    orig_size = np.array(raw_img.GetSize(), dtype=np.int)
    orig_spacing = raw_img.GetSpacing()
    new_size = np.array([x*(y/z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resampler.SetSize(new_size)
    
    isotropic_img = resampler.Execute(raw_img)
    sitk.WriteImage(isotropic_img, output_path, True)


#%% Function for cropping
def z_mid(mask, chosen_vert):
    #print(i, ' normal sample')
    indices= np.nonzero(mask == chosen_vert)
    lower = [np.min(i) for i in indices]
    upper = [np.max(i) for i in indices]
    
    return int((lower[0]+upper[0])/2)

    
def findZRange(img, mask):
    #list available vertebrae
    verts = np.unique(mask)
    print(verts)
    
    vert_low = verts[1]
    vert_up = verts[-1]
    
    print(vert_low, vert_up)
    z_range = [z_mid(mask, vert_low), z_mid(mask, vert_up)]
    print(z_range)
    return z_range

def crop_unref_vert(path, out_path, subset):
    img_path = os.path.join(path, subset, 'img')
    mask_path = os.path.join(path, subset, 'seg')
    weight_path = os.path.join(path, subset, 'weight')
    img_names =  [f for f in os.listdir(img_path) if f.endswith('.mhd')]
    
    
    for img_name in img_names:
        print('Processing ', img_name)
        img_name =  img_name
        mask_name = img_name.split('.')[0]+'_label.mhd'
        weight_name = img_name.split('.')[0]+'_weight.nrrd'
        
        img_file = os.path.join(img_path,  img_name)
        mask_file = os.path.join(mask_path, mask_name)
        weight_file = os.path.join(weight_path, weight_name)
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        weight = sitk.GetArrayFromImage(sitk.ReadImage(weight_file))
        
        
        z_range = findZRange(img, mask)
            
        sitk.WriteImage(sitk.GetImageFromArray(img[z_range[0]:z_range[1],:,:]), os.path.join(out_path,subset,'img',img_name), True)
        sitk.WriteImage(sitk.GetImageFromArray(mask[z_range[0]:z_range[1],:,:]),  os.path.join(out_path, subset,'seg', mask_name), True)
        sitk.WriteImage(sitk.GetImageFromArray(weight[z_range[0]:z_range[1],:,:]),  os.path.join(out_path, subset,'weight', weight_name), True)

#%% Function for calculating the weight
"""
Code taken from the kind answer of author Dr.Lessman (nikolas.lessmann@radboudumc.nl)
"""
def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=6):
    mask = np.asarray(mask)
    distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)    
    weights = alpha + beta*np.exp(-(distance_to_border**2/omega**2))
    return np.asarray(weights, dtype='float32')

#%%Start preprocessing
    
#Resampling   
root_path = "D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/CSI_dataset"
output_path = "D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/isotropic_dataset"

files = [x for x in os.listdir(os.path.join(root_path)) if 'raw' not in x]
for f in files:
    print('Resampling ' + f + '...')
    isotropic_resampler(os.path.join(root_path, f), os.path.join(output_path, f))


#Pre-calculate of weight of masks
mask_path =  'D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/isotropic_dataset/train/seg'
weight_path = 'D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/isotropic_dataset/train/weight'

for f in [f for f in os.listdir(mask_path) if f.endswith('.mhd')]:
  seg_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path,f)))
  weight = compute_distance_weight_matrix(seg_mask)
  sitk.WriteImage(sitk.GetImageFromArray(weight), os.path.join(weight_path, f.split('_')[0]+'_weight.nrrd'), True)
  print(f)
  
mask_path =  'D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/isotropic_dataset/isotropic_dataset/test/seg'
weight_path = 'D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/isotropic_dataset/test/weight'

for f in [f for f in os.listdir(mask_path) if f.endswith('.mhd')]:
  seg_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path,f)))
  weight = compute_distance_weight_matrix(seg_mask)
  sitk.WriteImage(sitk.GetImageFromArray(weight), os.path.join(weight_path, f.split('_')[0]+'_weight.nrrd'), True)
  print(f)

#Crop
path_data = "D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/isotropic_dataset"
out_path ="D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/crop_isotropic_dataset/"
crop_unref_vert(path_data, out_path, 'train')
crop_unref_vert(path_data, out_path, 'test')



    



