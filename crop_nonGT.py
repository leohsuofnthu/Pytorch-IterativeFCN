# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:41:30 2019

@author: Gabriel Hsu
"""
import os
import numpy as np
import SimpleITK as sitk

path_data = "./isotropic_dataset/"
out_path ="./crop_isotropic_dataset/"

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

#%%
crop_unref_vert(path_data, out_path, 'train')
crop_unref_vert(path_data, out_path, 'test')
