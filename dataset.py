# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:55:35 2019

@author: Gabriel Hsu
"""
from __future__ import print_function, division
import os 
from random import randint

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import SimpleITK as sitk

"""
The dataset of MICCAI 2014 Spine Challenge

"""

#%% Build the dataset 
class CSI_Dataset(Dataset):
    """xVertSeg Dataset"""
    
    def __init__(self, dataset_path, subset='train', linear_att=1.0, offset=1000.0):
        """
        Args:
            path_dataset(string): Root path to the whole dataset
            subset(string): 'train' or 'test' depend on which subset
        """
        
        self.dataset_path = dataset_path
        self.subset = subset
        self.linear_att = linear_att
        self.offset = offset
        
        
        self.img_path = os.path.join(dataset_path, subset, 'img')
        self.mask_path = os.path.join(dataset_path, subset, 'seg')
        
        
        self.img_names =  [f for f in os.listdir(self.img_path) if f.endswith('.mhd')]
        self.mask_names = [f for f in os.listdir(self.mask_path) if f.endswith('.mhd')]
     
    def __len__(self):
        return len(self.img_names)
    
    
    def __getitem__(self, idx):
        
        img_file = os.path.join(self.img_path, self.img_names[idx])
        mask_file =os.path.join(self.mask_path, self.mask_names[idx])
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        
        
        #linear transformation from 12bit reconstruction img to HU unit
        #depend on the original data (CSI data value is from 0 ~ 4095)
        img = img * self.linear_att - self.offset
        
        img_patch, ins_patch, gt_patch, c_label = extract_random_patch(img, 
                                                              mask)
        
        return img_patch, ins_patch, gt_patch, c_label
        
    
#%% Extract the 128*128*128 patch
def extract_random_patch(img, mask, patch_size=128):
    
    #list available vertebrae
    verts = np.unique(mask)
    print('mask values:', verts)
    chosen_vert = verts[randint(1, len(verts)-1)]
    print('chosen_vert:', chosen_vert)
    
    #create corresponde instance memory and ground truth
    ins_memory = np.copy(mask)
    ins_memory[ins_memory >= chosen_vert] = 0
    ins_memory[ins_memory > 0] = 1
#    print(np.unique(ins_memory))
    
    gt = np.copy(mask)
    gt[gt != chosen_vert] = 0
    gt[gt > 0] = 1
#    print(np.unique(gt))

    
    if np.random.rand() <= 0.25:
        print('Random non-vert Patch')
        patch_center = [np.random.randint(0, s) for s in img.shape]
        lower = [0, 0, 0]
        upper = [img.shape[0], img.shape[1], img.shape[2]]
        x = patch_center[0]
        y = patch_center[1]
        z = patch_center[2]
        
    else:
        print('Random vert Patch')
        indices = np.nonzero(mask == chosen_vert)
        lower = [np.min(i) for i in indices]
        upper = [np.max(i) for i in indices]
        #random center of patch
        x = randint(lower[0], upper[0])
        y = randint(lower[1], upper[1])
        z = randint(lower[2], upper[2])
    
    #extract the patch and padding
    x_low = int(max(x-patch_size/2, lower[0]))
    x_up = int(min(x+patch_size/2, upper[0]))
    
    y_low = int(max(y-patch_size/2, lower[1]))
    y_up = int(min(y+patch_size/2, upper[1]))
    
    z_low = int(max(z-patch_size/2, lower[2]))
    z_up = int(min(z+patch_size/2, upper[2]))
    
    x_pad, y_pad, z_pad = np.zeros(2), np.zeros(2), np.zeros(2)
    
    img_patch = img[x_low:x_up, y_low:y_up,z_low:z_up]
    ins_patch = ins_memory[x_low:x_up, y_low:y_up,z_low:z_up]
    gt_patch = gt[x_low:x_up, y_low:y_up,z_low:z_up]
    
    
    sitk.WriteImage(sitk.GetImageFromArray(img_patch), './img.nrrd', True)
    sitk.WriteImage(sitk.GetImageFromArray(ins_patch), './ins.nrrd', True)
    sitk.WriteImage(sitk.GetImageFromArray(gt_patch), './gt.nrrd', True)
    
    
    #give the label of completeness(partial or complete)
    vol = np.count_nonzero(gt == 1)
    print(vol)
    sample_vol = np.count_nonzero(gt_patch == 1 )
    print(sample_vol)
    
    print('vol:{:.6f}'.format(float(sample_vol/(vol+0.0001))))
    
    c_label = 0 if float(sample_vol/(vol+0.0001)) < 0.98 else 1

    #paddding the patch to 128*128*128
    if (patch_size - img_patch.shape[0])%2 == 1:
        x_pad[0] = int((patch_size - img_patch.shape[0])/2) 
        x_pad[1] = x_pad[0]+1
    else:
        x_pad[0] = (patch_size - img_patch.shape[0])/2
        x_pad[1] = x_pad[0]
        
    if (patch_size - img_patch.shape[1])%2 == 1:
        y_pad[0] = int((patch_size - img_patch.shape[1])/2) 
        y_pad[1] = y_pad[0]+1
    else:
        y_pad[0] = (patch_size - img_patch.shape[1])/2
        y_pad[1] = y_pad[0]    
        
    if (patch_size - img_patch.shape[2])%2 == 1:
        z_pad[0] = int((patch_size - img_patch.shape[2])/2) 
        z_pad[1] = z_pad[0]+1
    else:
        z_pad[0] = (patch_size - img_patch.shape[2])/2
        z_pad[1] = z_pad[0]
    
    x_pad = x_pad.astype(int)
    y_pad = y_pad.astype(int)
    z_pad = z_pad.astype(int)
    
    img_patch = np.pad(img_patch, ((x_pad[0], x_pad[1]), (y_pad[0], y_pad[1]), (z_pad[0], z_pad[1])), 'constant', constant_values= img.min())
    ins_patch = np.pad(ins_patch, ((x_pad[0], x_pad[1]), (y_pad[0], y_pad[1]), (z_pad[0], z_pad[1])), 'constant', constant_values=ins_memory.min())
    gt_patch = np.pad(gt_patch, ((x_pad[0], x_pad[1]), (y_pad[0], y_pad[1]), (z_pad[0], z_pad[1])), 'constant', constant_values=mask.min())
    
    img_patch = np.expand_dims(img_patch, axis=0)
    ins_patch = np.expand_dims(ins_patch, axis=0)
    gt_patch = np.expand_dims(gt_patch, axis=0)
    c_label = np.expand_dims(c_label, axis=0)
    
    return img_patch, ins_patch, gt_patch, c_label


#%%%
train_dataset = CSI_Dataset('D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/isotropic_dataset')

dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True)

img_patch, ins_patch, gt_patch, c_label = next(iter(dataloader_train))

print(c_label.item())











    