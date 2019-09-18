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
        
        #list available vertebrae
        verts = np.unique(mask)
        print(verts)
        chosen_vert = verts[randint(1, len(verts)-1)]
        print(chosen_vert)
        
        #create corresponde instance memory and ground truth
        ins_memory = np.copy(mask)
        ins_memory[ins_memory >= chosen_vert] = 0
        ins_memory[ins_memory > 0] = 1
        print(np.unique(ins_memory))
        
        gt = np.copy(mask)
        gt[gt != chosen_vert] = 0
        gt[gt > 0] = 1
        print(np.unique(gt))

        indices = np.nonzero(mask == chosen_vert)
        print(indices[1])
        lower = [np.min(i) for i in indices]
        upper = [np.max(i) for i in indices]
        
        print(lower)
        print(upper)
        
        #random center of patch
        x = randint(lower[0], upper[0])
        y = randint(lower[1], upper[1])
        z = randint(lower[2], upper[2])
        
        print(x,y,z)
        print(ins_memory[indices[0][0],indices[1][0],indices[2][0]])
        
        #extract the patch and padding
        patch_size = 128
        img_patch = img[x-patch_size/2:x+patch_size/2, y-patch_size/2:y+patch_size/2, z-patch_size/2:z+patch_size/2]
        
        return img, mask
    
#%% Extract the 128*128*128 patch
#def extract_patch(img, mask):
    
    


#%%%
train_dataset = CSI_Dataset('D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/isotropic_dataset')

dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True)

i, m = next(iter(dataloader_train))







    