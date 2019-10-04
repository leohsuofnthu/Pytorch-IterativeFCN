# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:04:23 2019

@author: Gabriel Hsu
"""

import os
import numpy as np
import pandas as pd

import SimpleITK as sitk

import torch
from torch.utils.data import Dataset, DataLoader


#%% Build the dataset 
class CSI_Dataset_Patched(Dataset):
    """xVertSeg Dataset"""
    
    def __init__(self, dataset_path, c_list , subset='train'):
        """
        Args:
            path_dataset(string): Root path to the whole dataset
            subset(string): 'train' or 'test' depend on which subset
        """
        
        self.dataset_path = dataset_path
        self.subset = subset
        self.c_list = c_list
        
        self.img_path = os.path.join(dataset_path, subset, 'img')
        self.ins_path = os.path.join(dataset_path, subset, 'ins')
        self.gt_path = os.path.join(dataset_path, subset, 'gt')
        self.weight_path = os.path.join(dataset_path, subset, 'weight')
        
        
        self.img_names =  [f for f in os.listdir(self.img_path) if f.endswith('.nrrd')]

     
    def __len__(self):
        return len(self.img_names)
    
    
    def __getitem__(self, idx):
    
        
        img_name =  self.img_names[idx]
        ins_name = self.img_names[idx].replace('img', 'ins')
        gt_name = self.img_names[idx].replace('img', 'gt')
        weight_name = self.img_names[idx].replace('img', 'weight')
        
        img_file = os.path.join(self.img_path,  img_name)
        ins_file = os.path.join(self.ins_path, ins_name)
        gt_file = os.path.join(self.gt_path,  gt_name)
        weight_file =  os.path.join(self.weight_path,  weight_name)
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        ins = sitk.GetArrayFromImage(sitk.ReadImage(ins_file))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
        weight =sitk.GetArrayFromImage(sitk.ReadImage(weight_file))
        
        num_id = (img_name.split('_')[1]).split('.')[0]
        c_label = self.c_list[int(num_id)]
        
        
        img = np.expand_dims(img, axis=0)
        ins = np.expand_dims(ins, axis=0)
        gt = np.expand_dims(gt, axis=0)
        weight = np.expand_dims(weight, axis=0)
        c_label = np.expand_dims(c_label, axis=0)
    
            
        return img, ins, gt, weight, c_label, img_name
    
#%%Test

#data_root = 'D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/patches'
#
#class_label = list(pd.read_excel(os.path.join(data_root, 'train_label.xlsx'))[0])   
#    
#train_dataset = CSI_Dataset_Patched('D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/patches', class_label, 'train')
#
#dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True)
#
#img_patch, ins_patch, gt_patch, weight, c_label = next(iter(dataloader_train))
#
#print(img_patch.shape)
#print(img_patch.shape)
#print(gt_patch.shape)
#print(weight.shape)
#print(c_label)
#
#
#img_patch = torch.squeeze(img_patch)
#ins_patch = torch.squeeze(ins_patch)
#gt_patch = torch.squeeze(gt_patch)
#weight = torch.squeeze(weight)
#
#
#
##produce 17000 training samples, and 3000 test sample
#
#sitk.WriteImage(sitk.GetImageFromArray(img_patch.numpy()), 'img.nrrd', True)
#sitk.WriteImage(sitk.GetImageFromArray(gt_patch.numpy()), 'gt.nrrd', True)
#sitk.WriteImage(sitk.GetImageFromArray(ins_patch.numpy()), 'ins.nrrd', True)
#sitk.WriteImage(sitk.GetImageFromArray(weight.numpy()), 'wei.nrrd', True)


