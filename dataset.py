# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:55:35 2019

@author: Gabriel Hsu

ref:https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations

"""
from __future__ import print_function, division

import os
import random

import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset

from data_augmentation import elastic_transform, gaussian_blur, gaussian_noise, random_crop

"""
The dataset of MICCAI 2014 Spine Challenge

"""


# %% Build the dataset
class CSI_Dataset(Dataset):
    """xVertSeg Dataset"""

    def __init__(self, dataset_path, subset='train', linear_att=1.0, offset=1000.0):
        """
        Args:
            path_dataset(string): Root path to the whole dataset
            subset(string): 'train' or 'test' depend on which subset
        """
        self.idx = 1

        self.dataset_path = dataset_path
        self.subset = subset
        self.linear_att = linear_att
        self.offset = offset

        self.img_path = os.path.join(dataset_path, subset, 'img')
        self.mask_path = os.path.join(dataset_path, subset, 'seg')
        self.weight_path = os.path.join(dataset_path, subset, 'weight')

        self.img_names = [f for f in os.listdir(self.img_path) if f.endswith('.mhd')]

    #        self.mask_names = [f for f in os.listdir(self.mask_path) if f.endswith('.mhd')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        mask_name = self.img_names[idx].split('.')[0] + '_label.mhd'
        weight_name = self.img_names[idx].split('.')[0] + '_weight.nrrd'

        img_file = os.path.join(self.img_path, img_name)
        mask_file = os.path.join(self.mask_path, mask_name)
        weight_file = os.path.join(self.weight_path, weight_name)

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        weight = sitk.GetArrayFromImage(sitk.ReadImage(weight_file))

        # linear transformation from 12bit reconstruction img to HU unit
        # depend on the original data (CSI data value is from 0 ~ 4095)
        # img = img * self.linear_att - self.offset

        # extract a traning patche
        img_patch, ins_patch, gt_patch, weight_patch, c_label = extract_random_patch(img,
                                                                                     mask, weight, self.idx,
                                                                                     self.subset)

        # patch normalization
        # img_patch = (img_patch - img_patch.mean()) / img_patch.std()

        self.idx += 1

        return img_patch, ins_patch, gt_patch, weight_patch, c_label


# %% Extract the 128*128*128 patch
def extract_random_patch(img, mask, weight, i, subset, patch_size=128):
    # list available vertebrae
    verts = np.unique(mask)
    #    print('mask values:', verts)
    chosen_vert = verts[random.randint(1, len(verts) - 1)]
    #    print('chosen_vert:', chosen_vert)

    # create corresponde instance memory and ground truth
    ins_memory = np.copy(mask)
    ins_memory[ins_memory <= chosen_vert] = 0
    ins_memory[ins_memory > 0] = 1
    #    print(np.unique(ins_memory))

    gt = np.copy(mask)
    gt[gt != chosen_vert] = 0
    gt[gt > 0] = 1
    #    print(np.unique(gt))

    flag_empty = False

    if True:
        # print(i, ' empty mask')
        patch_center = [np.random.randint(0, s) for s in img.shape]
        lower = [0, 0, 0]

        upper = [img.shape[0], img.shape[1], img.shape[2]]
        x = patch_center[2]
        y = patch_center[1]
        z = patch_center[0]

        # for ins
        gt = np.copy(mask)

        flag_empty = True

    else:
        # print(i, ' normal sample')
        indices = np.nonzero(mask == chosen_vert)
        lower = [np.min(i) for i in indices]
        upper = [np.max(i) for i in indices]
        # random center of patch
        x = random.randint(lower[2], upper[2])
        y = random.randint(lower[1], upper[1])
        z = random.randint(lower[0], upper[0])

    # extract the patch and padding
    x_low = int(x - patch_size / 2)
    x_up = int(x + patch_size / 2)

    # make sure patches are all inside image
    if x_low < 0:
        x_up -= x_low
        x_low = 0
    elif x_up > img.shape[2]:
        x_low -= (x_up - img.shape[2])
        x_up = img.shape[2]

    y_low = int(y - patch_size / 2)
    y_up = int(y + patch_size / 2)

    if y_low < 0:
        y_up -= y_low
        y_low = 0
    elif y_up > img.shape[1]:
        y_low -= (y_up - img.shape[1])
        y_up = img.shape[1]

    z_low = int(z - patch_size / 2)
    z_up = int(z + patch_size / 2)

    if z_low < 0:
        z_up -= z_low
        z_low = 0
    elif z_up > img.shape[0]:
        z_low -= (z_up - img.shape[0])
        z_up = img.shape[0]

    img_patch = img[z_low:z_up, y_low:y_up, x_low:x_up]
    ins_patch = ins_memory[z_low:z_up, y_low:y_up, x_low:x_up]
    gt_patch = gt[z_low:z_up, y_low:y_up, x_low:x_up]
    weight_patch = weight[z_low:z_up, y_low:y_up, x_low:x_up]

    """
    #paddding the patch to 128*128*128
    x_pad, y_pad, z_pad = np.zeros(2), np.zeros(2), np.zeros(2)
    
    if x_low == 0:
      x_pad[0] = int(patch_size - img_patch.shape[2]) 
    elif x_up == img.shape[2]:
      x_pad[1] = int(patch_size - img_patch.shape[2]) 
      
    if y_low == 0:
      y_pad[0] = int(patch_size - img_patch.shape[1]) 
    elif y_up == img.shape[1]:
      y_pad[1] = int(patch_size - img_patch.shape[1]) 
      
    if z_low == 0:
      z_pad[0] = int(patch_size - img_patch.shape[0]) 
    elif z_up == img.shape[0]:
      z_pad[1] = int(patch_size - img_patch.shape[0]) 
   
    x_pad = x_pad.astype(int)
    y_pad = y_pad.astype(int)
    z_pad = z_pad.astype(int)
    
    img_patch = np.pad(img_patch, ((z_pad[0], z_pad[1]), (y_pad[0], y_pad[1]), (x_pad[0], x_pad[1])), 'constant', constant_values=img.min())
    ins_patch = np.pad(ins_patch, ((z_pad[0], z_pad[1]), (y_pad[0], y_pad[1]), (x_pad[0], x_pad[1])), 'constant', constant_values=ins_memory.min())
    gt_patch = np.pad(gt_patch,   ((z_pad[0], z_pad[1]), (y_pad[0], y_pad[1]), (x_pad[0], x_pad[1])), 'constant', constant_values=gt.min())
    weight_patch = np.pad(weight_patch, ((z_pad[0], z_pad[1]), (y_pad[0], y_pad[1]), (x_pad[0], x_pad[1])), 'constant', constant_values=weight.min())
    """

    # the patches trained for producing empty mask
    if flag_empty:
        ins_patch = np.copy(gt_patch)
        ins_patch[ins_patch > 0] = 1
        gt_patch = np.zeros_like(ins_patch)
        weight = np.ones_like(ins_patch)

    # Randomly Data Augmentation
    # 50% chance elastic deformation
    if subset == 'train':
        if np.random.rand() > 0.5:
            print('elastic deform')
            img_patch, gt_patch, ins_patch, weight_patch = elastic_transform(img_patch, gt_patch, ins_patch,
                                                                             weight_patch, alpha=20, sigma=5)
        # 50% chance gaussian blur
        if np.random.rand() > 0.5:
            print('gaussian blur')
            img_patch = gaussian_blur(img_patch)
        # 50% chance gaussian noise
        if np.random.rand() > 0.5:
            print('gaussian noise')
            img_patch = gaussian_noise(img_patch)

        # 50% random crop along z-axis
        if np.random.rand() > 0.5:
            print('Random crop along z-axis')
            img_patch, ins_patch, gt_patch, weight_patch = random_crop(img_patch, ins_patch, gt_patch
                                                                       , weight_patch)

        """
        #50% random rotate 90, 180, or 270 degrees 
        if np.random.rand() > 0.5:
            print('rotate')
            img_patch, ins_patch, gt_patch, weight_patch = rotate(img_patch, ins_patch, gt_patch
            ,weight_patch)
        """
    # give the label of completeness(partial or complete)
    vol = np.count_nonzero(gt == 1)
    sample_vol = np.count_nonzero(gt_patch == 1)

    # print('visible volume:{:.6f}'.format(float(sample_vol/(vol+0.0001))))
    c_label = 0 if float(sample_vol / (vol + 0.0001)) < 0.98 else 1

    img_patch = np.expand_dims(img_patch, axis=0)
    ins_patch = np.expand_dims(ins_patch, axis=0)
    gt_patch = np.expand_dims(gt_patch, axis=0)
    weight_patch = np.expand_dims(weight_patch, axis=0)
    c_label = np.expand_dims(c_label, axis=0)

    return img_patch, ins_patch, gt_patch, weight_patch, c_label

# %% Test purpose
# import torch
# from torch.utils.data import Dataset, DataLoader
# train_dataset = CSI_Dataset('D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/crop_isotropic_dataset')
#
# dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True)
#
# img_patch, ins_patch, gt_patch, weight, c_label = next(iter(dataloader_train))
#
# print(img_patch.shape)
# print(gt_patch.shape)
#
#
# img_patch = torch.squeeze(img_patch)
# ins_patch = torch.squeeze(ins_patch)
# gt_patch = torch.squeeze(gt_patch)
# weight = torch.squeeze(weight)
#
#
##produce 17000 training samples, and 3000 test sample
#
# sitk.WriteImage(sitk.GetImageFromArray(img_patch.numpy()), './img.nrrd', True)
# sitk.WriteImage(sitk.GetImageFromArray(gt_patch.numpy()), './gt.nrrd', True)
# sitk.WriteImage(sitk.GetImageFromArray(ins_patch.numpy()), './ins.nrrd', True)
# sitk.WriteImage(sitk.GetImageFromArray(weight.numpy()), './wei.nrrd', True)
