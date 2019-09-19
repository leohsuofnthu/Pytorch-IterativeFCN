# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:30:56 2019

@author: Gabriel Hsu
"""

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import torch
from dataset import CSI_Dataset
from torch.utils.data import Dataset, DataLoader

from random import randint

import SimpleITK as sitk

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       
       Modified from: https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def gaussian_blur(image):
    return gaussian_filter(image, sigma=1)

def gaussian_noise(image):
     mean = 0
     var = 0.1
     sigma = var**0.5
     gauss = 50*np.random.normal(mean,sigma,image.shape)
     gauss = gauss.reshape(image.shape)
     return image + gauss
 
def crop_z(image, low_z, up_z):
    crop_img = np.copy(image)
    crop_img[:,:,:low_z] = image.min()
    crop_img[:,:,up_z:] = image.min()
    return crop_img

#%% Test Purpose
    
train_dataset = CSI_Dataset('D:/Project III- Iterative Fully Connected Network for Vertebrae Segmentation/Pytorch-IterativeFCN/isotropic_dataset')

dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=True)

img_patch, ins_patch, gt_patch, c_label = next(iter(dataloader_train))

img_patch = torch.squeeze(img_patch)
ins_patch = torch.squeeze(ins_patch)
gt_patch = torch.squeeze(gt_patch)


def_img_patch = crop_z(img_patch.numpy(), 12, 120)
def_gt_patch = crop_z(gt_patch.numpy(), 12, 120)
def_ins_patch = crop_z(ins_patch.numpy(), 12, 120)

sitk.WriteImage(sitk.GetImageFromArray(def_img_patch), 'df_img.nrrd', True)
sitk.WriteImage(sitk.GetImageFromArray(def_gt_patch), 'df_gt.nrrd', True)
sitk.WriteImage(sitk.GetImageFromArray(def_ins_patch), 'df_ins.nrrd', True)



sitk.WriteImage(sitk.GetImageFromArray(img_patch.numpy()), 'img.nrrd', True)
sitk.WriteImage(sitk.GetImageFromArray(gt_patch.numpy()), 'gt.nrrd', True)
sitk.WriteImage(sitk.GetImageFromArray(ins_patch.numpy()), 'ins.nrrd', True)

