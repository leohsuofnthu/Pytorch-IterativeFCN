# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:59:48 2019

@author: Gabriel Hsu
"""
import SimpleITK as sitk
import numpy as np
import torch
from scipy import ndimage

from model import iterativeFCN

# parameters
patch_size = 128

step = int(patch_size / 2)
sigma_x = 2
lim_alternate_times = 20
n_min = 1000

img_name = './case2.mhd'
img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))

# convert to HU
# test_numpy = test_numpy*1.0 - 1000.0

ins = np.zeros_like(img)
mask = np.zeros_like(img)

temp = np.zeros_like(img)

print('Image Shape: ', img.shape)

img_shape = img.shape


def extract(img, x, y, z, patch_size):
    offset = int(patch_size / 2)
    return img[z - offset:z + offset, y - offset:y + offset, x - offset:x + offset]


# Create FCN
print('Create Model!!')
model = iterativeFCN().to('cuda')
model.load_state_dict(torch.load('IterativeFCN_best_train.pth'))
print('Finish Loading Parameters!!')

# slide window with initial center coord
patch_size = 128
z = int(img.shape[0] - (patch_size / 2))
y = int(patch_size / 2)
x = int(patch_size / 2)

# x_t-1, x_t
c_now = [z, y, x]
c_prev = [0, 0, 0]

label = 100
iters = 0
ii = 0
# slide window check
print('Start Instance Segmentation')
while True:

    print('(Z, Y, X) Now:', z, y, x)

    if abs(x - patch_size / 2) < sigma_x and abs(y - patch_size / 2) < sigma_x and abs(z - patch_size / 2) < sigma_x:
        break

    # extract patch and instance memory
    img_patch = torch.tensor(np.expand_dims(extract(img, x, y, z, 128), axis=0))
    ins_patch = torch.tensor(np.expand_dims(extract(ins, x, y, z, 128), axis=0))

    # sitk.WriteImage(sitk.GetImageFromArray(extract(img, x, y, z , 128)), './img18'+str(ii)+'.nrrd', True)
    # sitk.WriteImage(sitk.GetImageFromArray(extract(ins, x, y, z , 128)), './ins18'+str(ii)+'.nrrd', True)

    input_patch = torch.cat((img_patch, ins_patch))
    input_patch = torch.unsqueeze(input_patch, dim=0)

    with torch.no_grad():
        S, C = model(input_patch.float().to('cuda'))

    S = torch.squeeze(S.round().to('cpu')).numpy()

    vol = np.count_nonzero(S)
    # sitk.WriteImage(sitk.GetImageFromArray(S), './gt18'+str(ii)+'.nrrd', True)

    ii += 1

    # check if vol > 1000
    if vol > n_min:
        c_prev[0] = c_now[0]
        c_prev[1] = c_now[1]
        c_prev[2] = c_now[2]

        center = ndimage.measurements.center_of_mass(S)
        center = [int(center[0]), int(center[1]), int(center[2])]
        print('Center relative to patch:', center)

        c_now[0] = z + (patch_size / 2) - (patch_size - center[0])
        c_now[1] = y - (patch_size / 2) + center[1]
        c_now[2] = x - (patch_size / 2) + center[2]
        print('Global Center:', c_now)

        # correction to be in-frame
        if (c_now[0] + patch_size / 2) > img.shape[0]:
            c_now[0] = img.shape[0] - (patch_size / 2)

        elif (c_now[0] - patch_size / 2) < 0:
            c_now[0] = (patch_size / 2)

        if (c_now[1] + patch_size / 2) > img.shape[1]:
            c_now[1] = img.shape[1] - (patch_size / 2)

        elif (c_now[1] - patch_size / 2) < 0:
            c_now[1] = (patch_size / 2)

        if (c_now[2] + patch_size / 2) > img.shape[2]:
            c_now[2] = img.shape[2] - (patch_size / 2)

        elif (c_now[2] - patch_size / 2) < 0:
            c_now[2] = (patch_size / 2)

        c_now[0] = int(c_now[0])
        c_now[1] = int(c_now[1])
        c_now[2] = int(c_now[2])
        print('Modified center:', c_now)
        print('Prev center', c_prev)

        if abs(c_now[0] - c_prev[0]) > sigma_x or abs(c_now[1] - c_prev[1]) > sigma_x or abs(
                c_now[2] - c_prev[2]) > sigma_x:
            iters += 1
            print('Not converge iterations', iters)

            if iters == 20:
                print('iteration == 20')
                # pick avg and dim as converge
                c_now[0] = int((c_now[0] + c_prev[0]) / 2)
                c_now[1] = int((c_now[1] + c_prev[0]) / 2)
                c_now[2] = int((c_now[2] + c_prev[0]) / 2)

                print('converge and seg')
                iters = 0
                # converge, update ins and mask

                z_low = int(c_now[0] - (patch_size / 2))
                z_up = int(c_now[0] + (patch_size / 2))
                y_low = int(c_now[1] - (patch_size / 2))
                y_up = int(c_now[1] + (patch_size / 2))
                x_low = int(c_now[2] - (patch_size / 2))
                x_up = int(c_now[2] + (patch_size / 2))

                r = S > 0
                ins[z_low:z_up, y_low:y_up, x_low:x_up][r] = 1
                mask[z_low:z_up, y_low:y_up, x_low:x_up][r] = label

                label += 100
                print("seg {}th verts complete!!".format(label))


        else:
            print('converge and seg')
            iters = 0

            # converge, update ins and mask
            z_low = int(c_now[0] - (patch_size / 2))
            z_up = int(c_now[0] + (patch_size / 2))
            y_low = int(c_now[1] - (patch_size / 2))
            y_up = int(c_now[1] + (patch_size / 2))
            x_low = int(c_now[2] - (patch_size / 2))
            x_up = int(c_now[2] + (patch_size / 2))

            r = S > 0
            ins[z_low:z_up, y_low:y_up, x_low:x_up][r] = 1
            mask[z_low:z_up, y_low:y_up, x_low:x_up][r] = label

            label += 100
            print("seg {}th verts complete!!".format(label))

        # same patch analyze again, center remain
        z = c_now[0]
        y = c_now[1]
        x = c_now[2]




    else:
        print('slide window')
        # continue slide windows
        if x + step > img_shape[2]:
            x = int(patch_size / 2)
            if y + step > img_shape[1]:
                y = int(patch_size / 2)
                z = z - step
            else:
                y = y + step
        else:
            x = x + step

print('Finish Segmentation!')
sitk.WriteImage(sitk.GetImageFromArray(mask), './pred_mask.nrrd', True)
