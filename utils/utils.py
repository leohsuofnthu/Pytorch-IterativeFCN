import numpy as np


def force_inside_img(x, patch_size, img_shape):
    x_low = int(x - patch_size / 2)
    x_up = int(x + patch_size / 2)
    if x_low < 0:
        x_up -= x_low
        x_low = 0
    elif x_up > img_shape[2]:
        x_low -= (x_up - img_shape[2])
        x_up = img_shape[2]
    return x_low, x_up
