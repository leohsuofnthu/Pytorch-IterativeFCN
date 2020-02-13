import os
import re
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

logging.basicConfig(level=logging.info())


# resample the CT images to isotropic
def isotropic_resampler(input_path, output_path):
    raw_img = sitk.ReadImage(input_path)
    new_spacing = [1, 1, 1]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(raw_img.GetDirection())
    resampler.SetOutputOrigin(raw_img.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)

    orig_size = np.array(raw_img.GetSize(), dtype=np.int)
    orig_spacing = raw_img.GetSpacing()
    new_size = np.array([x * (y / z) for x, y, z in zip(orig_size, orig_spacing, new_spacing)])
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resampler.SetSize(new_size)

    isotropic_img = resampler.Execute(raw_img)
    sitk.WriteImage(isotropic_img, output_path, True)


# Function for cropping
def z_mid(mask, chosen_vert):
    indices = np.nonzero(mask == chosen_vert)
    lower = [np.min(i) for i in indices]
    upper = [np.max(i) for i in indices]

    return int((lower[0] + upper[0]) / 2)


def findZRange(img, mask):
    # list available vertebrae
    verts = np.unique(mask)

    vert_low = verts[1]
    vert_up = verts[-1]

    z_range = [z_mid(mask, vert_low), z_mid(mask, vert_up)]
    logging.info('Range of Z axis %s' % z_range)
    return z_range


def crop_unref_vert(path, out_path, subset):
    img_path = os.path.join(path, subset, 'img')
    mask_path = os.path.join(path, subset, 'seg')
    weight_path = os.path.join(path, subset, 'weight')
    img_names = [f for f in os.listdir(img_path) if f.endswith('.mhd')]

    for img_name in img_names:
        logging.info('Cropping non-reference vertebrae of %s' % img_name)
        img_name = img_name
        mask_name = img_name.split('.')[0] + '_label.mhd'
        weight_name = img_name.split('.')[0] + '_weight.nrrd'

        img_file = os.path.join(img_path, img_name)
        mask_file = os.path.join(mask_path, mask_name)
        weight_file = os.path.join(weight_path, weight_name)

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        weight = sitk.GetArrayFromImage(sitk.ReadImage(weight_file))

        z_range = findZRange(img, mask)

        sitk.WriteImage(sitk.GetImageFromArray(img[z_range[0]:z_range[1], :, :]),
                        os.path.join(out_path, subset, 'img', img_name), True)
        sitk.WriteImage(sitk.GetImageFromArray(mask[z_range[0]:z_range[1], :, :]),
                        os.path.join(out_path, subset, 'seg', mask_name), True)
        sitk.WriteImage(sitk.GetImageFromArray(weight[z_range[0]:z_range[1], :, :]),
                        os.path.join(out_path, subset, 'weight', weight_name), True)


# calculate the weight via distance transform
def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=6):
    """
    Code from author : Dr.Lessman (nikolas.lessmann@radboudumc.nl)
    """
    mask = np.asarray(mask)
    distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)
    weights = alpha + beta * np.exp(-(distance_to_border ** 2 / omega ** 2))
    return np.asarray(weights, dtype='float32')


def calculate_weight(isotropic_path, subset):
    mask_path = os.path.join(isotropic_path, subset, 'seg')
    weight_path = os.path.join(isotropic_path, subset, 'weight')

    Path(mask_path).mkdir(parents=True, exist_ok=True)
    Path(weight_path).mkdir(parents=True, exist_ok=True)

    for f in [f for f in os.listdir(mask_path) if f.endswith('.mhd')]:
        seg_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path, f)))
        weight = compute_distance_weight_matrix(seg_mask)
        sitk.WriteImage(sitk.GetImageFromArray(weight), os.path.join(weight_path, f.split('_')[0] + '_weight.nrrd'),
                        True)
        logging.info("Calculating weight of %s" % f)


def create_folders(root, subsets, folders):
    for subset in subsets:
        for f in folders:
            Path(os.path.join(root, subset, f)).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='iterativeFCN')
    parser.add_argument('--dataset', type=str, default='./CSI_dataset', help='root path of CSI dataset ')
    parser.add_argument('--output_isotropic', type=str, default='./isotropic_dataset',
                        help='output path for isotropic images')
    parser.add_argument('--output_crop', type=str, default='./crop_isotropic_dataset',
                        help='output path for crop samples')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='ratio of train/test')
    args = parser.parse_args()

    # split data into train test folder
    folders = ['img', 'seg', 'weight']
    subsets = ['train', 'test']
    create_folders(args.output_isotropic, subsets, folders)
    create_folders(args.output_crop, subsets, folders)

    # resample the CSI dataset to isotropic dataset
    files = [x for x in os.listdir(os.path.join(args.dataset)) if 'raw' not in x]
    for f in files:
        case_id = re.findall(r'\d+', f)[0]
        logging.info('Resampling ' + f + '...')
        if int(case_id) < int(len(files)/2 * args.split_ratio):
            if '_label' in f:
                file_output = os.path.join(args.output_isotropic, 'train/seg', f)
            else:
                file_output = os.path.join(args.output_isotropic, 'train/img', f)
        else:
            if '_label' in f:
                file_output = os.path.join(args.output_isotropic, 'test/seg', f)
            else:
                file_output = os.path.join(args.output_isotropic, 'test/img', f)

        isotropic_resampler(os.path.join(args.dataset, f), file_output)

    # Pre Calculate the weight
    calculate_weight(args.output_isotropic, 'train')
    calculate_weight(args.output_isotropic, 'test')

    # Crop the image to remove the vertebrae that are not labeled in ground truth
    crop_unref_vert(args.output_isotropic, args.output_crop, 'train')
    crop_unref_vert(args.output_isotropic, args.output_crop, 'test')


if __name__ == '__main__':
    main()
