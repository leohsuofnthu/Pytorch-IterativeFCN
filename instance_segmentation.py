import os
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy import ndimage

import SimpleITK as sitk
from iterativeFCN import IterativeFCN

logging.basicConfig(level=logging.INFO)


def extract(img, x, y, z, patch_size):
    offset = int(patch_size / 2)
    return img[z - offset:z + offset, y - offset:y + offset, x - offset:x + offset]


def instance_segmentation(model, img_name, patch_size, sigma_x, lim_alternate_times, n_min, output_path):
    step = int(patch_size / 2)
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
    ins = np.zeros_like(img)
    mask = np.zeros_like(img)
    img_shape = img.shape

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
    logging.info('Start Instance Segmentation')
    while True:

        logging.info('(Z, Y, X) Now: (%s, %s, %s)' % (z, y, x))
        if abs(x - patch_size / 2) < sigma_x and abs(y - patch_size / 2) < sigma_x and abs(
                z - patch_size / 2) < sigma_x:
            break

        # extract patch and instance memory
        img_patch = torch.tensor(np.expand_dims(extract(img, x, y, z, 128), axis=0))
        ins_patch = torch.tensor(np.expand_dims(extract(ins, x, y, z, 128), axis=0))

        input_patch = torch.cat((img_patch, ins_patch))
        input_patch = torch.unsqueeze(input_patch, dim=0)

        with torch.no_grad():
            S, C = model(input_patch.float().to('cuda'))

        S = torch.squeeze(S.round().to('cpu')).numpy()
        vol = np.count_nonzero(S)

        ii += 1
        # check if vol > min_threshold
        if vol > n_min:
            c_prev[0] = c_now[0]
            c_prev[1] = c_now[1]
            c_prev[2] = c_now[2]

            center = ndimage.measurements.center_of_mass(S)
            center = [int(center[0]), int(center[1]), int(center[2])]
            logging.info('Center relative to patch:%s' % center)

            c_now[0] = z + (patch_size / 2) - (patch_size - center[0])
            c_now[1] = y - (patch_size / 2) + center[1]
            c_now[2] = x - (patch_size / 2) + center[2]
            logging.info('Global Center:%s' % c_now)

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
            logging.info('Modified center:%s' % c_now)
            logging.info('Prev center %s' % c_prev)

            if abs(c_now[0] - c_prev[0]) > sigma_x or abs(c_now[1] - c_prev[1]) > sigma_x or abs(
                    c_now[2] - c_prev[2]) > sigma_x:
                iters += 1
                logging.info('Not converge iterations %s' % iters)

                if iters == lim_alternate_times:
                    logging.info('iteration:%s' % lim_alternate_times)
                    # pick avg and dim as converge
                    c_now[0] = int((c_now[0] + c_prev[0]) / 2)
                    c_now[1] = int((c_now[1] + c_prev[0]) / 2)
                    c_now[2] = int((c_now[2] + c_prev[0]) / 2)

                    logging.info('converge and seg')
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
                    logging.info("seg {}th verts complete!!".format(label))
            else:
                logging.info('converge and seg')
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
                logging.info("seg {}th verts complete!!".format(label))

            # same patch analyze again, center remain
            z = c_now[0]
            y = c_now[1]
            x = c_now[2]
        else:
            logging.info('slide window')
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

    logging.info('Finish Segmentation!')
    sitk.WriteImage(sitk.GetImageFromArray(mask), output_path, True)


def main():
    parser = argparse.ArgumentParser(description='Iterative Fully Convolutional Network')
    parser.add_argument('--test_dir', type=str, default='./crop_isotropic_dataset/test/img',
                        help='folder of test images')
    parser.add_argument('--output_dir', type=str, default='./pred',
                        help='folder of pred masks')
    parser.add_argument('--weights', type=str, default='./weights/IterativeFCN_best_train.pth',
                        help='trained weights of model')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='patch_size of the model')
    parser.add_argument('--sigma', type=int, default=2,
                        help='patch_size of the model')
    parser.add_argument('--min_vol', type=int, default=1000,
                        help='min volume threshold')
    parser.add_argument('--max_alter', type=int, default=20,
                        help='max alternation of 2 centers')
    args = parser.parse_args()

    # Create FCN
    logging.info('Create Model and Loading Pretrained Weights')
    model = IterativeFCN().to('cuda')
    model.load_state_dict(torch.load(args.weights))

    # list the test images
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test_imgs = [x for x in os.listdir(os.path.join(args.test_dir)) if 'raw' not in x]
    for img in test_imgs:
        logging.info("Processing image: %s", img)
        output_path = os.path.join(args.output_dir, img.split('.')[0]+'_pred.nrrd')
        instance_segmentation(model, os.path.join(args.test_dir, img), args.patch_size, args.sigma, args.max_alter, args.min_vol, output_path)


if __name__ == '__main__':
    main()
