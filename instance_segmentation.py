import os
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy import ndimage
import SimpleITK as sitk

from iterativeFCN import IterativeFCN

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
)


def extract(img, x, y, z, patch_size):
    """Extract a cubic patch centered at (z, y, x) — array order is Z,Y,X."""
    offset = patch_size // 2
    z0, z1 = z - offset, z + offset
    y0, y1 = y - offset, y + offset
    x0, x1 = x - offset, x + offset
    return img[z0:z1, y0:y1, x0:x1]


def _clamp_center(c, patch_size, shape):
    half = patch_size // 2
    c = list(c)
    for i in range(3):
        if c[i] + half > shape[i]:
            c[i] = shape[i] - half
        if c[i] - half < 0:
            c[i] = half
        c[i] = int(c[i])
    return c


def instance_segmentation(model, device, img_name, patch_size, sigma_x,
                          lim_alternate_times, n_min, output_path,
                          max_steps=100000):
    """
    Sliding-window iterative instance segmentation (paper inference).

    For each candidate patch: predict binary vertebra mask S + completeness C.
    Refine window center to mask CoM until convergence, write instance into
    memory, then continue scanning.
    """
    step = patch_size // 2
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_name))).astype(np.float32)
    ins = np.zeros_like(img, dtype=np.float32)
    mask = np.zeros_like(img, dtype=np.float32)
    img_shape = img.shape

    # Start at cranial end, scan in +x, +y, -z (as in original code)
    z = int(img_shape[0] - step)
    y = int(step)
    x = int(step)
    z, y, x = _clamp_center([z, y, x], patch_size, img_shape)

    c_now = [z, y, x]
    c_prev = [0, 0, 0]

    label = 1
    iters = 0
    steps = 0
    model.eval()

    logging.info('Start Instance Segmentation: %s shape=%s', img_name, img_shape)

    while True:
        steps += 1
        if steps > max_steps:
            logging.warning('Hit max_steps=%s; stopping to avoid endless loop.', max_steps)
            break

        # Finished caudal sweep
        if z < step:
            logging.info('Reached end of volume (z < %s).', step)
            break

        logging.info('(Z, Y, X) Now: (%s, %s, %s)', z, y, x)

        patch = extract(img, x, y, z, patch_size)
        ins_patch = extract(ins, x, y, z, patch_size)
        if patch.shape != (patch_size, patch_size, patch_size):
            logging.warning('Incomplete patch at (%s,%s,%s); sliding on.', z, y, x)
            # slide
            x += step
            if x + step > img_shape[2]:
                x = step
                y += step
                if y + step > img_shape[1]:
                    y = step
                    z -= step
            z, y, x = _clamp_center([z, y, x], patch_size, img_shape)
            continue

        img_t = torch.tensor(np.expand_dims(patch, axis=0))
        ins_t = torch.tensor(np.expand_dims(ins_patch, axis=0))
        input_patch = torch.unsqueeze(torch.cat((img_t, ins_t)), dim=0).float().to(device)

        with torch.no_grad():
            S, C = model(input_patch)

        S = torch.squeeze(S.round().to('cpu')).numpy()
        vol = int(np.count_nonzero(S))

        if vol > n_min:
            c_prev = c_now.copy()
            center = ndimage.center_of_mass(S)
            if any(np.isnan(center)):
                logging.info('Invalid CoM; sliding window.')
            else:
                center = [int(center[0]), int(center[1]), int(center[2])]
                # Map patch-local CoM (z,y,x) to global
                c_now = [
                    int(z - step + center[0]),
                    int(y - step + center[1]),
                    int(x - step + center[2]),
                ]
                c_now = _clamp_center(c_now, patch_size, img_shape)
                logging.info('Global Center: %s (prev %s)', c_now, c_prev)

                moved = (
                    abs(c_now[0] - c_prev[0]) > sigma_x
                    or abs(c_now[1] - c_prev[1]) > sigma_x
                    or abs(c_now[2] - c_prev[2]) > sigma_x
                )

                if moved:
                    iters += 1
                    logging.info('Not converged (alternate %s/%s)', iters, lim_alternate_times)
                    if iters >= lim_alternate_times:
                        # Average both centers (fix: use each axis, not c_prev[0] thrice)
                        c_now = [
                            int((c_now[0] + c_prev[0]) / 2),
                            int((c_now[1] + c_prev[1]) / 2),
                            int((c_now[2] + c_prev[2]) / 2),
                        ]
                        c_now = _clamp_center(c_now, patch_size, img_shape)
                        iters = 0
                        _commit_mask(S, ins, mask, c_now, patch_size, label)
                        label += 1
                        logging.info('Forced converge; segmented instance %s', label - 1)
                else:
                    iters = 0
                    _commit_mask(S, ins, mask, c_now, patch_size, label)
                    label += 1
                    logging.info('Converged; segmented instance %s', label - 1)

                z, y, x = c_now
                continue

        # Slide window when no (valid) detection
        logging.info('Slide window')
        x += step
        if x + step > img_shape[2]:
            x = step
            y += step
            if y + step > img_shape[1]:
                y = step
                z -= step
        z, y, x = _clamp_center([z, y, x], patch_size, img_shape)

    logging.info('Finish Segmentation! instances=%s', label - 1)
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(output_path), True)


def _commit_mask(S, ins, mask, c_now, patch_size, label):
    half = patch_size // 2
    z_low, z_up = c_now[0] - half, c_now[0] + half
    y_low, y_up = c_now[1] - half, c_now[1] + half
    x_low, x_up = c_now[2] - half, c_now[2] + half
    r = S > 0
    ins[z_low:z_up, y_low:y_up, x_low:x_up][r] = 1
    mask[z_low:z_up, y_low:y_up, x_low:x_up][r] = label


def load_model(weights, device, num_channels=64):
    model = IterativeFCN(num_channels=num_channels).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Iterative FCN instance segmentation')
    parser.add_argument('--test_dir', type=str, default='./crop_isotropic_dataset/test/img',
                        help='folder of test images (.mhd/.nrrd)')
    parser.add_argument('--output_dir', type=str, default='./pred',
                        help='folder for predicted masks')
    parser.add_argument('--weights', type=str, default='./weights/IterativeFCN_pretrained.pth',
                        help='path to .pth weights')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='cubic patch size (paper uses 128)')
    parser.add_argument('--sigma', type=int, default=2,
                        help='convergence tolerance (voxels) between successive centers')
    parser.add_argument('--min_vol', type=int, default=1000,
                        help='min predicted voxels to accept a detection')
    parser.add_argument('--max_alter', type=int, default=20,
                        help='max CoM alternations before forced converge')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='safety cap against endless loops')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    logging.info('Device: %s', device)
    logging.info('Loading weights: %s', args.weights)
    model = load_model(args.weights, device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    test_imgs = sorted(
        x for x in os.listdir(args.test_dir)
        if x.endswith(('.mhd', '.nrrd', '.nii', '.nii.gz')) and 'raw' not in x
    )
    if not test_imgs:
        raise FileNotFoundError(f'No images found in {args.test_dir}')

    for img in test_imgs:
        logging.info('Processing image: %s', img)
        stem = img.split('.')[0]
        output_path = os.path.join(args.output_dir, f'{stem}_pred.nrrd')
        instance_segmentation(
            model, device,
            os.path.join(args.test_dir, img),
            args.patch_size, args.sigma, args.max_alter, args.min_vol,
            output_path, max_steps=args.max_steps,
        )


if __name__ == '__main__':
    main()
