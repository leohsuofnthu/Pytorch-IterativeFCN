import os
import argparse
import logging

import numpy as np
import SimpleITK as sitk

from utils.metrics import _dc

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate vertebra Dice vs labels')
    parser.add_argument('--label_dir', type=str, default='./crop_isotropic_dataset/test/seg',
                        help='folder of test label masks')
    parser.add_argument('--pred_dir', type=str, default='./pred',
                        help='folder of predicted masks')
    args = parser.parse_args()

    label_names = sorted(
        x for x in os.listdir(args.label_dir)
        if 'raw' not in x and not x.endswith('.raw')
    )
    scores = []
    n = 0
    for lname in label_names:
        label_path = os.path.join(args.label_dir, lname)
        stem = lname.split('.')[0].replace('_label', '')
        candidates = [
            os.path.join(args.pred_dir, f'{stem}_pred.nrrd'),
            os.path.join(args.pred_dir, f'{stem}.nrrd'),
            os.path.join(args.pred_dir, lname),
        ]
        pred_path = next((p for p in candidates if os.path.isfile(p)), None)
        if pred_path is None:
            logging.warning('No prediction found for %s; skipping', lname)
            continue

        logging.info('Process %s and %s', pred_path, label_path)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

        for vert_id in np.unique(label):
            if vert_id == 0:
                continue
            l_bin = (label == vert_id).astype(np.uint8)
            pred_vals = pred[label == vert_id]
            if pred_vals.size == 0:
                continue
            nonzero = pred_vals[pred_vals > 0]
            if nonzero.size == 0:
                scores.append(0.0)
                n += 1
                continue
            maj = int(np.bincount(nonzero.astype(int)).argmax())
            p_bin = (pred == maj).astype(np.uint8)
            scores.append(_dc(p_bin, l_bin))
            n += 1

    if n == 0:
        raise RuntimeError('No vertebra pairs evaluated. Check --label_dir / --pred_dir.')
    logging.info(
        'Average Dice Coefficient for %s individual vertebrae: %s',
        n, float(np.mean(scores)),
    )


if __name__ == '__main__':
    main()
