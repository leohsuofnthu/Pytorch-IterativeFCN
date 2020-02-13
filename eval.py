import os
import argparse
import logging
import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import dc

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Iterative Fully Convolutional Network')
    parser.add_argument('--label_dir', type=str, default='./crop_isotropic_dataset/test/seg',
                        help='folder of test label')
    parser.add_argument('--pred_dir', type=str, default='./pred',
                        help='folder of pred masks')
    args = parser.parse_args()

    labels = [os.path.join(args.label_dir, x) for x in os.listdir(os.path.join(args.label_dir)) if 'raw' not in x]
    preds = [os.path.join(args.pred_dir, x) for x in os.listdir(os.path.join(args.pred_dir)) if 'raw' not in x]

    n = 0
    avg_dc = 0.
    for l, p in zip(labels, preds):
        logging.info("Process %s and %s" % (p, l))
        label = sitk.GetArrayFromImage(sitk.ReadImage(l))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(p))
        for i in np.unique(label):
            l = label[label == i]
            p = pred[label == i]
            l[l > 0] = 1
            p[p > 0] = 1
            avg_dc += dc(p, l)
            n += 1

    logging.info("Average Dice Coefficient for %s individual vertebrae test : %s" % (n, avg_dc / n))


if __name__ == '__main__':
    main()
