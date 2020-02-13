import torch
import SimpleITK as sitk
from pathlib import Path
from data.dataset import CSIDataset
from torch.utils.data import Dataset, DataLoader

crop_img = '../crop_isotropic_dataset'
batch_size = 1

train_dataset = CSIDataset(crop_img)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

img_patch, ins_patch, gt_patch, weight, c_label = next(iter(train_dataloader))

img_patch = torch.squeeze(img_patch)
ins_patch = torch.squeeze(ins_patch)
gt_patch = torch.squeeze(gt_patch)
weight = torch.squeeze(weight)

assert img_patch.shape == (128, 128, 128)
assert ins_patch.shape == (128, 128, 128)
assert gt_patch.shape == (128, 128, 128)
assert weight.shape == (128, 128, 128)

# store patches for visualization
Path('./samples/').mkdir(parents=True, exist_ok=True)
sitk.WriteImage(sitk.GetImageFromArray(img_patch.numpy()), './samples/img.nrrd', True)
sitk.WriteImage(sitk.GetImageFromArray(gt_patch.numpy()), './samples/gt.nrrd', True)
sitk.WriteImage(sitk.GetImageFromArray(ins_patch.numpy()), './samples/ins.nrrd', True)
sitk.WriteImage(sitk.GetImageFromArray(weight.numpy()), './samples/wei.nrrd', True)