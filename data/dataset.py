import os

from torch.utils.data import Dataset
import SimpleITK as sitk

from utils.utils import extract_random_patch


class CSIDataset(Dataset):
    """MICCAI 2014 Spine Challange Dataset"""

    def __init__(self,
                 dataset_path,
                 subset='train',
                 empty_interval=5,
                 flag_patch_norm=False,
                 flag_linear=False,
                 linear_att=1.0,
                 offset=1000.0):

        self.idx = 1
        self.empty_interval = empty_interval
        self.flag_linear = flag_linear
        self.flag_patch_norm = flag_patch_norm

        self.dataset_path = dataset_path
        self.subset = subset
        self.linear_att = linear_att
        self.offset = offset

        self.img_path = os.path.join(dataset_path, subset, 'img')
        self.mask_path = os.path.join(dataset_path, subset, 'seg')
        self.weight_path = os.path.join(dataset_path, subset, 'weight')

        self.img_names = [f for f in os.listdir(self.img_path) if f.endswith('.mhd')]

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

        """
        linear transformation from 12bit reconstruction img to HU unit 
        depend on the original data (CSI data value is from 0 ~ 4095)
        """
        if self.flag_linear:
            img = img * self.linear_att - self.offset

        # extract a traning patche
        img_patch, ins_patch, gt_patch, weight_patch, c_label = extract_random_patch(img,
                                                                                     mask,
                                                                                     weight,
                                                                                     self.idx,
                                                                                     self.subset,
                                                                                     self.empty_interval)

        if self.flag_patch_norm:
            img_patch = (img_patch - img_patch.mean()) / img_patch.std()

        self.idx += 1

        return img_patch, ins_patch, gt_patch, weight_patch, c_label
