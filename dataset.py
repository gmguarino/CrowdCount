import torch
from torch.utils.data import Dataset
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import os

from kernel import generate_density_map
from utils import transforms

class ShanghaiTech(Dataset):
    
    def __init__(self, roots, train=True, tranforms=None):
        self.root = roots

        if train:
            self.folder = "train_data"
        else:
            self.folder = "test_data"
        self.tranforms = tranforms

        self.data_paths = [
            os.path.join(self.root[0], self.folder),
            os.path.join(self.root[1], self.folder)
        ]
        self.images = [
            os.path.join(self.data_paths[0], "images"),
            os.path.join(self.data_paths[1], "images")
        ]
        self.gts = [
            os.path.join(self.data_paths[0], "ground_truth"),
            os.path.join(self.data_paths[1], "ground_truth")
        ]


    def __len__(self):
        return len(os.listdir(self.images[0])) + len(os.listdir(self.images[1]))

    def __getitem__(self, idx):
        if idx >= len(os.listdir(self.images[0])):
            folder_idx = 1
            image_id = idx - len(os.listdir(self.images[0]))
        else:
            folder_idx = 0
        
            image_id = idx + 1
        
        image_path = os.path.join(self.images[folder_idx], "IMG_{}.jpg".format(image_id))
        gt_path = os.path.join(self.gts[folder_idx], "GT_IMG_{}.mat".format(image_id))
        im = Image.open(image_path)
        gt = io.loadmat(gt_path)
        count, density_map, positions = ShanghaiTech.process_ground_truth(gt, (im.size[1], im.size[0]))
        if self.tranforms is not None:
            im = self.tranforms(im, target=False)
            density_map = self.tranforms(torch.tensor(density_map, dtype=torch.float32), target=True)
        else:
            im = torch.tensor(np.array(im), dtype=torch.float32)
            density_map = torch.tensor(density_map, dtype=torch.float32)
        count = torch.tensor(count, dtype=torch.float32)
        target = dict()
        target["count"] = count
        target["map"] = density_map

        return im, target

    @staticmethod
    def process_ground_truth(ground_truth, shape):
        count = ground_truth['image_info'][0, 0][0, 0][1]
        positions = ground_truth['image_info'][0, 0][0, 0][0]
        density_map = generate_density_map(
            positions, 
            (shape[0], shape[1])
        )
        return count.item(), density_map, positions


# data = ShanghaiTech(["./part_A_final", "./part_B_final"])
# im, target = data.__getitem__(0)
# density_map = target["map"]
# plt.figure()

# plt.imshow(im.numpy().astype('int'))

# plt.figure()
# plt.imshow(density_map.numpy(), cmap='jet')

# plt.show()