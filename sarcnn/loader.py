"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling data loader implementation
"""

# Imports

import numpy as np
from PIL import Image, UnidentifiedImageError

from pathlib import Path
from random import randint

import torch
from torch.utils.data import Dataset

from sarcnn.utils import normalize_sar, inject_speckle_amplitude


class NoisyDataset(Dataset):
    def __init__(self, root_dir, transform=None, shuffle=False):
        super(NoisyDataset, self).__init__()
        root = Path(root_dir)
        self.transform = transform
        self.shuffle = shuffle

        self.images_files = [*root.glob('*.npy')]

    def __len__(self):
        return len(self.images_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        img_path = self.images_files[idx]

        try:
            img = np.load(img_path)
            img = torch.from_numpy(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.float()

            if self.transform:
                img = self.transform(img)

            speckle_img = inject_speckle_amplitude(img, 1)

            img = normalize_sar(img)
            img = img / 255.

            speckle_img = normalize_sar(speckle_img)
            speckle_img = speckle_img / 255.

        except UnidentifiedImageError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {img_path}.")
            print(f"Skipping index {idx}.")
            return self.skip_sample(idx)

        # success
        
        return img, speckle_img
