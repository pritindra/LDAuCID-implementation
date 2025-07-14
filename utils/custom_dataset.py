# utils/custom_dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None, labeled=True):
        self.data_dict = torch.load(file_path, map_location='cpu', weights_only=False)
        self.data = self.data_dict['data']  # tensor of shape (N, H, W, C) or (N, F)
        self.transform = transform
        self.labeled = labeled
        if labeled:
            self.targets = self.data_dict.get('targets', None)  # tensor of shape (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 3 and img.shape[-1] == 3:  # HWC image
            img = img.permute(2, 0, 1)  # Convert to CHW
        img = img.float() / 255.0  # Normalize

        if self.transform:
            img = self.transform(img)

        if self.labeled:
            label = self.targets[idx]
            return img, label
        else:
            return img


# Utility function to load dataset object
def load_dataset(domain_idx, train=True, transform=None):
    base = "../dataset/dataset/part_one_dataset"
    subdir = "train_data" if train else "eval_data"
    filename = f"{domain_idx}_{'train' if train else 'eval'}_data.tar.pth"
    path = os.path.join(base, subdir, filename)
    labeled = train and domain_idx == 1 or not train  # Only domain 1 is labeled in training, all eval are labeled
    return CustomDataset(path, transform=transform, labeled=labeled)
