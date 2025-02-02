import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SHLDataset(Dataset):
    def __init__(self, mode,transform=None):
        self.mode = mode
        self.transform = transform
        if mode == 'train':
            self.data = np.load(file=file_path)
            self.label = np.load(file=label_path)
        elif mode == 'val':
            self.data = np.load(file=file_path)
            self.label = np.load(file=label_path)
        elif mode == 'test':
            self.data = np.load(file=test_file_path)
        else:
            raise ValueError('只能处于train、val、test模式其中之一')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        if self.transform:
            data = self.transform(data)
        if self.mode == 'test':
            return data
        else:
            return data, label

