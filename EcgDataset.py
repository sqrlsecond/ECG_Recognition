from torch.utils.data import Dataset
import torch
import numpy as np


class EcgDataset(Dataset):
    def __init__(self, ecgs, labels):
        self.ecgs = torch.Tensor(ecgs)
        self.labels = torch.Tensor(labels)
    
    def __getitem__(self, idx):
        return self.ecgs[idx], self.label[idx]

    def __len__(self):
        return len(self.labels)


"""
arr = np.zeros((5,3,3))
tens = torch.Tensor(arr)
print(tens[0])
"""