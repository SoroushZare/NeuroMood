import torch
from torch.utils.data import Dataset
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]