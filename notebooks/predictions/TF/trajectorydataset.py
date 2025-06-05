import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y, X_mask=None, Y_mask=None):
        
        self.max_x = 12000
        self.max_y = 7000

        # Normaliza
        X[..., 0] /= self.max_x
        X[..., 1] /= self.max_y
        Y[..., 0] /= self.max_x
        Y[..., 1] /= self.max_y

        # Ensure X and Y are numpy arrays
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.X_mask = torch.tensor(X_mask, dtype=torch.float32) if X_mask is not None else None
        self.Y_mask = torch.tensor(Y_mask, dtype=torch.float32) if Y_mask is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.X_mask is not None:
            return self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx]
        return self.X[idx], self.Y[idx]