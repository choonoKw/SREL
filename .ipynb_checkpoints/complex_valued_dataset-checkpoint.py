from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch

class ComplexValuedDataset(Dataset):
    def __init__(self, mat_file):
        data = loadmat(mat_file)
        self.G_list = torch.tensor(data['G_list'], dtype=torch.complex64)
        self.H_list = torch.tensor(data['H_list'], dtype=torch.complex64)
        self.L = self.G_list.shape[0]  # Assume square matrices and same dimensions
        self.N = self.G_list.shape[2]  # Number of samples

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        G = self.G_list[:, :, idx]
        H = self.H_list[:, :, idx]
        phi_initial = torch.rand(G.shape[0], dtype=torch.float)
        return phi_initial, G, H