from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch

class ComplexValuedDataset(Dataset):
    def __init__(self, mat_file):
        data = loadmat(mat_file)
        self.G_M_list = torch.tensor(data['G_M_list'], dtype=torch.complex64)
        self.H_M_list = torch.tensor(data['H_M_list'], dtype=torch.complex64)
        self.L = self.G_M_list.shape[0]  # Assume square matrices and same dimensions
        self.M = self.G_M_list.shape[2] # Numer of targets
        self.N_trd = self.G_M_list.shape[3]  # Number of samples

    def __len__(self):
        return self.N_trd

    def __getitem__(self, idx):
        G = self.G_M_list[:, :, :, idx]
        H = self.H_M_list[:, :, :, idx]
        phi_initial = torch.rand(self.L, dtype=torch.float)
        return phi_initial, G, H