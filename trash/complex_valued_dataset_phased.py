from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch

class ComplexValuedDataset(Dataset):
    def __init__(self, mat_file, stage=1):
        data = loadmat(mat_file)
        self.G_M_list = torch.tensor(data['G_M_list'], dtype=torch.complex64)
        self.H_M_list = torch.tensor(data['H_M_list'], dtype=torch.complex64)
        self.phi_list = torch.tensor(data['phi_list'], dtype=torch.complex64)
        self.w_M_list = torch.tensor(data['w_M_list'], dtype=torch.complex64)
        
        # Ls, _, M, N_trd = self.G_M_list.shape
        # self.Ls = self.G_M_list.shape[0]
        
        # self.stage = stage
        
    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        if self.stage == 1:
            return 3*self.G_M_list.shape[-1]  # Number of training samples
            
        elif self.stage == 2:
            return self.G_M_list.shape[-1]  # Number of training samples

    def __getitem__(self, idx):
        if self.stage == 1:
            # adjust shape to train intra-target
            Ls, _, M, N_trd = self.G_M_list.shape
            Lw = self.w_M_list.shape[0]
            phi_repeat_list = self.phi_list.repeat_interleave(M, dim=1)
            w_list = self.w_M_list.permute(0, 2, 1).reshape(Ls, Ls, M*Lw)
            G_list = self.G_M_list.permute(0, 1, 3, 2).reshape(Ls, Ls, M*N_trd)
            H_list = self.H_M_list.permute(0, 1, 3, 2).reshape(Ls, Ls, M*N_trd)
            
            return phi_repeat_list, w_list, G_list, H_list
            
        elif self.stage == 2:
            phi = self.phi_list[:,idx]
            w_M = self.w_M_list[:,:,idx]
            G_M = self.G_M_list[:, :, :, idx]
            H_M = self.H_M_list[:, :, :, idx]
            
            return phi, w_M, G_M, H_M