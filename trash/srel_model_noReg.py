from model.estimate_eta import Estimate_eta
# from .estimate_rho import Estimate_rho
import torch
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F



class SREL(nn.Module):
    def __init__(self, constants):
        super(SREL, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        # Nt = constants['Nt']
        # N = constants['N']
        # Nr = constants['Nr']
        self.M = constants['M']
        # Lj = constants['Lj']
        # Lw = constants['Lw']
        # Lv = constants['Lv']
        self.Ls = constants['Nt']*constants['N']
        
        self.est_eta_module1 = \
            Estimate_eta(2*self.Ls + 2*constants['Lw'] + constants['Lv'], self.Ls)
            
        self.est_eta_module2 = \
            Estimate_eta(2*self.Ls + 2*constants['Lw'] + constants['Lv'], self.Ls)
        
    def forward(self, phi_batch, w_M_batch, v_M):
        batch_size = phi_batch.size(0)
        
        phi_optimal_batch = torch.zeros_like(phi_batch)
        
        for i in range(batch_size):
            phi = phi_batch[i]
            w_M = w_M_batch[i]
            
            s = self.compute_s(phi)
            eta_M = torch.zeros(self.Ls, self.M)
            for m in range(self.M):
                x = torch.cat((s.real, s.imag, w_M[:,m].real, w_M[:,m].imag, v_M[:,m]), dim=0)
                eta_M[:,m] = self.est_eta_module1(x) 
            
            eta = torch.sum(eta_M, dim=1)
            phi_optimal_batch[i,:] = phi - eta  # Update phi
            
        return phi_optimal_batch

    def compute_s(self, phi):
        magnitude = 1 / torch.sqrt(torch.tensor(self.Ls, dtype=torch.float))  # NtN = Ls = 128
        s = magnitude * torch.exp(1j * phi)
        return s