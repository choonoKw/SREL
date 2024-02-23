# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:20:26 2024

@author: jbk5816
"""

from model.estimate_eta import Estimate_eta
from model.estimate_rho import Estimate_rho
import torch
import torch.nn as nn
# import numpy as np

class SREL(nn.Module):
    def __init__(self, constants):
        super(SREL, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        
        self.est_eta_module1 = \
            Estimate_eta(2*self.Ls + 2*constants['Lw'] + constants['Lv'], self.Ls)
            
        self.est_rho_module1 = \
            Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Lv'], 1)
        
    def forward(self, phi_batch, w_M_batch, v_M):
        batch_size = phi_batch.size(0)
        
        phi_optimal_batch = torch.zeros_like(phi_batch)
        
        eta_M_batch = torch.zeros(batch_size, self.Ls, self.M)
        for i in range(batch_size):
            phi = phi_batch[i]
            w_M = w_M_batch[i]
            
            s = self.compute_s(phi)
            eta_M = torch.zeros(self.Ls, self.M)
            rho_M = torch.zeros(self.M)
            for m in range(self.M):
                x = torch.cat((s.real, s.imag, w_M[:,m].real, w_M[:,m].imag, v_M[:,m]), dim=0)
                eta_M[:,m] = self.est_eta_module1(x) 
                rho_M[m] = self.est_rho_module1(x).squeeze() 
            
            eta = torch.sum(rho_M*eta_M, dim=1)
            
            phi_optimal_batch[i,:] = phi - eta  # Update phi and save
            eta_M_batch[i,:,:] = eta_M # Save eta_M_batch
            
        return phi_optimal_batch, eta_M_batch

    def compute_s(self, phi):
        magnitude = 1 / torch.sqrt(torch.tensor(self.Ls, dtype=torch.float))  # NtN = Ls = 128
        s = magnitude * torch.exp(1j * phi)
        return s