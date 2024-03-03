# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:38:34 2024

@author: jbk5816
"""

from model.estimate_eta import Estimate_eta
from model.estimate_rho import Estimate_rho
import torch
import torch.nn as nn

class SREL_intra_rep(nn.Module):
    def __init__(self, constants):
        super(SREL_intra_rep, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        # Dynamically create the modules for estimating eta and rho
        self.est_eta_modules = Estimate_eta(2*self.Ls + 2*constants['Lw'] + constants['Ly'], self.Ls)
        
        self.est_rho_modules = Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
        
    def forward(self, phi_batch, w_batch, y):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64)
        eta_stack_batch = torch.zeros(batch_size, N_step, self.Ls) 
        
        for idx_batch in range(batch_size):
            phi0 = phi_batch[idx_batch]
            w = w_batch[idx_batch]
            
            
            # Repeat the update process N_step times
            phi = phi0
            for update_step in range(N_step):
                s = modulus*torch.exp(1j *phi)
                
                x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
                eta = self.est_eta_modules(x)
                beta = self.est_rho_modules(x)
                
                phi = phi - beta*eta  # Update phi
                
                # save on list
                s_stack_batch[idx_batch,update_step,:] = s
                eta_stack_batch[idx_batch,update_step,:] = eta
            
            s_stack_batch[idx_batch,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'eta_stack_batch': eta_stack_batch
        }
        return model_outputs