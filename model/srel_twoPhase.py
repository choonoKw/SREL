# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:09:49 2024

@author: jbk5816
"""

from model.estimate_eta import Estimate_eta
from model.estimate_rho import Estimate_rho
import torch
import torch.nn as nn

class SREL_intra(nn.Module):
    def __init__(self, constants):
        super(SREL_intra, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        
        # Dynamically create the modules for estimating eta and rho
        self.est_eta_modules = nn.ModuleList([
            Estimate_eta(2*self.Ls + 2*constants['Lw'] + constants['Lv'], self.Ls)
            for _ in range(self.N_step)
        ])
        self.est_rho_modules = nn.ModuleList([
            Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Lv'], 1)
            for _ in range(self.N_step)
        ])
        
    def forward(self, phi0_batch, w_M_batch, v_M):
        batch_size = phi0_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        
        # Initialize the list
        s_list_batch = torch.zeros(batch_size, self.Ls, N_step+1, dtype=torch.complex64)
        eta_M_list_batch = torch.zeros(batch_size, self.Ls, self.M, N_step) 
        
        for i in range(batch_size):
            phi0 = phi0_batch[i]
            w_M = w_M_batch[i]
            
            
            # Repeat the update process N_step times
            phi = phi0
            for update_step in range(N_step):
                # s = self.compute_s(phi)
                s = modulus*torch.exp(1j *phi)
                eta_M = torch.zeros(self.Ls, self.M)
                rho_M = torch.zeros(self.M)  # Ensure correct broadcasting shape
                
                for m in range(self.M):
                    x = torch.cat((s.real, s.imag, w_M[:,m].real, w_M[:,m].imag, v_M[:,m]), dim=0)
                    eta_M[:,m] = self.est_eta_module(x)
                    rho_M[m] = self.est_rho_module1(x).squeeze()  # Remove extra dimensions
                
                # Element-wise multiplication and sum across the M dimension
                eta = torch.sum(eta_M * rho_M, dim=1)
                eta = eta.to(self.device)
                phi = phi - eta  # Update phi
                
                # save on list
                s_list_batch[i,:,update_step] = s
                eta_M_list_batch[i,:,:,update_step] = eta_M
            
            s_list_batch[i,:,N_step] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
        model_outputs = {
            's_list_batch': s_list_batch,
            'eta_M_list_batch': eta_M_list_batch
        }
            
        # return {
        #     's_list_batch': s_list_batch,
        #     'eta_M_list_batch': eta_M_list_batch
        # }
        return model_outputs