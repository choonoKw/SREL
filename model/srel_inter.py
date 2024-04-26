# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:50:20 2024

@author: jbk5816
"""

# from model.estimate_eta import Estimate_eta
# from model.estimate_rho import Estimate_rho
from model.estimate_mu import Estimate_mu

import torch
import torch.nn as nn
from torch.nn import ModuleList

class SREL_rep_mu(nn.Module):
    def __init__(self, constants, model_intra_phase2):
        super(SREL_rep_mu, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra_phase2 = model_intra_phase2
        self.model_intra_phase1 = model_intra_phase2.model_intra_phase1
        
        # Dynamically create the modules for estimating eta and rho
        self.est_mu_modules = Estimate_mu(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
        
    def forward(self, phi0_batch, w_M_batch, y_M):
        batch_size = phi0_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        M = self.M
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        mu_stack_batch = torch.zeros(batch_size, N_step, M).to(self.device)
        
        if isinstance(self.model_intra_phase2.est_eta_modules, ModuleList):
            # model_intra_phase2 has various NN modules.
            
            if isinstance(self.model_intra_phase1.est_rho_modules, ModuleList):
                # model_intra_phase1 has various NN modules.
                pass
                
            else:
                # model_intra_phase1 has repeated NN module.
                pass
                
        else: # model_intra_phase2 has repeated NN module.
            if isinstance(self.model_intra_phase1.est_rho_modules, ModuleList):
                # model_intra_phase1 has various NN modules.
                pass
                
            else:
                # model_intra_phase1 has repeated NN module.
                
                for idx_batch in range(batch_size):
                    phi0 = phi0_batch[idx_batch]
                    w_M = w_M_batch[idx_batch]
                    
                    # Repeat the update process N_step times
                    phi = phi0
                    for update_step in range(N_step):
                        s = modulus*torch.exp(1j *phi)
                        
                        eta_net = torch.zeros(self.Ls).to(self.device)
                        
                        for m in range(M):
                            w = w_M[:,m]
                            y = y_M[:,m]
                            x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
                            eta = self.model_intra_phase2.est_eta_modules(x)
                            rho = self.model_intra_phase1.est_rho_modules(x)
                            mu = self.est_mu_modules(x)
                            
                            eta_net += mu*rho*eta
                        
                        
                        
                        phi = phi - eta_net  # Update phi
                        
                        # save on list
                        
                        s_stack_batch[idx_batch,update_step,:] = s
                        # eta_stack_batch[idx_batch,update_step,:] = eta
                    
                    s_stack_batch[idx_batch,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
            
        # return s_stack_batch
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'mu_stack_batch': mu_stack_batch
        }
        return model_outputs

class SREL_vary_mu(nn.Module):
    def __init__(self, constants, model_intra):
        super(SREL_vary_mu, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra = model_intra
        
        # Dynamically create the modules for estimating eta and rho
        self.est_mu_modules = nn.ModuleList([
            Estimate_mu(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
            for _ in range(self.N_step)
        ])
        
    def forward(self, phi0_batch, w_M_batch, y_M):
        batch_size = phi0_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        M = self.M
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        mu_stack_batch = torch.zeros(batch_size, N_step, M).to(self.device)
        
        for idx_batch in range(batch_size):
            phi0 = phi0_batch[idx_batch]
            w_M = w_M_batch[idx_batch]
            
            # Repeat the update process N_step times
            phi = phi0
            for update_step in range(N_step):
                s = modulus*torch.exp(1j *phi)
                
                eta_net = torch.zeros(self.Ls).to(self.device)
                
                for m in range(M):
                    w = w_M[:,m]
                    y = y_M[:,m]
                    x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
                    eta = self.model_intra.est_eta_modules[update_step](x)
                    rho = self.model_intra.est_rho_modules[update_step](x)
                    mu = self.est_mu_modules[update_step](x)
                    
                    eta_net += mu*rho*eta
                
                
                
                phi = phi - eta_net  # Update phi
                
                # save on list
                
                s_stack_batch[idx_batch,update_step,:] = s
                # eta_stack_batch[idx_batch,update_step,:] = eta
            
            s_stack_batch[idx_batch,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
            
        # return s_stack_batch
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'mu_stack_batch': mu_stack_batch
        }
        return model_outputs