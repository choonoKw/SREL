# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:51:22 2024

@author: jbk5816
"""

from model.estimate_rho_sred import Estimate_rho

from utils.input_standardize import standardize

import torch
import torch.nn as nn

class SRED_rho(nn.Module):
    def __init__(self, constants):
        super(SRED_rho, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        # Dynamically create the modules for estimating eta and rho
        self.est_rho_modules = Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
        
        
    def forward(self, phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        # M = self.M
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        rho_M_stack_batch = torch.zeros(batch_size, N_step, self.M).to(self.device)
        
        for idx_batch in range(batch_size):
            phi0 = phi_batch[idx_batch]
            w_M = w_M_batch[idx_batch]
            G_M = G_M_batch[idx_batch]
            H_M = H_M_batch[idx_batch]
            
            
            # Repeat the update process N_step times
            phi = phi0
            for update_step in range(N_step):
                s = modulus*torch.exp(1j *phi)
                
                eta_net = torch.zeros(self.Ls).to(self.device)
                        
                for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=2),
                                               torch.unbind(H_M, dim=2))):
                    
                    w = w_M[:,m]
                    y = y_M[:,m]
                    
                    x = standardize(s, w, y)
                    # x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
                    
                    
                    sGs = torch.vdot(s, torch.matmul(G, s))  # Equivalent to s'*G*s in MATLAB
                    sHs = torch.vdot(s, torch.matmul(H, s))  # Equivalent to s'*H*s in MATLAB
                    Gs = torch.matmul(G, s)  # G*s
                    Hs = torch.matmul(H, s)  # H*s
                    eta = torch.real(2 / (sHs ** 2) * torch.imag((sHs * Gs - sGs * Hs) * torch.conj(s)))
                    
                    
                    rho = self.est_rho_modules(x)
                    
                    eta_net += rho*eta
                    
                    rho_M_stack_batch[idx_batch,update_step,m] = rho
                
                phi = phi - eta_net  # Update phi
                
                # save on list
                s_stack_batch[idx_batch,update_step,:] = s
                # rho_stack_batch[idx_batch,update_step] = rho
            
            s_stack_batch[idx_batch,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
        
        # # Initialize the list
        # s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        # rho_sum_stack_batch = torch.zeros(batch_size, N_step).to(self.device)
        
        
        # for idx_batch in range(batch_size):
        #     phi0 = phi_batch[idx_batch]
        #     w_M = w_M_batch[idx_batch]
        #     G_M = G_M_batch[idx_batch]
        #     H_M = H_M_batch[idx_batch]
            
        #     # Repeat the update process N_step times
        #     phi = phi0
        #     for update_step in range(N_step):
        #         s = modulus*torch.exp(1j *phi)
                
        #         eta_net = torch.zeros(self.Ls).to(self.device)
                
        #         for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=2),
        #                                                 torch.unbind(H_M, dim=2))):
                    
        #             sGs = torch.vdot(s, torch.matmul(G, s))  # Equivalent to s'*G*s in MATLAB
        #             sHs = torch.vdot(s, torch.matmul(H, s))  # Equivalent to s'*H*s in MATLAB
        #             Gs = torch.matmul(G, s)  # G*s
        #             Hs = torch.matmul(H, s)  # H*s
        #             eta = torch.real(2 / (sGs ** 2) * torch.imag((sHs * Gs - sGs * Hs) * torch.conj(s)))
                    
        #             w = w_M[:,m]
        #             y = y_M[:,m]
        #             x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
        #             rho = self.est_rho_modules(x)
                    
        #             eta_net += rho*eta
                    
        #             rho_sum_stack_batch[idx_batch,update_step] += rho.item()
                
        #         phi = phi - eta_net  # Update phi
                
        #         # save on list
                
        #         s_stack_batch[idx_batch,update_step,:] = s
        #         # eta_stack_batch[idx_batch,update_step,:] = eta
            
        #     s_stack_batch[idx_batch,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
            
        # return s_stack_batch
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'rho_M_stack_batch': rho_M_stack_batch
        }
        return model_outputs
