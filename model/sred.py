# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:51:22 2024

@author: jbk5816
"""

from model.estimate_rho_sred import Estimate_rho
from model.estimate_rho import Estimate_rho_BN

# from utils.input_standardize import standardize

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
        self.est_rho_modules = nn.ModuleList([
            Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
            for _ in range(self.N_step)
        ])
        # self.est_rho_modules = Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
        
        
        
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
                    
                    # x = standardize(s, w, y)
                    x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
                    
                    
                    sGs = torch.abs(torch.vdot(s, torch.matmul(G, s)))  # Equivalent to s'*G*s in MATLAB
                    sHs = torch.abs(torch.vdot(s, torch.matmul(H, s)))  # Equivalent to s'*H*s in MATLAB
                    Gs = torch.matmul(G, s)  # G*s
                    Hs = torch.matmul(H, s)  # H*s
                    eta = torch.real(2 / (sHs ** 2) * torch.imag( (sHs * Gs - sGs * Hs) * torch.conj(s) ) )
                    
                    
                    rho = self.est_rho_modules[update_step](x)
                    
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
    
    
class SRED_rep_rho(nn.Module):
    def __init__(self, constants):
        super(SRED_rep_rho, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        # Dynamically create the modules for estimating eta and rho
        self.est_rho_module = Estimate_rho(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
        
        
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
                    
                    # x = standardize(s, w, y)
                    x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
                    
                    
                    sGs = torch.vdot(s, torch.matmul(G, s))  # Equivalent to s'*G*s in MATLAB
                    sHs = torch.vdot(s, torch.matmul(H, s))  # Equivalent to s'*H*s in MATLAB
                    Gs = torch.matmul(G, s)  # G*s
                    Hs = torch.matmul(H, s)  # H*s
                    eta = torch.real(2 / (sHs ** 2) * torch.imag((sHs * Gs - sGs * Hs) * torch.conj(s)))
                    
                    
                    rho = self.est_rho_module(x)
                    
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

def eta_sred_bn(s_batch,G_batch, H_batch):
    s_batch_unsqueezed = s_batch.unsqueeze(-1)
    
    Gs_batch = torch.bmm(G_batch, s_batch_unsqueezed)
    Hs_batch = torch.bmm(H_batch, s_batch_unsqueezed)
    
    sGs_batch = torch.einsum('bi,bi->b', s_batch.conj(), Gs_batch)
    sHs_batch = torch.einsum('bi,bi->b', s_batch.conj(), Hs_batch)

    
    
    # sGs = torch.vdot(s_batch, torch.matmul(G_batch, s_batch))  # Equivalent to s'*G*s in MATLAB
    # sHs = torch.vdot(s_batch, torch.matmul(H_batch, s_batch))  # Equivalent to s'*H*s in MATLAB
    # Gs = torch.matmul(G_batch, s_batch)  # G*s
    # Hs = torch.matmul(H_batch, s_batch)  # H*s
    eta_batch = torch.real(2 / (sHs_batch ** 2)
                           * torch.imag((sHs_batch * Gs_batch - sGs_batch * Hs_batch) * torch.conj(s_batch)))
    # for idx_batch in range(batch_size):
    #     phi0 = phi_batch[idx_batch]
    #     w_M = w_M_batch[idx_batch]
    #     G_M = G_M_batch[idx_batch]
    #     H_M = H_M_batch[idx_batch]
        
    #     eta_net = torch.zeros(self.Ls).to(self.device)
    #     for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=2),
    #                                        torch.unbind(H_M, dim=2))):
    return eta_batch
    
    
class SRED_rho_rep(nn.Module):
    def __init__(self, constants):
        super(SRED_rho_rep, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        # Dynamically create the modules for estimating eta and rho
        self.est_rho_modules = Estimate_rho_BN(2*self.Ls + 2*constants['Lw'] + constants['Ly'], 1)
        
        
    def forward(self, phi_batch, w_M_batch, y_M_batch, G_M_batch, H_M_batch):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        # M = self.M
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        rho_M_stack_batch = torch.zeros(batch_size, N_step, self.M).to(self.device)
        
        
        # for idx_batch in range(batch_size):
        #     phi = phi_batch[idx_batch]
        #     w_M = w_M_batch[idx_batch]
        #     G_M = G_M_batch[idx_batch]
        #     H_M = H_M_batch[idx_batch]
            
            
        # Repeat the update process N_step times
        for update_step in range(N_step):
            s_batch = modulus*torch.exp(1j *phi_batch)

            eta_net_batch = torch.zeros(batch_size,self.Ls).to(self.device)

            for m, (G_batch, H_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                       torch.unbind(H_M_batch, dim=3))):

                w_batch = w_M_batch[:,:,m]
                y_batch = y_M_batch[:,:,m]

                # x = standardize(s, w, y)
                x_batch = torch.cat((s_batch.real, s_batch.imag, w_batch.real, w_batch.imag, y_batch), dim=0)
                print(torch.size(x_batch))
                
                eta_batch = eta_sred_bn(s_batch, G_M_batch, H_M_batch)

                


                rho_batch = self.est_rho_modules(x_batch)

                eta_net_batch += rho_batch*eta_batch

                rho_M_stack_batch[:,update_step,m] = rho_batch

            phi_batch = phi_batch - eta_net_batch  # Update phi

            # save on list
            
            s_stack_batch[:,update_step,:] = s_batch
            # rho_stack_batch[idx_batch,update_step] = rho

        s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi_batch)  # Saving the final s after all updates
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'rho_M_stack_batch': rho_M_stack_batch
        }
        return model_outputs
