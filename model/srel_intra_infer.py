"""
Created on Tue April 2 2024

@author: jbk5816

Test SREL_intra models
"""

import torch
import torch.nn as nn
from torch.nn import ModuleList

from sred.functions import eta_sred


class SREL_intra_phase1_infer(nn.Module):
    def __init__(self, constants, model_intra_phase1):
        super(SREL_intra_phase1_infer, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra_phase1 = model_intra_phase1
        
    def forward(self, phi, w_M_batch, y_M, G_M, H_M):
        # batch_size = phi.size(0)
        N_step = self.N_step
        modulus = self.modulus
        # M = self.M
        Ls = self.Ls
        
        # Initialize the list
        # s_stack_batch = torch.zeros(batch_size, N_step+1, Ls, dtype=torch.complex64).to(self.device)
        # rho_M_stack_batch = torch.zeros(batch_size, N_step, M).to(self.device)
        
        if isinstance(self.model_intra_phase1.est_rho_modules, ModuleList):
            # model_intra_phase1_phase1 has various NN modules.
        
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s = modulus*torch.exp(1j *phi)    
                
                eta_net = torch.zeros(Ls).to(self.device)
                
                for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=-1),
                                                           torch.unbind(H_M, dim=-1))):
                    w = w_M_batch[:,:,m]
                    y = y_M[:,m]
                    # y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                    x = torch.cat((s.real, s.imag, w.real, w.imag, y))
                    
                    rho = self.model_intra_phase1.est_rho_modules[update_step](x)
                    
                    eta = eta_sred(G, H, s)
                    
                    eta_net += rho*eta
                    
                    # save
                    # rho_M_stack_batch[:,update_step,m] = rho.squeeze()
                
                
                # Update phi
                phi = phi - rho*eta_net  
                
                # s_stack_batch[:,update_step,:] = s
        
        else:
            # Repeat the update process N_step times
            for update_step in range(N_step):
                s = modulus*torch.exp(1j *phi)    
                
                eta_net = torch.zeros(Ls).to(self.device)
                
                for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=-1),
                                                           torch.unbind(H_M, dim=-1))):
                    w = w_M_batch[:,m]
                    y = y_M[:,m]
                    # y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                    x = torch.cat((s.real, s.imag, w.real, w.imag, y))
                    
                    rho = self.model_intra_phase1.est_rho_modules(x)
                    
                    eta = eta_sred(G, H, s)
                    
                    eta_net += rho*eta
                    
                    # save
                    # rho_M_stack_batch[:,update_step,m] = rho.squeeze()
                
                
                # Update phi
                phi = phi - rho*eta_net  
                
                # s_stack_batch[:,update_step,:] = s
        
        # s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
        # model_outputs = {
        #     's_stack_batch': s_stack_batch,
        #     'rho_M_stack_batch': rho_M_stack_batch
        # }
        
        return phi
    
    
class SREL_intra_phase2_infer(nn.Module):
    def __init__(self, constants, model_intra_phase2):
        super(SREL_intra_phase2_infer, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra_phase2 = model_intra_phase2
        
    def forward(self, phi, w_M_batch, y_M, G_M, H_M):
        # batch_size = phi.size(0)
        N_step = self.N_step
        modulus = self.modulus
        # M = self.M
        Ls = self.Ls
        device = self.device
        
        model_intra_phase2 = self.model_intra_phase2
        model_intra_phase1 = model_intra_phase2.model_intra_phase1
        # device = model_intra_phase2.device
        
        # Initialize the list
        # s_stack_batch = torch.zeros(batch_size, N_step+1, Ls, dtype=torch.complex64).to(device)
        # rho_M_stack_batch = torch.zeros(batch_size, N_step, M).to(device)
        
        if isinstance(model_intra_phase2.est_eta_modules, ModuleList):
            # model_intra_phase2 has various NN modules.
            
            if isinstance(model_intra_phase1.est_rho_modules, ModuleList):
                # model_intra_phase1 has various NN modules.
                
                # Repeat the update process N_step times
                for update_step in range(N_step):
                    s = modulus*torch.exp(1j *phi)    
                    
                    eta_net = torch.zeros(Ls).to(device)
                    
                    for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=-1),
                                                               torch.unbind(H_M, dim=-1))):
                        w = w_M_batch[:,:,m]
                        y = y_M[:,m]
                        # y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                        x = torch.cat((s.real, s.imag, w.real, w.imag, y))
                        
                        rho = model_intra_phase1.est_rho_modules[update_step](x)
                        
                        eta = model_intra_phase2[update_step](x)
                        
                        eta_net += rho*eta
                        
                        # save
                        # rho_M_stack_batch[:,update_step,m] = rho.squeeze()
                    
                    
                    # Update phi
                    phi = phi - rho*eta_net  
                    
                    # s_stack_batch[:,update_step,:] = s
                    
            else: # model_intra_phase1 repeat a single NN module
                
                # Repeat the update process N_step times
                for update_step in range(N_step):
                    s = modulus*torch.exp(1j *phi)    
                    
                    eta_net = torch.zeros(Ls).to(device)
                    
                    for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=-1),
                                                               torch.unbind(H_M, dim=-1))):
                        w = w_M_batch[:,:,m]
                        y = y_M[:,m]
                        # y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                        x = torch.cat((s.real, s.imag, w.real, w.imag, y))
                        
                        rho = model_intra_phase1.est_rho_modules[update_step](x)
                        
                        eta = model_intra_phase2.est_eta_modules(x)
                        
                        eta_net += rho*eta
                        
                        # save
                        # rho_M_stack_batch[:,update_step,m] = rho.squeeze()
                    
                    
                    # Update phi
                    phi = phi - rho*eta_net  
                    
                    # s_stack_batch[:,update_step,:] = s
                        
        
        else: # model_intra_phase2 repeat a single NN module
        
            if isinstance(model_intra_phase1.est_rho_modules, ModuleList):
                # model_intra_phase1 has various NN modules.
                
                # Repeat the update process N_step times
                for update_step in range(N_step):
                    s = modulus*torch.exp(1j *phi)    
                    
                    eta_net = torch.zeros(Ls).to(device)
                    
                    for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=-1),
                                                               torch.unbind(H_M, dim=-1))):
                        w = w_M_batch[:,:,m]
                        y = y_M[:,m]
                        # y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                        x = torch.cat((s.real, s.imag, w.real, w.imag, y))
                        
                        rho = model_intra_phase1.est_rho_modules[update_step](x)
                        
                        eta = model_intra_phase2.est_eta_modules(x)
                        
                        eta_net += rho*eta
                        
                        # save
                        # rho_M_stack_batch[:,update_step,m] = rho.squeeze()
                    
                    
                    # Update phi
                    phi = phi - rho*eta_net  
                    
                    # s_stack_batch[:,update_step,:] = s
            
            else: # model_intra_phase1 repeat a single NN module
                # Repeat the update process N_step times
                for update_step in range(N_step):
                    s = modulus*torch.exp(1j *phi)    
                    
                    eta_net = torch.zeros(Ls).to(device)
                    
                    for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=-1),
                                                               torch.unbind(H_M, dim=-1))):
                        w = w_M_batch[:,:,m]
                        y = y_M[:,m]
                        # y_batch = y.repeat(batch_size, 1) # cloned for batch_size times
                        x = torch.cat((s.real, s.imag, w.real, w.imag, y))
                        
                        rho = model_intra_phase1.est_rho_modules(x)
                        
                        eta = model_intra_phase2.est_eta_modules(x)
                        
                        eta_net += rho*eta
                        
                        # save
                        # rho_M_stack_batch[:,update_step,m] = rho.squeeze()
                    
                    
                    # Update phi
                    phi = phi - rho*eta_net  
                    
                    # s_stack_batch[:,update_step,:] = s
        
        # s_stack_batch[:,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
        # model_outputs = {
        #     's_stack_batch': s_stack_batch,
        #     'rho_M_stack_batch': rho_M_stack_batch
        # }
        
        return phi

# class SREL_intra_tester_prev(nn.Module):
#     def __init__(self, constants, model_intra):
#         super(SREL_intra_tester_prev, self).__init__()
#         # Unpack constants from the dictionary and store as attributes
#         self.M = constants['M']
#         self.Ls = constants['Nt']*constants['N']
#         self.N_step = constants['N_step']
#         self.modulus = constants['modulus']
        
#         self.model_intra = model_intra
        
#     def forward(self, phi, w_M_batch, y_M):
#         # batch_size = phi.size(0)
#         N_step = self.N_step
#         modulus = self.modulus
#         M = self.M
        
#         # Initialize the list
#         s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
#         mu_stack_batch = torch.zeros(batch_size, N_step, M).to(self.device)
        
#         for idx in range(batch_size):
#             phi0 = phi[idx]
#             w_M = w_M_batch[idx]
            
#             # Repeat the update process N_step times
#             phi = phi0
#             for update_step in range(N_step):
#                 s = modulus*torch.exp(1j *phi)
                
#                 eta_net = torch.zeros(self.Ls).to(self.device)
                
#                 for m in range(M):
#                     w = w_M[:,m]
#                     y = y_M[:,m]
#                     x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
#                     eta = self.model_intra.est_eta_modules[update_step](x)
#                     rho = self.model_intra.est_rho_modules[update_step](x)
                    
#                     eta_net += rho*eta
                
                
                
#                 phi = phi - eta_net  # Update phi
                
#                 s_stack_batch[idx,update_step,:] = s
#                 # eta_stack_batch[idx,update_step,:] = eta
            
#             s_stack_batch[idx,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
            
#         # return s_stack_batch
            
        
#         return s_stack_batch