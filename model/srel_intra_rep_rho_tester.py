import torch
import torch.nn as nn

class SREL_intra_rep_rho_tester(nn.Module):
    def __init__(self, constants, model_intra):
        super(SREL_intra_rep_rho_tester, self).__init__()
        # Unpack constants from the dictionary and store as attributes
        self.M = constants['M']
        self.Ls = constants['Nt']*constants['N']
        self.N_step = constants['N_step']
        self.modulus = constants['modulus']
        
        self.model_intra = model_intra
        
    def forward(self, phi_batch, w_M_batch, y_M):
        batch_size = phi_batch.size(0)
        N_step = self.N_step
        modulus = self.modulus
        M = self.M
        
        # Initialize the list
        s_stack_batch = torch.zeros(batch_size, N_step+1, self.Ls, dtype=torch.complex64).to(self.device)
        rho_M_stack_batch = torch.zeros(batch_size, N_step, M)
        
        for idx_batch in range(batch_size):
            phi0 = phi_batch[idx_batch]
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
                    rho = self.model_intra.est_rho_modules(x)
                    
                    eta_net += rho*eta
                
                    rho_M_stack_batch[idx_batch,update_step,m] = rho
                
                phi = phi - eta_net  # Update phi
                
                s_stack_batch[idx_batch,update_step,:] = s
                # eta_stack_batch[idx_batch,update_step,:] = eta
            
            s_stack_batch[idx_batch,N_step,:] = modulus*torch.exp(1j *phi)  # Saving the final s after all updates
            
            
        # return s_stack_batch
            
        model_outputs = {
            's_stack_batch': s_stack_batch,
            'rho_M_stack_batch': rho_M_stack_batch
        }
        return model_outputs
