# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:09:31 2024

@author: jbk5816
"""

import torch
# import numpy as np

# def reciprocal_sinr(G,H,s):
#     numerator = torch.abs(torch.vdot(s, torch.matmul(G, s)))
#     denominator = torch.abs(torch.vdot(s, torch.matmul(H, s)))
#     return numerator / denominator

def sum_of_sinr_reciprocal(G_M, H_M, s):
    f_sinr = 0.0
    for m, (G, H) in enumerate(zip(
            torch.unbind(G_M, dim=-1),torch.unbind(H_M, dim=-1)
            )): 
        numerator = torch.abs(torch.vdot(s, torch.matmul(G, s)))
        denominator = torch.abs(torch.vdot(s, torch.matmul(H, s)))
        
        f_sinr += numerator / denominator
        
        # f_sinr += reciprocal_sinr(G, H, s)
        
    return f_sinr

def eta_sred(G, H, s):
    Gs = torch.matmul(G, s)  # G*s
    Hs = torch.matmul(H, s)  # H*s
    
    sGs = torch.abs(torch.vdot(s, Gs)) 
    sHs = torch.abs(torch.vdot(s, Hs)) 
    
    eta = torch.real(2 / (sHs ** 2) * torch.imag( (sHs * Gs - sGs * Hs) * torch.conj(s) ) )
    
    return eta

def derive_w(struct_c,Psi_m,Gamma_m):
    Lj = struct_c.Lj;
    Nr = struct_c.Nr;
    M = struct_c.M;
    
    # Initialize tensors in PyTorch
    w_mList = torch.zeros((Lj*Nr, M), dtype=torch.complex64)
    W_m_tilde = torch.zeros((Nr, Lj, M), dtype=torch.complex64)

    for m in range(M):
        # Compute the eigenvalues and eigenvectors
        # PyTorch's eig does not support complex input directly for eigenvalue decomposition,
        # use torch.linalg.eig if available or perform operations on real and imaginary separately
        Psi_m_inv_Gamma = torch.linalg.inv(Psi_m[:, :, m]) @ Gamma_m[:, :, m]
        D, V = torch.linalg.eig(Psi_m_inv_Gamma)
        
        # Find the index of the maximum eigenvalue
        # Eigenvalues are returned as a vector, so take the real part if necessary
        i = torch.argmax(torch.abs(D))
        
        # Normalize the corresponding eigenvector and store it in w_mList
        norm_factor = torch.norm(V[:, i], p=2)
        w_mList[:, m] = V[:, i] / norm_factor
        
        # Reshape and store in W_m_tilde
        W_m_tilde[:, :, m] = w_mList[:, m].view(Nr, Lj)
        
    return w_mList, W_m_tilde

def derive_s(constants, phi, struct_c, struct_m):
    device = phi.device
    s = constants['modulus']*torch.exp(1j *phi) 
    
    Nt = struct_c.Nt
    M = struct_c.M
    Lj = struct_c.Lj
    
    lm = struct_m.lm.to(device)
    
    # Ensure s is unsqueezed to mimic column vector shape for concatenation
    s_extended = torch.cat(
        (s.unsqueeze(1), torch.zeros(
            Nt * (lm[M-1] - lm[0]), 1, dtype=torch.complex64)
        ), 0)
    
    # Reshape s_extended to Nt x Lj
    S_tilde = s_extended.view(Nt, Lj)
    
    return s, S_tilde