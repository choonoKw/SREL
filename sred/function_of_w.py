# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 01:35:25 2024

@author: jbk5816
"""

import torch
# import numpy as np

def make_Theta_M(struct_c,struct_m,W_M_tilde,aqaqh):
    device = W_M_tilde.device
    
    Nt = struct_c.Nt
    N = struct_c.N
    M = struct_c.M
    
    # target information
    lm = struct_m.lm
    delta_m = struct_m.delta_m
    delta_m_squared = delta_m ** 2
    
    
    Lj = struct_c.Lj
    
    # Initialize Theta_m
    Theta_m = torch.zeros(Nt * N, Nt * N, M, 
                          dtype=torch.complex64).to(device)
    
    for m in range(M):
        Theta_tilde = torch.zeros(Lj * Nt, Lj * Nt, dtype=W_M_tilde.dtype).to(device)
        rm = lm[m] - lm[0]
        for n1 in range(1, Lj - rm + 1):
            for n2 in range(1, Lj - rm + 1):
                Z = torch.outer(
                    W_M_tilde[:, rm + n1 - 1, m], W_M_tilde[:, rm + n2 - 1, m].conj()
                    )
                # Z = W_M_tilde[:, rm + n1 - 1, m].unsqueeze(-1) * W_M_tilde[:, rm + n2 - 1, m].unsqueeze(-1).conj().T
                for q1 in range(Nt):
                    for q2 in range(Nt):
                        index1 = q1 + Nt * (n1 - 1)
                        index2 = q2 + Nt * (n2 - 1)
                        Theta_tilde[index1, index2] = delta_m_squared[m] * torch.trace(
                            Z @ aqaqh[..., q1, q2, m]).to(device)
        
        # Hermitianized
        Theta_m[..., m] = (
            Theta_tilde[:Nt * N, :Nt * N] + Theta_tilde[:Nt * N, :Nt * N].conj().T
            ) / 2
    
    return Theta_m
    
def make_Theta_M_opt(struct_c,struct_m,W_M_tilde,AQAQH_M):
    device = W_M_tilde.device
    
    Nt = struct_c.Nt
    N = struct_c.N
    M = struct_c.M
    
    # target information
    lm = struct_m.lm
    delta_m = struct_m.delta_m
    delta_m_squared = delta_m ** 2
    
    
    Lj = struct_c.Lj
    
    # Initialize Theta_m
    Nt_range = torch.arange(Nt, dtype=torch.long, device=device)
    Theta_m = torch.zeros(Nt * N, Nt * N, M, 
                          dtype=torch.complex64).to(device)
    
    for m in range(M):
        Theta_tilde = torch.zeros(Lj * Nt, Lj * Nt, dtype=W_M_tilde.dtype).to(device)
        rm = lm[m] - lm[0]
        for n1 in range(1, Lj - rm + 1):
            x_n1 = W_M_tilde[:, rm+n1, m]
            for n2 in range(1, Lj - rm + 1):
                x_n2 = W_M_tilde[:, rm+n2, m]
                for i in Nt_range:
                    for j in Nt_range:
                        block = delta_m_squared[m] * kron(
                        eye(Nt, dtype=W_M_tilde.dtype, device=W_M_tilde.device), 
                        x_n2.unsqueeze(-1).T
                        ) @ AQAQH_M[:, :, m] @ kron(
                        eye(Nt, dtype=W_M_tilde.dtype, device=W_M_tilde.device), x_n1.unsqueeze(-1)
                        )
                        Theta_m_tilde[i + Nt*n1 : i + Nt*n1 + 1, j + Nt*n2 : j + Nt*n2 + 1] = block
#                Z = torch.outer(
 #                   W_M_tilde[:, rm + n1 - 1, m], W_M_tilde[:, rm + n2 - 1, m].conj()
  #                  )
                # Z = W_M_tilde[:, rm + n1 - 1, m].unsqueeze(-1) * W_M_tilde[:, rm + n2 - 1, m].unsqueeze(-1).conj().T
                # for q1 in range(Nt):
                  #   for q2 in range(Nt):
                  #       index1 = q1 + Nt * (n1 - 1)
                        # index2 = q2 + Nt * (n2 - 1)
                       #  Theta_tilde[index1, index2] = delta_m_squared[m] * torch.trace(
                         #    Z @ aqaqh[..., q1, q2, m]).to(device)
        
        # Hermitianized
        Theta_m[..., m] = (
            Theta_tilde[:Nt * N, :Nt * N] + Theta_tilde[:Nt * N, :Nt * N].conj().T
            ) / 2
    
    return Theta_m

def make_Phi_M(struct_c,struct_m,struct_k,w_M,W_M_tilde,aqaqh,bqbqh,Upsilon):
    device = w_M.device
    
    Nt = struct_c.Nt;
    N = struct_c.N;
    # Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    delta_m_squared = delta_m ** 2
    
    # interference information
    K = struct_c.K;
    r = struct_k.r
    sigma_k = struct_k.sigma_k;
    sigma_k_squared = sigma_k ** 2
    
    # sigma_v = struct_c.sigma_v;
    
    Lj = struct_c.Lj;
    
    # # Rearrange
    # for m in range(M):
        # W_M_tilde[:, :, m] = w_M[:, m].reshape(Lj,Nr).T

    # Make Phi_m
    Phi_m = torch.zeros((Nt * N, Nt * N, M), dtype=torch.complex64).to(device)
    for m in range(M):
        Phi_m_tilde = torch.zeros((Lj * Nt, Lj * Nt), dtype=torch.complex64).to(device)

        # Sum_{p=1,\neq m}^M
        pList = [p for p in range(M) if p != m]
        for p in pList:
            Phi_temp = torch.zeros((Lj * Nt, Lj * Nt), dtype=torch.complex64).to(device)
            rp = lm[p] - lm[0]
            for n1 in range(1, Lj - rp + 1):
                for n2 in range(1, Lj - rp + 1):
                    Z = torch.outer(
                        W_M_tilde[:, rp + n1 - 1, m], W_M_tilde[:, rp + n2 - 1, m].conj()
                        )
                    # Z = W_M_tilde[:, rp + n1 - 1, m].unsqueeze(-1) * W_M_tilde[:, rp + n2 - 1, m].unsqueeze(0).conj()
                    for q1 in range(Nt):
                        for q2 in range(Nt):
                            index1 = q1 + Nt * (n1 - 1)
                            index2 = q2 + Nt * (n2 - 1)
                            Phi_temp[index1, index2] = ( delta_m_squared[p] 
                                                        * torch.trace(Z @ aqaqh[..., q1, q2, p])
                                                        )
            Phi_m_tilde += Phi_temp

        # Sum_{k=1}^K
        for k in range(K):  # Loop from 0 to K-1
            Phi_temp = torch.zeros(Lj*Nt, Lj*Nt, dtype=W_M_tilde.dtype, device=device)
            rk_abs = abs(r[k])
            if r[k] > 0:
                for n1 in range(Lj-rk_abs):  # Python range is exclusive on the upper bound
                    for n2 in range(Lj-rk_abs):
                        Z = torch.outer(
                            W_M_tilde[:, rk_abs+n1, m], W_M_tilde[:, rk_abs+n2, m].conj()
                            )
                        # Z = W_M_tilde[:, rk_abs+n1, m].unsqueeze(-1) @ W_M_tilde[:, rk_abs+n2, m].conj().unsqueeze(0)
                        for q1 in range(Nt):
                            for q2 in range(Nt):
                                Phi_temp[q1 + Nt*n1, q2 + Nt*n2] = ( sigma_k_squared[k] 
                                                                    * torch.trace(Z @ bqbqh[:, :, q1, q2, k])
                                                                    )
            else:  # r[k] <= 0
                for n1 in range(rk_abs, Lj):  # Adjusted ranges for Python 0-indexing
                    for n2 in range(rk_abs, Lj):
                        Z = torch.outer(
                            W_M_tilde[:, n1-rk_abs, m], W_M_tilde[:, n2-rk_abs, m].conj()
                            )
                        # Z = W_M_tilde[:, n1-rk_abs, m].unsqueeze(-1) @ W_M_tilde[:, n2-rk_abs, m].conj().unsqueeze(0)
                        for q1 in range(Nt):
                            for q2 in range(Nt):
                                Phi_temp[q1 + Nt*n1, q2 + Nt*n2] = ( sigma_k_squared[k] 
                                                                    * torch.trace(Z @ bqbqh[:, :, q1, q2, k])
                                                                    )
        
            Phi_m_tilde += Phi_temp

        # Upsilon
        upsilon_term = (
            w_M[:, m].conj().unsqueeze(-1).T @ Upsilon @ w_M[:, m].unsqueeze(-1)
            ) * torch.eye(Lj * Nt, dtype=torch.complex64).to(device)
        Phi_m_tilde += upsilon_term

        Phi_m[:, :, m] = Phi_m_tilde[:Nt * N, :Nt * N]

    return Phi_m
