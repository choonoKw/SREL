# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 01:35:25 2024

@author: jbk5816
"""

import torch
# import numpy as np

def make_Theta_M(struct_c,struct_m,W_m_tilde,aqaqh):
    device = W_m_tilde.device
    
    Nt = struct_c.Nt
    N = struct_c.N
    M = struct_c.M
    
    # target information
    lm = struct_m.lm
    delta_m = struct_m.delta_m
    
    Lj = struct_c.Lj
    
    # Initialize Theta_m
    Theta_m = torch.zeros(Nt * N, Nt * N, M, 
                          dtype=torch.complex64).to(device)
    
    for m in range(M):
        Theta_m_tilde = torch.zeros(Lj * Nt, Lj * Nt, dtype=W_m_tilde.dtype).to(device)
        rm = lm[m] - lm[0]
        for n1 in range(1, Lj - rm + 1):
            for n2 in range(1, Lj - rm + 1):
                Z = W_m_tilde[:, rm + n1 - 1, m].unsqueeze(-1) * W_m_tilde[:, rm + n2 - 1, m].unsqueeze(-1).conj().transpose(0, 1)
                for q1 in range(Nt):
                    for q2 in range(Nt):
                        index1 = q1 + Nt * (n1 - 1)
                        index2 = q2 + Nt * (n2 - 1)
                        Theta_m_tilde[index1, index2] = (delta_m[m]**2) * torch.trace(Z @ aqaqh[..., q1, q2, m])
        
        # Hermitianized
        Theta_m[..., m] = (Theta_m_tilde[:Nt * N, :Nt * N] + Theta_m_tilde[:Nt * N, :Nt * N].transpose(0, 1)) / 2
    
    return Theta_m

def make_Phi_M(struct_c,struct_m,struct_k,w_mList,aqaqh,bqbqh,Upsilon):
    device = w_mList.device
    
    Nt = struct_c.Nt;
    N = struct_c.N;
    Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    
    # interference information
    K = struct_c.K;
    r = struct_k.r
    sigma_k = struct_k.sigma_k;
    
    # sigma_v = struct_c.sigma_v;
    
    Lj = struct_c.Lj;
    
    # Rearrange
    W_m_tilde = torch.zeros((Nr, Lj, M), dtype=torch.complex64).to(device)
    for m in range(M):
        W_m_tilde[:, :, m] = w_mList[:, m].view(Nr, Lj)

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
                    Z = W_m_tilde[:, rp + n1 - 1, m].unsqueeze(-1) * W_m_tilde[:, rp + n2 - 1, m].unsqueeze(0).conj()
                    for q1 in range(Nt):
                        for q2 in range(Nt):
                            index1 = q1 + Nt * (n1 - 1)
                            index2 = q2 + Nt * (n2 - 1)
                            Phi_temp[index1, index2] = (delta_m[p]**2) * torch.trace(Z @ aqaqh[..., q1, q2, p])
            Phi_m_tilde += Phi_temp

        # Sum_{k=1}^K
        for k in range(K):
            Phi_temp = torch.zeros((Lj * Nt, Lj * Nt), dtype=torch.complex64).to(device)
            rk = abs(r[k])
            for n1 in range(1, Lj - rk + 1):
                for n2 in range(1, Lj - rk + 1):
                    Z = W_m_tilde[:, rk + n1 - 1, m].unsqueeze(-1) * W_m_tilde[:, rk + n2 - 1, m].unsqueeze(0).conj()
                    for q1 in range(Nt):
                        for q2 in range(Nt):
                            index1 = q1 + Nt * (n1 - 1)
                            index2 = q2 + Nt * (n2 - 1)
                            Phi_temp[index1, index2] = (sigma_k[k]**2) * torch.trace(Z @ bqbqh[..., q1, q2, k])
            Phi_m_tilde += Phi_temp

        # Upsilon
        upsilon_term = (
            w_mList[:, m].unsqueeze(0).conj() @ Upsilon @ w_mList[:, m].unsqueeze(-1)
            ) * torch.eye(Lj * Nt, dtype=torch.complex64).to(device)
        Phi_m_tilde += upsilon_term

        Phi_m[:, :, m] = Phi_m_tilde[:Nt * N, :Nt * N]

    return Phi_m
