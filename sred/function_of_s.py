# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 00:28:44 2024

@author: jbk5816
"""

import torch
# import numpy as np

def make_Sigma(struct_c,struct_k,S_tilde,bqhbq):    
    Nr = struct_c.Nr;
    
    # interference information
    K = struct_c.K;
    r = struct_k.r;
    sigma_k = struct_k.sigma_k;
    
    Lj = struct_c.Lj;
    
    device = S_tilde.device
    
    # Initialize Sigma
    Sigma = torch.zeros((Lj * Nr, Lj * Nr), dtype=torch.complex64).to(device)

    # sum_{k=1}^K
    for k in range(K):
        Sigma_temp = torch.zeros((Lj * Nr, Lj * Nr), dtype=torch.complex64).to(device)
        rk = abs(r[k])
        if r[k] < 0:
            for n1 in range(1, Lj - rk + 1):
                for n2 in range(1, Lj - rk + 1):
                    Z = S_tilde[:, rk + n1 - 1].unsqueeze(-1) * S_tilde[:, rk + n2 - 1].unsqueeze(0).conj()
                    for q1 in range(1, Nr + 1):
                        for q2 in range(1, Nr + 1):
                            index1 = (q1 - 1) + Nr * (n1 - 1)
                            index2 = (q2 - 1) + Nr * (n2 - 1)
                            Sigma_temp[index1, index2] = sigma_k[k]**2 * torch.trace(Z @ bqhbq[:, :, q1 - 1, q2 - 1, k])
        else:  # rk > 0
            for n1 in range(rk + 1, Lj + 1):
                for n2 in range(rk + 1, Lj + 1):
                    Z = S_tilde[:, n1 - rk - 1].unsqueeze(-1) * S_tilde[:, n2 - rk - 1].unsqueeze(0).conj()
                    for q1 in range(1, Nr + 1):
                        for q2 in range(1, Nr + 1):
                            index1 = (q1 - 1) + Nr * (n1 - 1)
                            index2 = (q2 - 1) + Nr * (n2 - 1)
                            Sigma_temp[index1, index2] = sigma_k[k]**2 * torch.trace(Z @ bqhbq[:, :, q1 - 1, q2 - 1, k])

        Sigma += Sigma_temp

    return Sigma

def make_Gamma_M(struct_c,struct_m,S_tilde,aqhaq):
    device = S_tilde.device
    Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    
    Lj = struct_c.Lj;
    
    # Initialize Gamma_M
    Gamma_M = torch.zeros((Lj * Nr, Lj * Nr, M), dtype=torch.complex64).to(device)
    
    for m in range(M):
        Gamma_temp = torch.zeros((Lj * Nr, Lj * Nr), dtype=torch.complex64).to(device)
        rm = lm[m] - lm[0]
        for n1 in range(rm + 1, Lj + 1):
            for n2 in range(rm + 1, Lj + 1):
                Z = S_tilde[:, n1 - rm - 1].unsqueeze(-1) * S_tilde[:, n2 - rm - 1].unsqueeze(0).conj()
                for q1 in range(1, Nr + 1):
                    for q2 in range(1, Nr + 1):
                        index1 = (q1 - 1) + Nr * (n1 - 1)
                        index2 = (q2 - 1) + Nr * (n2 - 1)
                        Gamma_temp[index1, index2] = (delta_m[m]**2) * torch.trace(Z @ aqhaq[:, :, q1 - 1, q2 - 1, m])
    
        # Hermitianize Gamma_temp
        Gamma_M[:, :, m] = (Gamma_temp + Gamma_temp.transpose(0, 1).conj()) / 2
    
    return Gamma_M

def make_Psi_M(struct_c,struct_m,S_tilde,aqhaq,Sigma,Upsilon):
    device = S_tilde.device
    Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    
    Lj = struct_c.Lj;

    # Initialize Psi_M
    Psi_M = torch.zeros((Lj*Nr, Lj*Nr, M), dtype=torch.complex64).to(device)
    
    for m in range(M):
        # Exclude m from the list of p values
        p_list = [p for p in range(M) if p != m]
        for p in p_list:
            Psi_temp = torch.zeros((Lj*Nr, Lj*Nr), dtype=torch.complex64).to(device)
            rp = lm[p] - lm[0]
            for n1 in range(rp + 1, Lj + 1):
                for n2 in range(rp + 1, Lj + 1):
                    Z = S_tilde[:, n1 - rp - 1].unsqueeze(-1) * S_tilde[:, n2 - rp - 1].unsqueeze(0).conj()
                    for q1 in range(1, Nr + 1):
                        for q2 in range(1, Nr + 1):
                            index1 = (q1-1) + Nr*(n1-1)
                            index2 = (q2-1) + Nr*(n2-1)
                            Psi_temp[index1, index2] = (delta_m[p]**2) * torch.trace(Z @ aqhaq[:, :, q1 - 1, q2 - 1, p])
            Psi_M[:, :, m] += Psi_temp
    
        # Hermitianize Psi_M[:,:,m], then add Sigma and Upsilon
        Psi_M[:, :, m] = (Psi_M[:, :, m] + Psi_M[:, :, m].transpose(0, 1).conj()) / 2 + Sigma + Upsilon
    
    return Psi_M


