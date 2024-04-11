# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 00:28:44 2024

@author: jbk5816
"""

import torch
import numpy as np
from torch import kron, eye
from numpy.linalg import matrix_power

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
        rk_abs = abs(r[k])
        if r[k] < 0:
            for n1 in range(1, Lj - rk_abs + 1):
                for n2 in range(1, Lj - rk_abs + 1):
                    Z = S_tilde[:, rk_abs + n1 - 1].unsqueeze(-1) * S_tilde[:, rk_abs + n2 - 1].unsqueeze(0).conj()
                    for q1 in range(1, Nr + 1):
                        for q2 in range(1, Nr + 1):
                            index1 = (q1 - 1) + Nr * (n1 - 1)
                            index2 = (q2 - 1) + Nr * (n2 - 1)
                            Sigma_temp[index1, index2] = sigma_k[k]**2 * torch.trace(Z @ bqhbq[:, :, q1 - 1, q2 - 1, k])
        else:  # rk_abs > 0
            for n1 in range(rk_abs + 1, Lj + 1):
                for n2 in range(rk_abs + 1, Lj + 1):
                    Z = S_tilde[:, n1 - rk_abs - 1].unsqueeze(-1) * S_tilde[:, n2 - rk_abs - 1].unsqueeze(0).conj()
                    for q1 in range(1, Nr + 1):
                        for q2 in range(1, Nr + 1):
                            index1 = (q1 - 1) + Nr * (n1 - 1)
                            index2 = (q2 - 1) + Nr * (n2 - 1)
                            Sigma_temp[index1, index2] = sigma_k[k]**2 * torch.trace(Z @ bqhbq[:, :, q1 - 1, q2 - 1, k])

        Sigma += Sigma_temp

    return Sigma

def make_Sigma_opt(struct_c,struct_k,S_tilde,BQHBQ_K):    
    Nr = struct_c.Nr;
    
    # interference information
    K = struct_c.K;
    r = struct_k.r;
    sigma_k = struct_k.sigma_k;
    sigma_k_squared = sigma_k ** 2
    
    Lj = struct_c.Lj;
    
    device = S_tilde.device
    
    # Initialize Sigma
    Sigma = torch.zeros((Lj * Nr, Lj * Nr), dtype=torch.complex64).to(device)

    Nr_range = np.arange(Nr)
    
    # sum_{k=1}^K
    for k in range(K):
        Sigma_temp = torch.zeros((Lj * Nr, Lj * Nr), dtype=torch.complex64).to(device)
        rk_abs = abs(r[k])
        # print(k)
        if r[k] < 0:
            for n1 in range(1, Lj - rk_abs + 1):
                x_n1 = S_tilde[:,rk_abs+n1-1].unsqueeze(-1);
                for n2 in range(1, Lj - rk_abs + 1):
                    x_n2 = S_tilde[:,rk_abs+n2-1].unsqueeze(-1);
                    Sigma_temp[
                        np.ix_(Nr_range + Nr*(n1-1), Nr_range + Nr*(n2-1))
                        ] = sigma_k_squared[k] * (
                            kron(eye(Nr), x_n2.conj().T) @ BQHBQ_K[:,:,k] @ kron(eye(Nr), x_n1)
                            )
        else:  # r[k] > 0
            for n1 in range(rk_abs + 1, Lj + 1):
                x_n1 = S_tilde[:,n1-rk_abs-1].unsqueeze(-1);
                for n2 in range(rk_abs + 1, Lj + 1):
                    x_n2 = S_tilde[:,n2-rk_abs-1].unsqueeze(-1);
                    Sigma_temp[
                        np.ix_(Nr_range + Nr*(n1-1), Nr_range + Nr*(n2-1))
                        ] = sigma_k_squared[k] * (
                            kron(eye(Nr), x_n2.conj().T) @ BQHBQ_K[:,:,k] @ kron(eye(Nr), x_n1)
                            )

        Sigma += (Sigma_temp + Sigma_temp.conj().T)/2

    return Sigma

def make_Gamma_M(struct_c,struct_m,S_tilde,aqhaq):
    device = S_tilde.device
    Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    delta_m_squared = delta_m ** 2
    
    Lj = struct_c.Lj;
    
    # Initialize Gamma_M
    Gamma_M = torch.zeros((Lj * Nr, Lj * Nr, M), dtype=torch.complex64).to(device)
    
    range_Nr = range(1, Nr + 1)
    for m in range(M):
        Gamma_temp = torch.zeros((Lj * Nr, Lj * Nr), dtype=torch.complex64).to(device)
        rm = lm[m] - lm[0]
        for n1 in range(rm + 1, Lj + 1):
            for n2 in range(rm + 1, Lj + 1):
                Z = ( S_tilde[:, n1 - rm - 1].unsqueeze(-1) 
                     * S_tilde[:, n2 - rm - 1].unsqueeze(0).conj()
                     )
                for q1 in range_Nr:
                    for q2 in range_Nr:
                        index1 = (q1 - 1) + Nr * (n1 - 1)
                        index2 = (q2 - 1) + Nr * (n2 - 1)
                        Gamma_temp[index1, index2] = ( delta_m_squared[m] 
                                                      * torch.trace(
                                                          Z @ aqhaq[:, :, q1 - 1, q2 - 1, m]
                                                          ) 
                                                      )
    
        # Hermitianize Gamma_temp
        Gamma_M[:, :, m] = (Gamma_temp + Gamma_temp.conj().T) / 2
    
    return Gamma_M

def make_Gamma_M_opt(struct_c,struct_m,S_tilde,AQHAQ_M):
    device = S_tilde.device
    Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    delta_m_squared = delta_m ** 2
    
    Lj = struct_c.Lj;
    
    # Initialize Gamma_M
    Nr_range = np.arange(Nr)
    Gamma_M = torch.zeros((Lj * Nr, Lj * Nr, M), dtype=torch.complex64).to(device)
    
    for m in range(M):
        Gamma_temp = torch.zeros((Lj * Nr, Lj * Nr), dtype=torch.complex64).to(device)
        rm = lm[m] - lm[0]
        for n1 in range(rm + 1, Lj + 1):
            x_n1 = S_tilde[:,n1 - rm-1].unsqueeze(-1);
            for n2 in range(rm + 1, Lj + 1):
                x_n2 = S_tilde[:, n2 - rm -1].unsqueeze(-1);
                Gamma_temp[
                    np.ix_(Nr_range + Nr*(n1-1), Nr_range + Nr*(n2-1))
                    ] = delta_m_squared[m] * (
                        kron(eye(Nr), x_n2.conj().T) @ AQHAQ_M[:,:,m] @ kron(eye(Nr), x_n1)
                        )
        Gamma_M[:, :, m] = (Gamma_temp + Gamma_temp.conj().T) / 2
    
    return Gamma_M

def make_Psi_M(struct_c,struct_m,S_tilde,aqhaq,Sigma,Upsilon):
    device = S_tilde.device
    Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    delta_m_squared = delta_m ** 2
    
    Lj = struct_c.Lj;

    # Initialize Psi_M
    Psi_M = torch.zeros((Lj*Nr, Lj*Nr, M), dtype=torch.complex64).to(device)
    
    range_Nr = range(1, Nr + 1)
    for m in range(M):
        # Exclude m from the list of p values
        p_list = [p for p in range(M) if p != m]
        for p in p_list:
            Psi_temp = torch.zeros((Lj*Nr, Lj*Nr), dtype=torch.complex64).to(device)
            rp = lm[p] - lm[0]
            for n1 in range(rp + 1, Lj + 1):
                for n2 in range(rp + 1, Lj + 1):
                    Z = ( S_tilde[:, n1 - rp - 1].unsqueeze(-1) 
                         * S_tilde[:, n2 - rp - 1].unsqueeze(0).conj() )
                    for q1 in range_Nr:
                        for q2 in range_Nr:
                            index1 = (q1-1) + Nr*(n1-1)
                            index2 = (q2-1) + Nr*(n2-1)
                            Psi_temp[index1, index2] = ( delta_m_squared[p] 
                                                        * torch.trace(
                                                            Z @ aqhaq[:, :, q1 - 1, q2 - 1, p]
                                                            ) ) 
            Psi_M[:, :, m] += Psi_temp
    
        # Hermitianize Psi_M[:,:,m], then add Sigma and Upsilon
        Psi_M[:, :, m] = (
            Psi_M[:, :, m] + Psi_M[:, :, m].conj().T
            ) / 2 + Sigma + Upsilon
    
    return Psi_M

def make_Psi_M_opt(struct_c,struct_m,S_tilde,AQHAQ_M,Sigma,Upsilon):
    device = S_tilde.device
    Nr = struct_c.Nr;
    
    # target information
    M = struct_c.M;
    lm = struct_m.lm;
    delta_m = struct_m.delta_m;
    delta_m_squared = delta_m ** 2
    
    Lj = struct_c.Lj;

    # Initialize Psi_M
    Psi_M = torch.zeros((Lj*Nr, Lj*Nr, M), dtype=torch.complex64).to(device)
    
    Nr_range = np.arange(Nr)
    for m in range(M):
        # Exclude m from the list of p values
        p_list = [p for p in range(M) if p != m]
        for p in p_list:
            Psi_temp = torch.zeros((Lj*Nr, Lj*Nr), dtype=torch.complex64).to(device)
            rp = lm[p] - lm[0]
            for n1 in range(rp + 1, Lj + 1):
                x_n1 = S_tilde[:,n1 - rp -1].unsqueeze(-1);
                for n2 in range(rp + 1, Lj + 1):
                    x_n2 = S_tilde[:, n2 - rp -1].unsqueeze(-1);
                    Psi_temp[
                        np.ix_(Nr_range + Nr*(n1-1), Nr_range + Nr*(n2-1))
                        ] = delta_m_squared[p] * (
                            kron(eye(Nr), x_n2.conj().T) @ AQHAQ_M[:,:,p] @ kron(eye(Nr), x_n1)
                            )
            Psi_M[:, :, m] += Psi_temp
    
        # Hermitianize Psi_M[:,:,m], then add Sigma and Upsilon
        Psi_M[:, :, m] = (
            Psi_M[:, :, m] + Psi_M[:, :, m].conj().T
            ) / 2 + Sigma + Upsilon
    
    return Psi_M


