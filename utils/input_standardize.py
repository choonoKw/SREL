# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:37:48 2024

@author: jbk5816
"""

import torch

def standardize(s, w, y):
    """
    Standardize the input tensor x.

    Parameters:
    - x: A 1D or 2D tensor representing the concatenated components of the input.
    - means: A 1D tensor containing the mean of each component of the input.
    - stds: A 1D tensor containing the standard deviation of each component of the input.

    Returns:
    - x_std: The standardized version of x.
    """

    epsilon = 1e-8
    
    # Compute mean and standard deviation for each component
    s_real_mean, s_real_std = torch.mean(s.real, dim=0), torch.std(s.real, dim=0) + epsilon
    s_imag_mean, s_imag_std = torch.mean(s.imag, dim=0), torch.std(s.imag, dim=0) + epsilon
    w_real_mean, w_real_std = torch.mean(w.real, dim=0), torch.std(w.real, dim=0) + epsilon
    w_imag_mean, w_imag_std = torch.mean(w.imag, dim=0), torch.std(w.imag, dim=0) + epsilon
    y_mean, y_std = torch.mean(y, dim=0), torch.std(y, dim=0) + epsilon
    
    # Standardize each component
    s_real_standardized = (s.real - s_real_mean) / s_real_std
    s_imag_standardized = (s.imag - s_imag_mean) / s_imag_std
    w_real_standardized = (w.real - w_real_mean) / w_real_std
    w_imag_standardized = (w.imag - w_imag_mean) / w_imag_std
    y_standardized = (y - y_mean) / y_std
    
    # Concatenate standardized components
    x_standardized = torch.cat(
        (
            s_real_standardized, 
            s_imag_standardized, 
            w_real_standardized, 
            w_imag_standardized, 
            y_standardized), dim=0)
    
    return x_standardized