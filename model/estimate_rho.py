# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:09:07 2024

@author: jbk5816
"""

# import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimate_rho(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_rho, self).__init__()
        # Define layers for estimating step size rho
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Start with a more significant reduction
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Optional: add dropout for regularization
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),  # Output dimension matches rho 1
        )

    def forward(self, x):
        y = self.layers(x)
        # Apply Softplus to ensure positive output
        return F.softplus(y)