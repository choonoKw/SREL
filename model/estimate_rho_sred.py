# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:54:03 2024

@author: jbk5816
"""

import torch.nn as nn
import torch.nn.functional as F

class Estimate_rho(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_rho, self).__init__()
        # Define layers for estimating step size rho
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),  # Start with a more significant reduction
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),  # Optional: add dropout for regularization
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),  # Output dimension matches rho 1
        )

    def forward(self, x):
        y = self.layers(x)
        # print(x)
        # Apply Softplus to ensure positive output
        return F.softplus(y)