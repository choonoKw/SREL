# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:09:07 2024

@author: jbk5816
"""

import torch
import torch.nn as nn


class Estimate_eta(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Estimate_eta, self).__init__()
        # Define layers for estimating the real part of v
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # Output dimension matches phi 128
        )
        # # Define layers for estimating the imaginary part of v
        # self.imag_layers = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, input_dim),  # Output dimension matches phi
        # )

    def forward(self, x):
        return self.layers(x)
        # real = self.real_layers(s)
        # imag = self.imag_layers(s)
        # # Combine real and imag parts to form the complex-valued vector v
        # v = torch.complex(real, imag)
        # return v