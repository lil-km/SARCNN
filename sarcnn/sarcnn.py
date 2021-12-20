"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling model implementation
"""

import torch
from torch import nn
import torch.nn.functional as F


# helper functions

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# main classe

class SARCNN(nn.Module):
    def __init__(
        self,
        num_layers=17,
        num_features=64,
        kernel_size=3,
        padding=1,
        image_channels=1,
        image_size=64
    ):
        super(SARCNN, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(in_channels=num_features, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=True))
        
        self.dncnn = nn.Sequential(*layers)


    @torch.no_grad()
    @eval_decorator
    def denoise(self, y):
        return self(y)


    def forward(self, y, return_loss=False, x=None):
        n = self.dncnn(y)

        if not return_loss:
            return y-n
        
        # calculate the L2 loss

        return F.mse_loss(n, y-x)
