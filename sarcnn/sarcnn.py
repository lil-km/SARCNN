"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling model implementation
"""

import torch
from torch import nn
import torch.nn.functional as F

from sarcnn.perceptual_loss import PERCEPTUAL_LOSS


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
        image_channels=1
    ):
        super(SARCNN, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU())
        
        layers.append(nn.Conv2d(in_channels=num_features, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=True))
        
        self.sarcnn = nn.Sequential(*layers)
        self.p_loss = PERCEPTUAL_LOSS()


    @torch.no_grad()
    @eval_decorator
    def denoise(self, noisy_input):
        return self(noisy_input)


    def forward(self, noisy_input, ground_truth_input=None, return_loss=False, perceptual_loss=False, Lambda=1.):
        residual_noise = self.sarcnn(noisy_input)

        if perceptual_loss:
            # calculate the L2 loss, F.mse_loss(n, y-x)
            # calculate the L1 loss
            data_loss = F.l1_loss(residual_noise, noisy_input-ground_truth_input)
            perceptual_loss = self.p_loss(ground_truth_input, noisy_input-residual_noise)
            total_loss = data_loss + Lambda*perceptual_loss
            return total_loss, data_loss, perceptual_loss

        if return_loss:
            # calculate the L2 loss, F.mse_loss(n, y-x)
            # calculate the L1 loss
            data_loss = F.l1_loss(residual_noise, noisy_input-ground_truth_input)
            return data_loss

        return noisy_input-residual_noise

        
