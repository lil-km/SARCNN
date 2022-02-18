"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling model implementation
"""

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from sarcnn.perceptual_loss import PERCEPTUAL_LOSS
from sarcnn.utils import *

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

    # denoising function

    @torch.no_grad()
    @eval_decorator
    def denoise(self, noisy_input):
        return self(noisy_input)


    def test(self, test_files, save_dir, dataset_dir, device, real_sar):
        """Test SAR-CNN"""
        assert len(test_files) != 0, 'No testing data!'
        print("[*] start testing...")
        psnr = np.array([])
        ssim = np.array([])
        for idx in range(len(test_files)):
            print(test_files[idx])
            clean_image = load_sar_images(test_files[idx])

            if real_sar:
                "downsampling real image to reduce correlation"
                clean_image = torch.unsqueeze(clean_image, dim=0)
                clean_image = normalize_sar(clean_image)
                clean_image = clean_image / 255.
                clean_image_downsampled = clean_image[:,:,::2,::2]
                clean_image_downsampled = clean_image_downsampled.to(device)
                output_clean_image = self.denoise(clean_image_downsampled)
                noisy_image = clean_image

                groundtruth = denormalize_sar(clean_image[0][0])
                noisyimage = denormalize_sar(noisy_image[0][0])
                outputimage = denormalize_sar(output_clean_image[0][0])

                psnr = np.append(psnr, cal_psnr(outputimage.cpu(), groundtruth))
                ssim = np.append(ssim, cal_ssim(torch.unsqueeze(torch.unsqueeze(outputimage.cpu(), dim=0), dim=0), torch.unsqueeze(torch.unsqueeze(groundtruth, dim=0), dim=0)))

                imagename = test_files[idx].replace(dataset_dir+"/", "")
                print("Denoised image %s" % imagename)
            else:
                noisy_image = inject_speckle_amplitude(clean_image, 1)
                clean_image = torch.unsqueeze(clean_image, dim=0)
                clean_image = normalize_sar(clean_image)
                clean_image = clean_image / 255.

                noisy_image = torch.unsqueeze(noisy_image, dim=0)
                noisy_image = normalize_sar(noisy_image)
                noisy_image = noisy_image / 255.                

                noisy_image = noisy_image.to(device)
                output_clean_image = self.denoise(noisy_image)

                groundtruth = denormalize_sar(clean_image[0][0])
                noisyimage = denormalize_sar(noisy_image[0][0])
                outputimage = denormalize_sar(output_clean_image[0][0])

                psnr = np.append(psnr, cal_psnr(outputimage.cpu(), groundtruth))
                ssim = np.append(ssim, cal_ssim(torch.unsqueeze(torch.unsqueeze(outputimage.cpu(), dim=0), dim=0), torch.unsqueeze(torch.unsqueeze(groundtruth, dim=0), dim=0)))

                imagename = test_files[idx].replace(dataset_dir+"/", "")
                print("Denoised image %s" % imagename)

            save_sar_images(groundtruth, outputimage, noisyimage, imagename, save_dir, real_sar)

        return psnr, ssim

    def forward(self, noisy_input, ground_truth_input=None, return_loss=False, perceptual_loss=False, Lambda=1.):
        denoised_input = noisy_input - self.sarcnn(noisy_input)

        if perceptual_loss:
            # calculate the L2 loss, F.mse_loss(n, y-x)
            data_loss = F.l1_loss(denoised_input+cn, ground_truth_input)
            perceptual_loss = self.p_loss(denoised_input+cn, ground_truth_input)
            total_loss = data_loss + Lambda*perceptual_loss
            return total_loss, data_loss, perceptual_loss

        if return_loss:
            # calculate the L2 loss, F.mse_loss(n, y-x)
            data_loss = F.l1_loss(denoised_input+cn, ground_truth_input)
            return data_loss

        return denoised_input+cn
        