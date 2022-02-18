"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling utility functions
"""

# Imports
import numpy as np
from glob import glob
from PIL import Image

import scipy.ndimage
from scipy import special

import torch
import torch.nn.functional as F
from math import exp


# DEFINE PARAMETERS OF SPECKLE
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle


# sar normalization function

def normalize_sar(img):
    return ((torch.log(img + 0.2) - m) * 255. / (M - m)).float()

# sar denormalization function

def denormalize_sar(img):
    return torch.exp((M - m) * torch.clip(img.float(), 0., 1.) + m)

# compute psnr function

def cal_psnr(Shat, S):
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = torch.quantile(S, 0.99)
    res = 10 * torch.log10((P ** 2) / torch.mean(torch.abs(Shat - S) ** 2))
    return res

# clamp the dynamic range for sar visualization

def dynamic_clamp(image, threshold=None):
    if threshold == None:
        threshold = torch.mean(image) + 3*torch.std(image)

    image = torch.clip(image, 0, threshold)
    image = image/threshold*255
    return image

# inject synthetic speckle to clean sar image function

def inject_speckle_amplitude(img, L):
    rows, columns = img.shape[1], img.shape[2]
    s = torch.zeros((rows, columns))
    for k in range(0, L):
        gamma = torch.abs(torch.randn((rows, columns)) + torch.randn((rows, columns))*1j )**2/2
        s = s + gamma
    s_amplitude = torch.sqrt(s/L)
    ima_speckle_amplitude = torch.multiply(img, s_amplitude)
    return ima_speckle_amplitude



# compute SSIM loss, code from: https://github.com/Po-Hsun-Su/pytorch-ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def cal_ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

# load npy file as torch tensor

def load_sar_images(file):
        im = np.load(file)
        im = torch.from_numpy(im)
        im = torch.unsqueeze(im, dim=0)
        im = im.float()

        return im

# save numpy array as png image file

def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','png'))

def save_sar_images(groundtruth, residual, denoised, noisy, imagename, save_dir, real_sar_flag):
    choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92, 'lely':235.90, 'ramb':167.22, 'risoul':306.94, 'limagne':178.43}
    threshold = choices.get('%s' % imagename)
    if threshold==None: threshold = np.mean(groundtruth)+3*np.std(groundtruth)

    if not real_sar_flag: # simulated noisy images
        groundtruthfilename = f"{save_dir}/groundtruth_{imagename}"
        np.save(groundtruthfilename, groundtruth)
        store_data_and_plot(groundtruth, threshold, groundtruthfilename)

    denoised = denoised
    if real_sar_flag: denoised = scipy.ndimage.zoom(denoised, 2, order=1)
    denoisedfilename = f"{save_dir}/denoised_{imagename}"
    np.save(denoisedfilename, denoised)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    residualfilename = f"{save_dir}/residual_{imagename}"
    np.save(residualfilename, residual)
    store_data_and_plot(residual, np.mean(residual)+3*np.std(residual), residualfilename)
