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

import torch

class AWGN(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SPECKLE(object):
    def __init__(self, L):
        self.num_of_look = L
        
    def __call__(self, tensor):
        rows, columns = tensor.size()
        s = torch.zeros((rows, columns))
        for k in range(0, L):
            gamma = torch.abs(torch.randn((rows, columns)) + torch.randn((rows, columns))*1j )**2/2
            s = s + gamma

        s_amplitude = torch.sqrt(s/L)
        ima_speckle_amplitude = torch.multiply(tensor, s_amplitude)
        return ima_speckle_amplitude
    
    def __repr__(self):
        return self.__class__.__name__ + '(number of look={0})'.format(self.num_of_look)


class RandomRot90(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if not isinstance(img, torch.Tensor):
           img = torch.from_numpy(np.array(img))

        if torch.rand(1) < self.p:
            return torch.rot90(img, torch.randint(low=0,high=4,size=(1,)).item(), (1, 2))
        return img


# DEFINE PARAMETERS OF SPECKLE
M = 10.089038980848645
m = -1.429329123112601


def normalize_sar(img):
    return ((torch.log(img + 0.2) - m) * 255. / (M - m)).float()

def denormalize_sar(img):
    return torch.exp((M - m) * torch.clip(img.float(), 0., 1.) + m)

"""
Author: emanuele dalsasso
Estimate PSNR for SAR amplitude images
"""

def cal_psnr(Shat, S):
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = torch.quantile(S, 0.99)
    res = 10 * torch.log10((P ** 2) / torch.mean(torch.abs(Shat - S) ** 2))
    return res


def dynamic_clamp(image, threshold=None):
    if threshold == None:
        threshold = torch.mean(image) + 3*torch.std(image)

    image = torch.clip(image, 0, threshold)
    image = image/threshold*255
    return image


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:12:23 2018

@author: emasasso
"""

def inject_speckle_amplitude(img, L):
    rows, columns = img.shape[1], img.shape[2]
    s = torch.zeros((rows, columns))
    for k in range(0, L):
        gamma = torch.abs(torch.randn((rows, columns)) + torch.randn((rows, columns))*1j )**2/2
        s = s + gamma
    s_amplitude = torch.sqrt(s/L)
    ima_speckle_amplitude = torch.multiply(img, s_amplitude)
    return ima_speckle_amplitude

# im = np.load('denoised_lely.npy')
# speckled_image = injectspeckle_amplitude(im,1)
# np.save('noisy_lely.npy',speckled_image)



"""
example of a simple function normalizing SAR images
between 0 and 1. The max M and the min m have been
pre-estimated on a large dataset of real SAR images
and must not be modified
"""

files = glob('*.npy')
M = 10.089038980848645
m = -1.429329123112601
for n in range(len(files)):
	im = np.load(files[n])
	logim = np.log(im+np.spacing(1)) # avoid log(0)
	nlogim = ((logim-m) / (M - m))
	nlogim = nlogim.astype('float32')
	filename = './lognormdata/'+files[n]
	np.save(filename,nlogim)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:56:07 2018

@author: emasasso
"""

working_dir = "./Test/"
test_files = glob(working_dir+'*.npy')

# the thresholds have already been pre-estimated as mean+3*std 
# do not modify them
choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92, 'lely':235.90, 'ramb':167.22, 'risoul':306.94, 'limagne':178.43}
for filename in test_files:
    dim = np.load(filename)
    dim = np.squeeze(dim)
    for x in choices:
        if x in filename:
            threshold = choices.get(x)
        #if not: threshold= np.mean(dim)+3*np.std(dim)

    dim = np.clip(dim,0,threshold)
    dim = dim/threshold*255
    dim = Image.fromarray(dim.astype('float64')).convert('L')
    imagename = filename.replace("npy","png")
    dim.save(imagename)
