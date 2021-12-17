"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling model testing implementation
"""

# Imports

import argparse
import time
import numpy as np

from pathlib import Path

import torch
from PIL import Image

from sarcnn.sarcnn import SARCNN

# arguments parsing

parser = argparse.ArgumentParser(description='SARCNN Testing')
parser.add_argument('--image_path', type=str, required=True, help='test image path')
parser.add_argument('--model_path', type=str, required=True, help='pretrained model path')
parser.add_argument('--output_folder', type=str, required=True, help='output folder')

args = parser.parse_args()

# helper fns

def exists(val):
    return val is not None

def load_image(path):
    img = Image.open(path).convert('L')
    img = np.array(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.float()
    img = torch.unsqueeze(img, dim=0)
    img = img / 255.
    return img

def save_image(path, img):
    img = img.detach().cpu().numpy()
    img = np.squeeze(img)
    img = np.clip(img, 0, 1)
    img = img * 255.
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path+'sarcnn_output.jpg')

# constants

IMAGE_PATH = args.image_path
MODEL_PATH = args.model_path
OUTPUT_FOLDER = args.output_folder

# load SARCNN model

sarcnn = None
if exists(MODEL_PATH):
    sarcnn_path = Path(MODEL_PATH)
    assert sarcnn_path.exists(), 'pretrained sarcnn must exist'

    sarcnn = torch.load(str(sarcnn_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noisy_img = load_image(IMAGE_PATH)
noisy_img = noisy_img.to(device)

sarcnn_clean_img = sarcnn.denoise(noisy_img)

save_image(OUTPUT_FOLDER, sarcnn_clean_img)
