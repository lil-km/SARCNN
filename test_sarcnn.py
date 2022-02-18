"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling model testing implementation
"""

# Imports

import argparse
import time
import os
import numpy as np

from pathlib import Path

import torch
from PIL import Image

from sarcnn.sarcnn import SARCNN
from sarcnn.utils import *

# arguments parsing

parser = argparse.ArgumentParser(description='SARCNN Testing')
parser.add_argument('--model_path', type=str, required=True, help='pretrained model path')
parser.add_argument('--real_sar', dest='real_sar', action='store_true', help='real sar images flag')
parser.add_argument('--test_data', type=str, required=True, help='test data folder')
parser.add_argument('--test_dir', type=str, required=True, help='output folder')
parser.add_argument('--filename', type=str, required=True, help='test image filename')

args = parser.parse_args()

# helper fns

def exists(val):
    return val is not None

# constants

MODEL_PATH = args.model_path
REAL_SAR = args.real_sar
TEST_DATA = args.test_data
TEST_DIR = args.test_dir
FILENAME = args.filename

TEST_DIR = Path(TEST_DIR)
TEST_DIR.mkdir(parents = True, exist_ok = True)


def denoiser_test(model, device):
    if REAL_SAR:
        test_folder = TEST_DATA+"/real"
        test_files = glob((test_folder+f'/{FILENAME}*.npy').format('float32'))
        sarcnn.test(test_files, save_dir=TEST_DIR, dataset_dir=test_data, device=device, real_sar=REAL_SAR)
    else:
        test_folder = TEST_DATA+"/simulated"
        test_files = glob((test_folder+f'/{FILENAME}*.npy').format('float32'))
        psnr, ssim = sarcnn.test(test_files, save_dir=TEST_DIR, dataset_dir=test_data, device=device, real_sar=REAL_SAR)

        return psnr, ssim

# load SARCNN model

sarcnn = None
if exists(MODEL_PATH):
    sarcnn_path = Path(MODEL_PATH)
    assert sarcnn_path.exists(), 'pretrained sarcnn must exist'

    sarcnn = torch.load(str(sarcnn_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if REAL_SAR:
    denoiser_test(sarcnn, device)
else:
    psnr, ssim = denoiser_test(sarcnn, device)
    print(psnr)
    print(ssim)
