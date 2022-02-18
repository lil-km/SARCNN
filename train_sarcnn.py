"""
Project: sarcnn
Author: khalil MEFTAH
Date: 2021-12-10
SARCNN: Deep Neural Convolutional Network for SAR Images Despeckling model training implementation
"""

# Imports

import argparse
import time
import wandb
import numpy as np
import os
import random
import shutil

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms as T

from sarcnn.sarcnn import SARCNN
from sarcnn.loader import NoisyDataset
from sarcnn.utils import *

# arguments parsing

parser = argparse.ArgumentParser(description='SARCNN Training')
parser.add_argument('--train_image_folder', type=str, required=True,
                    help='path to your folder of images for learning the SARCNN')
parser.add_argument('--val_image_folder', type=str, required=True,
                    help='path to your folder of images for validating the SARCNN')
parser.add_argument('--plot_image_folder', type=str, required=True,
                    help='path to your folder of images for plot validating the SARCNN')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--output_file_name', type=str, default="SARCNN", help='output_file_name')
parser.add_argument('--wandb-name', type=str, default='sarcnn', help='name of wandb run')
parser.add_argument('--save_every_n_steps', default=500, type = int, help = 'Save a checkpoint every n steps')
parser.add_argument('--lambda_factor', type=float, default=1e-3, help='lambda weighting constant')
parser.add_argument('--perceptual_loss', dest='perceptual_loss', action='store_true', help='compute the perceptual loss')

args = parser.parse_args()

# helper functions

def train_img_transform():
    transform = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    RandomRot90(p=0.5)
            ])
    return transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# constants

TRAIN_DIR = args.train_image_folder
VAL_DIR = args.val_image_folder
PLOT_DIR = args.plot_image_folder
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
OUTPUT_FILE_NAME = args.output_file_name
WANDB_NAME = args.wandb_name
PERCEPTUAL = args.perceptual_loss
SAVE_EVERY_N_STEPS = args.save_every_n_steps
LAMBDA = args.lambda_factor


# create dataset and dataloader

train_dataset = NoisyDataset(root_dir=TRAIN_DIR, transform=train_img_transform(), shuffle=True)

assert len(train_dataset) > 0, 'dataset is empty'
print(f'{len(train_dataset)} images found for training.')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

val_dataset = NoisyDataset(root_dir=VAL_DIR, transform=None, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

plot_dataset = NoisyDataset(root_dir=PLOT_DIR, transform=None, shuffle=True)
plot_loader = DataLoader(plot_dataset, batch_size=8, shuffle=False, drop_last=True)

clean_img_plot, noisy_img_plot = next(iter(plot_loader))
clean_img_plot = clean_img_plot.to(device)
noisy_img_plot = noisy_img_plot.to(device)

# create SARCNN model

sarcnn = SARCNN(
    num_layers=19,
    num_features=64,
    kernel_size=3,
    padding=1,
    image_channels=1
)

sarcnn = sarcnn.to(device)

# optimizer

opt = Adam(sarcnn.parameters(), lr=LR)

# experiment tracker

model_config = dict(
    batch_size = BATCH_SIZE,
    learning_rate = LR,
    epochs = EPOCHS
)

run = wandb.init(
    project=args.wandb_name,  # 'sarcnn' by default
    config=model_config
)

# function to save the model

def save_model(path):
    torch.save(sarcnn, path)

save_model(f'./{OUTPUT_FILE_NAME}.pt')

# training loop

for epoch in range(EPOCHS):
    for i, (clean, noisy) in enumerate(train_loader):
        if i % 10 == 0:
            t = time.time()

        sarcnn.train()
        clean, noisy = map(lambda t: t.cuda(), (clean, noisy))

        # train with perceptual loss
        if PERCEPTUAL:
            loss, data_loss, perceptual_loss = sarcnn(noisy, clean, return_loss=True, perceptual_loss=PERCEPTUAL, Lambda=LAMBDA)
            loss.backward()
            opt.step()
            opt.zero_grad()

            log = {}

            if i % 10 == 0:
                log = {
                    'loss': loss.item(),
                    'data_loss': data_loss.item(),
                    'perceptual_loss': perceptual_loss.item(),
                    'epoch': epoch,
                    'iter': i
                }

            if i % 100 == 0:
                # denoise the image
                val_loss = 0
                val_data_loss = 0
                val_perceptual_loss = 0
                psnr = 0
                ssim = 0
                sarcnn.eval()
                with torch.no_grad():
                    for clean_img, noisy_img in val_loader:
                        clean_img, noisy_img = map(lambda t: t.cuda(), (clean_img, noisy_img))
                        val_loss += sarcnn(noisy_img, clean_img, return_loss=True, perceptual_loss=PERCEPTUAL, Lambda=LAMBDA)[0]
                        val_data_loss += sarcnn(noisy_img, clean_img, return_loss=True, perceptual_loss=PERCEPTUAL, Lambda=LAMBDA)[1]
                        val_perceptual_loss += sarcnn(noisy_img, clean_img, return_loss=True, perceptual_loss=PERCEPTUAL, Lambda=LAMBDA)[2]
                        psnr += cal_psnr(denormalize_sar(sarcnn(noisy_img)), denormalize_sar(clean_img))
                        ssim += cal_ssim(denormalize_sar(sarcnn(noisy_img)), denormalize_sar(clean_img))
                        
                    clean_img_plot_sarcnn = sarcnn(noisy_img_plot)
                    val_loss /= (BATCH_SIZE*len(val_loader))
                    val_data_loss /= (BATCH_SIZE*len(val_loader))
                    val_perceptual_loss /= (BATCH_SIZE*len(val_loader))
                    avg_psnr = psnr/len(val_loader)
                    avg_ssim = ssim/len(val_loader)

                log = {
                    **log,
                    'val_loss': val_loss.item(),
                    'val_data_loss': val_data_loss.item(),
                    'val_perceptual_loss': val_perceptual_loss.item(),
                    'psnr': avg_psnr.item(),
                    'ssim': avg_ssim.item()
                }

                compare_image = torch.cat((
                    dynamic_clamp(denormalize_sar(noisy_img_plot[:4])),
                    dynamic_clamp(denormalize_sar(clean_img_plot_sarcnn[:4])),
                    dynamic_clamp(denormalize_sar(noisy_img_plot[:4]-clean_img_plot_sarcnn[:4])),
                    dynamic_clamp(denormalize_sar(clean_img_plot[:4]))
                    ), dim=0)

                # save the image to wandb for visualization
                
                grid = make_grid(compare_image, normalize=False)
                ndarr = grid.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                pil_image = Image.fromarray(ndarr)

                log['image'] = wandb.Image(pil_image, caption="Denoised image")

        # train without perceptual loss
        else:
            loss= sarcnn(noisy, clean, return_loss=True, perceptual_loss=PERCEPTUAL, Lambda=LAMBDA)
            loss.backward()
            opt.step()
            opt.zero_grad()

            log = {}

            if i % 10 == 0:
                log = {
                    'loss': loss.item(),
                    'epoch': epoch,
                    'iter': i
                }

            if i % 100 == 0:
                # denoise the image
                val_loss = 0
                psnr = 0
                ssim = 0
                sarcnn.eval()
                with torch.no_grad():
                    for clean_img, noisy_img in val_loader:
                        clean_img, noisy_img = map(lambda t: t.cuda(), (clean_img, noisy_img))
                        val_loss += sarcnn(noisy_img, clean_img, return_loss=True, perceptual_loss=PERCEPTUAL, Lambda=LAMBDA)
                        psnr += cal_psnr(denormalize_sar(sarcnn(noisy_img)), denormalize_sar(clean_img))
                        ssim += cal_ssim(denormalize_sar(sarcnn(noisy_img)), denormalize_sar(clean_img))
                        
                    clean_img_plot_sarcnn = sarcnn(noisy_img_plot)
                    val_loss /= (BATCH_SIZE*len(val_loader))
                    avg_psnr = psnr/len(val_loader)
                    avg_ssim = ssim/len(val_loader)

                log = {
                    **log,
                    'val_loss': val_loss.item(),
                    'psnr': avg_psnr.item(),
                    'ssim': avg_ssim.item()
                }

                compare_image = torch.cat((
                    dynamic_clamp(denormalize_sar(noisy_img_plot[:4])),
                    dynamic_clamp(denormalize_sar(clean_img_plot_sarcnn[:4])),
                    dynamic_clamp(denormalize_sar(noisy_img_plot[:4]-clean_img_plot_sarcnn[:4])),
                    dynamic_clamp(denormalize_sar(clean_img_plot[:4]))
                    ), dim=0)

                # save the image to wandb for visualization
                
                grid = make_grid(compare_image, normalize=False)
                ndarr = grid.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                pil_image = Image.fromarray(ndarr)

                log['image'] = wandb.Image(pil_image, caption="Denoised image")            

        if i % 10 == 9:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f'sample_per_sec - {sample_per_sec}')

        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(f'./{OUTPUT_FILE_NAME}.pt')

        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end
    
    model_artifact = wandb.Artifact('trained-sarcnn', type='model', metadata=dict(model_config))
    run.log_artifact(model_artifact)


save_model(f'./{OUTPUT_FILE_NAME}-final.pt')
model_artifact = wandb.Artifact('trained-sarcnn', type='model', metadata=dict(model_config))
run.log_artifact(model_artifact)

wandb.finish()
