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
parser.add_argument('--image_folder', type=str, required=True,
                    help='path to your folder of images for learning the SARCNN')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch-size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--output_file_name', type=str, default = "SARCNN", help='output_file_name')
parser.add_argument('--wandb-name', type=str, default='sarcnn', help='name of wandb run')

args = parser.parse_args()

# helper functions

def train_img_transform(image_width, crop_size):
    transform = T.Compose([
                    T.CenterCrop((image_width, image_width)),
                    T.RandomCrop((crop_size, crop_size)),
                    # T.ToTensor()
                    # AWGN(0., std_value)
            ])
    return transform

def val_img_transform(image_width):
    transform = T.Compose([
                    T.CenterCrop((image_width, image_width)),
            ])
    return transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# constants

ROOT_DIR = args.image_folder
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
OUTPUT_FILE_NAME = args.output_file_name
WANDB_NAME = args.wandb_name

# create dataset and dataloader

ds = NoisyDataset(root_dir=ROOT_DIR, std=.25, mean=0.0, transform=train_img_transform(180, 40), shuffle=True)

assert len(ds) > 0, 'dataset is empty'
print(f'{len(ds)} images found for training.')

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

ds_test = NoisyDataset(root_dir=ROOT_DIR, std=.25, mean=0.0, transform=val_img_transform(180), shuffle=True)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

clean_img = next(iter(dl_test))
clean_img = clean_img.to(device)

sarcnn = SARCNN(
    num_layers=17,
    num_features=64,
    kernel_size=3,
    padding=1,
    image_channels=1,
    image_size=64
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
    project=args.wandb_name,  # 'dncnn' by default
    config=model_config
)

def save_model(path):
    # save_obj = sarcnn.state_dict()
    torch.save(sarcnn, path)

# save_model(f'./{OUTPUT_FILE_NAME}.pt')


# trainig

for epoch in range(EPOCHS):
    for i, clean in enumerate(dl):
        if i % 10 == 0:
            t = time.time()

        clean = clean.to(device)
        noise = torch.Tensor(clean.size()).normal_(mean=0, std=.25).to(device)
        noise = ((noise) / (M - m)).float()
        noisy = clean + noise
        loss = sarcnn(noisy, return_loss=True, x=clean)
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

            val_noise = torch.Tensor(clean_img.size()).normal_(mean=0, std=.25).to(device)
            val_noise = ((val_noise) / (M - m)).float()
            noisy_img = clean_img + val_noise
            clean_img_sarcnn = sarcnn.denoise(noisy_img)

            log = {
                **log,
            }

            compare_image = torch.cat((
                torch.unsqueeze(denormalize_sar(noisy_img)[0], dim=0),
                torch.unsqueeze(denormalize_sar(clean_img_sarcnn)[0], dim=0),
                torch.unsqueeze(denormalize_sar(clean_img)[0], dim=0)
                ), dim=0)

            # save the image to wandb
            
            grid = make_grid(compare_image, normalize=False)
            ndarr = grid.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            pil_image = Image.fromarray(ndarr)

            log['image'] = wandb.Image(pil_image, caption="Denoised image")

        if i % 10 == 9:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f'sample_per_sec - {sample_per_sec}')

        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end
    
    model_artifact = wandb.Artifact('trained-sarcnn', type='model', metadata=dict(model_config))
    run.log_artifact(model_artifact)


save_model(f'./{OUTPUT_FILE_NAME}-final.pt')
model_artifact = wandb.Artifact('trained-sarcnn', type='model', metadata=dict(model_config))
run.log_artifact(model_artifact)

wandb.finish()
