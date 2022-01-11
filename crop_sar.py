from pathlib import Path

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()

parser.add_argument('--inputs_dir', type = str, help='input directory')
parser.add_argument('--outputs_dir', type = str, help='output directory')
parser.add_argument('--patch_size', type = int, help='patch size')
parser.add_argument('--stride', type = int, help='stride')
parser.add_argument('--step', type = int, help='step')

args = parser.parse_args()

pat_size = args.patch_size
stride = args.stride
step = args.step

outputs_dir = Path(args.outputs_dir)
outputs_dir.mkdir(parents = True, exist_ok = True)

# for dirpath, dirnames, filenames in os.walk(args.inputs_dir):
#     structure = os.path.join(outputs_dir, dirpath[len(args.inputs_dir):])

#     if not os.path.isdir(structure):
#         os.mkdir(structure)

path = Path(args.inputs_dir)
image_files = [*path.glob('*.npy')]

print(f"number of data {len(image_files)} in {args.inputs_dir}")

count = 0
for f in image_files:
    img = np.load(f)
    H, W = img.shape
    # make patches from the image
    for x in range(0 + step, H - pat_size, stride):
        for y in range(0 + step, W - pat_size, stride):
            patch = img[x:x + pat_size, y:y + pat_size]
            patch_name = f"{f.stem}_{count}.npy"
            np.save(Path(outputs_dir) / patch_name, patch)
            count += 1

print(f"{count} patches are saved")
