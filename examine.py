from math import inf
import time
# import datetime
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, dataloader
from torch.optim import Adam

from torch import Tensor

import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# import torchgeo
import logging
from torchgeo.datasets import CDL, BoundingBox, landsat, stack_samples, Landsat7, Landsat8
from torchgeo.datasets.geo import pyproj
from torchgeo.samplers import RandomGeoSampler, Units
from matplotlib import pyplot as plt
from pathlib import Path

import torch.amp
from tqdm import tqdm

from GeoDSWrapper import GeoDSWrapper
from unet import unet as UNET

import numpy as np

import torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os, sys

BATCH_SIZE = 32
NUM_EPOCHS = 100

BANDS = [f"SR_B{i}" for i in range(2, 8)]
print(f"bands: {BANDS}")

# torch.cuda.set_per_process_memory_fraction(0.65, 0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# epoch_number = 0

cdl = CDL("data", download=True, checksum=True, years=[2023, 2022])

# landsat7 = Landsat7("data/shelby_landsat_2", bands=Landsat7.all_bands[:5])
landsat8_test = Landsat8("data/IA/L2/1023", bands=BANDS)
landsat_test = landsat8_test
dataset_test =  GeoDSWrapper(landsat_test & cdl)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def main(rank, world_size):
    # ddp_setup(rank, world_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, sampler=dataset_test.sampler, collate_fn=stack_samples, pin_memory=True, num_workers=0)
    DEVICE = "cuda:0"
    unet = UNET.UNet(len(BANDS), n_classes=1).to(DEVICE)
    state_dict = torch.load(sys.argv[1])
    unet.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        # unet.eval()

        for loc in dataset_test.sampler:
            # data = landsat8_test[loc]
            # data["mask"] = cdl[loc]["mask"]
            data = dataset_test[loc]
            # print(data)
            print(data.keys())
            # print(data["image"][(0,1,2)])
            # print(landsat8_test.rgb_bands)
            fig = landsat8_test.plot(data)
            fig.savefig("out.png")

            image = data["image"].to(DEVICE)
            mask = torch.where(data["mask"] <= 60, data["mask"], 0.0)
            mask = torch.where(mask >= 1, 1.0, 0.0).to(DEVICE)
            pred = unet(image.unsqueeze(0))
            pred = ((F.sigmoid(pred.cpu())) > 0.5).float()
            # pred = (((pred.cpu())) > 0.5).float()

            figPred, ax = plt.subplots(1, 1, figsize=(4,4))
            ax.imshow(pred.squeeze())
            ax.axis('off')

            figPred.savefig("pred.png")

            figActual, ax = plt.subplots(1, 1, figsize=(4,4))
            ax.imshow(mask.cpu().squeeze())
            ax.axis('off')
            figActual.savefig("truth.png")

            break

main(1, 2)
