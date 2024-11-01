import time
# import datetime
from datetime import datetime
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from torch import Tensor

import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# import torchgeo
import logging
from torchgeo.datasets import CDL, BoundingBox, stack_samples, Landsat7, Landsat8
from torchgeo.samplers import RandomGeoSampler
from matplotlib import pyplot as plt
from pathlib import Path

import torch.amp
from tqdm import tqdm

from unet import unet as UNET

import numpy as np

import torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# def ddp_setup(rank: int, world_size: int):
#     """
#     Args:
#        rank: Unique identifier of each process
#       world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     torch.cuda.set_device(rank)
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)

BATCH_SIZE = 32
NUM_EPOCHS = 100

BANDS = [f"SR_B{i}" for i in range(2, 8)]

# torch.cuda.set_per_process_memory_fraction(0.65, 0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# epoch_number = 0

cdl = CDL("data", download=True, checksum=True, years=[2023, 2022])

# landsat7 = Landsat7("data/shelby_landsat_2", bands=Landsat7.all_bands[:5])
landsat8_test = Landsat8("data/IA/L2/1022", bands=BANDS)
landsat_test = landsat8_test
dataset_test =  landsat_test & cdl

sampler_test = RandomGeoSampler(dataset_test, size=256, length=1000)

landsat8_train = Landsat8("data/IA/L2/1023", bands=BANDS)
landsat_train = landsat8_train
dataset_train =  landsat_train & cdl

sampler_train = RandomGeoSampler(dataset_train, size=256, length=2000)

CORN = 1

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False
UINT16_MAX = 65535.0

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
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, sampler=sampler_test, collate_fn=stack_samples, pin_memory=True, num_workers=0)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=sampler_train, collate_fn=stack_samples, pin_memory=True, num_workers=0)
    DEVICE = rank
    unet = UNET.UNet(len(BANDS), n_classes=1).to(DEVICE)
    # unet = DDP(unet, device_ids=[DEVICE])

    lossFunc = BCEWithLogitsLoss()
    scaler = torch.amp.grad_scaler.GradScaler()
    opt = torch.optim.RMSprop(unet.parameters(),
                              lr=0.001, weight_decay=1e-8, momentum=0.999, foreach=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=5)  # goal: maximize Dice score

    for epoch in tqdm(range(NUM_EPOCHS)):
        totalTrainLoss = 0
        totalTrainSteps = 0
        totalTestLoss = 0
        totalTestSteps = 0

        unet.train()

        def train_epoch():
            nonlocal totalTrainLoss
            nonlocal totalTrainSteps

            image = (batch["image"]).to(DEVICE)
            # print(image)
            mask = torch.where(batch["mask"] <= 60, batch["mask"], 0.0)
            mask = torch.where(mask >= 1, 1.0, 0.0).to(DEVICE)

            opt.zero_grad()

            pred = unet(image)
            loss = lossFunc(pred, mask)
            # loss += dice_loss(F.sigmoid(pred.squeeze(1)), mask.squeeze(1).float(), multiclass=False)

            # Zero your gradients for every batch!
            # loss.backward()
            scaler.scale(loss).backward()

            # Adjust learning weights
            # opt.step()
            scaler.step(opt)
            scaler.update()

            totalTrainLoss += loss.item()
            totalTrainSteps += 1

        if DEVICE == 0:
            for batch in tqdm(dataloader_train):
                train_epoch()
        else:
            for batch in dataloader_train:
                train_epoch()

        if DEVICE == 0:
            with torch.no_grad():
                unet.eval()

                for batch in tqdm(dataloader_test):
                    image = (batch["image"]).to(DEVICE)
                    mask = torch.where(batch["mask"] <= 60, batch["mask"], 0.0)
                    mask = torch.where(mask >= 1, 1.0, 0.0).to(DEVICE)

                    pred = unet(image)
                    # totalTestLoss += lossFunc(pred, mask)
                    mask_pred = (pred > 0.5).float()
                    # compute the Dice score
                    totalTestLoss  += dice_coeff(mask_pred, mask, reduce_batch_first=False)
                    totalTestSteps += 1

                    dir_checkpoint = Path("./checkpoints")
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict = unet.state_dict()
                    state_dict['mask_values'] = mask.cpu().detach()
                    torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                    logging.info(f'Checkpoint {epoch} saved!')

            avgTrainLoss = totalTrainLoss / totalTrainSteps
            avgTestLoss = totalTestLoss / totalTestSteps
            # scheduler.step(avgTestLoss)

            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, NUM_EPOCHS))
            print("Train loss: {:.6f}, Dice score: {:.4f}".format(
                avgTrainLoss, avgTestLoss))

# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size,), nprocs=world_size)

main(0, 1)
