import time
# import datetime
from datetime import datetime
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
# from torch.utils.tensorboard import SummaryWriter
# import torchgeo
import logging
from torchgeo.datasets import CDL, BoundingBox, stack_samples, Landsat7, Landsat8
from torchgeo.samplers import RandomGeoSampler
from matplotlib import pyplot as plt
from pathlib import Path

import torch.amp
from tqdm import tqdm

from unet import unet

import numpy as np

import torch

BATCH_SIZE = 48
NUM_EPOCHS = 100

BANDS = [f"B{i}" for i in range(2, 9)]

# torch.cuda.set_per_process_memory_fraction(0.65, 0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# epoch_number = 0

cdl = CDL("data", download=True, checksum=True)

# landsat7 = Landsat7("data/shelby_landsat_2", bands=Landsat7.all_bands[:5])
landsat8_test = Landsat8("data/jones_landsat", bands=BANDS)
landsat_test = landsat8_test
dataset_test =  landsat_test & cdl

sampler_test = RandomGeoSampler(dataset_test, size=256, length=500)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, sampler=sampler_test, collate_fn=stack_samples, num_workers=0)

landsat8_train = Landsat8("data/jones_landsat", bands=BANDS)
landsat_train = landsat8_train
dataset_train =  landsat_train & cdl

sampler_train = RandomGeoSampler(dataset_train, size=256, length=5000)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=sampler_train, collate_fn=stack_samples, num_workers=0)

CORN = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
UINT16_MAX = 65535.0

unet = unet.UNet(len(BANDS), n_classes=1).to(DEVICE)

lossFunc = BCEWithLogitsLoss()
scaler = torch.GradScaler()
opt = Adam(unet.parameters(), lr = 0.0002)

for epoch in tqdm(range(NUM_EPOCHS)):
    totalTrainLoss = 0
    totalTrainSteps = 0
    totalTestLoss = 0
    totalTestSteps = 0

    unet.train()

    for batch in tqdm(dataloader_train):
        image = (batch["image"]).to(DEVICE)
        # print(image)
        mask = torch.where(batch["mask"] == CORN, 1.0, 0.0).to(DEVICE)

        opt.zero_grad()

        pred = unet(image)
        loss = lossFunc(pred, mask)

        # Zero your gradients for every batch!
        opt.zero_grad()
        loss.backward()

        # Adjust learning weights
        opt.step()

        totalTrainLoss += loss.item()
        totalTrainSteps += 1

    with torch.no_grad():
        unet.eval()

        for batch in tqdm(dataloader_test):
            image = (batch["image"]).to(DEVICE)
            mask = torch.where(batch["mask"] == CORN, 1.0, 0.0).to(DEVICE)

            pred = unet(image)
            totalTestLoss += lossFunc(pred, mask)
            totalTestSteps += 1

            dir_checkpoint = Path("./checkpoints")
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = unet.state_dict()
            state_dict['mask_values'] = mask.cpu().detach()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    avgTrainLoss = totalTrainLoss / totalTrainSteps
    avgTestLoss = totalTestLoss / totalTestSteps

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(epoch + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(
        avgTrainLoss, avgTestLoss))
