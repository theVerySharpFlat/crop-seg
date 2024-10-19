import time
# import datetime
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import torchgeo
import logging
from torchgeo.datasets import CDL, BoundingBox, stack_samples, Landsat7, Landsat8
from torchgeo.samplers import RandomGeoSampler
from matplotlib import pyplot as plt

import torch.amp

from unet import unet

import numpy as np

import torch

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

cdl = CDL("data", download=True, checksum=True)

# landsat7 = Landsat7("data/shelby_landsat_2", bands=Landsat7.all_bands[:5])
landsat8 = Landsat8("data/shelby_landsat_2", bands=Landsat8.all_bands)

landsat = landsat8

dataset =  landsat & cdl

sampler = RandomGeoSampler(dataset, size=256, length=10000)
dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, collate_fn=stack_samples)


CORN = 1

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Change here to adapt to your data
# n_channels=3 for RGB images
# n_classes is the number of probabilities you want to get per pixel
print(f"nChannels={len(landsat8.default_bands)}")
print(f"channels={landsat8.default_bands}")
model = unet.UNet(n_channels=len(landsat8.default_bands), n_classes=1, bilinear=False)

optimizer = torch.optim.Adam(model.parameters(), 0.001)
loss_fn = torch.binary_cross_entropy_with_logits
grad_scaler = torch.amp.grad_scaler.GradScaler(enabled=True)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        inputs, labels = (data["image"], data["mask"])
        labels = torch.where(labels == 1, 1.0, 0.0)

        # Make predictions for this batch
        inputs = inputs.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        labels = labels.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels).squeeze().sum()
        print(f"labels dim: {labels.shape}")
        print(f"outputs dim: {outputs.shape}")
        print(f"loss: {loss}")
        # loss.backward()

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()


        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


for epoch in range(5):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # # Disable gradient computation and reduce memory consumption.
    # with torch.no_grad():
    #     for i, vdata in enumerate(validation_loader):
    #         vinputs, vlabels = vdata
    #         voutputs = model(vinputs)
    #         vloss = loss_fn(voutputs, vlabels)
    #         running_vloss += vloss
    #
    # avg_vloss = running_vloss / (i + 1)
    # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    #
    # # Log the running loss averaged per batch
    # # for both training and validation
    # writer.add_scalars('Training vs. Validation Loss',
    #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
    #                 epoch_number + 1)
    # writer.flush()
    #
    # # Track best performance, and save the model's state
    # if avg_vloss < best_vloss:
    #     best_vloss = avg_vloss
    #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
    #     torch.save(model.state_dict(), model_path)

    epoch_number += 1
