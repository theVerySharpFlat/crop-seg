from math import inf, trunc
import time
# import datetime
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, dataloader
from torch.optim import Adam

from torch import Tensor, count_nonzero

import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# import torchgeo
import logging
from torchgeo.datasets import CDL, BoundingBox, landsat, stack_samples, Landsat7, Landsat8
from torchgeo.datasets.geo import pyproj
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler, Units
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
import numpy as np
import cv2

BATCH_SIZE = 32
NUM_EPOCHS = 100

BANDS = [f"SR_B{i}" for i in range(2, 8)]
print(f"bands: {BANDS}")

# torch.cuda.set_per_process_memory_fraction(0.65, 0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# epoch_number = 0

cdl = CDL("data", download=True, checksum=True, years=[2023, 2023])

# landsat7 = Landsat7("data/shelby_landsat_2", bands=Landsat7.all_bands[:5])
landsat8_test = Landsat8("data/IA2/2022", bands=BANDS)
landsat_test = landsat8_test
dataset_test =  landsat_test & cdl

landsat8_clouds = Landsat8("data/IA2/2022", bands=["QA_PIXEL"])
landsat_clouds = landsat8_test
dataset_clouds =  landsat_clouds & cdl

ROI = [41.0628, 43.125, -95.4053, -91.6150, 1665460800, 1667102399]
transformer = pyproj.Transformer.from_crs("EPSG:4326", str(landsat8_test.crs))
BOUND_A = (41.5661, -94.7021)
BOUND_B = (42.3748, -93.9111)
BOUND_A_TRANS = transformer.transform(*BOUND_A)
BOUND_B_TRANS = transformer.transform(*BOUND_B)

ROI = BoundingBox(BOUND_A_TRANS[0], BOUND_B_TRANS[0], BOUND_A_TRANS[1], BOUND_B_TRANS[1], ROI[4], ROI[5])
print("ROI:", ROI)

sampler_test = RandomGeoSampler(dataset_test, size=256, length=1000)#, roi=ROI)

print("roi: ", sampler_test.roi)
print("crs: ", landsat8_test.crs)
print("crs: ", type(landsat8_test.crs))

test_wrapper_ds = GeoDSWrapper(dataset_test)



def remove_noise(image):
  kernel = np.ones((3,3),np.uint8)
  return cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel, iterations = 1)

def qa_good(qamask: np.ndarray):
    numel = qamask.size
    filler_mask = np.bitwise_and(qamask, 0b01)

    if np.count_nonzero(filler_mask) / numel > 0.01:
        return False

    cloud_mask = np.greater_equal(np.right_shift(np.bitwise_and(qamask, 0b11 << 8), 8), 0b10)

    print("cloud density:", (np.count_nonzero(cloud_mask) / numel))
    if (np.count_nonzero(cloud_mask) / numel) > 0.01:
        return False

    return True
def normalizeBandImage(tensor: Tensor, percentiles) -> Tensor:
    c, d = percentiles

    tensor = torch.mul(torch.sub(tensor, c), 1 / (d - c))

    return tensor

def normalizeBatch(batchImage: Tensor, bandPercentiles) -> Tensor:
    print("batch shape", batchImage.shape)
    for bandNum, percentiles in enumerate(bandPercentiles):
        for batchNum in range(batchImage.shape[0]):
            print("normalize batch num", batchNum)
            batchImage[batchNum][bandNum] = normalizeBandImage(batchImage[batchNum][bandNum], percentiles)

    return batchImage

def denoiseMask(mask: Tensor):
    assert len(mask.shape) <= 4 and len(mask.shape) >= 2

    prevNDims = len(mask.shape)

    while(len(mask.shape) < 4):
        mask = mask.unsqueeze(0)
    
    kernel = np.ones((2,2),np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i][j] = torch.from_numpy(cv2.morphologyEx(mask[i][j].cpu().numpy(), cv2.MORPH_OPEN, kernel, iterations = 1))

    while(len(mask.shape) > prevNDims):
        mask = mask.squeeze(0)

    return mask

def main(rank, world_size):
    bandPercentiles = [([ 7767., 11340.]), ([ 8312., 12746.]), ([ 7861., 14691.]), ([ 8873., 28555.]), ([ 9059., 23145.]), ([ 8147., 20632.])]
    # ddp_setup(rank, world_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, sampler=sampler_test, collate_fn=stack_samples, pin_memory=True, num_workers=0)
    DEVICE = "cuda:0"
    unet = UNET.UNet(len(BANDS), n_classes=1).to(DEVICE)
    state_dict = torch.load(sys.argv[1], map_location={"cuda:0": "cpu"})
    unet.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        # unet.eval()

        for loc in test_wrapper_ds.sampler:
            # data = landsat8_test[loc]
            data = test_wrapper_ds[loc]
            # print(data)
            print(data.keys())
            # print(data["image"][(0,1,2)])
            # print(landsat8_test.rgb_bands)
            fig = landsat8_test.plot(data)
            fig.savefig("out.png")

            image = data["image"].to(DEVICE)
            print("b1:", image.squeeze().cpu().numpy()[1])
            print("percentiles:", np.percentile(image.squeeze().cpu().numpy()[1], [0, 1, 99, 100]))
            image = normalizeBatch(image.unsqueeze(0), bandPercentiles).squeeze()
            mask = torch.where(data["mask"] <= 60, data["mask"], 0.0)
            mask = torch.where(mask >= 1, 1.0, 0.0)
            denoisedMask = denoiseMask(mask)
            # mask_pre = mask

            mask = torch.from_numpy((mask.squeeze().numpy())).unsqueeze(0)
            mask = mask.to(DEVICE)

            pred = unet(image.unsqueeze(0))
            pred = ((F.sigmoid(pred.cpu())) > 0.5).float()

            figPred, ax = plt.subplots(1, 5, figsize=(10,10))

            clouds = landsat8_clouds[loc]["image"].squeeze().long().numpy()
            print(landsat8_clouds[loc]["image"].squeeze())
            clouds: np.ndarray = clouds.astype(np.uint16)
            print(clouds)
            clouds = np.greater_equal(np.right_shift(np.bitwise_and(clouds, 0b11 << 8), 8), 0b10)
            # clouds = torch.bitwise_and(clouds, 0b1 << 3)
            # clouds = torch.where(clouds > 0, 1.0, 0.0)

            print("b1:", image.squeeze().cpu().numpy()[1])
            print("percentiles:", np.percentile(image.squeeze().cpu().numpy()[1], [0, 1, 99, 100]))

            ax[0].imshow(pred.squeeze())
            ax[1].imshow(mask.cpu().squeeze())
            ax[2].imshow(denoisedMask.cpu().squeeze())
            ax[3].imshow(clouds)
            ax[4].imshow(image.squeeze().cpu().numpy()[[2, 1, 0]].transpose((1, 2, 0)))
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[3].axis('off')
            ax[4].axis('off')

            figPred.show()
            plt.show(block=False)

            zeroCount = torch.count_nonzero(image)
            print(f"zerocount: {zeroCount}")
            if (zeroCount < 390000):
                print("shit")

            if (not qa_good(landsat8_clouds[loc]["image"].squeeze().long().numpy().astype(np.uint16))):
                print("crap")

            # figPred.savefig("pred.png")
            #
            # figActual, ax = plt.subplots(1, 1, figsize=(4,4))
            # ax.imshow(mask.cpu().squeeze())
            # ax.axis('off')
            # figActual.savefig("truth.png")
            input()
            plt.close(fig="all")

main(1, 2)
