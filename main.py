import sys
import time
import cv2
# import datetime
from datetime import datetime
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from torch import Tensor, distributed

import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# import torchgeo
import logging
from torchgeo.datasets import CDL, BoundingBox, stack_samples, Landsat7, Landsat8, Landsat
from torchgeo.samplers import RandomGeoSampler
from matplotlib import pyplot as plt
from pathlib import Path

import torch.amp
from tqdm import tqdm

# from GeoDSWrapper import GeoDSWrapper
import mpBatcher
from sampleChooser import AdaptiveGeoSampler, chooseSamples
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

BATCH_SIZE = 48
NUM_EPOCHS = 1000

BANDS = [f"SR_B{i}" for i in range(2, 8)]

# torch.cuda.set_per_process_memory_fraction(0.65, 0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
# epoch_number = 0

# def dsGen():
#     cdl = CDL("data", download=True, checksum=True, years=[2023, 2022])
#
#     landsat8 = Landsat8("data/IA2/2023", bands=BANDS)
#     landsat_train = landsat8
#     dataset_train =  landsat_train & cdl
#     return dataset_train

# sampler_train = RandomGeoSampler(dataset_train, size=256, length=5000)

CORN = 1

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PIN_MEMORY = True if DEVICE == "cuda" else False
UINT16_MAX = 65535.0

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

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

# def tversky_loss(delta = 0.7, smooth = 0.000001):
 #    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	# Link: https://arxiv.org/abs/1706.05721
 #    Parameters
 #    ----------
 #    delta : float, optional
 #        controls weight given to false positive and false negatives, by default 0.7
 #    smooth : float, optional
 #        smoothing constant to prevent division by zero errors, by default 0.000001
 #    """
 #    def loss_function(y_true, y_pred):
 #        # axis = identify_axis(y_true.shape)
 #        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
 #        tp = y_true * y_pred
 #        fn = y_true * (1-y_pred)
 #        fp = ((1-y_true) * y_pred)
 #        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
 #        # Average class scores
 #        tversky_loss = torch.mean(1-tversky_class)
	#
 #        return tversky_loss
	#
 #    return loss_function

def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.

    Returns:
        tversky_loss: the Tversky loss.

    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.long().squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def computePercentiles(ds, qa: Landsat, band: int):
    sampler = AdaptiveGeoSampler(ds, qa, 100, 1)

    points = []

    for sample in sampler:
        data = ds[sample]

        lin: np.ndarray = data["image"][band].numpy().flatten()

        points.extend(lin.tolist())

    return (np.percentile(points, [1, 99]))

def normalizeBandImage(tensor: Tensor, percentiles, clip) -> Tensor:
    c, d = percentiles

    tensor = torch.mul(torch.sub(tensor, c), 1 / (d - c))

    if clip:
        tensor = torch.clip(tensor, 1.0, 0.0)

    return tensor

def normalizeBatch(batchImage: Tensor, bandPercentiles, clip = False) -> Tensor:
    for bandNum, percentiles in enumerate(bandPercentiles):
        for batchNum in range(batchImage.shape[0]):
            batchImage[batchNum][bandNum] = normalizeBandImage(batchImage[batchNum][bandNum], percentiles, clip)

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
    cdl = CDL("data", download=True, checksum=True, years=[2023, 2022])

    landsat8 = Landsat8("data/IA2/2023", bands=BANDS) | Landsat8("data/IA2/2022", bands=BANDS)
    landsat8_test = Landsat8("data/IA2/2023", bands=BANDS) | Landsat8("data/IA2/2022", bands=BANDS)
    qa_test = Landsat8("data/IA2/2023", bands=["QA_PIXEL"]) | Landsat8("data/IA2/2022", bands=["QA_PIXEL"])
    landsat_test = landsat8_test

    dataset_test =  landsat_test & cdl

    landsat_train = landsat8
    qa_train = Landsat8("data/IA2/2023", bands=["QA_PIXEL"]) | Landsat8("data/IA2/2022", bands=["QA_PIXEL"])

    dataset_train =  landsat_train & cdl

    # mgr = mpBatcher.MPDatasetManager(dsGen=dsGen, nBuckets=4, nWorkers=1)
    sampler_test = AdaptiveGeoSampler(dataset_test, qa_test, 500, BATCH_SIZE)
    sampler_train = AdaptiveGeoSampler(dataset_train, qa_train, 1000, BATCH_SIZE)

    test_samples = [sample for sample in sampler_test]

    # sampler_train = mpBatcher.MPDatasetSampler(mgr, 1000, BATCH_SIZE)
    # computePercentiles(dataset_train, qa_train, 2)
    ddp_setup(rank, world_size)
    DEVICE = rank
    unet = UNET.UNet(len(BANDS), n_classes=1).to(DEVICE)
    unet = DDP(unet, device_ids=[DEVICE])

    torch.distributed.barrier()
    if len(sys.argv) >= 2:
        unet.module.load_state_dict(torch.load(sys.argv[1], map_location={'cuda:0': f'cuda:{rank}'}))

    # lossFunc = tversky_loss#BCEWithLogitsLoss()
    # lossFunc = BCEWithLogitsLoss()
    # scaler = torch.amp.grad_scaler.GradScaler()
    opt = torch.optim.AdamW(unet.parameters(), lr=0.00005, foreach=True, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=5)  # goal: maximize Dice score

    bandPercentiles = []
    bandPercentiles = [([ 7767., 11340.]), ([ 8312., 12746.]), ([ 7861., 14691.]), ([ 8873., 28555.]), ([ 9059., 23145.]), ([ 8147., 20632.])]
    if False:
        if DEVICE == 0:
            print("computing band percentiles...")
            for i, _ in tqdm(enumerate(BANDS), total=len(BANDS)):
                bandPercentiles.append(computePercentiles(dataset_train, qa_train, i))
            print(bandPercentiles)
        else:
            for i, _ in (enumerate(BANDS)):
                bandPercentiles.append(computePercentiles(dataset_train, qa_train, i))

    for epoch in tqdm(range(NUM_EPOCHS)):
        dataloader_test = DataLoader(dataset_test, batch_size=1, sampler=test_samples, collate_fn=stack_samples, pin_memory=True, num_workers=0)
        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=sampler_train, collate_fn=stack_samples, pin_memory=True, num_workers=0)

        totalTrainLoss = 0
        totalTrainSteps = 0
        totalTestLoss = 0
        totalTestSteps = 0

        if epoch != 0:
            unet.train()

            def train_epoch():
                nonlocal totalTrainLoss
                nonlocal totalTrainSteps

                image = (batch["image"]).to(DEVICE)
                image = normalizeBatch(image, bandPercentiles)

                mask = torch.where(batch["mask"] <= 60, batch["mask"], 0.0)
                mask = torch.where(mask >= 1, 1.0, 0.0).to(DEVICE)
                mask = denoiseMask(mask).to(DEVICE)

                opt.zero_grad()

                pred = unet(image)
                loss = dice_loss(F.sigmoid(pred.squeeze(1)), mask.squeeze(1).float(), multiclass=False)

                loss.backward()
                opt.step()

                totalTrainLoss += loss.item()
                totalTrainSteps += 1

            if DEVICE == 0:
                for batch in tqdm(dataloader_train):
                    train_epoch()
            else:
                for batch in dataloader_train:
                    train_epoch()



        #if DEVICE == 0:
        with torch.no_grad():
            unet.eval()

            balances =[]
            scores = []

            def run_test():
                nonlocal totalTestLoss
                nonlocal totalTestSteps
                nonlocal balances
                nonlocal scores

                image = (batch["image"]).to(DEVICE)
                image = normalizeBatch(image, bandPercentiles)
                mask = torch.where(batch["mask"] <= 60, batch["mask"], 0.0)
                mask = torch.where(mask >= 1, 1.0, 0.0).to(DEVICE)
                # print(mask.shape)
                # lol
                mask = denoiseMask(mask).to(DEVICE).squeeze().unsqueeze(0).unsqueeze(0)
                # print(mask.shape)

                nonzeroes = torch.where(mask > 0.0, 1.0, 0.0).count_nonzero()
                total = torch.numel(mask)

                pred = unet(image)
                # totalTestLoss += lossFunc(pred, mask)
                mask_pred = (F.sigmoid(pred) > 0.5).float()
                # compute the Dice score
                score = dice_coeff(mask_pred, mask, reduce_batch_first=False)
                totalTestLoss  += score
                totalTestSteps += 1

                balances.append(nonzeroes.item() / total)
                scores.append(score.item())

            if DEVICE==0:
                for batch in tqdm(dataloader_test):
                    run_test()
            else:
                for batch in dataloader_test:
                    run_test()
        # plt.close('all')
        # plt.scatter(balances, scores)
        # plt.show()
        if DEVICE == 0:
            dir_checkpoint = Path("./checkpoints")
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = unet.module.state_dict()
            # state_dict['mask_values'] = mask.cpu().detach()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        avgTrainLoss = 0.0
        if epoch != 0:
            avgTrainLoss = totalTrainLoss / totalTrainSteps
        avgTestLoss = totalTestLoss / totalTestSteps
        # scheduler.step(avgTestLoss)

        # print the model training and validation information
        if DEVICE == 0:
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, NUM_EPOCHS))
            print("Train loss: {:.6f}, Dice score: {:.4f}".format(
                avgTrainLoss, avgTestLoss))

        def updateConcentrationsFromScoresAndDensities(scores, densities, nBuckets):
            assert len(scores) == len(densities)

            totalErrors = np.zeros(nBuckets)
            totalErrorCounts = np.zeros(nBuckets)

            for score, density in zip(scores, densities):
                err = 1.0 - score

                targetBucket = int(np.interp(density, [0.0, 1.0], [0, nBuckets - 1]))

                totalErrors[targetBucket] += err
                totalErrorCounts[targetBucket] += 1.0

            with np.errstate(divide='ignore', invalid="ignore"):
                # averageErrors = totalErrors / totalErrorCounts
                # averageErrors[totalErrorCounts == 0] = 0.0
                # totalAverageError = averageErrors.sum()

                proportions = totalErrors / totalErrors.sum()

                return list(proportions)

        # sampler_train.concentrations = updateConcentrationsFromScoresAndDensities(scores, balances, 5)
        # sampler_train.concentrations = [0.25, 0.25, 0.25, 0.25, 0.0]
        if DEVICE == 0:
            print("concentrations:",sampler_train.concentrations)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("world size:", world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    distributed.destroy_process_group()

# main(0, 1)
