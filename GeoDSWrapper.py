import torch
from torch import nonzero
from torch.utils.data import Dataset, sampler
from torchgeo.datasets import BoundingBox
import numpy as np
import cv2
from torchgeo.samplers import RandomGeoSampler


class GeoDSWrapper(Dataset):
    def __init__(self, geods, patchSize = 256, denoise=False, nPerEpoch = 1000, minNonzero = 390000):
        self.ds = geods
        self.minNonzero = minNonzero
        self.sampler = RandomGeoSampler(self.ds, size=patchSize, length=nPerEpoch)
        self.denoise = denoise

    def goodSample(self, image) -> bool:
        nonzeroCount = torch.count_nonzero(image)
        return nonzeroCount.item() > self.minNonzero

    def denoiseMask(self, mask):
        if not self.denoise:
            return mask
        else:
            kernel = np.ones((3,3),np.uint8)
            return torch.from_numpy(cv2.morphologyEx(mask.squeeze().numpy(), cv2.MORPH_OPEN,kernel, iterations = 1)).unsqueeze(0)


    def __getitem__(self, idx: BoundingBox):
        # while True:
        data = self.ds[idx]

        while not self.goodSample(data["image"]):
            data = self.ds[next(iter(self.sampler))]

        mask = torch.where(data["mask"] <= 60, data["mask"], 0.0)
        mask = torch.where(mask >= 1, 1.0, 0.0)
        data["mask"] = mask
        data["mask"] = self.denoiseMask(data["mask"])

        return data
