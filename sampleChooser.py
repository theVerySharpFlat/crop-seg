import torch
from torch._prims_common import Tensor
from torchgeo.datasets import BoundingBox, Landsat
from torchgeo.samplers import RandomGeoSampler
from collections.abc import Iterator
import numpy as np
import math
from tqdm import tqdm

def chooseSamples(dataset, randomSampler: RandomGeoSampler, n):
    minNonzero = 390000

    def goodSample(image) -> bool:
        nonzeroCount = torch.count_nonzero(image)
        return nonzeroCount.item() > minNonzero

    for _ in range(n):
        bbox = next(iter(randomSampler))
        data = dataset[bbox]

        if goodSample(data["image"]):
            yield bbox

class AdaptiveGeoSampler():
    def __init__(self, dataset, qads, n, batchSize, patchSize = 256):
        self.ds = dataset
        self.patchSize = patchSize
        self.batchSize = batchSize
        self.sampler = RandomGeoSampler(dataset, patchSize)
        self.n = n
        self.concentrations = [1.0]
        self.qads = qads

    def density(self, mask: Tensor) -> float:
        m = torch.where(torch.logical_and(mask <= 60, mask >= 1), 1.0, 0.0) 
        nonzeroCount = m.count_nonzero().item()
        total = torch.numel(m)

        return nonzeroCount / total

    def updateConcentrationsFromScoresAndDensities(self, scores, densities, nBuckets):
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

            self.concentrations = list(proportions)

    def __iter__(self) -> Iterator[BoundingBox]:
        minNonzero = 390000

        def goodSample(image) -> bool:
            nonzeroCount = torch.count_nonzero(image)
            return nonzeroCount.item() > minNonzero

        def qa_good(qamask: np.ndarray):
            numel = qamask.size
            filler_mask = np.bitwise_and(qamask, 0b01)

            if np.count_nonzero(filler_mask) / numel > 0.01:
                return False

            cloud_mask = np.greater_equal(np.right_shift(np.bitwise_and(qamask, 0b11 << 8), 8), 0b10)

            # print("cloud density:", (np.count_nonzero(cloud_mask) / numel))
            if (np.count_nonzero(cloud_mask) / numel) > 0.01:
                return False

            return True


        for _ in range(self.n // self.actualBatchSize()):
            bucketCounts = np.zeros(self.actualBatchSize(), int)
            for _ in (range(self.actualBatchSize())):
                while True:
                    bbox = next(iter(self.sampler))

                    qamask = self.qads[bbox]["image"].squeeze().long().numpy().astype(np.uint16)

                    if (not qa_good(qamask)):
                        continue

                    data = self.ds[bbox]

                    maskDensity = self.density(data["mask"])

                    targetBucket = int(np.interp(maskDensity, [0.0, 1.0], [0, len(self.concentrations) - 1]))
                    maxInBucket = math.floor(self.concentrations[targetBucket] * self.batchSize)

                    if bucketCounts[targetBucket] >= maxInBucket:
                        continue

                    if goodSample(data["image"]):
                        yield bbox
                    else:
                        continue

                    bucketCounts[targetBucket] += 1

                    break
    def actualBatchSize(self) -> int:
        total = 0
        for concentration in self.concentrations:
            total += int(concentration * self.batchSize)

        return int(total)

    def __len__(self) -> int:
        total = self.actualBatchSize()
        return int(total * (self.n // total))
