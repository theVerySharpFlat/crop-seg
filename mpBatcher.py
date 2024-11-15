import torch
from torch import Tensor
from torchgeo.datasets import BoundingBox, Landsat
from torchgeo.samplers import RandomGeoSampler
from collections.abc import Iterator
import numpy as np
import math
from tqdm import tqdm
import random
from torchgeo.datasets import CDL, BoundingBox, stack_samples, Landsat7, Landsat8
from torchgeo.samplers import RandomGeoSampler

import multiprocessing

def batchGenerator(args):
    buckets, dsGen, minNonzero, patchSize = args
    # print("start!")

    ds = dsGen
    def density(mask: Tensor) -> float:
        m = torch.where(torch.logical_and(mask <= 60, mask >= 1), 1.0, 0.0) 
        nonzeroCount = m.count_nonzero().item()
        total = torch.numel(m)

        return nonzeroCount / total

    def goodSample(image) -> bool:
        nonzeroCount = torch.count_nonzero(image)
        return nonzeroCount.item() > minNonzero

    sampler = RandomGeoSampler(ds, patchSize)

    while True:
        bbox = next(iter(sampler))
        data = ds[bbox]

        maskDensity = density(data["mask"])

        targetBucket = int(np.interp(maskDensity, [0.0, 1.0], [0, len(buckets) - 0.1]))

        bucket: multiprocessing.Queue = buckets[targetBucket]
        if bucket.full():
            continue

        try:
            if goodSample(data["image"]):
                bucket.put_nowait(bbox)
        except:
            pass

class MPDatasetManager:
    def __init__(self, dsGen, nBuckets, nWorkers = 16, patchSize = 256):
        self.dsGen = dsGen
        self.ds = dsGen()
        self.patchSize = patchSize

        self.buckets: list[multiprocessing.Queue] = []
        for _ in range(nBuckets):
            self.buckets.append(multiprocessing.Queue(maxsize=1024))

        self.workers = []

        for _ in range(nWorkers):
            p = multiprocessing.Process(target=batchGenerator, args=((self.buckets, dsGen(), 390000, patchSize),))
            p.start()
            self.workers.append(p)

        # self.pool = multiprocessing.Pool()
        # self.res = self.pool.map(batchGenerator, [(self.buckets, dsGen, 390000, patchSize) for _ in range(nWorkers)])

class MPDatasetSampler:
    def __init__(self, datasetManager: MPDatasetManager, n, batchSize):
        self.mg = datasetManager
        self.n = n
        self.batchSize = batchSize
        self.concentrations = [1.0 / len(self.mg.buckets) for _ in range(len(self.mg.buckets))]
        
    def actualBatchSize(self) -> int:
        total = 0
        for concentration in self.concentrations:
            total += int(concentration * self.batchSize)

        return int(total)

    def __len__(self) -> int:
        total = self.actualBatchSize()
        return int(total * (self.n // total))

    def __iter__(self) -> Iterator[BoundingBox]:

        for _ in range(self.n // self.actualBatchSize()):
            batch: list[BoundingBox] = []
            # print("begin")
            for i, (c, bucket) in enumerate(zip(self.concentrations, self.mg.buckets)):
                qty = int(c * self.batchSize)
                # print("bucket:", i)
                for j in (range(qty)):
                    # print("element:", j, "/", qty)
                    batch.append(bucket.get())

            # print("yield!")
            random.shuffle(batch)

            for bbox in batch:
                yield bbox

# if __name__ == "__main__":
#     BANDS = [f"SR_B{i}" for i in range(2, 8)]
#
#     def dsGen():
#         cdl = CDL("data", download=True, checksum=True, years=[2023, 2022])
#
#         # landsat7 = Landsat7("data/shelby_landsat_2", bands=Landsat7.all_bands[:5])
#         landsat8 = Landsat8("data/IA2/2023", bands=BANDS)
#         landsat8_test = Landsat8("data/IA2/2022", bands=BANDS)
#         # landsat9_test = Landsat8("data/IA/L2/1023", bands=BANDS)
#         landsat_test = landsat8_test
#         # dataset_test =  GeoDSWrapper(landsat_test & cdl, nPerEpoch=100)
#         dataset_test =  landsat_test & cdl
#         # sampler_test = RandomGeoSampler(dataset_test, size=256, length=1000)
#
#         # landsat9_train = Landsat8("data/IA/L2/1023", bands=BANDS)
#         landsat_train = landsat8
#         dataset_train =  landsat_train & cdl
#         return dataset_train
#
#     mgr = MPDatasetManager(dsGen=dsGen, nBuckets=4, nWorkers=16)
#
#     BATCH_SIZE = 48
#     sampler = MPDatasetSampler(mgr, 1000, BATCH_SIZE)
#     sampler.concentrations = [0.250588312672225, 0.42580514038372963, 0.18370643010468168, 0.13990011683936374]
#
#     for sample in tqdm(sampler):
#         pass
