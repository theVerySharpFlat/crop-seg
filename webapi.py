import io
import typing
import PIL
from fastapi.responses import JSONResponse, Response
from numpy.lib import math
import uvicorn
import pyproj
import numpy as np

from torch import Tensor, inf, minimum

import torch

from fastapi import FastAPI
from pydantic import BaseModel
from unet import unet as UNET
import torch.nn.functional as F

from dateutil.rrule import rrule, MONTHLY
from datetime import datetime

import argparse
from torchgeo.datasets import BoundingBox, Landsat8

from matplotlib import pyplot as plt

import re
import os

from PIL import Image

# parser = argparse.ArgumentParser()
# parser.add_argument("weights", help="path to weights file")
# parser.add_argument("port", help="port to host server on", type=int)
# args = parser.parse_args()
WEIGHTS_FILE="./checkpoints/checkpoint_epoch53.pth"

app = FastAPI()


class BboxArgs(BaseModel):
    east: float
    west: float
    north: float
    south: float


@app.get("/")
def read_root():
    return {"Hello": "World"}
    # return JSONResponse(content={"Hello": "World"})


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BANDS = [f"SR_B{i}" for i in range(2, 8)]
# landsat8_23 = Landsat8("data/IA2/2023", bands=BANDS)
landsat8 = Landsat8("data/IA2/2023", bands=BANDS) | Landsat8("data/IA2/2022", bands=BANDS)
qa_ds = Landsat8("data/IA2/2023", bands=["QA_PIXEL"]) | Landsat8("data/IA2/2022", bands=["QA_PIXEL"])

# print(f"resolution: {landsat8_23.res}")
# print(f"index: {landsat8.index}")

# for entry in landsat8_23.index.intersection(tuple(landsat8_23.bounds), objects=True):
#     print(f"entry: {entry.object}")
#     pass

transformer = pyproj.Transformer.from_crs("EPSG:4326", str(landsat8.crs), allow_ballpark=False, only_best=True, accuracy=1.0)
print("CRS: " + str(landsat8.crs))
revTransformer = pyproj.Transformer.from_crs(str(landsat8.crs), "EPSG:4326", allow_ballpark=False, only_best=True, accuracy=1.0)

unet = UNET.UNet(len(BANDS), n_classes=1).to(DEVICE)
unet.load_state_dict(torch.load(WEIGHTS_FILE, {"cuda:0": DEVICE}))

bandPercentiles = [([ 7767., 11340.]), ([ 8312., 12746.]), ([ 7861., 14691.]), ([ 8873., 28555.]), ([ 9059., 23145.]), ([ 8147., 20632.])]
def normalizeBandImage(tensor: Tensor, percentiles) -> Tensor:
    c, d = percentiles

    tensor = torch.mul(torch.sub(tensor, c), 1 / (d - c))

    return tensor

def normalizeBatch(batchImage: Tensor, bandPercentiles) -> Tensor:
    for bandNum, percentiles in enumerate(bandPercentiles):
        for batchNum in range(batchImage.shape[0]):
            batchImage[batchNum][bandNum] = normalizeBandImage(batchImage[batchNum][bandNum], percentiles)
    return batchImage

def months(start_month, start_year, end_month, end_year):
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    return [int(d.strftime('%s')) for d in rrule(MONTHLY, dtstart=start, until=end)]

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

def goodSample(image) -> bool:
    nonzeroCount = torch.count_nonzero(image)
    return nonzeroCount.item() > 390000

time_pairs = []
m = months(1, 2022, 12, 2023)
for i, month in enumerate(m):
    if i >= len(m) - 1:
        break
    time_pairs.append((m[i], m[i+1]))

# counter = 0
def run_pred(minx, maxx, miny, maxy):
    global counter
    preds = []
    for tp in time_pairs:
        try:
            bbox = BoundingBox(minx, maxx, miny, maxy, tp[0], tp[1])
        except Exception as e:
            print(f"bounding box error: {e}")
            return None, {"error": "bounding box invalid"}

        try:
            qa = qa_ds[bbox]
            qamask = qa["image"].squeeze().long().numpy().astype(np.uint16)
            if (not qa_good(qamask)):
                continue

            img = landsat8[bbox]["image"].to(DEVICE)
            if (not goodSample(img)):
                continue
            img = normalizeBatch(img.unsqueeze(0), bandPercentiles).squeeze()
        except Exception as e:
            continue

        with torch.no_grad():
            img = img.unsqueeze(0)
            # print(img.shape)
            pred = unet(img)
            preds.append(pred)
            # pred = ((F.sigmoid(pred.cpu())) > 0.5).float()

            
            # figPred, ax = plt.subplots(1, 1, figsize=(4,4))
            # ax.imshow(pred.squeeze())
            # ax.axis('off')

            # fig = landsat8_23.plot(landsat8[bbox])
            # fig.savefig(f"{counter}.png")
            # figPred.savefig("serverpred.png")

            # fig = landsat8_23.plot(landsat8[bbox])
            # fig.savefig("actual.png")
            # figPred.savefig("serverpred.png")


        # print("here3")
        # return {"message": "success???"}
    # return {"error": "no index in dataset!"}

    if len(preds) == 0:
        return None, {"error": "no index in dataset!"}

    pred = preds[0]
    for i in range(1, len(preds)):
        pred += preds[i]

    pred /= float(len(preds))

    pred = ((F.sigmoid(pred.cpu())) > 0.5).float()

    # 
    # figPred, ax = plt.subplots(1, 1, figsize=(4,4))
    # ax.imshow(pred.squeeze())
    # ax.axis('off')
    # figPred.savefig(f"{counter}.png")
    # counter +=1

    return pred, {"message": "success???"}

@app.get("/segment")
def segment(south: float, west: float, north: float, east: float):
    lower = transformer.transform(south, west)
    upper = transformer.transform(north, east)

    if inf in lower or inf in upper:
        return {"error": "CRS transform error!"}

    # print("lower: ", lower)
    # bounds = transformer.transform_bounds(item.south, item.west, item.north, item.east)
    bounds = (*lower, *upper)
    revBounds = revTransformer.transform_bounds(*bounds)
    # print("latlong bounds: ", item)
    # print("bounds:", bounds)
    # print("revbounds:", revBounds)
    if not bounds[0] < bounds[2]:
        return {"error": "south must be less than north!"}

    if not bounds[1] < bounds[3]:
        return {"error": "west must be less than east"}

    # for entry in landsat8_23.index.intersection((bounds[0], bounds[2], bounds[1], bounds[3], landsat8.bounds.mint, landsat8.bounds.maxt), objects=True):
    #     print(f"entry: {entry.object}")
    #     pass

    nXIters = int(math.ceil((bounds[2] - bounds[0]) / (landsat8.res * 256)))
    nYIters = int(math.ceil((bounds[3] - bounds[1]) / (landsat8.res * 256)))

    if nXIters * nYIters > 16:
        return {"error": "That's a lot of tiles to compute! GPUs don't grow on trees, you know"}
    # nXIters = 2
    # nYIters = 1

    # print(f"iters: {nXIters}, {nYIters}")

    netImg = np.zeros((nYIters * 256, nXIters * 256))

    for i in range(nXIters):
        for j in range(nYIters):
            result, ret = run_pred(bounds[0] + i * landsat8.res * 256, bounds[0] + (i + 1) * landsat8.res * 256, bounds[1] + j * landsat8.res * 256, bounds[1] + (j + 1) * landsat8.res * 256)

            if result == None:
                return ret

            # print(f"access: {(nYIters - 1 - j) * 256}:{(nYIters - 1 - (j + 1)) * 256}")
            netImg[(nYIters - j - 1) * 256:(nYIters - (j)) * 256, i * 256:(i+1) * 256] = result.cpu().numpy()


    img = Image.fromarray(np.uint8(netImg[0:int((bounds[3] - bounds[1]) / (landsat8.res)), 0:int((bounds[2] - bounds[0]) / landsat8.res)] * 255), mode="L")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return Response(content=img_bytes.getvalue(), media_type="image/png")
    # return run_pred(bounds[0], 0, bounds[1], 0)
    # bbox = BoundingBox(bounds[0], bounds[2], bounds[1], bounds[3], landsat8.bounds.mint, landsat8.bounds.maxt)
    # for tp in time_pairs:
    #     bbox = BoundingBox(bounds[0], bounds[0] + 30 * 256, bounds[1], bounds[1] + 30 * 256, tp[0], tp[1])
    #
    #     print("here: ", tp)
    #     try:
    #         qa = qa_test[bbox]
    #         qamask = qa["image"].squeeze().long().numpy().astype(np.uint16)
    #         if (not qa_good(qamask)):
    #             print("qa bad! " + tp)
    #             continue
    #
    #         img = landsat8[bbox]["image"].to(DEVICE)
    #         img = normalizeBatch(img.unsqueeze(0), bandPercentiles).squeeze()
    #     except Exception as e:
    #         print(f"exception: {e}")
    #         continue
    #
    #     with torch.no_grad():
    #         img = img.unsqueeze(0)
    #         print(img.shape)
    #         pred = unet(img)
    #         pred = ((F.sigmoid(pred.cpu())) > 0.5).float()
    #         
    #         figPred, ax = plt.subplots(1, 1, figsize=(4,4))
    #         ax.imshow(pred.squeeze())
    #         ax.axis('off')
    #
    #         fig = landsat8_23.plot(landsat8[bbox])
    #         fig.savefig("actual.png")
    #         figPred.savefig("serverpred.png")
    #
    #
    #     print("here3")
    #     return {"message": "success???"}
    # return {"error": "no index in dataset!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="trace")
