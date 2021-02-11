import torch
import cv2
import numpy as np
import torchvision.transforms as T
from models.mixnet import MixNet
from utils.torch_utils import select_device
import random
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

with open(r"examples/result.json", mode="r") as json_obj:
    data = json.load(json_obj)

json_obj.close()

device = select_device("")

model = MixNet(arch="l", num_classes=2)
checkpoint = torch.load(r"weights/PKLot_CNR_MixNet_L_celoss_0.02104.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

tf = T.Compose([
    # T.Resize(120, 120),
    T.ToTensor(),  # normalize to [0, 1]
])

frame = cv2.imread(r"examples/IMG_2078.JPG")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
orig_size = frame.shape[:2]

x_upscale = orig_size[1] / 100
y_upscale = orig_size[0] / 100

plt.imshow(frame)
# plt.figure(figsize=(10, 10))
plt.axis('off')

for bbox in data[1]["completions"][0]["result"]:
    x, y = bbox["value"]["x"] * x_upscale, bbox["value"]["y"] * y_upscale
    w, h = bbox["value"]["width"] * x_upscale, bbox["value"]["height"] * y_upscale

    x, y = int(x), int(y)
    w, h = int(w), int(h)
    crop = frame[y:y + h, x:x + w]
    crop = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_AREA)

    crop = tf(crop)
    crop = crop.unsqueeze(0).to(device)
    outputs = model(crop)
    # outputs = torch.sigmoid(outputs)
    _, predicted = torch.max(outputs, 1)

    if predicted.item() == 1:
        plt.gca().add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    elif predicted.item() == 0:
        plt.gca().add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none'))

plt.show()
# plt.savefig(r"results/" + str(r"IMG_2078.JPG"), dpi=400, pad_inches=0, bbox_inches='tight',)


