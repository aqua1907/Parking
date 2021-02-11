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

original_size = (1944, 2592)

images_list = []
root_dir = r"inputs/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750"
root_dir = Path(root_dir)
for root, dirs, files in os.walk(root_dir):
    for name in files:
        images_list.append(os.path.join(root, name))

random.shuffle(images_list)

device = select_device("")

model = MixNet(arch="l", num_classes=1)
checkpoint = torch.load(r"weights/PKLot_CNR_MixNet_L_0.01746.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

tf = T.Compose([
    # T.Resize(120, 120),
    T.ToTensor(),  # normalize to [0, 1]
])

dir_w_csvs = r"inputs/CNR-EXT_FULL_IMAGE_1000x750"
for image_path in images_list:
    split_path = image_path.split("\\")
    camera = split_path[-2] + ".csv"
    path_to_csv = os.path.join(dir_w_csvs, camera)

    df = pd.read_csv(path_to_csv)
    frame = cv2.imread(image_path)
    # frame = cv2.resize(frame, (2592, 1944))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_size = frame.shape[:2]
    x_scale = frame_size[1] / original_size[1]
    y_scale = frame_size[0] / original_size[0]

    plt.imshow(frame)
    # plt.figure(figsize=(10, 10))
    plt.axis('off')

    for i, row in df.iterrows():
        x = int(row["X"] * x_scale)
        y = int(row["Y"] * y_scale)
        w = int(row["W"] * x_scale)
        h = int(row["H"] * y_scale)

        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, cv2.LINE_AA)

        crop = frame[y:y + h, x:x + w]
        crop = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_AREA)

        crop = tf(crop)
        crop = crop.unsqueeze(0).to(device)
        outputs = model(crop)
        outputs = torch.sigmoid(outputs)

        if outputs.item() > 0.5:
            plt.gca().add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
        else:
            plt.gca().add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none'))

    # plt.savefig(r"results/" + str(split_path[-1]), dpi=400, pad_inches=0, bbox_inches='tight',)
    plt.show()
    # cv2.imshow("frame", frame)
    # cv2.imwrite(r"examples/" + str(split_path[-1]), frame)
    # cv2.waitKey(0)
