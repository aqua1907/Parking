import torch
import cv2
import numpy as np
import torchvision.transforms as T
from models.mixnet import MixNet
from utils.torch_utils import select_device
from utils.general import xml_to_dict
import random
from pathlib import Path
import os

# IMAGE_PATH = r"PKLot/PKLot/UFPR04/Rainy/2012-12-21/2012-12-21_18_20_14.jpg"
# XML_PATH = r"PKLot/PKLot/UFPR04/Rainy/2012-12-21/2012-12-21_18_20_14.xml"

images_list = []
root_dir = r"inputs/PKLot/PKLot"
root_dir = Path(root_dir)
for root, dirs, files in os.walk(root_dir):
    for name in files:
        if name.endswith(".jpg"):
            images_list.append(os.path.join(root, name))
        else:
            continue

random.shuffle(images_list)

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

# park_lots_cnts = xml_to_dict(XML_PATH)
# random.shuffle(park_lots_cnts)
# frame = cv2.imread(IMAGE_PATH)

for image_path in images_list:
    frame = cv2.imread(image_path)
    splits = image_path.split("\\")
    filename_im = splits[-1]
    filename = filename_im.split(".")[0]
    splits[-1] = filename + ".xml"
    xml_path = os.path.join(*splits)
    park_lots_cnts = xml_to_dict(xml_path)

    for item in park_lots_cnts:
        pts = np.array(item["points"], dtype=np.int32)
        x1, y1 = min(pts[:, 0]), min(pts[:, 1])
        x2, y2 = max(pts[:, 0]), max(pts[:, 1])
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_AREA)
        # cv2.imshow("crop", crop)
        # cv2.waitKey(0)

        crop = tf(crop)
        crop = crop.unsqueeze(0).to(device)
        outputs = model(crop)
        # outputs = torch.sigmoid(outputs)
        _, predicted = torch.max(outputs, 1)

        pts = pts.reshape((-1, 1, 2))
        if predicted.item() == 1:
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        elif predicted.item() == 0:
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    # cv2.imwrite(r"results/"+str(filename_im), frame)
    cv2.waitKey(0)



