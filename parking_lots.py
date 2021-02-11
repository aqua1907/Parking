import cv2
import torch
import numpy as np
from utils.general import cv2_show, detect
from utils.torch_utils import select_device
from models.experimental import attempt_load
from xml.dom import minidom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


CONFIDENCE = 0.3
IOU_THRESHOLD = 0.6
SIZE = 1280

MODEL_PATH = r"weights/yolov5l.pt"
IMAGE_PATH = r"parking_map.jpg"
XML_PATH = r"PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_09_55_29.xml"

device = select_device("")
half = device.type != "cpu"

yolo5 = attempt_load(MODEL_PATH, map_location=device)
CLASSES = yolo5.module.names if hasattr(yolo5, 'module') else yolo5.names
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

yolo5.to(device)
if half:
    yolo5.half()


image = cv2.imread(IMAGE_PATH)

outs = detect(image, yolo5, device, half, conf=CONFIDENCE, iou_thresh=IOU_THRESHOLD, size=SIZE)

for out in outs:
    if out is not None:
        cv2_show(image, out, CLASSES, COLORS)

# park_lots_cnts = xml_to_dict(XML_PATH)

# plt.imshow(image)
# ax = plt.gca()

# for item in park_lots_cnts:
#     pts = np.array(item["points"], dtype=np.int32)
#     pts = pts.reshape((-1, 1, 2))
#     cv2.polylines(image, [pts], True, (0, 255, 0), 1)

    # cx, cy = item['rectangle']['x'], item['rectangle']['y']
    # w, h = item['rectangle']['w'], item['rectangle']['h']
    # angle = item['rectangle']['angle']
    #
    # plt.gca().add_patch(Rectangle((cx, cy), w, h, linewidth=1, edgecolor='g', facecolor='none', angle=angle))

# pts = np.array(park_lots_cnts[0]["points"], dtype=np.int32)
# print(pts)
# x1, y1 = min(pts[:, 0]), min(pts[:, 1])
# x2, y2 = max(pts[:, 0]), max(pts[:, 1])
# crop = image[y1:y2, x1:x2]
# plt.savefig("figure.png",  dpi=90, bbox_inches='tight')
cv2.imshow("window", cv2.resize(image, (1920, 1080)))
# cv2.imshow("crop", crop)
cv2.waitKey(0)


cv2.destroyAllWindows()
