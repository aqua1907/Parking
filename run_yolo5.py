import cv2
from models.experimental import attempt_load
from utils.general import detect, cv2_show
from utils.torch_utils import select_device
import numpy as np
from timeit import default_timer as timer

device = select_device("")
half = device.type != "cpu"
rescale_size = 1280
conf = 0.4

detection_weights = r"weights/yolov5s.pt"
video_path = r"examples/la_highway_driving1.mp4"

yolo5 = attempt_load(detection_weights, map_location=device)
CLASSES = yolo5.module.names if hasattr(yolo5, 'module') else yolo5.names
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

yolo5.to(device)
if half:
    yolo5.half()

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(f"results/la_highway_demo0_{rescale_size}p_conf{conf}.mp4", fourcc, 25.0, size)


while cap.isOpened():
    start = timer()
    ret, frame = cap.read()

    outs = detect(frame, yolo5, device, half, conf=conf, iou_thresh=0.6, size=rescale_size)

    for out in outs:
        if out is not None:
            cv2_show(frame, out, CLASSES, COLORS)

    fps = 1 / (timer() - start)
    fps = "FPS: {:.1f}".format(fps)
    cv2.putText(frame, fps, (1000, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow("window", frame)
    # writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

