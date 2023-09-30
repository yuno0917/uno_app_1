from ultralytics import YOLO
import torch
import cv2
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils import ops
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

#画像をYOLOの入力として適切な形に変換するための前処理を行う関数
def preprocess(img, size=640):
        img = LetterBox(size, True)(image=img)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img.unsqueeze(0)
#YOLOの出力から、非最大抑制を用いて最終的な検出結果を抽出する関数
def postprocess(preds, img, orig_img):
    preds = ops.non_max_suppression(preds,
                                    0.25,
                                    0.8,
                                    agnostic=False,
                                    max_det=100)

    for i, pred in enumerate(preds):
        shape = orig_img.shape
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

    return preds

def drow_bbox(pred, names, annotator):
    for *xyxy, conf, cls in reversed(pred):
        c = int(cls)  # integer class
        label =  f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))


while True:
    ret, img = cap.read()
    origin = deepcopy(img)
    annotator = Annotator(origin,line_width=1,example=str(model.model.names))
    img = preprocess(img)
    preds = model.model(img, augment=False)
    preds = postprocess(preds,img,origin)
    drow_bbox(preds[0], model.model.names, annotator)
    cv2.imshow("test",origin)
    cv2.waitKey(1)
    
