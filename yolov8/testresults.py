import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

cap = cv2.VideoCapture(0)


# 1フレームずつ取得する。
ret, frame = cap.read()
#フレームが取得できなかった場合は、画面を閉じる
results = model(frame, save=True) 

print(results)

  
#rendered_frame = results.render()[0]
  
  
# Escキーを入力されたら画面を閉じる
cap.release()
cv2.destroyAllWindows()