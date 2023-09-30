import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

cap = cv2.VideoCapture(0)

while True:
  # 1フレームずつ取得する。
  ret, frame = cap.read()
  #フレームが取得できなかった場合は、画面を閉じる
  if not ret:
    break
    
  results = model(frame, save=True) 

  img = results[1].orig_img
  
  
  # ウィンドウに出力
  cv2.imshow("yolo object detect",img)
  #os.remove("C:\\Users\\kouki\\yolov8\\runs\\predict\\image0.jpg")
  
  key = cv2.waitKey(1)
  # Escキーを入力されたら画面を閉じる
  if key == 27:
    break
cap.release()
cv2.destroyAllWindows()