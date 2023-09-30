#画像一枚ずつ取得して変換を連続して行っている

import cv2
from ultralytics import YOLO
import pyautogui 

model = YOLO("best.pt")
#model = YOLO("yolov8n.pt")
#model = YOLO("yolov8x.pt")

#cap = cv2.VideoCapture(0)
while True:
  screen_shot = pyautogui.screenshot() 
  # 1フレームずつ取得する。
  #ret, frame = screen_shot.read()
  #フレームが取得できなかった場合は、画面を閉じる
  #if not ret:
    #break
    
  results = model(screen_shot, save=True) 
  
  img = results[0].plot()
  resized_img = cv2.resize(img, None, fx=0.5, fy=0.5)
  
  
  # ウィンドウに出力
  cv2.imshow("yolo object detect",resized_img)
  
  key = cv2.waitKey(1)
  # Escキーを入力されたら画面を閉じる
  if key == 27:
    break
cap.release()
cv2.destroyAllWindows()


