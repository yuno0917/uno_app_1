#画像一枚ずつ取得して変換を連続して行っている

import cv2
from ultralytics import YOLO
#best.pt=焼肉モデル 
#yolov8n=もとから用意されたモデル
#model = YOLO("best.pt")
model = YOLO("yolov8n.pt")
#model = YOLO("yolov8x.pt")

txt_path = "C:\\Users\\kouki\\yolov8\\runs\\detect\\predict\\labels\\image0.txt"

cap = cv2.VideoCapture(0)

while True:
  # 1フレームずつ取得する。
  ret, frame = cap.read()
  #フレームが取得できなかった場合は、画面を閉じる
  if not ret:
    break
  #save_txt=True,save_conf=Trueで.txtが作られる
  results = model(frame, save=True,save_txt=True,save_conf=True) 

  img = results[0].plot()
  
  
  # ウィンドウに出力
  cv2.imshow("yolo object detect",img)

  with open(txt_path, 'r') as f:
    lines = f.readlines()
  # first_numberが種類でlast_numberが精度
  for line in lines:
      parts = line.strip().split()
      first_number = parts[0]
      last_number = parts[-1]
      print(f"First Number: {first_number}, Last Number: {last_number}")

  
  key = cv2.waitKey(1)
  # Escキーを入力されたら画面を閉じる
  if key == 27:
    break
cap.release()
cv2.destroyAllWindows()


