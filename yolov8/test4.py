import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
#model = YOLO("yolov8x.pt")

cap = cv2.VideoCapture(0)

while True:
  # 1フレームずつ取得する。
  ret, frame = cap.read()
  #フレームが取得できなかった場合は、画面を閉じる
  if not ret:
    break
    
  results = model(frame, save=True) 
  img_path = "C:\\Users\\kouki\\yolov8\\runs\\detect\\predict\\image0.jpg"

  img = cv2.imread(img_path)
  
  
  # ウィンドウに出力
  cv2.imshow("yolo object detect",img)
  
  key = cv2.waitKey(1)
  # Escキーを入力されたら画面を閉じる
  if key == 27:
    break
cap.release()
cv2.destroyAllWindows()


"""
import os
import shutil

folder_path = "C:\\Users\\kouki\\yolov8\\runs\\detect\\predict"

# フォルダが存在するかを確認
if os.path.exists(folder_path):
    # フォルダを削除
    shutil.rmtree(folder_path)
    print(f"'{folder_path}' が削除されました。")
else:
    print(f"'{folder_path}' は存在しません。")
"""