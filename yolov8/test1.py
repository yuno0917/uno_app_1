from ultralytics import YOLO
model = YOLO("yolov8x.pt")

results = model("C:\\Users\\kouki\\yolov8\\test2.jpg", save=True) 