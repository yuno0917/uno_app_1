"""
@software{yolov8_ultralytics,
  author       = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title        = {YOLO by Ultralytics},
  version      = {8.0.0},
  year         = {2023},
  url          = {https://github.com/ultralytics/ultralytics},
  orcid        = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license      = {AGPL-3.0}
}
"""
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=r"C:\Users\kouki\yolov8\datasets\dataset.yaml", epochs=100, imgsz=640, batch=12, device=0)
    
    # Export the model
    model.export(format="torchscript")
