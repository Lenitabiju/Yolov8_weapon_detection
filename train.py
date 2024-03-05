from ultralytics import YOLO

model = YOLO("./model/yolov8n.pt")


model.train(data="data.yaml", epochs=50) 