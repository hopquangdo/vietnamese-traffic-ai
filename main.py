from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # hoặc yolov8m.pt, yolov8l.pt nếu cần chính xác hơn

results = model.predict(source="video.mp4", classes=[2, 3, 5, 7], show=True)
# class 2: car, 3: motorbike, 5: bus, 7: truck
