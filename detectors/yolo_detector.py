from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path='models/helmet_yolov8.pt'):
        self.model = YOLO(model_path)
        self.class_map = self.model.names

    def detect(self, frame, track=True):
        results = self.model.track(frame, persist=True, conf=0.3, verbose=False) if track else self.model.predict(frame, conf=0.3)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_map[cls_id]
                track_id = int(box.id[0]) if box.id is not None else None
                detections.append((x1, y1, x2, y2, conf, cls_name, track_id))
        return detections
