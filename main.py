import cv2
import pandas as pd
from datetime import datetime
from detectors.yolo_detector import YOLODetector

detector = YOLODetector('models/helmet_yolov8.pt')
cap = cv2.VideoCapture('data/test_videos/sample.mp4')

violation_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    for (x1, y1, x2, y2, conf, cls_name) in detections:
        color = (0, 255, 0)
        label = cls_name

        if "without" in cls_name.lower() and "helmet" in cls_name.lower():
            color = (0, 0, 255)  # Red box
            label += " 🚫"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cropped = frame[y1:y2, x1:x2]
            filename = f"logs/violation_{timestamp.replace(' ', '_').replace(':','-')}.jpg"
            cv2.imwrite(filename, cropped)
            violation_log.append({"time": timestamp, "label": cls_name, "image": filename})

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if violation_log:
    df = pd.DataFrame(violation_log)
    df.to_csv("logs/violations.csv", index=False)
