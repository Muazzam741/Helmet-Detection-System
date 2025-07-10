import cv2
import pandas as pd
import os
from datetime import datetime
from detectors.yolo_detector import YOLODetector

def process_video(video_path, detector, output_dir="logs/frames"):
    import os
    import pandas as pd
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    violations = []
    seen_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        for (x1, y1, x2, y2, conf, cls_name, track_id) in detections:
            if "without" in cls_name.lower() and "helmet" in cls_name.lower():
                if track_id is not None and track_id in seen_ids:
                    continue
                if track_id is not None:
                    seen_ids.add(track_id)

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                crop = frame[y1:y2, x1:x2]
                filename = f"{output_dir}/violation_{timestamp.replace(' ', '_').replace(':','-')}.jpg"
                cv2.imwrite(filename, crop)
                violations.append({
                    "timestamp": timestamp,
                    "track_id": track_id,
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "image": filename
                })
    cap.release()
    if violations:
        df = pd.DataFrame(violations)
        df.to_csv("logs/violations.csv", index=False)

    return violations
