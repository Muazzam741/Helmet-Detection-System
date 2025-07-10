import streamlit as st
import pandas as pd
from detectors.yolo_detector import YOLODetector
from utils.ui_utils import process_video
import os

st.set_page_config(page_title="🚨 Road Safety Assistant", layout="wide")
st.title("🛵 Helmet Violation Detector")

# Upload video
video_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if video_file:
    with open("data/test_videos/input.mp4", "wb") as f:
        f.write(video_file.read())

    st.success("Video uploaded successfully.")
    detector = YOLODetector("models/helmet_yolov8.pt")

    with st.spinner("🔍 Processing video..."):
        results = process_video("data/test_videos/input.mp4", detector)

    if results:
        st.subheader("Violations Detected")
        df = pd.DataFrame(results)
        st.dataframe(df[["timestamp", "class", "confidence"]])

        st.subheader("📸 Violation Snapshots")
        cols = st.columns(3)
        for i, row in df.iterrows():
            with cols[i % 3]:
                st.image(row["image"], caption=f"{row['timestamp']} ({row['confidence']})", use_column_width=True)
    else:
        st.success("No helmet violations detected!")

else:
    st.info("Upload a video to begin detection.")
