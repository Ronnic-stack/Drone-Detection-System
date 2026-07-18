import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Drone Detection System", layout="wide")
st.title("Drone Detection Dashboard")

# --- Model Loading ---
@st.cache_resource
def load_model():
    return YOLO("runs/detect/backend/drone_model_v1/weights/best.pt")

model = load_model()

# --- Sidebar Controls ---
st.sidebar.header("Inference Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6)

# Added 'jpg', 'jpeg', 'png' to allowed types
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["mp4", "mov", "jpg", "jpeg", "png"])

# --- Main Logic ---
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()

    # --- Handle Video ---
    if file_type in ["mp4", "mov"]:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, conf=conf_threshold, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
            annotated_frame = results[0].plot(labels=True, conf=True)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=False)
        cap.release()
        os.remove(tfile.name)

    # --- Handle Images ---
    elif file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        # Convert RGB to BGR for OpenCV/YOLO
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        results = model.predict(frame_bgr, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot(labels=True, conf=True)
        
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Detection Result", use_container_width=True)

else:
    st.info("Please upload an image or video file via the sidebar to begin.")