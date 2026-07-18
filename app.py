import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="Drone Detection System", layout="wide")
st.title("Drone Detection Dashboard")

@st.cache_resource
def load_model():
    return YOLO("runs/detect/backend/drone_model_v1/weights/best.pt")

model = load_model()

st.sidebar.header("Inference Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6)

input_source = st.sidebar.radio("Select Input Source", ["Upload File", "Live Camera"])

if input_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["mp4", "mov", "jpg", "jpeg", "png"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type in ["mp4", "mov"]:
            st.info("Click 'Process Video' to analyze. It will play smoothly once finished.")
            
            if st.button("Process Video"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = model.track(frame, conf=conf_threshold, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
                        annotated_frame = results[0].plot(labels=True, conf=True)
                        
                        out.write(annotated_frame)

                        frame_count += 1
                        progress_percentage = min(frame_count / total_frames, 1.0)
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Processing frame {frame_count} of {total_frames}...")
                        
                finally:
                    # Safely close everything
                    cap.release()
                    out.release()
                    os.remove(tfile.name)
                
                status_text.text("Processing complete! Playing video...")
                
                with open(output_path, 'rb') as video_file:
                    st.video(video_file.read())
                
                os.remove(output_path)

        elif file_type in ["jpg", "jpeg", "png"]:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                frame = np.array(image)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                results = model.predict(frame_bgr, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot(labels=True, conf=True)
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Detection Result", use_container_width=False)
            except Exception as e:
                st.error(f"Could not process image: {e}")

    else:
        st.info("Please upload an image or video file via the sidebar to begin.")

elif input_source == "Live Camera":
    st.info("Select your camera index and click 'Start' to open your webcam. Click 'Stop' to end the feed.")
    
    camera_index = st.sidebar.number_input("Camera Device Index", min_value=0, max_value=5, value=0, step=1)
    
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    col1, col2 = st.columns(2)
    if col1.button("Start Camera"):
        st.session_state.camera_active = True
        st.rerun() 
        
    if col2.button("Stop Camera"):
        st.session_state.camera_active = False
        st.rerun()
    
    stframe = st.empty()

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(camera_index)
        
        try:
            while cap.isOpened() and st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error(f"Failed to access the camera at index {camera_index}.")
                    break
                
                results = model.track(frame, conf=conf_threshold, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
                annotated_frame = results[0].plot(labels=True, conf=True)
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=False)
                
        finally:
            cap.release()