import cv2
import sys
from ultralytics import YOLO

def detect_drones(video_path="test_video.mp4", model_path="runs/detect/backend/drone_model_v1/weights/best.pt"):
    print(f"[INFO] Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    print(f"[INFO] Initializing video stream for {video_path}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}. Check if the file exists and the path is correct.")
        sys.exit(1)

    print("[INFO] Inference active. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[INFO] Video stream ended.")
            break

        results = model.track(frame, conf=0.6, iou=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)

        annotated_frame = results[0].plot(labels=True, conf=True)

        cv2.imshow("Drone Tracker", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User terminated the inference stream.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_drones()