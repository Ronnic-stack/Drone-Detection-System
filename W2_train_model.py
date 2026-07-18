from ultralytics import YOLO
import time

def train_drone_model():
    print("Loading base YOLOv8 Nano architecture...")
    model = YOLO("yolov8n.pt") 

    print("Igniting Apple Silicon GPU for training...")
    start_time = time.time()

    model.train(
        data="dataset.yaml",
        epochs=30,           
        imgsz=640,          
        batch=16,           
        device="mps",        
        project="backend",   
        name="drone_model_v1"
    )

    elapsed = (time.time() - start_time) / 60
    print(f"\n--- Training Complete ---")
    print(f"Total training time: {elapsed:.2f} minutes")
    print("Your trained model weights are saved at: backend/drone_model_v1/weights/best.pt")

if __name__ == "__main__":
    train_drone_model()