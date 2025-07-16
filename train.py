import os
import datetime
from ultralytics import YOLO

def train():
    # Set up paths
    data_yaml = "C:/Users/User/Desktop/hanging/Competition/T4G 022025/Deploy2/data.yaml"
    model_name = "yolov8n-seg.pt"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_model_name = f"{timestamp}.pt"
    output_model_path = os.path.join("trained_models", output_model_name)

    # Ensure output directory exists
    os.makedirs("trained_models", exist_ok=True)

    # Load model
    model = YOLO(model_name)

    print(f"🧠 Starting training, model will be saved as: {output_model_path}")

    # Train
    model.train(
        data=data_yaml,
        epochs=10,
        imgsz=640,
        batch=4,
        device='cuda',
        workers=0,
        patience=5,
        save=True,
        val=False,  # ❌ No evaluation
        project="trained_models",
        name=timestamp
    )

    # Copy best.pt to central place
    trained_result_dir = os.path.join("trained_models", timestamp)
    best_model_in_project = os.path.join(trained_result_dir, "weights", "best.pt")
    if os.path.exists(best_model_in_project):
        os.rename(best_model_in_project, output_model_path)
        print(f"✅ Model saved to {output_model_path}")
    else:
        print("⚠ Training completed but best.pt was not found.")

if __name__ == "__main__":
    train()
