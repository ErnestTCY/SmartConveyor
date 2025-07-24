import os
import datetime
from ultralytics import YOLO
import pandas as pd
import json

def train():
    # Set up paths
    data_yaml = "C:/Users/User/Desktop/hanging/Competition/T4G 022025/dep5/data.yaml"
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
        epochs=50,
        imgsz=640,
        batch=4,
        device='cuda',
        workers=0,
        patience=5,
        save=True,
        val=True,  
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
    
    # 4) Compute & save best‐accuracy metrics
    results_csv = os.path.join(trained_result_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"⚠ results.csv not found at {results_csv}")
        return

    # Load the training history
    df = pd.read_csv(results_csv)
    # Detect which precision/recall columns we have
    if 'metrics/precision' in df.columns and 'metrics/recall' in df.columns:
        prec_col, rec_col = 'metrics/precision', 'metrics/recall'
    elif 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        prec_col, rec_col = 'metrics/precision(B)', 'metrics/recall(B)'
    else:
        print("⚠ Couldn't find precision/recall columns in results.csv")
        print("Available columns:", df.columns.tolist())
        return

    # Compute accuracy and pick best epoch
    df['accuracy'] = (df[prec_col] + df[rec_col]) / 2
    best_idx       = df['accuracy'].idxmax()
    best_row       = df.loc[best_idx, ['epoch', prec_col, rec_col, 'accuracy']]

    # Normalize keys for JSON
    best_metrics = {
        'epoch':    int(best_row['epoch']),
        'precision': float(best_row[prec_col]),
        'recall':    float(best_row[rec_col]),
        'accuracy':  float(best_row['accuracy'])
    }

    # Write best_metrics.json beside results.csv
    json_path = os.path.join(trained_result_dir, "best_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(best_metrics, f, indent=4)

    print(f"✅ Saved best metrics to {json_path}")
    print(best_metrics)
if __name__ == "__main__":
    train()
