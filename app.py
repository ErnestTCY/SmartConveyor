import os
import json
import shutil
import threading
import asyncio
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from ultralytics import YOLO
import websockets
import paho.mqtt.client as mqtt
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import glob
import requests
import base64
import logging
# === App Setup ===
app = Flask(__name__, template_folder='templates/htmls', static_folder='static')
load_dotenv()

logging.basicConfig(filename='flask_upload.log', level=logging.INFO)
# === MQTT Settings ===
MQTT_BROKER = 'broker.hivemq.com'
MQTT_PORT = 1883
MQTT_TOPIC = 'robotic_arm/command'
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# === Gemini API Setup ===
genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))

# Add to global section
current_temperature = {"value": 0.0}

# Inside the MQTT setup:
def on_message(client, userdata, msg):
    global current_temperature
    topic = msg.topic
    payload = msg.payload.decode()

    if topic == "sensor/temperature":
        try:
            current_temperature["value"] = round(float(payload), 2)
        except ValueError:
            print("⚠️ Invalid temperature received:", payload)


# === Image Directory Management Functions ===
def get_latest_timestamp_directory(serial_number):
    """Get the latest timestamp directory for a given serial number"""
    base_path = os.path.join('static', 'product_images', serial_number)
    if not os.path.exists(base_path):
        return None
    
    timestamp_dirs = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))]
    
    if not timestamp_dirs:
        return None
    
    # Sort by timestamp (newest first)
    timestamp_dirs.sort(reverse=True)
    return timestamp_dirs[0]

def get_images_from_timestamp_directory(serial_number, timestamp_dir):
    """Get all images from a specific timestamp directory"""
    dir_path = os.path.join('static', 'product_images', serial_number, timestamp_dir)
    if not os.path.exists(dir_path):
        return []
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(dir_path) 
              if f.lower().endswith(image_extensions)]
    
    # Use forward slashes for URL paths
    return [f'product_images/{serial_number}/{timestamp_dir}/{img}' 
            for img in images]

def get_latest_images_for_product(serial_number):
    """Get all images from the latest timestamp directory for a product"""
    latest_timestamp = get_latest_timestamp_directory(serial_number)
    if not latest_timestamp:
        return []
    
    return get_images_from_timestamp_directory(serial_number, latest_timestamp)

def create_timestamp_directory(serial_number, timestamp=None):
    """Create a new timestamp directory for a serial number"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    dir_path = os.path.join('static', 'product_images', serial_number, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    return timestamp

def save_image_to_timestamp_directory(serial_number, image_file, timestamp=None):
    """Save an image to the appropriate timestamp directory"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    dir_path = os.path.join('static', 'product_images', serial_number, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    
    # Save the image
    image_path = os.path.join(dir_path, image_file.filename)
    image_file.save(image_path)
    
    # Return URL path with forward slashes
    return f'product_images/{serial_number}/{timestamp}/{image_file.filename}'

def get_all_timestamp_directories(serial_number):
    """Get all timestamp directories for a serial number, sorted by newest first"""
    base_path = os.path.join('static', 'product_images', serial_number)
    if not os.path.exists(base_path):
        return []
    
    timestamp_dirs = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))]
    
    # Sort by timestamp (newest first)
    timestamp_dirs.sort(reverse=True)
    return timestamp_dirs

def get_product_image_history(serial_number):
    """Get complete image history for a product"""
    timestamp_dirs = get_all_timestamp_directories(serial_number)
    history = {}
    
    for timestamp_dir in timestamp_dirs:
        images = get_images_from_timestamp_directory(serial_number, timestamp_dir)
        if images:
            history[timestamp_dir] = images
    
    return history

# === YOLOv8 Label Storage and Processing ===
def get_labels_directory(serial_number, timestamp_dir):
    """Get the labels directory path for a specific timestamp"""
    return os.path.join('static', 'product_images', serial_number, timestamp_dir, 'labels')

def save_yolo_labels(serial_number, timestamp_dir, image_filename, results):
    """Save YOLOv8 detection results as JSON labels"""
    labels_dir = get_labels_directory(serial_number, timestamp_dir)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Extract detection data
    detections = []
    for result in results:
        if result.boxes:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]

                detections.append({
                    "class": result.names[class_id] if hasattr(result, "names") else f"class_{class_id}",
                    "confidence": round(conf, 2),
                    "position": {"x": int(xyxy[0]), "y": int(xyxy[1])},
                    "size": {"width": int(width), "height": int(height)},
                    "bbox": xyxy
                })
    
    # Save as JSON
    label_filename = os.path.splitext(image_filename)[0] + '_labels.json'
    label_path = os.path.join(labels_dir, label_filename)
    
    with open(label_path, 'w') as f:
        json.dump({
            "image_filename": image_filename,
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "total_detections": len(detections)
        }, f, indent=2)
    
    return label_path

def load_yolo_labels(serial_number, timestamp_dir, image_filename):
    """Load saved YOLOv8 labels for an image"""
    labels_dir = get_labels_directory(serial_number, timestamp_dir)
    label_filename = os.path.splitext(image_filename)[0] + '_labels.json'
    label_path = os.path.join(labels_dir, label_filename)
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            return json.load(f)
    return None

def process_all_images_in_timestamp(serial_number, timestamp_dir):
    """Process all images in a timestamp directory and save YOLOv8 labels"""
    images = get_images_from_timestamp_directory(serial_number, timestamp_dir)
    all_detections = []
    
    for image_path in images:
        # Convert URL path to file system path
        fs_image_path = os.path.join('static', image_path)
        
        if os.path.exists(fs_image_path):
            # Run YOLOv8 detection
            results = model.predict(source=fs_image_path, conf=0.5)
            
            # Extract filename from path
            image_filename = os.path.basename(image_path)
            
            # Save labels
            label_path = save_yolo_labels(serial_number, timestamp_dir, image_filename, results)
            
            # Collect detections for summary
            detections = []
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].tolist()
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        width = xyxy[2] - xyxy[0]
                        height = xyxy[3] - xyxy[1]

                        detections.append({
                            "image": image_filename,
                            "class": result.names[class_id] if hasattr(result, "names") else f"class_{class_id}",
                            "confidence": round(conf, 2),
                            "position": f"x={int(xyxy[0])}, y={int(xyxy[1])}",
                            "size": f"{int(width)}x{int(height)}"
                        })
            
            all_detections.extend(detections)
            print(f"✅ Processed {image_filename}: {len(detections)} detections")
    
    return all_detections

def get_crack_description_from_labels(serial_number, timestamp_dir):
    """Get crack descriptions from saved labels instead of running YOLOv8 again"""
    images = get_images_from_timestamp_directory(serial_number, timestamp_dir)
    all_detections = []
    
    for image_path in images:
        image_filename = os.path.basename(image_path)
        label_data = load_yolo_labels(serial_number, timestamp_dir, image_filename)
        
        if label_data:
            for detection in label_data['detections']:
                all_detections.append({
                    "image": image_filename,
                    "class": detection['class'],
                    "confidence": detection['confidence'],
                    "position": f"x={detection['position']['x']}, y={detection['position']['y']}",
                    "size": f"{detection['size']['width']}x{detection['size']['height']}"
                })
    
    return all_detections

def get_crack_description(image_path):
    results = model.predict(source=image_path, conf=0.5)
    crack_descriptions = []

    for result in results:
        if result.boxes:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]

                crack_descriptions.append({
                    "class": result.names[class_id] if hasattr(result, "names") else f"class_{class_id}",
                    "confidence": round(conf, 2),
                    "position": f"x={int(xyxy[0])}, y={int(xyxy[1])}",
                    "size": f"{int(width)}x{int(height)}"
                })

    return crack_descriptions

def generate_gemini_reasoning(product):
    try:
        serial_number = product['serial_number']
        
        # Get the latest timestamp directory
        latest_timestamp = get_latest_timestamp_directory(serial_number)
        if not latest_timestamp:
            return "No images found for analysis."
        
        # Check if labels exist, if not, process all images in the timestamp
        images = get_images_from_timestamp_directory(serial_number, latest_timestamp)
        if not images:
            return "No images found in the latest timestamp directory."
        
        # Check if labels exist for the first image
        first_image_filename = os.path.basename(images[0])
        existing_labels = load_yolo_labels(serial_number, latest_timestamp, first_image_filename)
        
        if existing_labels is None:
            # Process all images and save labels
            print(f"🔄 Processing all images in {latest_timestamp} for {serial_number}...")
            all_detections = process_all_images_in_timestamp(serial_number, latest_timestamp)
        else:
            # Use existing labels
            print(f"📋 Using existing labels for {serial_number}...")
            all_detections = get_crack_description_from_labels(serial_number, latest_timestamp)
        
        # Format crack information for all images
        if all_detections:
            crack_info_parts = []
            for i, detection in enumerate(all_detections):
                crack_info_parts.append(
                    f"- Image: {detection['image']}, Class: {detection['class']}, "
                    f"Confidence: {detection['confidence']}, Position: ({detection['position']}), "
                    f"Size: {detection['size']}"
                )
            crack_info = "\n".join(crack_info_parts)
        else:
            crack_info = "No cracks detected by YOLOv8 in any of the images."

        prompt = f"""
        You are an expert solar board diagnostics assistant.
        Analyze the cause of damage to the solarboard below:

        Serial Number: {serial_number}
        Model Name: {product['model_name']}
        Status: {product['status']}
        Specs: {product.get('specs_json', {})}
        Timestamp: {product.get('timestamp')}
        Images Analyzed: {len(images)} images from timestamp {latest_timestamp}

        YOLOv8 Detection Summary (All Images):
        {crack_info}

        Based on the above detection and production parameters, identify likely causes for the crack(s) and give actionable recommendations.

        Production parameters to consider:
        - Lamination Pressure (ideal: 50–100 N/cm²; >120 N/cm² risks cracking)
        - Lamination Temperature (140–155°C; >160°C stresses cells)
        - Soldering Temperature (240–260°C; >270°C risks thermal fracture)
        - Cell Stringing Speed (~0.5–1.2 m/s; too fast causes misalignment stress)
        - Handling Force (<5 N; >10 N can crack corners)
        - Vacuum Level before lamination (≤1 mbar recommended)
        - Cooling Rate post-lamination (1–3°C/min; >5°C/min induces thermal mismatch)
        - Cell Thickness (<150 μm increases fragility)

        Include estimated faulty parameter values if possible.
        """

        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        return "Gemini reasoning failed."


# === Product Data Handling ===
DATA_FILE = 'Database/products.json'
def load_products():
    with open(DATA_FILE, 'r') as f:
        return json.load(f)
def save_products(products):
    with open(DATA_FILE, 'w') as f:
        json.dump(products, f, indent=2)

# === Camera Stream (WebSocket) ===
latest_frame = None
async def receive_frames():
    global latest_frame
    uri = "ws://10.10.13.45:8765"  # Replace with actual camera websocket URL
    try:
        async with websockets.connect(uri, max_size=None) as websocket:
            print("[INFO] Connected to camera stream...")
            while True:
                frame_bytes = await websocket.recv()
                np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    latest_frame = buffer.tobytes()
    except Exception as e:
        print(f"[ERROR] Camera connection failed: {e}")
def start_camera_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(receive_frames())
threading.Thread(target=start_camera_loop, daemon=True).start()

# === YOLO Model Loading ===
def load_latest_model():
    model_dir = "trained_models"
    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not pt_files:
        print("⚠ No .pt model found in trained_models/. Using fallback.")
        return YOLO("yolov8n-seg.pt")  # fallback
    latest = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    print(f"✅ Loaded latest model: {latest}")
    return YOLO(os.path.join(model_dir, latest))
model = load_latest_model()

# === Self-Learn/Training Paths ===
NEW_IMG_DIR = 'static/new_images'
PRED_DIR = 'static/predictions'
TRAIN_IMG_DIR = 'train/images'
TRAIN_LABEL_DIR = 'train/labels'

# === Flask Routes ===
@app.route('/')
def home():
    return redirect(url_for('analytics'))

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/livestream')
def livestream():
    return render_template('livestream.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_motor', methods=['POST'])
def update_motor():
    try:
        data = request.get_json()
        motor = int(data.get('motor'))
        angle = int(data.get('angle'))
        action = data.get('action', 'absolute')

        if 1 <= motor <= 5 and 0 <= angle <= 180:
            command = {
                "motor": motor,
                "action": action,
                "angle": angle
            }

            mqtt_client.publish("robotic_arm/command", json.dumps(command))
            print(f"[MQTT] Sent motor update: {command}")
            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"error": "Invalid motor or angle"}), 400
    except Exception as e:
        print(f"[ERROR] update_motor: {e}")
        return jsonify({"error": "Exception occurred"}), 500


@app.route('/send_mqtt', methods=['POST'])
def send_mqtt():
    action = request.form.get('action')
    message = ''
    if action == 'start':
        message = 'automate'
    elif action == 'stop':
        message = 'stop'
    elif action == 'initialize':
        message = 'initialize'
    elif action == 'save_location':
        message = 'save'
    if message:
        mqtt_client.publish(MQTT_TOPIC, json.dumps({"action": message}))
        print(f"[MQTT] Sent: {message}")
    return redirect(url_for('livestream'))

@app.route('/database')
def database():
    products = load_products()
    
    # Update each product with latest images
    for product in products:
        serial_number = product["serial_number"]
        latest_images = get_latest_images_for_product(serial_number)
        product["latest_images"] = latest_images
        
        # Update image_path to use latest image if available
        if latest_images:
            product["image_path"] = latest_images[0]
        elif product.get("image_path"):
            # Fallback to old structure
            filename = os.path.basename(product["image_path"])
            product["image_path"] = f"product_images/{filename}"
    
    return render_template('database.html', products=products)

@app.route('/product/<serial_number>', methods=['GET', 'POST'])
def product_detail(serial_number):
    products = load_products()
    product = next((p for p in products if p["serial_number"] == serial_number), None)
    if not product:
        return "Product not found", 404
    
    # Get latest images for the product
    latest_images = get_latest_images_for_product(serial_number)
    product["images"] = latest_images
    
    # Get image history for the product
    image_history = get_product_image_history(serial_number)
    product["image_history"] = image_history
    
    # Update the main image_path to use the latest image if available
    if latest_images:
        product["image_path"] = latest_images[0]  # Use the first image as main
    elif product.get("image_path"):
        # Fallback to old structure if no new images found
        filename = os.path.basename(product["image_path"])
        product["image_path"] = f"product_images/{filename}"
    
    if request.method == 'POST':
        product["model_name"] = request.form["model_name"]
        product["status"] = request.form["status"]
        product["reasoning"] = request.form.get("reasoning", product.get("reasoning", ""))
        save_products(products)
        return redirect(url_for('product_detail', serial_number=serial_number))
    return render_template('product.html', product=product)

@app.route('/generate_reasoning/<serial_number>', methods=['POST'])
def generate_reasoning(serial_number):
    products = load_products()
    product = next((p for p in products if p["serial_number"] == serial_number), None)
    if not product:
        return {"error": "Product not found"}, 404
    reasoning = generate_gemini_reasoning(product)
    product["reasoning"] = reasoning
    save_products(products)
    return {"reasoning": reasoning}

@app.route('/upload_image/<serial_number>', methods=['POST'])
def upload_image(serial_number):
    """Upload new images for a product to the latest timestamp directory"""
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return {"error": "No image file selected"}, 400
    
    try:
        # Save image to timestamp directory
        image_path = save_image_to_timestamp_directory(serial_number, image_file)
        
        # Update product in database if it exists
        products = load_products()
        product = next((p for p in products if p["serial_number"] == serial_number), None)
        
        if product:
            # Update timestamp to current time
            product["timestamp"] = datetime.now().isoformat()
            save_products(products)
        
        return {"success": True, "image_path": image_path}
    
    except Exception as e:
        return {"error": f"Failed to upload image: {str(e)}"}, 500

@app.route('/get_image_history/<serial_number>')
def get_image_history(serial_number):
    """Get image history for a product"""
    history = get_product_image_history(serial_number)
    return {"history": history}

@app.route('/process_images/<serial_number>', methods=['POST'])
def process_images(serial_number):
    """Manually trigger processing of all images in the latest timestamp directory"""
    try:
        latest_timestamp = get_latest_timestamp_directory(serial_number)
        if not latest_timestamp:
            return {"error": "No timestamp directory found for this product"}, 404
        
        # Process all images and save labels
        all_detections = process_all_images_in_timestamp(serial_number, latest_timestamp)
        
        return {
            "success": True,
            "message": f"Processed {len(all_detections)} detections from {latest_timestamp}",
            "detections": all_detections
        }
    
    except Exception as e:
        return {"error": f"Failed to process images: {str(e)}"}, 500

@app.route('/get_labels/<serial_number>')
def get_labels(serial_number):
    """Get all saved labels for a product's latest timestamp"""
    try:
        latest_timestamp = get_latest_timestamp_directory(serial_number)
        if not latest_timestamp:
            return {"error": "No timestamp directory found for this product"}, 404
        
        images = get_images_from_timestamp_directory(serial_number, latest_timestamp)
        all_labels = {}
        
        for image_path in images:
            image_filename = os.path.basename(image_path)
            label_data = load_yolo_labels(serial_number, latest_timestamp, image_filename)
            if label_data:
                all_labels[image_filename] = label_data
        
        return {
            "success": True,
            "timestamp": latest_timestamp,
            "labels": all_labels
        }
    
    except Exception as e:
        return {"error": f"Failed to get labels: {str(e)}"}, 500

# === Self-Learn/YOLOv8 Routes ===
@app.route('/selflearn')
def selflearn():
    images = [f for f in os.listdir(NEW_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    return render_template('selflearn.html', images=images)

@app.route('/infer/<filename>')
def infer(filename):
    img_path = os.path.join(NEW_IMG_DIR, filename)
    os.makedirs(PRED_DIR, exist_ok=True)
    model.predict(
        source=img_path,
        conf=0.5,
        save=True,
        project='static',
        name='predictions',
        exist_ok=True
    )
    return redirect(url_for('review', filename=filename))

@app.route('/review/<filename>')
def review(filename):
    return render_template('review.html', filename=filename)

@app.route('/accept/<filename>')
def accept(filename):
    src_img = os.path.join(NEW_IMG_DIR, filename)
    label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(TRAIN_LABEL_DIR, label_filename)
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    results = model.predict(source=src_img, conf=0.5)
    for r in results:
        if r.boxes or r.masks:
            r.save_txt(label_path)
        else:
            print(f"⚠ No predictions for {filename}, label not saved.")
    shutil.copy(src_img, os.path.join(TRAIN_IMG_DIR, filename))
    os.remove(src_img)
    return redirect(url_for('selflearn'))

@app.route('/infer_all')
def infer_all():
    images = [f for f in os.listdir(NEW_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    os.makedirs(PRED_DIR, exist_ok=True)
    for img in images:
        model.predict(
            source=os.path.join(NEW_IMG_DIR, img),
            conf=0.5,
            save=True,
            project='static',
            name='predictions',
            exist_ok=True
        )
    return redirect(url_for('review_all'))

@app.route('/review_all')
def review_all():
    detected_images = [
        f for f in os.listdir(PRED_DIR)
        if f.lower().endswith(('.jpg', '.png'))
    ]
    return render_template('review_all.html', images=detected_images)

@app.route('/delete/<filename>', methods=['GET'])
def delete_image(filename):
    img_path = os.path.join(NEW_IMG_DIR, filename)
    if os.path.exists(img_path):
        os.remove(img_path)
    pred_img_path = os.path.join(PRED_DIR, filename)
    if os.path.exists(pred_img_path):
        os.remove(pred_img_path)
    next_url = request.args.get('next')
    return redirect(next_url or url_for('selflearn'))

is_training = False
@app.route('/accept_all', methods=['POST'])
def accept_all():
    images = [
        f for f in os.listdir(PRED_DIR)
        if f.lower().endswith(('.jpg', '.png'))
    ]
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    for filename in images:
        pred_img_path = os.path.join(PRED_DIR, filename)
        new_img_path = os.path.join(NEW_IMG_DIR, filename)
        label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(TRAIN_LABEL_DIR, label_filename)
        if os.path.exists(new_img_path):
            results = model.predict(source=new_img_path, conf=0.5)
            for r in results:
                if r.boxes or r.masks:
                    r.save_txt(label_path)
            shutil.copy(new_img_path, os.path.join(TRAIN_IMG_DIR, filename))
            os.remove(new_img_path)
        if os.path.exists(pred_img_path):
            os.remove(pred_img_path)
    return redirect(url_for('review_all'))

@app.route('/train', methods=['POST'])
def retrain():
    global model, is_training
    is_training = True
    os.system("python train.py")
    model_dir = "trained_models"
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if pt_files:
        latest = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
        model = YOLO(os.path.join(model_dir, latest))
        print(f"✅ Reloaded model: {latest}")
    else:
        print("⚠ No .pt models found after training.")
    is_training = False
    return "done"

@app.route('/train_status')
def train_status():
    return {'training': is_training}

@app.route('/upload_yolo_images', methods=['POST'])
def upload_yolo_images():
    data = request.get_json()
    serial_number = str(data.get('serial_number'))
    timestamp = str(data.get('timestamp'))
    images = data.get('images', {})

    if not serial_number or not images:
        print('[UPLOAD ERROR] Missing serial_number or images in request')
        return jsonify({'error': 'Missing serial_number or images'}), 400

    # Directory for product images
    product_dir = os.path.join('static', 'product_images', serial_number, timestamp)
    os.makedirs(product_dir, exist_ok=True)
    # Directory for new images
    new_images_dir = os.path.join('static', 'new_images')
    os.makedirs(new_images_dir, exist_ok=True)

    saved_files = []
    received_filenames = []
    yolo_thermal_filenames = []
    for key, img_info in images.items():
        filename = img_info['filename']
        img_b64 = img_info['data']
        if not filename or not img_b64:
            continue
        img_bytes = base64.b64decode(img_b64)
        if key in ["yolo", "thermal"]:
            # Save to product_images/<serial>/<timestamp>/
            product_path = os.path.join(product_dir, filename)
            with open(product_path, 'wb') as f:
                f.write(img_bytes)
            saved_files.append(product_path)
            received_filenames.append(filename)
            yolo_thermal_filenames.append(filename)
            # === Run YOLO detection and save JSON labels instantly for YOLO images ===
            if key == "yolo":
                try:
                    # Run YOLO detection
                    results = model.predict(source=product_path, conf=0.5)
                    # Save labels as JSON
                    save_yolo_labels(serial_number, timestamp, filename, results)
                    print(f"[YOLO LABELS] Saved for {filename}")
                except Exception as e:
                    print(f"[YOLO ERROR] Could not process {filename}: {e}")
        elif key in ["light", "no_light"]:
            # Save to static/new_images/
            new_image_path = os.path.join(new_images_dir, filename)
            with open(new_image_path, 'wb') as f:
                f.write(img_bytes)
            saved_files.append(new_image_path)
            received_filenames.append(filename)
        else:
            # Default: save to product_dir
            product_path = os.path.join(product_dir, filename)
            with open(product_path, 'wb') as f:
                f.write(img_bytes)
            saved_files.append(product_path)
            received_filenames.append(filename)

    print(f"[UPLOAD RECEIVED] Serial: {serial_number} | Timestamp: {timestamp} | Files: {received_filenames}")

    # === Update products.json ===
    try:
        products = load_products()
    except Exception:
        products = []

    # Find product by serial number
    product = next((p for p in products if p.get("serial_number") == serial_number), None)
    # Only use a product image for image_path (not light/no_light)
    image_path = f"product_images/{serial_number}/{timestamp}/{yolo_thermal_filenames[0]}" if yolo_thermal_filenames else ""
    if product:
        # Update timestamp, image_path, and set status/model_name/reasoning to 'unknown yet'
        product["timestamp"] = timestamp
        product["image_path"] = image_path
        product["status"] = "unknown yet"
        product["model_name"] = "unknown yet"
        product["reasoning"] = "unknown yet"
    else:
        # Add new product entry
        new_product = {
            "serial_number": serial_number,
            "timestamp": timestamp,
            "status": "unknown yet",
            "model_name": "unknown yet",
            "reasoning": "unknown yet",
            "image_path": image_path
        }
        products.append(new_product)
    save_products(products)
    # === End update products.json ===

    return jsonify({'success': True, 'saved': saved_files})

@app.route('/api/analytics_data')
def analytics_data():
    """API endpoint to provide business analytics for solar board statistics."""
    try:
        products = load_products()
        total_boards = len(products)
        cracked_boards = 0
        healthy_boards = 0
        unknown_boards = 0
        total_images = 0
        images_per_board = []
        status_counts = {}

        for product in products:
            status = product.get('status', '').lower()
            if 'crack' in status:
                cracked_boards += 1
            elif 'healthy' in status or 'ok' in status:
                healthy_boards += 1
            else:
                unknown_boards += 1
            # Count images
            serial_number = product.get('serial_number')
            if serial_number:
                image_history = get_product_image_history(serial_number)
                num_images = sum(len(imgs) for imgs in image_history.values())
                images_per_board.append(num_images)
                total_images += num_images
            # Count status
            status_counts[status] = status_counts.get(status, 0) + 1

        avg_images_per_board = round(total_images / total_boards, 2) if total_boards else 0
        cracked_percent = round((cracked_boards / total_boards) * 100, 2) if total_boards else 0
        healthy_percent = round((healthy_boards / total_boards) * 100, 2) if total_boards else 0

        return jsonify({
            'total_boards': total_boards,
            'cracked_boards': cracked_boards,
            'healthy_boards': healthy_boards,
            'unknown_boards': unknown_boards,
            'avg_images_per_board': avg_images_per_board,
            'total_images': total_images,
            'cracked_percent': cracked_percent,
            'healthy_percent': healthy_percent,
            'status_counts': status_counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 