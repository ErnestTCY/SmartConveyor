import os
import json
import shutil
import threading
import asyncio
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
from ultralytics import YOLO
import websockets
import paho.mqtt.client as mqtt
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from datetime import  timedelta
import glob
import requests
import base64
import logging
import time
import socket
from collections import Counter
from pathlib import Path  
from collections import defaultdict
from pdf_generator import generate_report
from flask import flash


# === App Setup ===
app = Flask(__name__, template_folder='templates/htmls', static_folder='static')
load_dotenv()

logging.basicConfig(filename='flask_upload.log', level=logging.INFO)
# === MQTT Settings ===
MQTT_BROKER = 'broker.hivemq.com'
MQTT_PORT = 1883
MQTT_TOPIC = 'robotic_arm/command'
MQTT_HEARTBEAT_TOPIC = 'robotic_arm/heartbeat'
mqtt_client = mqtt.Client()

# MQTT connection state
mqtt_connected = threading.Event()

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info('[MQTT] Connected successfully')
        mqtt_connected.set()
    else:
        logging.warning(f'[MQTT] Connection failed with code {rc}')
        mqtt_connected.clear()

def on_disconnect(client, userdata, rc):
    logging.warning(f'[MQTT] Disconnected (rc={rc})')
    mqtt_connected.clear()
    # Try to reconnect in a loop
    while not mqtt_connected.is_set():
        try:
            logging.info('[MQTT] Attempting to reconnect...')
            client.reconnect()
            break
        except Exception as e:
            logging.error(f'[MQTT] Reconnect failed: {e}')
            time.sleep(5)

def on_message(client, userdata, msg):
    global current_temperature
    topic = msg.topic
    payload = msg.payload.decode()

    if topic == "sensor/temperature":
        try:
            current_temperature["value"] = round(float(payload), 2)
        except ValueError:
            print("⚠️ Invalid temperature received:", payload)

mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message

# Start MQTT loop in background
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
def start_mqtt_loop():
    mqtt_client.loop_forever()
threading.Thread(target=start_mqtt_loop, daemon=True).start()

# Heartbeat function
def mqtt_heartbeat():
    while True:
        if mqtt_connected.is_set():
            try:
                mqtt_client.publish(MQTT_HEARTBEAT_TOPIC, json.dumps({'status': 'alive', 'timestamp': datetime.now().isoformat()}))
                logging.info('[MQTT] Heartbeat sent')
            except Exception as e:
                logging.error(f'[MQTT] Heartbeat publish failed: {e}')
        # Only sleep and check again, do not send heartbeat if not connected
        time.sleep(15)
threading.Thread(target=mqtt_heartbeat, daemon=True).start()

# === Gemini API Setup ===
genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))

# Add to global section
current_temperature = {"value": 0.0}

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

def get_thermal_images_from_timestamp_directory(serial_number, timestamp_dir):
    dir_path = os.path.join('static', 'product_images', serial_number, timestamp_dir, 'thermal')
    if not os.path.exists(dir_path):
        return []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(dir_path) if f.lower().endswith(image_extensions)]
    return [f'product_images/{serial_number}/{timestamp_dir}/thermal/{img}' for img in images]

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
    
    # Update status.json after saving label
    update_status_json(serial_number, timestamp_dir)
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
            results = model.predict(source=fs_image_path, conf=0.5, verbose=False)
            
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
    results = model.predict(source=image_path, conf=0.5, verbose=False)
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

def save_reasoning(serial_number, timestamp_dir, reasoning_text):
    reasoning_path = os.path.join('static', 'product_images', serial_number, timestamp_dir, 'reasoning.json')
    with open(reasoning_path, 'w') as f:
        json.dump({
            'serial_number': serial_number,
            'timestamp': timestamp_dir,
            'reasoning': reasoning_text
        }, f, indent=2)
    return reasoning_path

def load_reasoning(serial_number, timestamp_dir):
    reasoning_path = os.path.join('static', 'product_images', serial_number, timestamp_dir, 'reasoning.json')
    if os.path.exists(reasoning_path):
        with open(reasoning_path, 'r') as f:
            return json.load(f).get('reasoning', None)
    return None


def generate_gemini_reasoning(product, timestamp=None):
    try:
        serial_number = product['serial_number']
        if not timestamp:
            timestamp = get_latest_timestamp_directory(serial_number)
        if not timestamp:
            return "No images found for analysis."

        # — Gather image paths & crack detections as before —
        images = get_images_from_timestamp_directory(serial_number, timestamp)
        if not images:
            return "No images found in the selected timestamp directory."

        first_image = os.path.basename(images[0])
        existing_labels = load_yolo_labels(serial_number, timestamp, first_image)
        if existing_labels is None:
            all_detections = process_all_images_in_timestamp(serial_number, timestamp)
        else:
            all_detections = get_crack_description_from_labels(serial_number, timestamp)

        # Format crack info
        if all_detections:
            crack_info = "\n".join(
                f"- Image: {d['image']}, Class: {d['class']}, "
                f"Conf: {d['confidence']}, Pos: ({d['position']}), Size: {d['size']}"
                for d in all_detections
            )
        else:
            crack_info = "No cracks detected by YOLOv8 in any of the images."

        # — NEW: load all thermal JSONs under /thermal/area*.json —
        thermal_dir = os.path.join(
            'static', 'product_images', serial_number, timestamp, 'thermal'
        )
        thermal_data = {}
        if os.path.isdir(thermal_dir):
            for jf in sorted(glob.glob(os.path.join(thermal_dir, "area*.json"))):
                try:
                    payload = json.loads(Path(jf).read_text())
                    area_name = Path(jf).stem       # e.g. "area1"
                    thermal_data[area_name] = payload.get("raw_grid", [])
                except Exception:
                    # skip bad files
                    continue

        # Format thermal info into text for the prompt
        if thermal_data:
            thermal_info = "\n".join(
                f"- {area}: {grid}"
                for area, grid in thermal_data.items()
            )
        else:
            thermal_info = "No thermal JSON data available for this timestamp."

        # — Status lookup as before —
        status = (
            product.get('status_by_timestamp', {})
                   .get(timestamp)
            or load_status(serial_number, timestamp)
            or 'unknown'
        )

        # — Build the prompt, embedding the crack_info AND thermal_info —
        prompt = f"""
Analysis of Solarboard Damage
Serial Number: {serial_number}
Timestamp: {timestamp}

You are an expert solar board diagnostics assistant.

Model Name: {product.get('model_name','<unknown>')}
Status: {status}
Images Analyzed: {len(images)}

YOLOv8 Detection Summary:
{crack_info}

Thermal Sensor Data (raw grids):
{thermal_info}

Based on the above detection and thermal readings, identify likely causes for any cracks or hotspots and give actionable recommendations.

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
        reasoning_text = response.text.strip()

        # save out so you can re‑load later
        save_reasoning(serial_number, timestamp, reasoning_text)
        return reasoning_text

    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        return "Gemini reasoning failed."


# === Product Data Handling ===
DATA_FILE = 'Database/products.json'
def load_products():
    with open(DATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Malformed JSON in {DATA_FILE}: {e}")

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

def update_status_json(serial_number, timestamp_dir):
    labels_dir = get_labels_directory(serial_number, timestamp_dir)
    status = 'normal'
    for label_file in glob.glob(os.path.join(labels_dir, '*_labels.json')):
        with open(label_file, 'r') as f:
            data = json.load(f)
        for det in data.get('detections', []):
            cls = str(det.get('class', '')).lower()
            if 'crack' in cls:
                status = 'cracked'
                break
            elif 'scratch' in cls:
                status = 'scratch'
                # keep scanning in case there is also a crack
        if status == 'cracked':
            break

    status_path = os.path.join('static', 'product_images', serial_number, timestamp_dir, 'status.json')
    with open(status_path, 'w') as f:
        json.dump({'status': status}, f)
    return status


def load_status(serial_number, timestamp_dir):
    status_path = os.path.join('static', 'product_images', serial_number, timestamp_dir, 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as f:
            return json.load(f).get('status', 'unknown')
    return 'unknown'

# Patch save_yolo_labels to update status.json after saving a label
def save_yolo_labels(serial_number, timestamp_dir, image_filename, results):
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
    
    # Update status.json after saving label
    update_status_json(serial_number, timestamp_dir)
    return label_path

def update_all_products_status_by_timestamp():
    try:
        products = load_products()
    except FileNotFoundError:
        products = []

    changed = False
    for product in products:
        serial_number = product.get("serial_number")
        if not serial_number:
            continue
        base_path = os.path.join('static', 'product_images', serial_number)
        if not os.path.exists(base_path):
            continue
        status_by_timestamp = {}
        for timestamp in os.listdir(base_path):
            ts_path = os.path.join(base_path, timestamp)
            if not os.path.isdir(ts_path):
                continue
            status_path = os.path.join(ts_path, 'status.json')
            if os.path.exists(status_path):
                try:
                    with open(status_path, 'r') as f:
                        status = json.load(f).get('status', 'unknown')
                except Exception:
                    status = 'unknown'
                status_by_timestamp[timestamp] = status
        if status_by_timestamp:
            if product.get("status_by_timestamp") != status_by_timestamp:
                product["status_by_timestamp"] = status_by_timestamp
                changed = True
    if changed:
        save_products(products)


def update_all_products_thermal_data() -> None:
    """
    Walk every   static/product_images/<serial>/<timestamp>/thermal/area*.json
    and copy each file’s `raw_grid` into

        product["thermal_by_timestamp"][<timestamp>][<area>] = raw_grid

    • Creates the nested dictionaries on demand.
    • Writes to Database/products.json only when at least one new
      grid is added or an existing grid is updated.
    """
    try:
        products = load_products()
    except FileNotFoundError:
        products = []
    changed = False

    root_dir = Path("static") / "product_images"

    for product in products:
        serial = product.get("serial_number")
        if not serial:
            continue

        serial_path = root_dir / serial
        if not serial_path.is_dir():
            continue

        # ── iterate over each timestamp folder ──────────────────────────
        for ts_dir in serial_path.iterdir():
            if not ts_dir.is_dir():
                continue

            thermal_dir = ts_dir / "thermal"
            if not thermal_dir.is_dir():
                continue

            # ── read every area*.json inside the thermal folder ────────
            for jf in thermal_dir.glob("area*.json"):
                area_name = jf.stem            # e.g.  "area1"
                try:
                    raw_grid = json.loads(jf.read_text())["raw_grid"]
                except Exception:
                    # corrupt or malformed file – skip it gracefully
                    continue

                # Ensure nested dicts exist
                ts_dict   = product.setdefault("thermal_by_timestamp", {}) \
                                        .setdefault(ts_dir.name, {})

                # Add / update only if different or missing
                if area_name not in ts_dict or ts_dict[area_name] != raw_grid:
                    ts_dict[area_name] = raw_grid
                    changed = True

    # Save back to disk only once, and only if something actually changed
    if changed:
        save_products(products)
def get_product_thermal_history(serial_number):
    history = {}
    for ts in get_all_timestamp_directories(serial_number):
        imgs = get_thermal_images_from_timestamp_directory(serial_number, ts)
        if imgs:
            history[ts] = imgs
    return history

# === Flask Routes ===
@app.route('/')
def home():
    return redirect(url_for('analytics'))

@app.route('/analytics')
def analytics():
    update_all_products_status_by_timestamp()  # Ensure status_by_timestamp is up to date
    update_all_products_thermal_data()

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
    try:
        products = load_products()
    except FileNotFoundError:
        products = []
    for product in products:
        serial_number = product["serial_number"]
        latest_timestamp = get_latest_timestamp_directory(serial_number)
        latest_images = get_latest_images_for_product(serial_number)
        print(f"{serial_number} | latest_timestamp: {latest_timestamp} | latest_images: {latest_images}")
        product["latest_images"] = latest_images
        product["timestamp"] = latest_timestamp if latest_timestamp else "-"
        # Load status from status.json
        if latest_timestamp:
            product["status"] = load_status(serial_number, latest_timestamp)
        else:
            product["status"] = 'unknown'
    return render_template('database.html', products=products)


@app.route('/product/<serial_number>', methods=['GET','POST'])
def product_detail(serial_number):
    products = load_products()
    product  = next((p for p in products if p.get('serial_number')==serial_number), None)
    if not product:
        abort(404)

    if request.method == 'POST':
        new_name = request.form.get('model_name','').strip()
        if new_name:
            product['model_name'] = new_name
            save_products(products)
        return redirect(url_for('product_detail', serial_number=serial_number))

    # on GET, just read what's in products.json
    latest_images      = get_latest_images_for_product(serial_number)
    image_history      = get_product_image_history(serial_number)
    thermal_history    = get_product_thermal_history(serial_number)
    status_by_timestamp= product.get('status_by_timestamp', {})
    current_status     = product.get('status', 'unknown')

    # inject for template
    product['images']             = latest_images
    product['image_history']      = image_history
    product['thermal_history']    = thermal_history
    product['status_by_timestamp']= status_by_timestamp

    return render_template(
        'product.html',
        product=product,
        status=current_status
    )



@app.route('/generate_reasoning/<serial_number>', methods=['POST'])
def generate_reasoning(serial_number):
    try:
        products = load_products()
    except FileNotFoundError:
        products = []
    product = next((p for p in products if p["serial_number"] == serial_number), None)
    if not product:
        return {"error": "Product not found"}, 404
    # Get timestamp from POST data (JSON or form) or query param
    timestamp = request.form.get('timestamp') or request.args.get('timestamp')
    if not timestamp and request.is_json:
        timestamp = request.get_json().get('timestamp')
    if not timestamp:
        timestamp = get_latest_timestamp_directory(serial_number)
    if not timestamp:
        return {"error": "No images found for this product"}, 404
    # Generate reasoning for the specified timestamp
    reasoning = generate_gemini_reasoning(product, timestamp)
    return {"reasoning": reasoning}

@app.route('/get_reasoning/<serial_number>/<timestamp>', methods=['GET'])
def get_reasoning(serial_number, timestamp):
    reasoning = load_reasoning(serial_number, timestamp)
    if reasoning:
        return {"reasoning": reasoning}
    else:
        return {"reasoning": None, "message": "Click the button below to generate reasoning."}

@app.route('/get_status/<serial_number>/<timestamp>', methods=['GET'])
def get_status(serial_number, timestamp):
    status = load_status(serial_number, timestamp)
    return {"status": status}

@app.route('/upload_image/<serial_number>', methods=['POST'])
def upload_image(serial_number):
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400
    image_file = request.files['image']
    if image_file.filename == '':
        return {"error": "No image file selected"}, 400
    try:
        # Save image to timestamp directory
        image_path = save_image_to_timestamp_directory(serial_number, image_file)
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
        verbose=False,
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
    results = model.predict(source=src_img, conf=0.5, verbose=False)
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
            verbose=False ,
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
            results = model.predict(source=new_img_path, conf=0.5, verbose=False)
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
    data = request.get_json() or {}
    serial_number = str(data.get('serial_number', '')).strip()
    timestamp     = str(data.get('timestamp', '')).strip()
    images        = data.get('images', {})
    model_name    = data.get('model_name', '').strip()

    if not serial_number or not timestamp or not images:
        return jsonify({'error': 'Missing serial_number, timestamp, or images'}), 400

    # 1) Save incoming images
    base_dir    = os.path.join('static', 'product_images', serial_number, timestamp)
    labels_dir  = os.path.join(base_dir, 'labels')
    thermal_dir = os.path.join(base_dir, 'thermal')
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(thermal_dir, exist_ok=True)

    saved = []
    for key, info in images.items():
        fn = info.get('filename')
        b64 = info.get('data')
        if not fn or not b64:
            continue
        raw = base64.b64decode(b64)

        if key == 'thermal':
            path = os.path.join(thermal_dir, fn)
            # also write raw_grid JSON
            grid = info.get('raw_grid')
            if grid:
                loc = fn.split('_')[2]
                with open(os.path.join(thermal_dir, f"area{loc}.json"), 'w') as jf:
                    json.dump({
                        'serial_number': serial_number,
                        'timestamp': timestamp,
                        'raw_grid': grid
                    }, jf, indent=2)
        elif key in ('light','no_light'):
            new_dir = os.path.join('static','new_images')
            os.makedirs(new_dir, exist_ok=True)
            path = os.path.join(new_dir, fn)
        else:
            # covers 'yolo' and any others
            path = os.path.join(base_dir, fn)

        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(raw)
        saved.append(path)

    # 2) Determine new status (crack > scratch > normal)
    new_status = update_status_json(serial_number, timestamp)

    # 3) Sync into products.json
    products = load_products()
    prod = next((p for p in products if p.get('serial_number')==serial_number), None)
    if not prod:
        prod = {
            'serial_number': serial_number,
            'model_name': model_name or 'unknown yet',
            'status': new_status,
            'status_by_timestamp': {}
        }
        products.append(prod)
    else:
        # only overwrite model_name if provided
        if model_name:
            prod['model_name'] = model_name
        prod['status'] = new_status

    # record per‑timestamp status
    sbt = prod.setdefault('status_by_timestamp', {})
    sbt[timestamp] = new_status

    save_products(products)

    # 4) Run YOLO on any “yolo_*.jpg” with no existing labels
    for fn in os.listdir(base_dir):
        if fn.startswith('yolo_') and fn.lower().endswith('.jpg'):
            img_path   = os.path.join(base_dir, fn)
            label_path = os.path.join(labels_dir, fn.replace('.jpg','_labels.json'))
            if not os.path.exists(label_path):
                try:
                    results = model.predict(source=img_path, conf=0.5, verbose=False)
                    save_yolo_labels(serial_number, timestamp, fn, results)
                except Exception as e:
                    print(f"[YOLO ERROR] {fn}: {e}")

    return jsonify({
        'success': True,
        'saved_images': saved,
        'status': new_status
    })


@app.route('/cracked_board_rate')
def cracked_board_rate():
    return render_template('cracked_board_rate.html')


@app.route('/api/cracked_board_rate_data')
def cracked_board_rate_data():
    try:
        # 1) get “days” from query (0 = all‑time)
        days = request.args.get('days', default=0, type=int)

        # 2) load your product list
        products = load_products()
        total_boards = len(products)

        # 3) prepare counters
        status_counts  = Counter()
        history_counts = Counter()
        history_raw    = {}       # collect timestamp → status

        # 4) compute cutoff
        now    = datetime.now()
        cutoff = now - timedelta(days=days) if days > 0 else None

        # 5) iterate all products
        for p in products:
            sbt = p.get('status_by_timestamp', {})
            filtered_statuses = []

            for ts, st in sbt.items():
                # parse ISO timestamp
                try:
                    dt = datetime.fromisoformat(ts)
                except ValueError:
                    dt = datetime.strptime(ts, '%Y-%m-%dT%H-%M-%S')

                # only keep if inside the window
                if cutoff is None or dt >= cutoff:
                    status_lower = st.lower()
                    filtered_statuses.append(status_lower)
                    history_raw[ts] = status_lower

            # update aggregate counts
            history_counts.update(filtered_statuses)
            if 'cracked' in filtered_statuses:
                status_counts['cracked'] += 1
            elif 'normal' in filtered_statuses:
                status_counts['healthy'] += 1
            else:
                status_counts['unknown'] += 1

        # 6) return everything
        return jsonify({
            'total_boards':   total_boards,
            'status_counts':  dict(status_counts),
            'history_counts': dict(history_counts),
            'history_raw':    history_raw
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/inspection_accuracy')
def inspection_accuracy():
    return render_template('inspection_accuracy.html')


@app.route('/api/inspection_timestamps')
def inspection_timestamps():
    base = 'trained_models'
    if not os.path.isdir(base):
        return jsonify([])

    runs = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ]
    runs.sort()   # so oldest→newest
    return jsonify(runs)


@app.route('/thermal_trends')
def thermal_trends():
    return render_template('thermal_trends.html')

@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    period = request.form.get('period')
    
    if period not in ('daily','weekly','monthly'):
        flash('Please select a valid report period.', 'danger')
        return redirect(url_for('analytics'))

    try:
        pdf_path = generate_report(
        period,
        inspection_line="Smart Conveyor Automated System",
        inspector="Automated System"  # or any other name you want
    )
    except Exception as e:
        flash(f"Failed to generate PDF: {e}", 'danger')
        return redirect(url_for('analytics'))

    # serve file as download
    return send_file(pdf_path,
                     as_attachment=True,
                     download_name=os.path.basename(pdf_path),
                     mimetype='application/pdf')


@app.route('/api/thermal_trends_data')
def thermal_trends_data():
    # parse days= query (0 = all time)
    days = request.args.get('days', default=0, type=int)
    products = load_products()   # :contentReference[oaicite:0]{index=0}

    now    = datetime.now()
    cutoff = now - timedelta(days=days) if days > 0 else None

    # collect all readings by area
    area_readings = defaultdict(list)
    for p in products:
        for ts, area_map in p.get('thermal_by_timestamp', {}).items():
            # parse timestamp
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                dt = datetime.strptime(ts, '%Y-%m-%dT%H-%M-%S')
            if cutoff is None or dt >= cutoff:
                for area, grid in area_map.items():
                    # flatten the grid and extend
                    flat = [v for row in grid for v in row]
                    area_readings[area].extend(flat)

    # compute average temperature per area
    area_avg = {
        area: (sum(vals)/len(vals)) if vals else None
        for area, vals in area_readings.items()
    }

    # compute distribution counts
    below = normal = above = 0
    for vals in area_readings.values():
        for v in vals:
            if v < 25:      below  += 1
            elif v > 38:    above  += 1
            else:           normal += 1

    return jsonify({
        'area_avg':     area_avg,
        'distribution': {'below': below, 'normal': normal, 'above': above}
    })


@app.route('/api/inspection_accuracy_data')
def inspection_accuracy_data():
    ts = request.args.get('timestamp')
    path = os.path.join('trained_models', ts, 'best_metrics.json')
    if not os.path.isfile(path):
        return jsonify({'error': 'Not found'}), 404

    with open(path) as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/api/best_metrics')
def best_metrics_api():
    import os, json
    from flask import jsonify

    base = 'trained_models'
    if not os.path.isdir(base):
        return jsonify({'error': 'trained_models directory not found'}), 404

    # find all timestamp‐named subfolders
    runs = [d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]
    if not runs:
        return jsonify({'error': 'No training runs found'}), 404

    runs.sort()  # lexicographic works for YYYYmmdd_HHMMSS
    latest = runs[-1]
    path   = os.path.join(base, latest, 'best_metrics.json')
    if not os.path.isfile(path):
        return jsonify({'error': 'best_metrics.json not found'}), 404

    with open(path) as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/api/products_data')
def products_data():
    # returns the full products.json array
    try:
        products = load_products()
    except FileNotFoundError:
        products = []
    return jsonify(products)


@app.route('/api/analytics_data')
def analytics_data():
    try:
        with open(DATA_FILE) as f:
            products = json.load(f)
        total_boards = len(products)

        status_counts = {}
        history_counts = Counter()

        for p in products:
            # classification by ever‑seen status
            sts = [s.lower() for s in p.get('status_by_timestamp', {}).values()]
            if 'cracked' in sts:
                cls = 'cracked'
            elif 'scratch' in sts:
                cls = 'scratch'
            elif 'normal' in sts:
                cls = 'healthy'
            else:
                cls = 'unknown'
            status_counts[cls] = status_counts.get(cls, 0) + 1
            history_counts.update(sts)

        c = status_counts.get
        cracked  = c('cracked',  0)
        scratch  = c('scratch',  0)
        healthy  = c('healthy',  0)
        unknown  = c('unknown',  0)

        def pct(x): return round(x/total_boards*100,2) if total_boards else 0
        return jsonify({
            'total_boards':    total_boards,
            'cracked_boards':  cracked,
            'scratch_boards':  scratch,
            'healthy_boards':  healthy,
            'unknown_boards':  unknown,
            'cracked_percent': pct(cracked),
            'scratch_percent': pct(scratch),
            'healthy_percent': pct(healthy),
            'status_counts':   status_counts,
            'history_counts':  dict(history_counts),
            'defect_rate':     pct(cracked + scratch)
        })
    except Exception as e:
        print("read error")
        return jsonify({'error': str(e)}), 500




if __name__ == "__main__":
    host_ip = socket.gethostbyname(socket.gethostname())
    port = 5000
    print(f"\n============================================\nServer running at: http://{host_ip}:{port}\n============================================\n")
    app.run(host="0.0.0.0", port=port, debug=False) 

