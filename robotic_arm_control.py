import numpy as np
from adafruit_pca9685 import PCA9685
import board
import busio
import time
import yaml
import threading
from collections import deque
import pickle
import tkinter as tk
from tkinter import ttk
import paho.mqtt.client as mqtt
import json
import adafruit_amg88xx
from multiprocessing import shared_memory
import base64
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from io import BytesIO
import cv2
from gpiozero import DigitalOutputDevice
from time import sleep
import adafruit_vl53l0x
import requests
import json
from datetime import datetime
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random



# === Config ===
zoom_factor = 8  # 8x8 -> 64x64
vmin, vmax = 28, 35  # Adjust these for your environment

# Load configuration
with open('robotic_arm_config.yml', 'r') as file:
	config = yaml.safe_load(file)

# Servo configuration
servo_min = 150
servo_max = 625

# Initial positions
initial_servo_angles = {
	1: 95,   # Base
	2: 80,  # Shoulder
	3: 57,   # Elbow
	4: 90,   # Wrist
	5: 180   # Grab
}

servo_angles = initial_servo_angles.copy()
light = DigitalOutputDevice(24, active_high=False, initial_value=False)
conveyor = DigitalOutputDevice(23, active_high=False, initial_value=False)
saved_locations = {} 
AUTOMATION_FILE = "automation.txt"
serial_no = None
stop_requested = False
REGISTER_SIZE = 13

logging.basicConfig(
    filename='robotic_arm.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def generate_and_send_image():
    data = np.array(amg.pixels)
    interpolated_data = zoom(data, zoom_factor, order=0)

    fig, ax = plt.subplots()
    im = ax.imshow(interpolated_data, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.axis('off')  # Remove axis

    # Save to BytesIO instead of disk
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Encode to base64
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    raw_grid = [[round(temp, 1) for temp in row] for row in data]

    # Create and send payload
    payload = json.dumps({
        "type": "thermal_image",
        "timestamp": time.time(),
        "image": img_b64,
        "raw_grid": raw_grid 
    })

    mqtt_client.publish("robotic_arm/thermal_image", payload)
    print("📤 Thermal image sent over MQTT")
    

def generate_heat_image_base64(amg, zoom_factor=10, vmin=20, vmax=40):
    data = np.array(amg.pixels)
    data = add_random_hotspots(data.tolist(), num_spots=3, hot_temp=30.0)
    interpolated_data = zoom(data, zoom_factor, order=0)

    fig, ax = plt.subplots()
    ax.imshow(interpolated_data, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode('utf-8')
    raw_grid = [[round(temp, 2) for temp in row] for row in data]  # Optional: round to 2 decimals

    return encoded, raw_grid
  
def add_random_hotspots(grid, num_spots=3, hot_temp=38.0):
    """Injects `num_spots` hot points into the 8x8 thermal grid."""
    for _ in range(num_spots):
        row = random.randint(0, 7)
        col = random.randint(0, 7)
        grid[row][col] = round(hot_temp + random.uniform(0, 2), 2)  # Slightly vary
    return grid
    
def write_register(condition, value):
    # First, try to clean up any old shared‑memory segment by that name.
    try:
        old = shared_memory.SharedMemory(name=condition)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[WARN] Could not unlink old segment {condition}: {e}")

    # Now create a fresh one; if it already sneaked in again, force‑unlink + retry.
    try:
        shm = shared_memory.SharedMemory(name=condition, create=True, size=REGISTER_SIZE)
    except FileExistsError:
        try:
            existing = shared_memory.SharedMemory(name=condition)
            existing.close()
            existing.unlink()
        except Exception as e:
            print(f"[WARN] Could not force‑unlink existing {condition}: {e}")
        # retry creation
        shm = shared_memory.SharedMemory(name=condition, create=True, size=REGISTER_SIZE)

    # Write your value into it
    buffer = np.ndarray((REGISTER_SIZE,), dtype=np.uint8, buffer=shm.buf)
    buffer.fill(0)
    value_bytes = str(value).encode()[:REGISTER_SIZE]
    buffer[:len(value_bytes)] = np.frombuffer(value_bytes, dtype=np.uint8)

    shm.close()
    
def read_register(sensor):
    """
    Read an integer value back from a 10‑byte shared‑memory register,
    or return None if it doesn’t yet exist (or is zero‑length).
    """
    try:
        shm = shared_memory.SharedMemory(name=sensor)
        # If we somehow opened a zero‑size segment, bail out:
        if shm.size == 0:
            shm.close()
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        # no segment, or worthless zero‑length segment → treat as “no value”
        return None

    # Otherwise we have a valid 10‑byte region:
    buf = np.ndarray((REGISTER_SIZE,), dtype=np.uint8, buffer=shm.buf)
    raw = buf.tobytes().rstrip(b"\x00")
    shm.close()
    val = raw.decode(errors="ignore")
    print(f"[DEBUG] Register {sensor} = '{val}'")
    return int(val) if val.isdigit() else None
 
def read_image_from_register(name):
    try:
        shm = shared_memory.SharedMemory(name=name)
        img_bytes = bytes(shm.buf[:])
        shm.close()
        np_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    except FileNotFoundError:
        print(f"[ERROR] Shared memory for '{name}' not found.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read image from '{name}': {e}")
        return None

def angle_to_pwm(angle):
	return int((servo_min + (angle / 180.0) * (servo_max - servo_min)) / 4096 * 65535)

def set_servo_angle(channel, target_angle):
	global servo_angles
	
	current_angle = int(servo_angles[channel])
	step = 1 if int(target_angle) > int(current_angle) else -1
	
	while current_angle != int(target_angle):
		pca.channels[channel].duty_cycle = angle_to_pwm(current_angle)
		current_angle += step
		servo_angles[channel] = current_angle
		time.sleep(0.02)
	
	pca.channels[channel].duty_cycle = angle_to_pwm(target_angle)
	servo_angles[channel] = target_angle

def initialize_servos():
	print("Initializing servos...")
	
	# Sort keys to move servo 2 last
	ordered_channels = [ch for ch in initial_servo_angles if ch != 2] + [2]

	for channel in ordered_channels:
		angle = initial_servo_angles[channel]
		set_servo_angle(channel, angle)
		time.sleep(0.2)
	
	print("Initialization complete")


def save_location(slot):
	# Load existing data
	locations = {}
	try:
		with open(AUTOMATION_FILE, 'r') as f:
			for line in f:
				if ':' in line:
					loc, angles = line.strip().split(':')
					locations[int(loc)] = list(map(int, angles.split(',')))
	except FileNotFoundError:
		pass  # No existing file yet

	# Update or insert current servo angles
	locations[slot] = [servo_angles[ch] for ch in range(1, 6)]

	# Write back to file (overwrite old)
	with open(AUTOMATION_FILE, 'w') as f:
		for loc in sorted(locations.keys(), key=lambda x: (x != "barcode", x)):
			angle_list = ','.join(map(str, locations[loc]))
			f.write(f"{loc}:{angle_list}\n")

	print(f"✅ Saved current servo angles to location {slot} in {AUTOMATION_FILE}")



def automate_sequence(delay_between_locations=1.0, movement_speed=0.01):
	global stop_requested
	stop_requested = False
	
	try:
		with open(AUTOMATION_FILE, 'r') as f:
			lines = f.readlines()
	except FileNotFoundError:
		print(f"⚠️ File {AUTOMATION_FILE} not found.")
		return

	if not lines:
		print("⚠️ No locations saved.")
		return

	print("🚀 Starting automation...")
	timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	while not stop_requested:
		conveyor.on()
		distance = vl53.range  # Distance in mm
		print(f"Distance: {distance} mm")
		if(distance < 70):
			conveyor.off()
			serial_no = None
			time.sleep(0.5)
			sorted_lines = sorted(lines, key=lambda l: (not l.startswith("barcode:"), l))
			for line in sorted_lines:
				if stop_requested:
					break
				if ':' not in line:
					continue
				loc_num, angle_str = line.strip().split(':')
				angles = list(map(int, angle_str.split(',')))
				print(f"🔄 Moving to Location {loc_num}")

				for ch in range(1, 6):
					current_angle = int(servo_angles[ch])
					target_angle = int(angles[ch - 1])
					step = 1 if target_angle > current_angle else -1

					for angle in range(current_angle, target_angle + step, step):
						pca.channels[ch].duty_cycle = angle_to_pwm(angle)
						servo_angles[ch] = angle
						time.sleep(movement_speed)
						
				if loc_num == "0" or loc_num == 0:
					write_register("barcode", 1)
					while not stop_requested:
						serial_no = read_register("barcode_result");
						print("Barcode Detection:", serial_no)
						print("waiting serial no")
						if serial_no not in [None, 0, '0', '', 'None']:
							break
				else:
					post_location_routine(serial_no, loc_num, timestamp)

				time.sleep(delay_between_locations)
				
			print("✅ Automation complete.")
			print("⏳ Waiting for object to leave...")
			initialize_servos()
			distance = vl53.range
			print(f"Distance: {distance} mm")
			while distance < 70 and not stop_requested:
				distance = vl53.range
				print(f"Distance: {distance} mm")
				conveyor.on()
				
			print("🟢 Ready for next object.")
			write_register("reset", 1)
			time.sleep(1)
			clear_all_registers()


def post_location_routine(serial_no, loc_num, timestamp):
	print("📍 Running post-location routine...")
	time.sleep(1)
	
	write_register("yolo", 1)
	while True:
		print("waiting yolo result")
		if read_register("yolo") == 0:
			break
			
	write_register("frame", 1)
	while True:
		print("waiting data frame")
		if read_register("frame") == 0:
			break
			
	light.on()
	time.sleep(2)
	write_register("frame2", 1)
	while True:
		print("waiting data frame")
		if read_register("frame") == 0:
			break
			
	light.off()
	light_frame = read_image_from_register("frame_light")
	no_light_frame = read_image_from_register("frame_no_light")
	heat_frame, raw_grid = generate_heat_image_base64(amg, zoom_factor=10, vmin=20, vmax=40)
	yolo_result_mqtt = read_image_from_register("yolo_result")
	mqtt_yolo_result(yolo_result_mqtt,no_light_frame, light_frame, heat_frame, raw_grid, serial_no, loc_num, timestamp)
	
	# Optionally wait a bit to allow MQTT to finish sending
	time.sleep(0.5)
	
	print("✅ Post-location routine complete.")

def clear_all_registers():
	global serial_no
	serial_no = None
	keys_to_clear = [
		"yolo", "barcode", "barcode_result",
		"yolo_result", "automate",
		# Add other keys used during automation
	]
	for key in keys_to_clear:
		write_register(key, 0)  # Or "" or None depending on type
	


def send_images_to_flask(serial_no, timestamp, images_payload, raw_grid):
	def _send():
		try:
			session = requests.Session()
			retry = Retry(
				total=3,
				backoff_factor=0.5,
				status_forcelist=[500, 502, 503, 504],
				allowed_methods=["POST"]
			)
			adapter = HTTPAdapter(max_retries=retry)
			session.mount("http://", adapter)

			response = session.post(
				"http://10.10.16.168:5000/upload_yolo_images",
#				"http://10.10.21.194:5000/upload_yolo_images",
				json={"serial_number": serial_no, "timestamp": timestamp, "images": images_payload, "raw_heat_data": raw_grid},
				timeout=5  # seconds
			)
			print(f"✅ Flask upload response: {response.status_code} - {response.text}")
		except requests.exceptions.Timeout:
			print("⚠️ Flask upload timed out.")
		except requests.exceptions.RequestException as e:
			print(f"❌ Flask upload failed: {e}")

	# Launch in background thread
	threading.Thread(target=_send, daemon=True).start()


def mqtt_yolo_result(yolo_mqtt, no_light_frame, light_frame, heat_frame, raw_grid, serial_no, loc_num, timestamp):
	def encode_image_with_name(image, img_type):
		try:
			if image is None or image.size == 0:
				raise ValueError(f"{img_type} image is empty or None.")
			timestamp = int(time.time())
			filename = f"{img_type}_{serial_no}_{loc_num}_{timestamp}.jpg"
			success, buffer = cv2.imencode('.jpg', image)
			if not success:
				raise ValueError(f"Failed to encode {img_type} image.")
			img_b64 = base64.b64encode(buffer).decode('utf-8')
			return filename, img_b64
		except Exception as e:
			print(f"[ERROR] Encoding {img_type} image failed: {e}")
			return None, None


	images_payload = {}

	if yolo_mqtt is not None:
		yolo_name, yolo_b64 = encode_image_with_name(yolo_mqtt, "yolo")
		images_payload["yolo"] = {
			"filename": yolo_name,
			"data": yolo_b64
		}

	no_light_name, no_light_b64 = encode_image_with_name(no_light_frame, "no_light")
	light_name, light_b64 = encode_image_with_name(light_frame, "light")
	heat_name = f"thermal_{serial_no}_{loc_num}_{timestamp}.png"  # Thermal is base64 PNG

	images_payload.update({
		"no_light": {
			"filename": no_light_name,
			"data": no_light_b64
		},
		"light": {
			"filename": light_name,
			"data": light_b64
		},
		"thermal": {
			"filename": heat_name,
			"data": heat_frame
		}
	})

	payload = {
		"type": "yolo_result",
		"serial_number": serial_no,
		"location_number": f"location_{loc_num}",
		"timestamp": timestamp,
		"images": images_payload
	}
	print(raw_grid)
	try:
		mqtt_client.publish("robotic_arm/yolo_result", json.dumps(payload))
		print("📤 Frame images sent over MQTT" + (" with YOLO" if yolo_mqtt is not None else " (YOLO skipped)"))
		logging.info(f"Sent image payload for serial {serial_no} at {timestamp}")
	except Exception as e:
		logging.error(f"[MQTT] Failed to publish image payload: {e}")
	try:
		send_images_to_flask(serial_no, timestamp, images_payload, raw_grid)
	except Exception as e:
		print(f"❌ Failed to send image to Flask: {e}")
	



    
# MQTT Callbacks
def on_connect(client, userdata, flags, reasonCode, properties=None):
	print("? MQTT connected with reason code:", reasonCode)
	topic = "robotic_arm/command"
	client.subscribe(topic)
	print(f"? Subscribed to topic: {topic}")

def on_disconnect(client, userdata, rc):
	print(f"⚠️ [MQTT] Disconnected (rc={rc}) — will try to reconnect")

def keep_alive_loop():
	while True:
		mqtt_client.publish("robotic_arm/heartbeat", "ping")
		time.sleep(15)

def mqtt_reconnect_loop():
    while True:
        if not mqtt_client.is_connected():
            print("🔄 MQTT disconnected. Reconnecting...")
            try:
                mqtt_client.reconnect()
            except Exception as e:
                print(f"❌ MQTT reconnect failed: {e}")
        time.sleep(10)


  
def stop_all():
	global stop_requested
	stop_requested = True
	print("🛑 Stop pressed: resetting servos and clearing registers")
	conveyor.off()
	light.off()
	initialize_servos()
	clear_all_registers()
    
def on_message(client, userdata, msg):
	try:
		payload = msg.payload.decode()
		print(f"📩 [MQTT] Message received on topic '{msg.topic}': {payload}")

		try:
			data = json.loads(payload)
		except json.JSONDecodeError:
			print("⚠️ Invalid JSON received.")
			return

		if isinstance(data, dict):
			motor = data.get("motor")
			action = data.get("action")

			# Handle absolute action (from Node-RED slider)
			if action == "absolute" and "angle" in data:
				angle = int(data["angle"])
				print(f"🎚 Setting motor {motor} to angle {angle}")
				set_servo_angle(motor, angle)
				return

			# Optional: legacy support for ON/OFF + left/right
			state = data.get("state")
			if motor and action and state:
				print(f"➡️ Motor: {motor}, Action: {action}, State: {state}")
				if state == "ON":
					if action == "left":
						set_servo_angle(motor, servo_angles[motor] - 5)
					elif action == "right":
						set_servo_angle(motor, servo_angles[motor] + 5)
			elif action == "initialize":
				initialize_servos() 
			elif action == "automate":
				threading.Thread(target=automate_sequence, daemon=True).start()
			elif action == "stop":
				stop_all()
						

		else:
			print("⚠️ Payload is not a JSON object.")

	except Exception as e:
		print(f"❌ Error in on_message: {e}")
        
mqtt_client = mqtt.Client(protocol=mqtt.MQTTv311, transport="tcp")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect
threading.Thread(target=mqtt_reconnect_loop, daemon=True).start()
threading.Thread(target=keep_alive_loop, daemon=True).start()

try:
    mqtt_broker = 'broker.hivemq.com'
    mqtt_port = config.get('mqtt_port', 1883)
    mqtt_client.connect(mqtt_broker, mqtt_port)
    mqtt_client.loop_start()
    print(f"? Connected to MQTT broker at {mqtt_broker}:{mqtt_port}")
except Exception as e:
    print(f"? Warning: Failed to connect to MQTT broker ({mqtt_broker}:{mqtt_port}): {e}")
    mqtt_client = None

def on_slider_change(channel, val):
	angle = int(float(val))
	set_servo_angle(channel, angle)

def build_gui():
	root = tk.Tk()
	root.title("Robotic Arm Control")

	sliders = {}
	for ch in range(1, 6):
		frame = ttk.Frame(root, padding=10)
		frame.pack()
		
		ttk.Label(frame, text=f"Servo {ch}").pack(anchor='w')
		slider = ttk.Scale(frame, from_=0, to=180, orient='horizontal',
						   command=lambda val, ch=ch: on_slider_change(ch, val))
		slider.set(servo_angles[ch])
		slider.pack(fill='x', padx=20)
		slider.config(length=400)
		sliders[ch] = slider

	# Location Save Section
	loc_frame = ttk.Frame(root, padding=10)
	loc_frame.pack()

	ttk.Label(loc_frame, text="Select Location:").grid(row=0, column=0, padx=5)

	location_var = tk.IntVar()
	location_dropdown = ttk.Combobox(loc_frame, textvariable=location_var, values=[0,1,2,3,4,5,6], width=10)
	location_dropdown.current(0)
	location_dropdown.grid(row=0, column=1, padx=5)
	location_dropdown.current(0)

	save_button = ttk.Button(loc_frame, text="Save Location", command=lambda: save_location(location_var.get()))
	save_button.grid(row=0, column=2, padx=5)

	auto_button = ttk.Button(loc_frame, text="Automate", command=lambda: threading.Thread(target=automate_sequence, daemon=True).start())
	auto_button.grid(row=0, column=3, padx=5)
	
	thermal_button = ttk.Button(loc_frame, text="Print Heat Grid", command=generate_and_send_image)
	thermal_button.grid(row=0, column=4, padx=5)

	# Other Buttons
	button_frame = ttk.Frame(root, padding=10)
	button_frame.pack()

	ttk.Button(button_frame, text="Initialize", command=initialize_servos).grid(row=0, column=0, padx=5, pady=5)
	ttk.Button(button_frame, text="Stop", command=stop_all).grid(row=0, column=1, padx=5, pady=5)
	ttk.Button(button_frame, text="Exit", command=lambda: [pca.deinit(), root.quit()]).grid(row=0, column=2, padx=5, pady=5)

	root.mainloop()

  

def initialize_hardware_with_retry():
    global pca, amg, vl53
    while True:
        try:
            i2c = busio.I2C(3, 2, frequency=100000)
            while not i2c.try_lock():
                pass  # Wait for I2C lock
            i2c.unlock()  # Immediately unlock to prevent lockup on next use

            # Try initializing devices
            pca = PCA9685(i2c)
            pca.frequency = 60

            amg = adafruit_amg88xx.AMG88XX(i2c, 0x69)
            vl53 = adafruit_vl53l0x.VL53L0X(i2c)

            print("✅ I2C devices initialized successfully.")
            return  # Success — break the loop

        except Exception as e:
            print(f"❌ I2C device initialization failed: {e}")
            print("🔄 Retrying in 2 seconds...")
            time.sleep(2)  # Wait before retrying
          
# Initialize hardware
initialize_hardware_with_retry()
initialize_servos()
print("Arm control ready (GUI and MQTT only)")
build_gui()
