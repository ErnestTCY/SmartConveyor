import cv2
import time
import asyncio
import websockets
from ultralytics import YOLO
from multiprocessing import shared_memory
import numpy as np
import atexit
from pyzbar.pyzbar import decode
import logging
import sys
import os

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("AnomalyDetection")


# Globals
connected_clients = set()
clients_lock = asyncio.Lock()
model = YOLO("surebest.pt")
ai_enabled = True
conf_threshold = 0.2
REGISTER_SIZE = 13


def write_register(condition, value):
    try:
        old = shared_memory.SharedMemory(name=condition)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Could not unlink old segment {condition}: {e}")

    try:
        shm = shared_memory.SharedMemory(name=condition, create=True, size=REGISTER_SIZE)
    except FileExistsError:
        try:
            existing = shared_memory.SharedMemory(name=condition)
            existing.close()
            existing.unlink()
        except Exception as e:
            log.warning(f"Force-unlink failed for {condition}: {e}")
        shm = shared_memory.SharedMemory(name=condition, create=True, size=REGISTER_SIZE)

    buffer = np.ndarray((REGISTER_SIZE,), dtype=np.uint8, buffer=shm.buf)
    buffer.fill(0)
    value_bytes = str(value).encode()[:REGISTER_SIZE]
    buffer[:len(value_bytes)] = np.frombuffer(value_bytes, dtype=np.uint8)
    shm.close()


def read_register(sensor):
    try:
        shm = shared_memory.SharedMemory(name=sensor)
        if shm.size == 0:
            shm.close()
            return None
    except (FileNotFoundError, ValueError):
        return None

    buf = np.ndarray((10,), dtype=np.uint8, buffer=shm.buf)
    raw = buf.tobytes().rstrip(b"\x00")
    shm.close()
    val = raw.decode(errors="ignore")
    return int(val) if val.isdigit() else None


def write_image_to_register(name, image):
    ok, buffer = cv2.imencode('.jpg', image)
    if not ok:
        log.error("Failed to encode image.")
        return

    image_bytes = buffer.tobytes()
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=len(image_bytes))
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        if shm.size < len(image_bytes):
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(name=name, create=True, size=len(image_bytes))

    shm.buf[:len(image_bytes)] = image_bytes
    shm.close()


def clear_all_registers():
    keys_to_clear = [
        "yolo", "barcode", "barcode_result",
        "yolo_result", "automate", "reset",
        "frame", "frame2"
    ]
    for key in keys_to_clear:
        write_register(key, 0)


async def video_loop():
    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        log.critical("Camera not found.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed.")
            await asyncio.sleep(0.1)
            continue

        display_frame = frame.copy()

        if read_register("yolo"):
            log.info("YOLO detection triggered.")
            try:
                results = model.predict(source=frame, stream=True, conf=conf_threshold)
                for r in results:
                    display_frame = r.plot()
            except Exception as e:
                log.error(f"YOLO error: {e}")
            write_image_to_register("yolo_result", display_frame)
            write_register("yolo", 0)

        if read_register("frame"):
            write_image_to_register("frame_no_light", display_frame)
            write_register("frame", 0)

        if read_register("frame2"):
            write_image_to_register("frame_light", display_frame)
            write_register("frame2", 0)

        if read_register("barcode"):
            log.info("Barcode scan triggered.")
            text = None
            start_time = time.time()

            while time.time() - start_time < 10:
                ret2, frame2 = cap.read()
                if not ret2:
                    continue

                barcodes = decode(frame2)
                for bc in barcodes:
                    data = bc.data.decode('utf-8')
                    text = f"{data} ({bc.type})"
                    break
                if text:
                    break
                await asyncio.sleep(0)

            if not text:
                text = "1234567890012"
                log.warning(f"Barcode timeout – fallback: {text}")
            write_register("barcode_result", text)
            write_register("barcode", 0)

        if read_register("reset"):
            clear_all_registers()

        # FPS overlay
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam + AI" if ai_enabled else "Webcam", display_frame)

        # Send raw frame to clients
        try:
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = jpeg.tobytes()
            async with clients_lock:
                targets = connected_clients.copy()

            disconnected = set()
            for client in targets:
                try:
                    await client.send(jpeg_bytes)
                except Exception as e:
                    log.warning(f"WebSocket send error: {e}")
                    disconnected.add(client)

            if disconnected:
                async with clients_lock:
                    connected_clients.difference_update(disconnected)

        except Exception as e:
            log.error(f"Unexpected error in video loop: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()


async def handler(ws):
    log.info("Client connected: %s", ws.remote_address)
    async with clients_lock:
        connected_clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        async with clients_lock:
            connected_clients.discard(ws)
        log.info("Client disconnected: %s", ws.remote_address)


async def main():
    clear_all_registers()
    log.info("WebSocket server starting at ws://0.0.0.0:8765")
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=None):
        await video_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutdown requested.")
