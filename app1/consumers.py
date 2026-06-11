import base64
import json
import cv2
import torch
import pygame
import asyncio
import time
import pytz
import sqlite3
import urllib.request
import urllib.parse
import urllib.error
from collections import deque
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
from datetime import datetime


CONFIDENCE_THRESHOLD = 0.6  # Raised to 0.6 to reduce false detections

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7564093390:AAGJU5INCiI1hjOah4rWDIw1oS1zCQcQEmI"
TELEGRAM_CHAT_ID = "5154424717"

# Fine amounts per violation type
FINE_AMOUNTS = {
    'No Helmet': 500,
    'NO-Safety Vest': 300,
}

def send_telegram_message(message):
    """Send a text alert message to Telegram."""
    try:
        text = urllib.parse.quote(message)
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={text}&parse_mode=HTML"
        urllib.request.urlopen(url, timeout=5)
        print(f"Telegram message sent.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def send_telegram_photo(image_bytes, caption):
    """Send a violation photo with caption to Telegram."""
    try:
        import io
        import multipart_form_data_builder as mfdb
    except ImportError:
        pass

    try:
        boundary = b"----TelegramBoundary"
        caption_encoded = caption.encode("utf-8")
        image_bytes_data = image_bytes

        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="chat_id"\r\n\r\n' +
            TELEGRAM_CHAT_ID.encode() + b"\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="caption"\r\n\r\n' +
            caption_encoded + b"\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="photo"; filename="violation.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n" +
            image_bytes_data + b"\r\n"
            b"--" + boundary + b"--\r\n"
        )

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
        print("Telegram photo sent successfully.")
    except Exception as e:
        print(f"Failed to send Telegram photo: {e}")
        # Fallback: send text only
        send_telegram_message(caption)

# Initialize the model
model = YOLO('app1/best.pt')

# Set up video capture (webcam)
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Pygame setup for sound alert
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("app1/alert.mp3")

# Database setup
def setup_database():
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        class_id INTEGER,
                        class_name TEXT,
                        confidence REAL,
                        track_id INTEGER,
                        bbox TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
    conn.commit()
    return conn, cursor

conn, cursor = setup_database()

class VideoStreamConsumer(AsyncWebsocketConsumer):
    active_connections = set()
    current_class_ids = []
    tracked_objects = {}
    object_entry_time = {}   # key: (track_id, class_id) → start time
    alerted_objects = set()  # key: (track_id, class_id) → already alerted
    object_counter = 0
    ALERT_TIME_THRESHOLD = 3  # seconds before alert triggers
    latest_frame = None       # store latest frame for photo capture

    async def connect(self):
        print("WebSocket connection established")
        self.active_connections.add(self)
        await self.accept()
        self.streaming = True
        if len(self.active_connections) == 1:
            asyncio.create_task(self.stream_video())

    async def disconnect(self, close_code):
        print("WebSocket connection closed")
        self.active_connections.discard(self)
        self.streaming = False

    async def receive(self, text_data):
        data = json.loads(text_data)
        if data.get('action') == 'set_class_ids':
            self.current_class_ids = data.get('class_ids', [])
            print(f"Class IDs set to {self.current_class_ids}")
        elif data.get('action') == 'set_confidence_threshold':
            new_threshold = data.get('confidence_threshold')
            if isinstance(new_threshold, (int, float)) and 0 <= new_threshold <= 1:
                global CONFIDENCE_THRESHOLD
                CONFIDENCE_THRESHOLD = new_threshold
                print(f"Confidence threshold set to {CONFIDENCE_THRESHOLD}")

    def calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2
        inter_x1 = max(x1, x1_b)
        inter_y1 = max(y1, y1_b)
        inter_x2 = min(x2, x2_b)
        inter_y2 = min(y2, y2_b)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area != 0 else 0

    def assign_tracking_ids(self, results, frame_id, current_frame):
        detections = []
        current_time = time.time()
        current_alert_keys = set()

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())

            if confidence > CONFIDENCE_THRESHOLD and (not self.current_class_ids or class_id in self.current_class_ids):
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': model.names[class_id]
                }

                # Assign tracking ID
                assigned = False
                for track_id, tracked_data in list(self.tracked_objects.items()):
                    tracked_bbox = tracked_data[-1][1]
                    iou = self.calculate_iou(detection['bbox'], tracked_bbox)
                    if iou > 0.3:
                        self.tracked_objects[track_id].append((frame_id, detection['bbox']))
                        detection['track_id'] = track_id
                        assigned = True
                        break

                if not assigned:
                    self.object_counter += 1
                    track_id = self.object_counter
                    self.tracked_objects[track_id] = deque([(frame_id, detection['bbox'])])
                    detection['track_id'] = track_id

                # Save to database
                self.save_detection_to_db(detection)

                # Violation alert logic — only for No Helmet (2) and No Safety Vest (1)
                if class_id in [1, 2]:
                    alert_key = (track_id, class_id)
                    current_alert_keys.add(alert_key)

                    if alert_key not in self.object_entry_time:
                        # First time seeing this violation
                        self.object_entry_time[alert_key] = current_time
                    else:
                        duration = current_time - self.object_entry_time[alert_key]

                        if duration > self.ALERT_TIME_THRESHOLD and alert_key not in self.alerted_objects:
                            violation_name = model.names[class_id]
                            fine = FINE_AMOUNTS.get(violation_name, 200)

                            riyadh_tz = pytz.timezone("Asia/Riyadh")
                            alert_time = datetime.now(riyadh_tz).strftime("%Y-%m-%d %H:%M:%S")

                            print(f"🚨 VIOLATION! {violation_name} | Track ID: {track_id} | Duration: {duration:.1f}s")

                            # Play sound alert
                            pygame.mixer.Sound.play(alert_sound)

                            # Build alert caption with fine
                            caption = (
                                f"🚨 PPE VIOLATION DETECTED!\n\n"
                                f"❌ Violation: {violation_name}\n"
                                f"💰 Fine Amount: ₹{fine}\n"
                                f"🆔 Person ID: {track_id}\n"
                                f"📊 Confidence: {confidence:.0%}\n"
                                f"⏱ Duration: {duration:.1f} seconds\n"
                                f"🕐 Time: {alert_time}\n\n"
                                f"⚠️ Please wear PPE immediately!"
                            )

                            # Capture current frame as photo
                            frame_copy = current_frame.copy()
                            # Draw red box on violation area in the captured photo
                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.putText(frame_copy, f"VIOLATION: {violation_name}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame_copy, f"Fine: Rs.{fine}", (x1, y2 + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            _, img_buffer = cv2.imencode('.jpg', frame_copy)
                            image_bytes = img_buffer.tobytes()

                            # Send photo + caption to Telegram
                            asyncio.create_task(
                                asyncio.to_thread(send_telegram_photo, image_bytes, caption)
                            )

                            # Mark as alerted so we don't spam
                            self.alerted_objects.add(alert_key)

                detections.append(detection)

        # Clean up entry times for objects no longer visible
        keys_to_remove = [k for k in self.object_entry_time if k not in current_alert_keys]
        for k in keys_to_remove:
            del self.object_entry_time[k]
            self.alerted_objects.discard(k)  # Allow re-alert if they come back

        return detections

    def save_detection_to_db(self, detection):
        class_id = detection["class_id"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        track_id = detection["track_id"]
        bbox = str(detection["bbox"])
        riyadh_tz = pytz.timezone("Asia/Riyadh")
        timestamp = datetime.now(riyadh_tz).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''INSERT INTO detections (class_id, class_name, confidence, track_id, bbox, timestamp)
                      VALUES (?, ?, ?, ?, ?, ?)''',
                       (class_id, class_name, confidence, track_id, bbox, timestamp))
        conn.commit()
        print(f"Detection saved: {class_name} (ID: {track_id}, Conf: {confidence:.2f}, Time: {timestamp})")

    async def stream_video(self):
        frame_id = 0
        while self.active_connections:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            results = model(frame, verbose=False)
            detections = self.assign_tracking_ids(results, frame_id, frame)

            # Draw bounding boxes on the frame
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                track_id = detection['track_id']

                # Red for violations, Green for safe
                if detection['class_id'] in [1, 2]:
                    color = (0, 0, 255)   # Red = Violation
                    status = "VIOLATION"
                else:
                    color = (0, 255, 0)   # Green = Safe
                    status = "SAFE"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                label = f"{class_name} {confidence:.0%}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            message = json.dumps({'frame': frame_data, 'detected_objects': detections})

            tasks = [connection.send(text_data=message) for connection in self.active_connections]
            await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(0.02)
            frame_id += 1

        print("Video stream ended")

    def close_database(self):
        cursor.close()
        conn.close()
        print("Database connection closed")
