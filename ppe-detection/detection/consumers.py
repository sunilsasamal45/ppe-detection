from channels.generic.websocket import AsyncWebsocketConsumer
import json
import cv2
import numpy as np
from .utils import detect_ppe

class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.video_capture = cv2.VideoCapture(0)  # Start capturing video from the webcam
        self.class_ids = []
        self.confidence_threshold = 0.3

        # Start the video streaming loop
        self.send_video_stream()

    async def disconnect(self, close_code):
        self.video_capture.release()  # Release the video capture object

    async def receive(self, text_data):
        data = json.loads(text_data)
        self.class_ids = data.get('class_ids', [])
        self.confidence_threshold = data.get('confidence_threshold', 0.3)

    async def send_video_stream(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            detections = detect_ppe(frame, self.class_ids, self.confidence_threshold)
            frame = self.draw_detections(frame, detections)

            # Encode the frame to send over WebSocket
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()

            await self.send(bytes_data=frame_data)

    def draw_detections(self, frame, detections):
        for detection in detections:
            class_id, confidence, bbox = detection
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {class_id} Conf: {confidence:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame