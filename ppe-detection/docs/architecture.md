# Architecture of the PPE Detection System

## Overview
The PPE Detection System is designed to provide real-time monitoring and detection of Personal Protective Equipment (PPE) compliance using advanced technologies such as Django, YOLO (You Only Look Once), and WebSockets. The architecture is modular, allowing for easy maintenance and scalability.

## Components

### 1. Django Framework
The core of the application is built on the Django framework, which provides a robust backend for handling requests, managing databases, and serving web pages.

### 2. ASGI and WebSockets
The application utilizes ASGI (Asynchronous Server Gateway Interface) to enable real-time communication through WebSockets. This allows for low-latency video streaming and immediate feedback on PPE violations.

### 3. YOLO Object Detection
The YOLO model is integrated into the application for real-time object detection. It processes video frames to identify whether individuals are wearing the required PPE, such as helmets and safety vests.

### 4. OpenCV
OpenCV is used for video processing and manipulation. It captures live video feeds and prepares frames for the YOLO model to analyze.

### 5. Detection Log Dashboard
The dashboard provides insights into detection statistics, including:
- Total detections
- Class distribution of detected PPE
- Historical logs of detections with timestamps and confidence levels

### 6. User Interface
The frontend is built using HTML, CSS, and JavaScript, providing an interactive user interface for users to start video streaming, adjust detection settings, and view logs.

## Data Flow
1. **Video Capture**: The system captures live video from a camera using OpenCV.
2. **Frame Processing**: Each frame is sent to the YOLO model for detection.
3. **Detection Results**: The results are processed and sent back to the frontend via WebSockets.
4. **User Notifications**: If a PPE violation is detected, an audio alert is triggered, and the event is logged in the dashboard.
5. **Dashboard Updates**: The detection log dashboard updates automatically to reflect the latest statistics and logs.

## Future Enhancements
- Expand the range of detectable PPE classes.
- Improve detection speed and accuracy through model optimization.
- Implement cloud storage solutions for log data and model management.

This architecture ensures a comprehensive and efficient approach to PPE detection, enhancing workplace safety through real-time monitoring and alerts.