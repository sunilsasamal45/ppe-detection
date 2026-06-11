# 📘 Complete Project Notes
# Real-Time PPE Detection with Django and YOLOv8

---

## 📌 TABLE OF CONTENTS
1. What is this Project?
2. What is PPE?
3. Technologies Used
4. What is YOLO?
5. What is YOLOv8n?
6. Model Architecture (Deep Level)
7. Preprocessing Steps
8. How Detection Works (Step by Step)
9. What is Django?
10. What is Django Channels & WebSocket?
11. What is Daphne?
12. Project Folder Structure
13. File-by-File Explanation
14. Database Explanation
15. Telegram Alert System
16. How to Run the Project
17. How the Frontend Works
18. Full Data Flow (Zero to Deep)
19. Common Errors and Fixes

---

## 1. 🔍 What is this Project?

This is a **Real-Time PPE (Personal Protective Equipment) Detection System**.

It uses a **webcam** to capture live video, runs an **AI model (YOLOv8)** to detect
whether workers are wearing safety equipment like helmets and safety vests,
and sends **alerts** (sound + Telegram message) when violations are detected.

### What it does:
- Opens your webcam in real-time
- Detects helmets, no-helmet, safety vest, no-vest, shoes
- Draws colored boxes around detected objects
  - 🟢 Green = Safe (Helmet, Safety Vest)
  - 🔴 Red = Violation (No Helmet, No Vest)
- Saves all detections to a database
- Plays an alert sound for violations
- Sends a Telegram message for violations
- Shows detection logs with statistics

---

## 2. 🦺 What is PPE?

PPE = Personal Protective Equipment

Equipment workers must wear for safety:
- **Helmet** = protects head from injury
- **Safety Vest** = makes worker visible
- **Shoes** = protects feet

In construction sites, factories, and warehouses,
not wearing PPE can cause serious accidents.
This system automatically monitors and alerts supervisors.

---

## 3. 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10 | Main programming language |
| Django 5.0.7 | Web framework (handles URLs, views, templates) |
| Django Channels 4.0 | WebSocket support for real-time streaming |
| Daphne 4.1.2 | ASGI server (runs Django with WebSocket) |
| YOLOv8n (Ultralytics) | AI model for object detection |
| OpenCV | Captures webcam frames, draws boxes |
| PyTorch 2.2.2 | Deep learning framework (runs YOLOv8) |
| Pygame 2.6.0 | Plays alert sound |
| SQLite | Database to store detections |
| Telegram Bot API | Sends violation alerts to phone |
| HTML/CSS/JavaScript | Frontend interface |

---

## 4. 🎯 What is YOLO?

YOLO = **You Only Look Once**

### Old way (before YOLO):
- Scan image many times at different locations
- Very slow (cannot do real-time)

### YOLO way:
- Look at the entire image **only once**
- Divide image into a grid
- Each grid cell predicts bounding boxes and class probabilities
- All predictions happen in a **single forward pass**
- Very fast → perfect for real-time video

### YOLO versions:
- YOLOv1 (2016) → first version
- YOLOv3 → popular, good accuracy
- YOLOv5 → widely used
- **YOLOv8 (2023)** → latest, best accuracy + speed ✅ (used in this project)

---

## 5. 🧠 What is YOLOv8n?

YOLOv8 has 5 sizes:

| Model | Parameters | Speed | Accuracy |
|-------|-----------|-------|---------|
| YOLOv8n (Nano) | 3.2M | Fastest | Good |
| YOLOv8s (Small) | 11.2M | Fast | Better |
| YOLOv8m (Medium) | 25.9M | Medium | Good |
| YOLOv8l (Large) | 43.7M | Slow | Very Good |
| YOLOv8x (XLarge) | 68.2M | Slowest | Best |

This project uses **YOLOv8n** (Nano):
- Only **3,011,823 parameters**
- **225 layers**
- **8.2 GFLOPs** computation
- Best for real-time webcam on normal computers
- Trained on custom PPE dataset (saved as best.pt)

---

## 6. 🏗️ Model Architecture (Deep Level)

YOLOv8n has 3 main parts:

### Part 1: BACKBONE (Feature Extractor)
Extracts important features from the image.

```
Input Image (640x640x3)
        ↓
Conv Layer (64 filters, 3x3, stride 2)  → 320x320
        ↓
Conv Layer (128 filters, 3x3, stride 2) → 160x160
        ↓
C2f Block (3 repeats, 128 channels)     → feature extraction
        ↓
Conv Layer (256 filters, stride 2)      → 80x80
        ↓
C2f Block (6 repeats, 256 channels)
        ↓
Conv Layer (512 filters, stride 2)      → 40x40
        ↓
C2f Block (6 repeats, 512 channels)
        ↓
Conv Layer (1024 filters, stride 2)     → 20x20
        ↓
C2f Block (3 repeats, 1024 channels)
        ↓
SPPF (Spatial Pyramid Pooling Fast)     → multi-scale features
```

**C2f Block** = Cross Stage Partial with 2 convolutions
- Improves gradient flow during training
- Faster than older CSP blocks

**SPPF** = Spatial Pyramid Pooling Fast
- Captures features at multiple scales
- Handles objects of different sizes

### Part 2: NECK (Feature Pyramid Network)
Combines features from different scales to detect both small and large objects.

```
Large features (20x20) ──→ Upsample ──→ 40x40
                                           + Concat with backbone 40x40
                                           ↓ C2f
                                           → Upsample → 80x80
                                                          + Concat with backbone 80x80
                                                          ↓ C2f (small object features)
```

This is called **FPN + PAN** (Feature Pyramid Network + Path Aggregation Network)

### Part 3: HEAD (Detection)
Takes features from 3 scales and outputs predictions.

```
Scale 1: 80x80 → detects SMALL objects
Scale 2: 40x40 → detects MEDIUM objects
Scale 3: 20x20 → detects LARGE objects
        ↓
Detect Layer
        ↓
Output: [x, y, w, h, confidence, class_scores] for each box
```

### Final Output per detection:
- **x, y** = center of bounding box
- **w, h** = width and height of box
- **confidence** = how sure the model is (0.0 to 1.0)
- **class** = which PPE class (0-4)

---

## 7. 🔄 Preprocessing Steps

Every webcam frame goes through these steps before entering the model:

### Step 1: Capture Frame
```python
ret, frame = video_capture.read()
# frame shape: (480, 640, 3) → Height x Width x BGR channels
```

### Step 2: Resize
```
640x480 → resized to 640x640 (with letterboxing/padding)
```

### Step 3: Color Convert
```
BGR (OpenCV format) → RGB (YOLO format)
```

### Step 4: Normalize
```
Pixel values: 0-255 → 0.0 to 1.0
(divide each pixel by 255)
```

### Step 5: Add Batch Dimension
```
Shape: (640, 640, 3) → (1, 3, 640, 640)
Format: [batch, channels, height, width]
```

### Step 6: Send to Model
```python
results = model(frame, verbose=False)
```

### Step 7: Post-processing (NMS)
**NMS = Non-Maximum Suppression**
- Model generates many overlapping boxes
- NMS keeps only the best box for each object
- Removes duplicates using IoU (Intersection over Union) threshold

### Step 8: Filter by Confidence
```python
if confidence > CONFIDENCE_THRESHOLD:  # 0.3 = 30%
    # keep this detection
```

---

## 8. 🔍 How Detection Works (Step by Step)

```
Webcam
  ↓
Frame captured (30 FPS)
  ↓
YOLOv8n processes frame
  ↓
Raw detections (many boxes)
  ↓
NMS filters duplicates
  ↓
Confidence filter (> 0.3)
  ↓
Class filter (if user selected specific classes)
  ↓
IoU Tracking (assign Track IDs to each person)
  ↓
Draw boxes on frame (Green=safe, Red=violation)
  ↓
Save detection to SQLite database
  ↓
Check if violation lasted > 3 seconds
  ↓ YES
Play alert sound (pygame) + Send Telegram message
  ↓
Encode frame as JPEG → Base64
  ↓
Send via WebSocket to browser
  ↓
Browser displays frame in <img> tag
```

### What is IoU Tracking?
IoU = Intersection over Union

Used to track the same person across frames:
```
IoU = (Area of overlap) / (Area of union)

If IoU > 0.3 → same object → keep same Track ID
If IoU < 0.3 → new object → assign new Track ID
```

This is a simple custom tracker (not DeepSORT or ByteTrack).

---

## 9. 🌐 What is Django?

Django is a **Python web framework** that makes building websites easy.

### Key concepts:
- **Model** = database structure (defined in models.py)
- **View** = logic of what to show (defined in views.py)
- **Template** = HTML pages (in templates/ folder)
- **URL** = routes (defined in urls.py)
- **Settings** = configuration (settings.py)

### Django handles:
- Routing URLs to the right view
- Rendering HTML templates
- Serving static files (CSS, JS)
- Database operations
- Admin panel

---

## 10. 🔌 What is Django Channels & WebSocket?

### Normal HTTP (Django default):
```
Browser → Request → Django → Response → Browser
(one-way, browser must ask every time)
```

### WebSocket (Django Channels):
```
Browser ←→ Django (two-way, always connected)
(server can push data to browser anytime)
```

WebSocket is used for the **live video stream** because:
- Video needs continuous data push (30 frames/second)
- HTTP cannot do this efficiently
- WebSocket keeps connection open and streams frames

### How it works in this project:
1. Browser connects to `ws://127.0.0.1:8000/ws/video_stream/`
2. Server continuously sends encoded video frames
3. Browser receives frames and displays them in `<img>` tag

---

## 11. 🚀 What is Daphne?

- Django's default server (`runserver`) only handles **HTTP**
- **Daphne** is an **ASGI server** that handles both HTTP and WebSocket
- ASGI = Asynchronous Server Gateway Interface
- Required for Django Channels to work

### Why not use runserver?
```
runserver → HTTP only → WebSocket returns 404 → No video stream
Daphne    → HTTP + WebSocket → Video stream works ✅
```

---

## 12. 📁 Project Folder Structure

```
Real-Time-PPE-Detection/
│
├── app1/                        ← Main Django app
│   ├── consumers.py             ← WebSocket + YOLO detection logic
│   ├── views.py                 ← HTTP views (detection logs page)
│   ├── urls.py                  ← URL routes for app1
│   ├── routing.py               ← WebSocket URL routes
│   ├── models.py                ← Database models
│   ├── admin.py                 ← Admin panel config
│   ├── best.pt                  ← Trained YOLOv8n model weights
│   ├── alert.mp3                ← Alert sound file
│   └── migrations/              ← Database migration files
│
├── ppe/                         ← Django project config
│   ├── settings.py              ← All settings (database, apps, etc.)
│   ├── urls.py                  ← Main URL routing
│   ├── asgi.py                  ← ASGI config (WebSocket routing)
│   └── wsgi.py                  ← WSGI config (for HTTP)
│
├── templates/                   ← HTML templates
│   ├── ppe_detection.html       ← Main video stream page
│   └── detection_logs.html      ← Detection history page
│
├── static/                      ← Static files
│   ├── css/style.css            ← Styling
│   └── js/script.js             ← Frontend JavaScript
│
├── manage.py                    ← Django management commands
├── requirements.txt             ← Python package list
├── db.sqlite3                   ← Django database
├── detections.db                ← Detection logs database
└── .gitignore                   ← Git ignore rules
```

---

## 13. 📄 File-by-File Explanation

### consumers.py (Most Important File)
This is the heart of the project.

```python
# 1. Loads YOLOv8 model at startup
model = YOLO('app1/best.pt')

# 2. Opens webcam
video_capture = cv2.VideoCapture(0)

# 3. VideoStreamConsumer class handles:
#    - WebSocket connect/disconnect
#    - Receiving messages from browser (class filter, confidence)
#    - Streaming video frames
#    - Running YOLO detection
#    - Tracking objects with IoU
#    - Saving to database
#    - Playing alert sound
#    - Sending Telegram alerts
```

### views.py
Handles HTTP pages (not WebSocket):
- `ppe_detection()` → renders the main video page
- `detection_list()` → renders the logs page with statistics
- `fetch_detections()` → returns JSON data for the logs

### settings.py
Key configurations:
```python
INSTALLED_APPS = ['channels', 'app1']  # Enables channels and app1
ASGI_APPLICATION = 'ppe.asgi.application'  # Points to ASGI config
CHANNEL_LAYERS = {'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}}
# InMemoryChannelLayer = no Redis needed, stores in RAM
```

### asgi.py
Routes traffic to correct handler:
```python
ProtocolTypeRouter({
    "http": Django HTTP handler,
    "websocket": WebSocket handler → VideoStreamConsumer
})
```

### script.js
Frontend logic:
- Opens WebSocket connection to server
- Receives base64 encoded frames
- Sets them as `<img>` src to display video
- Sends class filter and confidence settings to server

---

## 14. 🗄️ Database Explanation

### Two databases are used:

**db.sqlite3** → Django's default database
- Stores users, sessions, admin data
- Managed by Django migrations

**detections.db** → Custom detections database
- Stores every PPE detection
- Created manually in consumers.py

### detections table structure:
```sql
CREATE TABLE detections (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id  INTEGER,    -- 0=Helmet, 1=NoVest, 2=NoHelmet, 3=Vest, 4=Shoes
    class_name TEXT,      -- Human readable class name
    confidence REAL,      -- Detection confidence (0.0 to 1.0)
    track_id  INTEGER,    -- Unique ID for tracking same person
    bbox      TEXT,       -- Bounding box "(x1, y1, x2, y2)"
    timestamp TIMESTAMP   -- When detected (Asia/Riyadh timezone)
)
```

---

## 15. 📱 Telegram Alert System

### How it works:
1. Violation detected (No Helmet or No Vest)
2. Violation persists for **more than 3 seconds**
3. System calls `send_telegram_alert()` function
4. Function sends HTTP request to Telegram Bot API
5. Bot delivers message to your Telegram account

### Alert message format:
```
⚠️ PPE VIOLATION ALERT!
🚨 Violation: No Helmet
🆔 Track ID: 5
📊 Confidence: 0.87
🕐 Time: 2026-06-11 10:30:45
```

### Telegram Bot API URL format:
```
https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={MESSAGE}
```

### Configured in consumers.py:
```python
TELEGRAM_BOT_TOKEN = "7564093390:AAGJU5..."
TELEGRAM_CHAT_ID = "5154424717"
```

---

## 16. ▶️ How to Run the Project

### Every time you want to run:

**Step 1:** Open CMD (Command Prompt)

**Step 2:** Navigate to project folder:
```
cd "c:\Users\sunil\Favorites\Downloads\Real-Time-PPE-Detection-with-Django-and-Yolo-main (1) (2)\Real-Time-PPE-Detection-with-Django-and-Yolo-main"
```

**Step 3:** Start the server:
```
python -m daphne -b 127.0.0.1 -p 8000 ppe.asgi:application
```

**Step 4:** Open browser:
```
http://127.0.0.1:8000/
```

**To stop:** Press `Ctrl + C` in CMD

### Available pages:
| URL | Page |
|-----|------|
| http://127.0.0.1:8000/ | Live PPE Detection |
| http://127.0.0.1:8000/detections/ | Detection Logs |
| http://127.0.0.1:8000/admin/ | Django Admin |

---

## 17. 🖥️ How the Frontend Works

### ppe_detection.html page:
1. Page loads → JavaScript connects to WebSocket
2. WebSocket connects to `ws://127.0.0.1:8000/ws/video_stream/`
3. Server starts streaming base64 encoded JPEG frames
4. JavaScript decodes and shows each frame in `<img>` tag
5. Creates illusion of live video at ~30 FPS

### User controls:
- **Confidence Slider** → sets minimum detection confidence (0.1 to 1.0)
- **Class Dropdown** → select which PPE classes to detect
- **Set Classes button** → sends selected classes to server
- **Start/Stop buttons** → control streaming

---

## 18. 🔁 Full Data Flow (Zero to Deep)

```
WEBCAM
  │
  │ cv2.VideoCapture(0).read()
  ▼
RAW FRAME (480x640 BGR numpy array)
  │
  │ model(frame)
  ▼
YOLO PREPROCESSING
  ├─ Resize to 640x640
  ├─ BGR → RGB
  ├─ Normalize (÷255)
  └─ Add batch dim [1,3,640,640]
  │
  ▼
BACKBONE (Feature Extraction)
  ├─ Conv layers extract edges, shapes, textures
  ├─ C2f blocks learn complex features
  └─ SPPF captures multi-scale context
  │
  ▼
NECK (Feature Fusion)
  ├─ FPN: top-down pathway (large → small scale)
  └─ PAN: bottom-up pathway (small → large scale)
  │
  ▼
HEAD (Prediction)
  ├─ 3 detection scales (80x80, 40x40, 20x20)
  └─ Outputs: boxes + confidence + class scores
  │
  ▼
POST-PROCESSING
  ├─ NMS (remove duplicate boxes)
  └─ Confidence filter (> 0.3)
  │
  ▼
TRACKING (IoU-based)
  └─ Assign Track IDs to each detection
  │
  ▼
DECISION
  ├─ class_id in [1,2]? → START TIMER
  │   └─ Timer > 3 sec? → ALERT!
  │       ├─ Play sound (pygame)
  │       └─ Send Telegram message
  │
  ▼
SAVE TO DATABASE (detections.db)
  │
  ▼
DRAW ON FRAME
  ├─ Green box → Safe (Helmet, Vest)
  └─ Red box → Violation (No Helmet, No Vest)
  │
  ▼
ENCODE FRAME
  └─ JPEG compression → Base64 string
  │
  ▼
WEBSOCKET SEND
  └─ JSON: {frame: base64, detected_objects: [...]}
  │
  ▼
BROWSER RECEIVES
  └─ img.src = "data:image/jpeg;base64,..." → VIDEO DISPLAYED
```

---

## 19. ❗ Common Errors and Fixes

### Error 1: WebSocket 404
```
Not Found: /ws/video_stream/
```
**Fix:** You are using `python manage.py runserver` instead of Daphne.
Use: `python -m daphne -b 127.0.0.1 -p 8000 ppe.asgi:application`

---

### Error 2: Camera not opening
```
Error: Failed to capture frame
```
**Fix:** Another app is using the camera. Close Zoom, Teams, or any camera app and restart.

---

### Error 3: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Fix:** Run: `pip install -r requirements.txt`

---

### Error 4: Telegram alert not received
**Fix:**
- Make sure you started a chat with your bot on Telegram first
- Search your bot name and press Start
- Check internet connection

---

### Error 5: Model not detecting anything
**Fix:**
- Lower the confidence slider to 0.1
- Make sure no class filter is selected (or select all)
- Ensure good lighting on the webcam

---

## 📊 Summary

| Feature | Details |
|---------|---------|
| Model | YOLOv8n |
| Parameters | 3,011,823 |
| Classes | 5 (Helmet, No Helmet, Vest, No Vest, Shoes) |
| Detection Speed | ~30 FPS on CPU |
| Alert Trigger | Violation > 3 seconds |
| Alert Channels | Sound (pygame) + Telegram |
| Database | SQLite (detections.db) |
| Server | Daphne (ASGI) |
| Streaming | WebSocket |
| Framework | Django 5.0.7 |

---

*Project by: Sunil Sasmal*
*GitHub: https://github.com/sunilsasamal45/ppe-detection*
