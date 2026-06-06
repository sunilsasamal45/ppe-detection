#!/bin/bash

# Start the video streaming process using OpenCV and the Django server

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Start the Django server in the background
daphne -b 0.0.0.0 -p 8000 ppe.asgi:application &

# Give the server a moment to start
sleep 5

# Start the video stream
python -m detection.consumers.VideoStreamConsumer