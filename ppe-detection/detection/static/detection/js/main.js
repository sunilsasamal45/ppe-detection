// This file contains the JavaScript code for the detection application.

document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('start-stream');
    const videoElement = document.getElementById('video-stream');
    const detectionLog = document.getElementById('detection-log');
    const confidenceThresholdInput = document.getElementById('confidence-threshold');
    const classSelector = document.getElementById('class-selector');

    const socket = new WebSocket('ws://' + window.location.host + '/ws/video/');

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDetectionLog(data);
        updateVideoStream(data);
    };

    socket.onclose = function(event) {
        console.error('WebSocket closed unexpectedly');
    };

    startButton.onclick = function() {
        socket.send(JSON.stringify({
            'action': 'start_stream',
            'confidence_threshold': confidenceThresholdInput.value,
            'class_id': classSelector.value
        }));
    };

    function updateDetectionLog(data) {
        const logEntry = document.createElement('div');
        logEntry.innerHTML = `Detected: ${data.class_name} with confidence ${data.confidence}% at ${data.timestamp}`;
        detectionLog.appendChild(logEntry);
    }

    function updateVideoStream(data) {
        videoElement.src = URL.createObjectURL(data.video_blob);
        videoElement.play();
    }
});