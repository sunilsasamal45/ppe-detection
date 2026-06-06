from django.shortcuts import render
from django.http import JsonResponse
from .models import DetectionLog

def index(request):
    return render(request, 'detection/index.html')

def get_detection_logs(request):
    logs = DetectionLog.objects.all().order_by('-timestamp')[:10]
    data = [
        {
            'id': log.id,
            'class_id': log.class_id,
            'class_name': log.class_name,
            'confidence': log.confidence,
            'track_id': log.track_id,
            'bounding_box': log.bounding_box,
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        }
        for log in logs
    ]
    return JsonResponse(data, safe=False)