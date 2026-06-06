from rest_framework import serializers
from .models import DetectionLog

class DetectionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionLog
        fields = ['id', 'class_id', 'class_name', 'confidence', 'track_id', 'bounding_box', 'timestamp']