from django.db import models

class Detection(models.Model):
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    track_id = models.IntegerField()
    bounding_box = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.class_name} (ID: {self.class_id}) - Confidence: {self.confidence:.2f}"