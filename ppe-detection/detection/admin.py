from django.contrib import admin
from .models import DetectionLog

@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'class_id', 'class_name', 'confidence', 'track_id', 'timestamp')
    search_fields = ('class_name',)
    list_filter = ('class_id', 'timestamp')
    ordering = ('-timestamp',)