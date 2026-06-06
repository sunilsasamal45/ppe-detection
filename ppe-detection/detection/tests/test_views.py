from django.test import TestCase
from django.urls import reverse
from .models import DetectionLog

class DetectionLogViewTests(TestCase):

    def setUp(self):
        # Create a sample detection log entry for testing
        DetectionLog.objects.create(
            class_id=3,
            class_name='Safety Vest',
            confidence=45.44,
            track_id=3,
            bounding_box='(253, 276, 501, 480)',
            timestamp='2025-03-25 09:58:03'
        )

    def test_detection_log_view(self):
        response = self.client.get(reverse('detection_log'))  # Adjust the URL name as needed
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Safety Vest')
        self.assertTemplateUsed(response, 'detection/index.html')  # Adjust the template name as needed

    def test_detection_log_count(self):
        response = self.client.get(reverse('detection_log'))  # Adjust the URL name as needed
        self.assertContains(response, 'Total Detections: 1')  # Adjust based on your implementation

    def test_detection_log_details(self):
        response = self.client.get(reverse('detection_log'))  # Adjust the URL name as needed
        self.assertContains(response, 'Confidence: 45.44')
        self.assertContains(response, 'Track ID: 3')