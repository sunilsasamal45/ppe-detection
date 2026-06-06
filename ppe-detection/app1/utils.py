def load_model(model_path):
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

def preprocess_frame(frame):
    # Resize frame to the expected input size for the model
    return cv2.resize(frame, (640, 640))

def postprocess_detections(detections, confidence_threshold):
    results = []
    for *box, conf, cls in detections.xyxy[0]:
        if conf >= confidence_threshold:
            results.append({
                'bounding_box': box.tolist(),
                'confidence': conf.item(),
                'class_id': int(cls.item())
            })
    return results

def play_alert_sound(alert_sound_path):
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(alert_sound_path)
    pygame.mixer.music.play()