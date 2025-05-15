# waste_classifier_webcam.py
# Webcam classifier for multi-class waste detection

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import mobilenet_v2
import cv2
import numpy as np
from PIL import Image
import time

# 1. SYSTEM SETUP
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n Initializing waste classifier on {device}...")

# 2. MODEL CONFIGURATION
class_names = ['cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic']
num_classes = len(class_names)
CONFIDENCE_THRESHOLD = 0.65
SOUND_ENABLED = False

# Generate distinct colors for each class
np.random.seed(42)
class_colors = [tuple(np.random.randint(100, 255, 3).tolist()) for _ in class_names]

# 3. SOUND SYSTEM
try:
    import sounddevice as sd
    import soundfile as sf

    def generate_beep(freq=1000, duration=0.3, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * freq * t) * 0.5
        return wave.astype(np.float32)

    # Generate one beep per class (descending pitch)
    beeps = [generate_beep(1000 - i * 100) for i in range(num_classes)]

    def play_sound(class_index):
        if 0 <= class_index < len(beeps):
            sd.play(beeps[class_index], samplerate=44100)
            sd.wait()

except Exception as e:
    print(f" Sound system unavailable: {str(e)}")
    SOUND_ENABLED = False

# 4. LOAD MODEL
def load_model(model_path):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    return model

# 5. PROCESS FRAME
def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(Image.fromarray(rgb).resize((224, 224)))
    return tensor.unsqueeze(0).to(device)

def get_detection_zone(frame):
    h, w = frame.shape[:2]
    size = min(h, w) // 2
    x1, y1 = (w - size) // 2, (h - size) // 2
    x2, y2 = x1 + size, y1 + size
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# 6. RUN CLASSIFIER
def run_classifier(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Webcam unavailable")
        return

    print("\n System ready - Point waste items at the center of your webcam")
    print(f" Sound feedback: {'ON' if SOUND_ENABLED else 'OFF'}")
    print(f" Confidence threshold: {CONFIDENCE_THRESHOLD*100}%")

    last_sound_time = 0
    sound_cooldown = 2.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame = cv2.flip(frame, 1)
        roi, (x1, y1, x2, y2) = get_detection_zone(frame)

        if roi.size > 0:
            with torch.no_grad():
                outputs = model(process_frame(roi))
                probs = torch.softmax(outputs, dim=1)[0]
                confidence, pred_idx = torch.max(probs, 0)
                confidence = confidence.item()
                pred_idx = pred_idx.item()

                if confidence > CONFIDENCE_THRESHOLD:
                    color = class_colors[pred_idx]
                    label = f"{class_names[pred_idx].upper()} ({confidence*100:.1f}%)"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, label, (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                    if SOUND_ENABLED and (current_time - last_sound_time) > sound_cooldown:
                        play_sound(pred_idx)
                        last_sound_time = current_time
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)

        cv2.putText(frame, "Place waste in center box", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 255), 2)

        cv2.imshow('Waste Classifier', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n System shutdown")

# 7. ENTRY POINT
if __name__ == '__main__':
    try:
        import sounddevice
        import soundfile
    except ImportError:
        print("\n Installing required sound packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice", "soundfile"])
        print(" Sound packages installed")

    model = load_model("/Users/sonwabise/Documents/Anaconda/Python/venv/Multi Class classification/model_evaluation/waste_classifier_final.pth")  # Make sure this matches your trained model
    run_classifier(model)
