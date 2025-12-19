import gradio as gr
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

detector = MTCNN()
model = load_model("deepfake_model.h5")

def analyze_video(video):
    cap = cv2.VideoCapture(video)
    scores = []
    count = 0

    while cap.isOpened() and count < 15:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces:
            x, y, w, h = faces[0]['box']
            face = rgb[y:y+h, x:x+w]
            face = cv2.resize(face, (224,224))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            score = model.predict(face)[0][0]
            scores.append(score)
            count += 1

    cap.release()

    if len(scores) == 0:
        return "No face detected ❌"

    avg = np.mean(scores)
    if avg > 0.5:
        return f"FAKE VIDEO ❌ (Confidence {avg:.2f})"
    else:
        return f"REAL VIDEO ✅ (Confidence {1-avg:.2f})"

gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(),
    outputs="text",
    title="DeepShield AI – Fake Video Detector",
    description="Online AI system to detect AI-generated fake videos"
).launch()
