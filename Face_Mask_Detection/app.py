from __future__ import annotations

import atexit
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = BASE_DIR / "mask_recognition.h5"

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
model = load_model(str(MODEL_PATH), compile=False)

camera = cv2.VideoCapture(0)
camera_lock = Lock()


def detect_and_annotate(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    face_batches = []
    face_boxes = []

    for (x, y, w, h) in faces:
        face_frame = frame[y : y + h, x : x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = preprocess_input(face_frame)
        face_batches.append(face_frame)
        face_boxes.append((x, y, w, h))

    predictions = []
    if face_batches:
        batch = np.array(face_batches, dtype="float32")
        predictions = model.predict(batch, verbose=0)

    for (box, pred) in zip(face_boxes, predictions):
        x, y, w, h = box
        mask_score, no_mask_score = pred

        if mask_score >= no_mask_score:
            label = "Mask"
            color = (16, 185, 129)
            confidence = mask_score
        else:
            label = "No Mask"
            color = (239, 68, 68)
            confidence = no_mask_score

        text = f"{label}: {confidence * 100:.2f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame


def generate_frames():
    while True:
        with camera_lock:
            ok, frame = camera.read()

        if not ok:
            fallback = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                fallback,
                "Camera not available",
                (140, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            frame = fallback
        else:
            frame = detect_and_annotate(frame)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@atexit.register
def cleanup_camera() -> None:
    with camera_lock:
        if camera.isOpened():
            camera.release()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
