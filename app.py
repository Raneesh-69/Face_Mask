from __future__ import annotations

import base64
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, flash, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = BASE_DIR / "mask_recognition.h5"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "face-mask-detection")

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
model = load_model(str(MODEL_PATH), compile=False)


def classify_face(face_frame: np.ndarray) -> tuple[str, float, tuple[int, int, int]]:
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (224, 224))
    face_frame = img_to_array(face_frame)
    face_frame = preprocess_input(face_frame)

    prediction = model.predict(np.expand_dims(face_frame, axis=0), verbose=0)[0]
    mask_score, no_mask_score = prediction

    if mask_score >= no_mask_score:
        return "Mask", float(mask_score), (16, 185, 129)

    return "No Mask", float(no_mask_score), (239, 68, 68)


def process_image(image_bytes: bytes) -> tuple[str | None, float | None, str | None, str | None]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return None, None, None, "Please upload a valid image file."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None, None, None, "No face detected in the uploaded image."

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_frame = frame[y : y + h, x : x + w]
    label, confidence, color = classify_face(face_frame)

    text = f"{label}: {confidence * 100:.2f}%"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        text,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None, None, None, "Could not process the uploaded image."

    encoded_image = base64.b64encode(buffer).decode("utf-8")
    image_data = f"data:image/jpeg;base64,{encoded_image}"

    return label, confidence, image_data, None


@app.route("/", methods=["GET", "POST"])
def index():
    result_label = None
    confidence = None
    image_data = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename.strip() == "":
            flash("Please choose an image to upload.", "error")
        else:
            result_label, confidence, image_data, error_message = process_image(uploaded_file.read())
            if error_message:
                flash(error_message, "error")

    return render_template(
        "index.html",
        result_label=result_label,
        confidence=confidence,
        image_data=image_data,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)