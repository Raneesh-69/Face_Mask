# Live Face Mask Detection (Webcam)

Simple portfolio-ready app that detects mask/no-mask in live webcam feed.

## Features

- Live webcam stream in browser (Flask)
- Face detection using Haar Cascade
- Mask prediction using your trained `mask_recognition.h5`
- Clean and minimal UI

## Project Files

- `app.py` -> Flask app (main)
- `templates/index.html` -> UI page
- `static/style.css` -> UI styles
- `mask_recognition.h5` -> trained mask classifier
- `haarcascade_frontalface_default.xml` -> face detector

## Run Locally

1. Create and activate virtual environment (if not active)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start app:
   ```bash
   python app.py
   ```
4. Open:
   - http://127.0.0.1:5000

Press `Ctrl + C` in terminal to stop.

## Deployment Notes

Webcam access needs a machine/browser that has camera hardware and permission.

Good portfolio options:

- Deploy Flask on Render/Railway/VM
- Add screenshots or a short GIF on your portfolio page
- Mention tech stack: TensorFlow, OpenCV, Flask

## Tip for Portfolio

Record a 10-20 second screen capture showing:

- Face without mask -> red label
- Face with mask -> green label
  This clearly shows your model and real-time inference.
