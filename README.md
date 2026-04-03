# 😷 Face Mask Recognition System

A real-time face mask detection system built using **OpenCV**, **TensorFlow/Keras**, and **Flask**. This application detects faces from a webcam feed and classifies whether the person is wearing a mask or not, displaying the result directly in a web interface.

---

---

##🚀 Live Demo

🔗 Portfolio Website: https://raneesh-portfolio.netlify.app/

---

---

## 🚀 Features

- 🎥 Real-time face detection using Haar Cascade
- 🧠 Deep learning model for mask classification
- 🌐 Live video streaming in browser using Flask
- 📦 Lightweight and easy to deploy
- 🔒 Thread-safe camera handling

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **TensorFlow / Keras**
- **Flask**
- **NumPy**

---

## 📂 Project Structure

```
├── app.py
├── mask_recognition.h5
├── haarcascade_frontalface_default.xml
├── templates/
│   └── index.html
└── README.md
```

---

#

## 🧠 How It Works

1. Webcam captures live video
2. OpenCV detects faces using Haar Cascade
3. Each face is preprocessed and passed to the model
4. Model predicts:
   - Mask 😷
   - No Mask ❌

5. Bounding boxes and labels are displayed on screen

---

## 📸 Output

- Green box → Mask detected
- Red box → No mask detected
- Confidence score shown above face

---

## 🔌 API Endpoints

| Endpoint      | Description         |
| ------------- | ------------------- |
| `/`           | Web interface       |
| `/video_feed` | Live video stream   |
| `/health`     | Server health check |

---

## 🧹 Cleanup

The application safely releases the camera resource when stopped to prevent hardware issues.

---

🤝 Connect With Me
I’m always open to collaborations, internships, and project opportunities.

📌 GitHub: https://github.com/Raneesh-69
📌 LinkedIn: https://www.linkedin.com/in/pitamber-joga-79656a351
📌 Email: prjoga9@gmail.com
