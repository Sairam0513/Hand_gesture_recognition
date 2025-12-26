
✋ Hand Gesture Recognition System

A real-time hand gesture recognition application built using MediaPipe and OpenCV, capable of detecting multiple gestures using hand landmark analysis and finger counting logic.

📌 Features

Real-time hand tracking using webcam

Detects 9+ gestures

Smooth and stable gesture output

Works efficiently on CPU

Beginner & interview friendly logic

✋ Gestures Supported
Gesture	Description
✊ Fist	Stop
☝️ Point	Pointing
👍 Thumbs Up	Like / Yes
✌️ Victory	Peace
👌 OK	OK sign
🔢 Three	Three fingers
🔢 Four	Four fingers
✋ Open Palm	Open hand
👋 Wave	Hello / Bye
🧠 How It Works

MediaPipe detects 21 hand landmarks

Finger states are determined by comparing tip and joint positions

Gesture classification is done using rule-based logic

Gesture smoothing is applied to reduce noise and flickering

🛠 Tech Stack

Python

OpenCV

MediaPipe

Computer Vision

▶️ Installation & Run
pip install mediapipe opencv-python protobuf==3.20.3
python gesture_recognition.py

📷 Output

Live webcam feed

Hand landmarks drawn

Detected gesture displayed in real time

🎯 Applications

Human–Computer Interaction

Touchless interfaces

Smart systems

Assistive technologies

Gesture-controlled applications

👤 Author

Sai Ram
Instagram: Futurix 🚀
Explaining AI & Tech concepts for Techies
