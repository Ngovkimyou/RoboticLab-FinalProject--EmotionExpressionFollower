# 📌 README.md — Emotion Controlled Robot using Raspberry Pi + TensorFlow Lite
## 🤖 Overview

This project implements a real-time emotion recognition robot that reacts to human facial expressions.
Using a Raspberry Pi camera, TensorFlow Lite model, OpenCV face detection, and a custom robot controller, the robot performs physical movements based on 5 detected emotions:

Happy → Move Forward

Sad → Move Backward

Angry → Turn Right

Surprise → Turn Left

Neutral → Stop

A Flask web server provides a live camera stream, along with real-time emotion and action display.

<br>  

## 🚀 Features
- Real-time facial emotion detection
- 5 emotion classes: Angry, Happy, Neutral, Sad, Surprise
- TensorFlow Lite optimized for Raspberry Pi
- Haar cascade face detection
- Movement control through serial communication
- Live camera feed on browser (MJPEG stream)
- Real-time action + emotion text overlay
- Data augmentation pipeline simulating Raspberry Pi camera conditions

<br> 

## Requirements 
Install virtual environment
```python
python3 -m venv venv
source venv/bin/activate
```

Install PySerial ( for robot control )
```python
pip install pyserial
```

Install 

## Usage  
Run the main script 
```python
python3 run_robot.py
```

Open the live camera feed
```python
http://<your-raspberrypi-ip>:5000
```
