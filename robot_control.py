# run_robot.py (Updated — Emotion Display + Web UI + Raspberry Pi Safe)
import time
import threading
from collections import deque
from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import tensorflow as tf
from auppbot import AUPPBot

# ==========================
# CONFIGURATION
# ==========================
TFLITE_PATH = r"/home/aupp/Documents/boo/mobilenet_boo.tflite"
CAM_INDEX   = 0
FLIP        = False
SERIAL_PORT = "/dev/ttyUSB0"
BAUD        = 115200
SMOOTH_LEN  = 8

EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]
ACTION_MAP = {
    "angry":    "turn_right",
    "happy":    "forward",
    "neutral":  "stop",
    "sad":      "backward",
    "surprise": "turn_left"
}

# ==========================
# ROBOT CONTROL WRAPPERS
# ==========================
bot = None
def init_bot():
    global bot
    try:
        bot = AUPPBot(port=SERIAL_PORT, baud=BAUD)
    except Exception as e:
        print("Warning: could not open serial port for robot:", e)
        bot = None

def move_forward(speed=20):
    if bot:
        bot.motor1.forward(speed); bot.motor2.forward(speed)
        bot.motor3.forward(speed); bot.motor4.forward(speed)

def move_backward(speed=20):
    if bot:
        bot.motor1.backward(speed); bot.motor2.backward(speed)
        bot.motor3.backward(speed); bot.motor4.backward(speed)

def move_left(speed=20):
    if bot:
        bot.motor1.backward(speed); bot.motor2.backward(speed)
        bot.motor3.forward(speed); bot.motor4.forward(speed)

def move_right(speed=20):
    if bot:
        bot.motor1.forward(speed); bot.motor2.forward(speed)
        bot.motor3.backward(speed); bot.motor4.backward(speed)

def stop_robot():
    if bot:
        bot.motor1.stop(); bot.motor2.stop()
        bot.motor3.stop(); bot.motor4.stop()

def execute_action_by_name(name):
    if name == "forward":
        move_forward()
    elif name == "backward":
        move_backward()
    elif name == "turn_left":
        move_left()
    elif name == "turn_right":
        move_right()
    else:
        stop_robot()

# ==========================
# LOAD TFLITE MODEL
# ==========================
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
in_shape = input_details["shape"]
in_dtype = input_details["dtype"]

# ==========================
# OPENCV FACE DETECTOR
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================
# SHARED STATE
# ==========================
current_frame = None
current_emotion = "no_face"
_current_action_name = "stop"

# ==========================
# HELPERS
# ==========================
def preprocess_face(face_img):
    th, tw = in_shape[1], in_shape[2]

    face = cv2.resize(face_img, (tw, th))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    dtype = in_dtype
    quant = input_details.get("quantization", (0.0,0))

    if dtype == np.float32:
        arr = face_rgb.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0).astype(np.float32)

    scale, zp = quant
    float_arr = face_rgb.astype(np.float32) / 255.0
    q = np.round(float_arr / scale).astype(np.int32) + int(zp)

    if np.issubdtype(dtype, np.uint8):
        q = np.clip(q, 0, 255).astype(np.uint8)
    else:
        q = np.clip(q, -128, 127).astype(np.int8)

    return np.expand_dims(q, axis=0)

# ==========================
# INFERENCE LOOP
# ==========================
def inference_loop(cam_index=CAM_INDEX, flip=FLIP):
    global current_frame, current_emotion, _current_action_name

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("ERROR: camera not found at index", cam_index)
        return

    smooth_q = deque(maxlen=SMOOTH_LEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        # Rotate 180 degrees if camera is upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        if flip:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion = "no_face"
        display_frame = frame.copy()

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.size > 100:
                inp = preprocess_face(face_crop)
                interpreter.set_tensor(input_details["index"], inp)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details["index"])[0]

                # DEBUG — send raw outputs so we can fix mapping
                print("RAW MODEL OUTPUT:", out)

                smooth_q.append(out)
                avg = np.mean(smooth_q, axis=0)
                emotion = EMOTIONS[int(np.argmax(avg))]

            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Add emotion text onto frame
        cv2.putText(display_frame, f"Emotion: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        current_frame = display_frame
        current_emotion = emotion

        # Map emotion -> robot action
        action_name = ACTION_MAP.get(emotion, "stop")
        _current_action_name = action_name
        execute_action_by_name(action_name)

        time.sleep(0.01)

# ==========================
# FLASK SERVER
# ==========================
app = Flask(__name__)
HTML = """
<html>
<head><title>Robot Camera</title></head>
<body style="background:#111;color:#fff;text-align:center;font-family:Arial;">
<h1>Robot Live View</h1>
<img src="/video_feed" style="width:80%;border-radius:8px;border:3px solid #333;" />

<div id="emotion" style="font-size:32px;margin-top:10px;color:#0af;">
Emotion: loading...
</div>

<div id="action" style="font-size:28px;margin-top:10px;">
Action: loading...
</div>

<script>
setInterval(() => {
  fetch('/emotion').then(r=>r.text()).then(t=>document.getElementById('emotion').innerText='Emotion: '+t);
  fetch('/action').then(r=>r.text()).then(t=>document.getElementById('action').innerText='Action: '+t);
}, 300);
</script>

</body>
</html>
"""

def generate_mjpeg():
    global current_frame
    while True:
        if current_frame is None:
            time.sleep(0.01)
            continue
        
        ret, jpg = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action')
def get_action():
    return _current_action_name

@app.route('/emotion')
def get_emotion():
    return current_emotion

# ==========================
# MAIN ENTRY
# ==========================
if __name__ == '__main__':
    print("Initializing robot interface...")
    init_bot()

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    print("Starting web server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
