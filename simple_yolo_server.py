#!/usr/bin/env python3
"""
Simple YOLO Server for Raspberry Pi using OpenCV DNN
- No ultralytics dependency required
- Uses OpenCV's built-in DNN module
- Works with basic packages only
"""
import cv2
import numpy as np
import base64
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template_string
import warnings
import webbrowser
import subprocess
import os
# import torch  # Removed for Pi 3A 32-bit compatibility

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Configuration optimized for Pi 3A 32-bit
CAMERA_INDEX = 0

# Optimized settings for Pi 3A performance
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 3  # Reduced FPS for Pi 3A stability
PI_IP = "0.0.0.0"
PI_PORT = 5000

# Detection thresholds optimized for Pi 3A
CONFIDENCE_THRESHOLD = 0.6  # Higher threshold to reduce false positives
NMS_THRESHOLD = 0.4

# Global variables
camera = None
latest_frame = None
latest_detections = []
frame_count = 0
fps = 0
last_time = time.time()
detection_running = False

# COCO class names - use the requested list (keeps earlier entries removed/normalized)
CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Person detection only - class ID 0 is 'person' in COCO
PERSON_CLASS_ID = 0

# Voice announcement settings
VOICE_ENABLED = True
last_announcement_time = {}
ANNOUNCEMENT_COOLDOWN = 5  # seconds between announcements for same person

# ===============================
# LOAD MODELS (YOLOv3-tiny + gender .pt)
# ===============================

def load_yolo_model():
    """Load YOLOv3-tiny via OpenCV DNN if available (optimized for Pi 3A)"""
    cfg = "yolov3-tiny.cfg"
    weights = "yolov3-tiny.weights"
    try:
        print("Loading YOLOv3-tiny model for Pi 3A...")
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✅ YOLOv3-tiny model loaded successfully!")
        return net
    except Exception as e:
        print(f"[INFO] YOLOv3-tiny not available ({e}). Using simple detection mode.")
        print("[INFO] To use YOLOv3-tiny, download yolov3-tiny.cfg and yolov3-tiny.weights")
        return None

def load_gender_model():
    """Gender classification disabled for Pi 3A 32-bit compatibility"""
    print("[INFO] Gender classification using heuristic method only (PyTorch not available on Pi 3A 32-bit)")
    return None

# load models at startup (will be used in detection worker)
yolo_net = load_yolo_model()
gender_model = load_gender_model()

# ===============================
# OBJECT DETECTION (YOLOv3-tiny)
# ===============================
def detect_objects_opencv(frame, net):
    """Detect objects using YOLOv3-tiny (OpenCV DNN). Returns detections and annotated frame."""
    if net is None:
        # fallback to simple contours-based detector
        return detect_objects_simple(frame)

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except:
        # older OpenCV might return as list of lists
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            if len(scores) == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    detections = []

    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            label = f"{CLASSES[class_id] if class_id < len(CLASSES) else class_id}: {confidence:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.append({
                "class": CLASSES[class_id] if class_id < len(CLASSES) else str(class_id),
                "confidence": round(confidence, 2),
                "bbox": [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            })

    return detections, frame

# ===============================
# GENDER CLASSIFICATION (PyTorch)
# ===============================
def predict_gender(face_crop):
    """Predict gender using heuristic method only (PyTorch not available on Pi 3A 32-bit)."""
    # Always return Unknown since PyTorch model is not available
    return "Unknown"

def classify_gender(frame, bbox):
    """Gender classification using heuristic method (optimized for Pi 3A)"""
    x, y, w, h = bbox
    
    # Extract person region
    person_region = frame[y:y+h, x:x+w]
    
    if person_region.size == 0:
        return "Unknown"
    
    # Improved heuristic method for Pi 3A
    try:
        # Focus on upper body region for clothing analysis
        upper_region = person_region[:int(h*0.4), :]
        if upper_region.size > 0:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
            
            # Analyze color characteristics
            avg_color = np.mean(upper_region, axis=(0, 1))
            brightness = np.mean(avg_color)
            
            # Analyze hue distribution
            hue_values = hsv[:, :, 0].flatten()
            hue_mean = np.mean(hue_values)
            hue_std = np.std(hue_values)
            
            # Simple classification based on brightness and color patterns
            if brightness < 80:
                return "Male"  # Darker clothing
            elif brightness > 180:
                return "Female"  # Brighter clothing
            elif hue_std > 30:  # More varied colors
                return "Female"
            else:
                return "Male"
    except Exception as e:
        print(f"[WARN] Gender classification failed: {e}")
    
    return "Unknown"

def announce_person(gender, bbox):
    """Announce when a person is detected"""
    global last_announcement_time, ANNOUNCEMENT_COOLDOWN
    
    if not VOICE_ENABLED:
        return
    
    # Create a unique ID for this person based on position
    x, y, w, h = bbox
    person_id = f"{x//50}_{y//50}"  # Grid-based ID to avoid duplicate announcements
    
    current_time = time.time()
    
    # Check if we've announced this person recently
    if person_id in last_announcement_time:
        if current_time - last_announcement_time[person_id] < ANNOUNCEMENT_COOLDOWN:
            return
    
    # Update last announcement time
    last_announcement_time[person_id] = current_time
    
    # Create announcement text
    if gender == "Male":
        message = "Male person detected"
    elif gender == "Female":
        message = "Female person detected"
    else:
        message = "Person detected"
    
    print(f"🔊 ANNOUNCING: {message}")
    
    # Use espeak for text-to-speech (install with: sudo apt install espeak)
    try:
        subprocess.Popen(['espeak', '-s', '150', '-v', 'en', message], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("⚠️ espeak not installed. Install with: sudo apt install espeak")
    except Exception as e:
        print(f"⚠️ Voice announcement failed: {e}")

def detect_objects_simple(frame):
    """Simple object detection using OpenCV (optimized for Pi 3A)"""
    detections = []
    
    try:
        # Convert to grayscale for better performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and draw rectangles around detected objects
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Higher threshold for Pi 3A to reduce false positives
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio to detect person-like objects
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 3.0:  # Person-like aspect ratio
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detections.append({
                        'class': 'person',
                        'confidence': 0.7,
                        'bbox': [x, y, x + w, y + h]
                    })
    except Exception as e:
        print(f"[WARN] Simple detection failed: {e}")
    
    return detections, frame

def initialize_camera():
    """Initialize camera"""
    global camera
    print("Initializing camera...")
    # Try V4L2 backend first (better for Pi cameras)
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    
    # If V4L2 fails, try default backend
    if not camera.isOpened():
        print("[INFO] V4L2 backend failed, trying default backend...")
        camera = cv2.VideoCapture(CAMERA_INDEX)
    
    if not camera.isOpened():
        print("[ERROR] Cannot open camera")
        return False
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    # Additional settings for Pi camera stability
    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    except:
        pass  # Some properties might not be supported
    
    print("[OK] Camera initialized")
    return True

def detection_worker():
    """Background thread for continuous detection"""
    global latest_frame, latest_detections, frame_count, fps, last_time, detection_running, yolo_net
    
    print("Starting detection worker...")
    detection_running = True
    
    # Use already-loaded yolo_net (loaded at module import)
    net = yolo_net
    
    frame_delay = 1.0 / TARGET_FPS
    
    while detection_running:
        loop_start = time.time()
        
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue
        
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("[WARNING] Failed to capture frame")
            time.sleep(0.1)
            continue
        
        # Run detection
        try:
            detections, annotated = detect_objects_opencv(frame.copy(), net)
            
            # Run gender detection for people (heuristic method only)
            for det in detections:
                if det.get("class") == "person":
                    x1, y1, x2, y2 = det.get("bbox", [0,0,0,0])
                    # ensure ints and bounds
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(annotated.shape[1]-1, x2); y2 = min(annotated.shape[0]-1, y2)
                    if x2 > x1 and y2 > y1:
                        # Use heuristic gender classification (PyTorch not available on Pi 3A)
                        gender = classify_gender(annotated, [x1, y1, x2-x1, y2-y1])
                        # annotate on frame and the detection record
                        cv2.putText(annotated, gender, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        det['gender'] = gender
                        
                        # Announce person detection
                        announce_person(gender, [x1, y1, x2-x1, y2-y1])
                    else:
                        det['gender'] = "Unknown"
            
            # Add FPS counter and stats
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = current_time
            
            cv2.putText(annotated, f'FPS: {fps}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f'Objects: {len(detections)}', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f'Time: {datetime.now().strftime("%H:%M:%S")}', (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Store latest frame and detections
            latest_frame = annotated
            latest_detections = detections
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
        
        # Maintain target FPS
        elapsed = time.time() - loop_start
        sleep_time = max(0, frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Pi YOLO Server</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            h1 {
                text-align: center;
                color: #fff;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                margin-bottom: 30px;
            }
            .status {
                background: rgba(0, 255, 0, 0.2);
                border: 2px solid #00ff00;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                margin: 20px 0;
                font-weight: bold;
                font-size: 18px;
            }
            .button {
                display: inline-block;
                padding: 12px 24px;
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                margin: 10px 5px;
                transition: all 0.3s ease;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }
            .button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            }
            .actions {
                text-align: center;
                margin: 30px 0;
            }
            .info-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👥 People Detection Server</h1>
            
            <div class="status">
                ✓ Server Running - People Detection Active
            </div>
            
            <div class="info-card">
                <h3>📊 Server Information</h3>
                <p><strong>Detection:</strong> People Only (Men/Women)</p>
                <p><strong>Gender Classification:</strong> Heuristic Color Analysis (Pi 3A Optimized)</p>
                <p><strong>Camera:</strong> Built-in Webcam</p>
                <p><strong>Resolution:</strong> 320x240</p>
                <p><strong>Target FPS:</strong> 3</p>
            </div>
            
            <div class="actions">
                <a href="/stream" class="button">📺 View Live Stream</a>
                <a href="/api/detections" class="button">🔍 API Endpoint</a>
                <a href="/health" class="button">❤️ Health Check</a>
                <button onclick="toggleVoice()" class="button" id="voiceButton">🔊 Voice: ON</button>
            </div>
            
            <script>
            function toggleVoice() {
                fetch('/toggle_voice', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    const button = document.getElementById('voiceButton');
                    button.textContent = data.voice_enabled ? '🔊 Voice: ON' : '🔇 Voice: OFF';
                    alert(data.message);
                });
            }
            </script>
            
            <div class="info-card">
                <h3>📖 How to Use</h3>
                <ol>
                    <li><strong>Live Stream:</strong> Click "View Live Stream" to see detection</li>
                    <li><strong>API Access:</strong> Use /api/detections for programmatic access</li>
                    <li><strong>Health Check:</strong> Monitor server status with /health</li>
                    <li><strong>PC Access:</strong> Access from PC using Pi's IP address on port 5000</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/stream')
def stream_viewer():
    """Live stream viewer page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Detection Stream</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: #1a1a1a;
                font-family: Arial, sans-serif;
                color: white;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #00ff00;
                text-shadow: 0 0 15px #00ff00;
                margin-bottom: 30px;
            }
            .stream-container {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 0 30px rgba(0,255,0,0.4);
                margin: 20px 0;
                text-align: center;
            }
            #stream {
                width: 100%;
                max-width: 800px;
                height: auto;
                border-radius: 10px;
                border: 2px solid #00ff00;
            }
            .controls {
                background: #333;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #00ff00;
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 10px;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .back-button {
                display: inline-block;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px;
                transition: background 0.3s;
            }
            .back-button:hover {
                background: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Live Detection Stream</h1>
            
            <div class="stream-container">
                <img id="stream" src="/video_feed" />
            </div>
            
            <div class="controls">
                <p><span class="status-indicator"></span>LIVE - Streaming from Raspberry Pi Camera</p>
                <p>Optimized object detection for Pi 3A 32-bit</p>
                <a href="/" class="back-button">← Back to Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    '''

def generate_stream():
    """Generate MJPEG stream from latest frames"""
    global latest_frame
    
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint"""
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """API endpoint to get current detections"""
    global latest_detections, fps
    
    return jsonify({
        'success': True,
        'detections': latest_detections,
        'count': len(latest_detections),
        'fps': fps,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    global camera, fps, VOICE_ENABLED
    
    camera_status = "connected" if camera and camera.isOpened() else "disconnected"
    
    return jsonify({
        'status': 'ok',
        'model': 'opencv_dnn',
        'mode': 'live_detection',
        'camera': camera_status,
        'fps': fps,
        'voice_enabled': VOICE_ENABLED,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/toggle_voice', methods=['POST'])
def toggle_voice():
    """Toggle voice announcements on/off"""
    global VOICE_ENABLED
    
    VOICE_ENABLED = not VOICE_ENABLED
    status = "enabled" if VOICE_ENABLED else "disabled"
    
    return jsonify({
        'success': True,
        'voice_enabled': VOICE_ENABLED,
        'message': f"Voice announcements {status}"
    })

def cleanup():
    """Cleanup function"""
    global camera, detection_running
    
    print("\nShutting down...")
    detection_running = False
    
    if camera:
        camera.release()
    
    print("Cleanup complete.")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🍓 Simple Pi Detection Server")
    print("="*60)
    
    # Initialize camera
    if not initialize_camera():
        print("[ERROR] Failed to initialize camera")
        exit(1)
    
    # Start detection thread
    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()
    
    print(f"Server starting on {PI_IP}:{PI_PORT}")
    print(f"Web interface: http://[PI_IP]:{PI_PORT}")
    print(f"Live stream: http://[PI_IP]:{PI_PORT}/stream")
    print(f"API endpoint: http://[PI_IP]:{PI_PORT}/api/detections")
    print("="*60 + "\n")
    
    # Auto-open browser after 1.5 seconds
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PI_PORT}/stream")).start()
    
    try:
        app.run(host=PI_IP, port=PI_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()
