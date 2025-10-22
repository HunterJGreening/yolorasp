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

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 10  # Lower FPS for Pi
PI_IP = "0.0.0.0"
PI_PORT = 5000

# Global variables
camera = None
latest_frame = None
latest_detections = []
frame_count = 0
fps = 0
last_time = time.time()
detection_running = False

# COCO class names (80 classes)
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_yolo_model():
    """Load YOLO model using OpenCV DNN"""
    print("Loading YOLO model...")
    
    try:
        # Try to load YOLOv8 ONNX model
        net = cv2.dnn.readNet("yolov8n.onnx")
        print("YOLOv8 ONNX model loaded successfully!")
        return net
    except:
        print("YOLOv8 model not found. Using simple detection...")
        return None

def detect_objects_opencv(frame, net):
    """Detect objects using OpenCV DNN"""
    if net is None:
        return [], frame
    
    height, width = frame.shape[:2]
    
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get detections
    outputs = net.forward()
    
    detections = []
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{CLASSES[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'class': CLASSES[class_id],
                'confidence': round(confidence, 2),
                'bbox': [x, y, x + w, y + h]
            })
    
    return detections, frame

def detect_objects_simple(frame):
    """Simple object detection using OpenCV (no YOLO model needed)"""
    detections = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Simple edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around detected objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'class': 'object',
                'confidence': 0.8,
                'bbox': [x, y, x + w, y + h]
            })
    
    return detections, frame

def initialize_camera():
    """Initialize camera"""
    global camera
    print("Initializing camera...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    
    if not camera.isOpened():
        print("[ERROR] Cannot open camera")
        return False
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("[OK] Camera initialized")
    return True

def detection_worker():
    """Background thread for continuous detection"""
    global latest_frame, latest_detections, frame_count, fps, last_time, detection_running
    
    print("Starting detection worker...")
    detection_running = True
    
    # Load model
    net = load_yolo_model()
    
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
            if net is not None:
                detections, annotated = detect_objects_opencv(frame.copy(), net)
            else:
                detections, annotated = detect_objects_simple(frame.copy())
            
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
            <h1>🍓 Simple Pi Detection Server</h1>
            
            <div class="status">
                ✓ Server Running - Basic Detection Active
            </div>
            
            <div class="info-card">
                <h3>📊 Server Information</h3>
                <p><strong>Detection:</strong> OpenCV DNN (No ultralytics needed)</p>
                <p><strong>Camera:</strong> Built-in Webcam</p>
                <p><strong>Resolution:</strong> 640x480</p>
                <p><strong>Target FPS:</strong> 10</p>
            </div>
            
            <div class="actions">
                <a href="/stream" class="button">📺 View Live Stream</a>
                <a href="/api/detections" class="button">🔍 API Endpoint</a>
                <a href="/health" class="button">❤️ Health Check</a>
            </div>
            
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
                <p>Basic object detection with OpenCV</p>
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
    global camera, fps
    
    camera_status = "connected" if camera and camera.isOpened() else "disconnected"
    
    return jsonify({
        'status': 'ok',
        'model': 'opencv_dnn',
        'mode': 'live_detection',
        'camera': camera_status,
        'fps': fps,
        'timestamp': datetime.now().isoformat()
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
