#!/usr/bin/env python3
"""
Unified YOLO Server for Raspberry Pi
- Runs YOLO detection directly on Pi with webcam
- Provides web interface accessible from PC
- Real-time streaming with detection overlay
"""
import os
# Fix for PyTorch 2.6 - must be set BEFORE importing torch
os.environ['TORCH_WEIGHTS_ONLY'] = '0'

import cv2
import numpy as np
import base64
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template_string
from ultralytics import YOLO
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Configuration
CAMERA_INDEX = 0  # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15  # Target FPS for detection
PI_IP = "0.0.0.0"  # Listen on all interfaces
PI_PORT = 5000

# Global variables for camera and detection
camera = None
latest_frame = None
latest_detections = []
frame_count = 0
fps = 0
last_time = time.time()
detection_running = False

# Load YOLO model
print("Loading YOLO model...")
try:
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative method...")
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")

def initialize_camera():
    """Initialize camera with optimal settings"""
    global camera
    print("Initializing camera...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    
    if not camera.isOpened():
        print("[ERROR] Cannot open camera")
        return False
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    # Set buffer size to reduce latency
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("[OK] Camera initialized")
    return True

def detection_worker():
    """Background thread for continuous detection"""
    global latest_frame, latest_detections, frame_count, fps, last_time, detection_running
    
    print("Starting detection worker...")
    detection_running = True
    
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
        
        # Run YOLO detection
        try:
            results = model(frame, imgsz=416, verbose=False)
            
            # Extract detection info
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': round(conf, 2),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            # Create annotated image
            annotated = results[0].plot()
            
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
        <title>Raspberry Pi YOLO Server</title>
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
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .info-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
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
            .stats {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
                border-left: 4px solid #00ff00;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍓 Raspberry Pi YOLO Detection Server</h1>
            
            <div class="status">
                ✓ Server Running - Live Detection Active
            </div>
            
            <div class="info-grid">
                <div class="info-card">
                    <h3>📊 Server Information</h3>
                    <div class="stats">
                        <p><strong>Model:</strong> YOLOv8n (nano)</p>
                        <p><strong>Camera:</strong> Built-in Webcam</p>
                        <p><strong>Resolution:</strong> 640x480</p>
                        <p><strong>Target FPS:</strong> 15</p>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>🌐 Network Access</h3>
                    <div class="stats">
                        <p><strong>Pi IP:</strong> 0.0.0.0 (all interfaces)</p>
                        <p><strong>Port:</strong> 5000</p>
                        <p><strong>Mode:</strong> Live Streaming</p>
                        <p><strong>Status:</strong> Ready</p>
                    </div>
                </div>
            </div>
            
            <div class="actions">
                <a href="/stream" class="button">📺 View Live Stream</a>
                <a href="/api/detections" class="button">🔍 API Endpoint</a>
                <a href="/health" class="button">❤️ Health Check</a>
            </div>
            
            <div class="info-card">
                <h3>📖 How to Use</h3>
                <ol>
                    <li><strong>Live Stream:</strong> Click "View Live Stream" to see real-time detection</li>
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
        <title>Live YOLO Detection Stream</title>
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
            <h1>🎥 Live YOLO Detection Stream</h1>
            
            <div class="stream-container">
                <img id="stream" src="{{ url_for('video_feed') }}" />
            </div>
            
            <div class="controls">
                <p><span class="status-indicator"></span>LIVE - Streaming from Raspberry Pi Camera</p>
                <p>Real-time object detection with YOLOv8n</p>
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
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Send placeholder if no frame yet
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
        'model': 'yolov8n',
        'mode': 'live_detection',
        'camera': camera_status,
        'fps': fps,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def detect_upload():
    """API endpoint for external image upload (backward compatibility)"""
    global latest_frame
    
    try:
        # Get image from request
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO detection
        results = model(img, imgsz=416, verbose=False)
        
        # Extract detection info
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': round(conf, 2),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        # Create annotated image
        annotated = results[0].plot()
        
        # Encode and return
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'annotated_image': img_base64
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
    print("🍓 Raspberry Pi YOLO Detection Server")
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
    
    try:
        app.run(host=PI_IP, port=PI_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        cleanup()
    finally:
        cleanup()
