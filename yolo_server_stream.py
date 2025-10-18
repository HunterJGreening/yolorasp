import os
# Fix for PyTorch 2.6 - must be set BEFORE importing torch
os.environ['TORCH_WEIGHTS_ONLY'] = '0'

from flask import Flask, request, jsonify, Response, render_template_string
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import warnings
import time
from datetime import datetime

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

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

# Store latest frame globally
latest_frame = None
latest_detections = []
frame_count = 0
fps = 0
last_time = time.time()

@app.route('/detect', methods=['POST'])
def detect():
    global latest_frame, latest_detections, frame_count, fps, last_time
    
    try:
        # Get image from request
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO detection
        results = model(img, imgsz=416, verbose=False)  # Smaller size for speed
        
        # Extract detection info
        detections = []
        for r in results:
            boxes = r.boxes
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
        
        # Add FPS counter
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
        
        # Store latest frame for web viewer
        latest_frame = annotated
        latest_detections = detections
        
        # Encode and return
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'annotated_image': img_base64,
            'fps': fps
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_stream():
    """Generate MJPEG stream from latest frames"""
    global latest_frame
    
    while True:
        if latest_frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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

@app.route('/stream')
def stream_viewer():
    """Web page to view the live stream"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Live Stream</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: #1a1a1a;
                font-family: Arial, sans-serif;
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #00ff00;
                text-shadow: 0 0 10px #00ff00;
            }
            .stream-container {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,255,0,0.3);
                margin: 20px 0;
            }
            #stream {
                width: 100%;
                height: auto;
                border-radius: 5px;
            }
            .info {
                background: #333;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
            .status {
                color: #00ff00;
                font-weight: bold;
            }
            .detections {
                margin-top: 10px;
                padding: 10px;
                background: #2a2a2a;
                border-left: 3px solid #00ff00;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 YOLO Live Detection Stream</h1>
            
            <div class="stream-container">
                <img id="stream" src="{{ url_for('video_feed') }}" />
            </div>
            
            <div class="info">
                <p class="status">● LIVE - Streaming from Raspberry Pi</p>
                <p>Server: 192.168.1.107:5000</p>
                <p>Model: YOLOv8n (nano)</p>
                <div class="detections">
                    <p><strong>Real-time object detection with bounding boxes</strong></p>
                    <p>Objects are detected and annotated as frames arrive from the Pi</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'model': 'yolov8n',
        'mode': 'streaming',
        'fps': fps
    })

@app.route('/')
def dashboard():
    return '''
    <html>
    <head>
        <title>YOLO Detection Server</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            .status { color: green; font-weight: bold; font-size: 18px; }
            .info { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .button { display: inline-block; padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }
            .button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <h1>🚀 YOLO Detection Server</h1>
        <p class="status">✓ Server is running (Streaming Mode)</p>
        
        <div class="info">
            <h3>Server Information</h3>
            <p><strong>Model:</strong> YOLOv8n (nano)</p>
            <p><strong>Mode:</strong> Live Streaming</p>
            <p><strong>Status:</strong> Ready to receive video</p>
            <p><strong>Address:</strong> 192.168.1.107:5000</p>
        </div>
        
        <div class="info">
            <h3>Quick Actions</h3>
            <a href="/stream" class="button">📺 View Live Stream</a>
            <a href="/health" class="button">🔍 Check Health</a>
        </div>
        
        <div class="info">
            <h3>How to Use</h3>
            <ol>
                <li>Keep this server running</li>
                <li>Start the Pi streaming client: <code>python3 stream_to_pc.py</code></li>
                <li>Click "View Live Stream" above to watch in your browser</li>
            </ol>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎥 YOLO Live Streaming Server")
    print("="*50)
    print("Status: Ready")
    print("Address: http://0.0.0.0:5000")
    print("Stream Viewer: http://192.168.1.107:5000/stream")
    print("PC IP: 192.168.1.107")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

