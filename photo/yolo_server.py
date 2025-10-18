import os
# Fix for PyTorch 2.6 - must be set BEFORE importing torch
os.environ['TORCH_WEIGHTS_ONLY'] = '0'

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load YOLO model (change to your model)
print("Loading YOLO model...")
try:
    # Try loading with the environment variable set
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative method...")
    # Fallback: use weights_only=False in torch.load
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get image from request
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO detection
        results = model(img, imgsz=640, verbose=False)
        
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
        
        # Optionally create annotated image
        annotated = results[0].plot()
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'yolov8n'})

@app.route('/')
def dashboard():
    return '''
    <html>
    <head>
        <title>YOLO Detection Server</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .status { color: green; font-weight: bold; }
            .info { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🚀 YOLO Detection Server</h1>
        <p class="status">✓ Server is running</p>
        <div class="info">
            <h3>Server Information</h3>
            <p><strong>Model:</strong> YOLOv8n (nano)</p>
            <p><strong>Status:</strong> Ready to receive images</p>
            <p><strong>Endpoint:</strong> POST /detect</p>
        </div>
        <h3>Quick Test</h3>
        <p><a href="/health">Check Server Health</a></p>
        <h3>Next Steps</h3>
        <ol>
            <li>Make sure this server keeps running</li>
            <li>Note your PC's IP address: 192.168.1.107</li>
            <li>Configure your Raspberry Pi to send images here</li>
        </ol>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 YOLO Detection Server")
    print("="*50)
    print("Status: Ready")
    print("Address: http://0.0.0.0:5000")
    print("PC IP: 192.168.1.107")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
