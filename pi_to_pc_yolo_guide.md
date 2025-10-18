# Raspberry Pi 3 A+ → PC YOLO Remote Processing Guide

A simple guide to set up your Raspberry Pi 3 A+ as a camera that sends images to your PC for YOLO processing.

**How it works:**
1. Pi captures images/video with camera
2. Pi sends images to your PC over network
3. PC runs YOLO detection (fast and powerful)
4. PC sends results back to Pi (optional)

---

## Part 1: Set Up Your PC for YOLO

### 1.1 Install Python and Dependencies

**On your Windows PC:**

1. Install Python 3.10+ from https://www.python.org/downloads/
2. Open Command Prompt and install YOLO:

```powershell
pip install ultralytics opencv-python pillow flask
```

### 1.2 Create YOLO Server on PC

Create a folder for your project:
```powershell
cd C:\Users\hunte\Desktop\rasp
```

Create a file called `yolo_server.py`:

```python
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load YOLO model (change to your model)
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # or 'best.pt' for your custom model
print("Model loaded!")

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
        results = model(img, imgsz=640)
        
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

if __name__ == '__main__':
    print("\n=== YOLO Server Ready ===")
    print("Waiting for images from Raspberry Pi...")
    print("Server running on: http://0.0.0.0:5000")
    print("========================\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 1.3 Find Your PC's IP Address

In Command Prompt, run:
```powershell
ipconfig
```

Look for **"IPv4 Address"** under your active network adapter.
Example: `192.168.1.100`

**Write this down!** You'll need it for the Pi.

### 1.4 Start YOLO Server

```powershell
python yolo_server.py
```

Leave this running! The server is now waiting for images.

---

## Part 2: Set Up Your Raspberry Pi

### 2.1 Install Python and Camera Tools

**In PuTTY, connect to your Pi and run:**

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-opencv
sudo apt install -y python3-picamera2
pip3 install requests pillow
```

### 2.2 Enable Camera

```bash
sudo raspi-config
```

Navigate to:
- **Interface Options** → **Camera** → **Enable**
- Reboot: `sudo reboot`

Reconnect via PuTTY after reboot.

### 2.3 Create Project Folder

```bash
cd ~
mkdir yolo_client
cd yolo_client
```

### 2.4 Create Pi Camera Client

Create a file to send images to PC:

```bash
nano capture_and_send.py
```

Paste this code (right-click in PuTTY to paste):

```python
#!/usr/bin/env python3
import requests
import cv2
from picamera2 import Picamera2
import time
import sys
import base64
import json

# Configuration
PC_IP = "192.168.1.100"  # CHANGE THIS to your PC's IP address!
PC_PORT = 5000
SERVER_URL = f"http://{PC_IP}:{PC_PORT}/detect"

def test_connection():
    """Test if PC server is reachable"""
    try:
        response = requests.get(f"http://{PC_IP}:{PC_PORT}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Connected to PC server!")
            return True
    except:
        print("✗ Cannot connect to PC server!")
        print(f"  Make sure server is running on {PC_IP}:{PC_PORT}")
        return False
    return False

def capture_image():
    """Capture image from Pi Camera"""
    print("Starting camera...")
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)  # Let camera warm up
    
    print("Capturing image...")
    frame = picam2.capture_array()
    picam2.stop()
    
    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def send_image_for_detection(image):
    """Send image to PC for YOLO detection"""
    print(f"Sending to PC at {SERVER_URL}...")
    
    # Encode image as JPEG
    _, img_encoded = cv2.imencode('.jpg', image)
    
    # Send to server
    files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        response = requests.post(SERVER_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Server returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error sending image: {e}")
        return None

def save_annotated_image(result_data, filename='result.jpg'):
    """Save the annotated image from server"""
    if 'annotated_image' in result_data:
        img_base64 = result_data['annotated_image']
        img_bytes = base64.b64decode(img_base64)
        
        with open(filename, 'wb') as f:
            f.write(img_bytes)
        print(f"✓ Saved annotated image: {filename}")

def main():
    print("\n=== Raspberry Pi YOLO Client ===")
    print(f"PC Server: {PC_IP}:{PC_PORT}\n")
    
    # Test connection first
    if not test_connection():
        sys.exit(1)
    
    # Capture image
    image = capture_image()
    print(f"✓ Image captured ({image.shape[1]}x{image.shape[0]})")
    
    # Save original
    cv2.imwrite('original.jpg', image)
    print("✓ Saved: original.jpg")
    
    # Send for detection
    result = send_image_for_detection(image)
    
    if result and result.get('success'):
        detections = result['detections']
        count = result['count']
        
        print(f"\n{'='*40}")
        print(f"DETECTED {count} OBJECTS:")
        print(f"{'='*40}")
        
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['class']} - {det['confidence']*100:.1f}%")
            print(f"   Location: {det['bbox']}")
        
        print(f"{'='*40}\n")
        
        # Save annotated image
        save_annotated_image(result, 'detected.jpg')
        
        print("✓ Detection complete!")
        print("  Transfer 'detected.jpg' to view results")
    else:
        print("✗ Detection failed!")

if __name__ == "__main__":
    main()
```

**IMPORTANT:** Edit line 10 to use your PC's IP address!

Save and exit (Ctrl+X, Y, Enter)

### 2.5 Make Script Executable

```bash
chmod +x capture_and_send.py
```

---

## Part 3: Run Detection!

### 3.1 Start Server on PC

**On your Windows PC:**
```powershell
cd C:\Users\hunte\Desktop\rasp
python yolo_server.py
```

Leave it running!

### 3.2 Capture and Detect from Pi

**In PuTTY on your Pi:**
```bash
cd ~/yolo_client
python3 capture_and_send.py
```

### 3.3 View Results

The Pi will save two images:
- `original.jpg` - The captured image
- `detected.jpg` - With YOLO detections drawn

**To transfer to your PC:**
```powershell
scp hunte@192.168.1.109:~/yolo_client/detected.jpg C:\Users\hunte\Desktop\
```

Or use WinSCP to browse and download files.

---

## Part 4: Test with USB Webcam (Alternative)

If you're using a USB webcam instead of Pi Camera:

```bash
nano capture_webcam.py
```

```python
#!/usr/bin/env python3
import requests
import cv2
import sys

PC_IP = "192.168.1.100"  # CHANGE THIS!
PC_PORT = 5000
SERVER_URL = f"http://{PC_IP}:{PC_PORT}/detect"

def capture_webcam():
    """Capture from USB webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        sys.exit(1)
    
    print("Webcam opened. Capturing...")
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    else:
        print("Error capturing frame")
        sys.exit(1)

# Use same send_image_for_detection function from above
# ... (copy the rest from capture_and_send.py)
```

---

## Part 5: Continuous Detection (Video Stream)

For real-time detection, create a loop version:

```bash
nano continuous_detect.py
```

```python
#!/usr/bin/env python3
import requests
import cv2
from picamera2 import Picamera2
import time

PC_IP = "192.168.1.100"  # CHANGE THIS!
PC_PORT = 5000
SERVER_URL = f"http://{PC_IP}:{PC_PORT}/detect"
INTERVAL = 5  # Seconds between detections

def send_image(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        response = requests.post(SERVER_URL, files=files, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                count = result['count']
                print(f"[{time.strftime('%H:%M:%S')}] Detected: {count} objects")
                for det in result['detections']:
                    print(f"  - {det['class']} ({det['confidence']*100:.0f}%)")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Starting continuous detection...")
    print(f"Server: {PC_IP}:{PC_PORT}")
    print(f"Interval: {INTERVAL} seconds")
    print("Press Ctrl+C to stop\n")
    
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)
    
    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            send_image(frame)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
```

Run it:
```bash
python3 continuous_detect.py
```

---

## Part 6: Troubleshooting

### Problem: "Cannot connect to PC server"

**Solutions:**
1. Check PC firewall - allow port 5000:
   - Windows: Settings → Firewall → Allow an app
   - Or temporarily disable firewall to test
2. Verify PC IP address: `ipconfig`
3. Make sure yolo_server.py is running on PC
4. Test from PC browser: http://localhost:5000/health

### Problem: "Camera not found"

**Solutions:**
1. Check camera connection
2. Enable in raspi-config: `sudo raspi-config`
3. Test camera: `libcamera-hello`
4. For USB webcam: `ls /dev/video*` (should see /dev/video0)

### Problem: "Slow detection"

**Solutions:**
1. Reduce image size before sending
2. Use smaller YOLO model (yolov8n)
3. Increase interval in continuous mode
4. Use lower camera resolution

### Problem: "Out of memory on Pi"

This shouldn't happen with this approach! The Pi only captures images, it doesn't run YOLO.

---

## Part 7: Advanced Features

### 7.1 Save Detection History

Modify `yolo_server.py` to log all detections:

```python
import json
from datetime import datetime

# Add to detect() function after detections:
log_entry = {
    'timestamp': datetime.now().isoformat(),
    'count': len(detections),
    'detections': detections
}

with open('detection_log.json', 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
```

### 7.2 Add Email Alerts

When specific objects detected:

```python
import smtplib
from email.message import EmailMessage

def send_alert(detection_info):
    # Configure your email
    msg = EmailMessage()
    msg['Subject'] = f"Alert: {detection_info['count']} objects detected"
    msg['From'] = "your-email@gmail.com"
    msg['To'] = "your-email@gmail.com"
    msg.set_content(str(detection_info))
    
    # Send (configure SMTP settings)
    # ...
```

### 7.3 Web Dashboard

Add to `yolo_server.py`:

```python
@app.route('/')
def dashboard():
    return '''
    <html>
    <body>
        <h1>YOLO Detection Server</h1>
        <p>Status: Running</p>
        <p><a href="/health">Check Health</a></p>
    </body>
    </html>
    '''
```

Access from any device: http://YOUR_PC_IP:5000/

---

## Quick Reference

### Start Everything:

**On PC:**
```powershell
cd C:\Users\hunte\Desktop\rasp
python yolo_server.py
```

**On Pi (single shot):**
```bash
cd ~/yolo_client
python3 capture_and_send.py
```

**On Pi (continuous):**
```bash
cd ~/yolo_client
python3 continuous_detect.py
```

### Transfer Files from Pi to PC:

```powershell
scp hunte@192.168.1.109:~/yolo_client/*.jpg C:\Users\hunte\Desktop\
```

---

## Performance Expectations

| Setup | Detection Speed | Notes |
|-------|----------------|-------|
| PC (YOLOv8n) | ~0.02-0.1 sec | Very fast |
| PC (YOLOv8s) | ~0.05-0.2 sec | Still fast |
| Network transfer | ~0.1-0.5 sec | Depends on WiFi |
| Pi capture | ~2-3 sec | Camera init + capture |
| **Total** | **~3-5 sec** | Much better than on-Pi! |

---

## Benefits of This Approach

✅ **Pi 3 A+ works perfectly** - no resource issues  
✅ **Fast detection** - uses your powerful PC  
✅ **Easy to update** - change model on PC anytime  
✅ **Multiple Pis** - can connect several Pis to one PC  
✅ **Scalable** - add features on PC side easily  
✅ **Reliable** - Pi won't crash or run out of memory  

---

## Next Steps

1. ✅ Set up YOLO server on PC
2. ✅ Set up camera client on Pi
3. ✅ Test with single image
4. ✅ Try continuous detection
5. ✅ Add your custom YOLO model to PC
6. Consider adding web dashboard or alerts

**Your Pi is now a smart camera! 🎥✨**

