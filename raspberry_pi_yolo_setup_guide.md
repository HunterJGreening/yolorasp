# Raspberry Pi 3 A+ YOLO Setup Guide

A simple guide to set up your Raspberry Pi 3 A+ and run YOLOv8 object detection.

## ⚠️ Important Note
The Raspberry Pi 3 A+ has only 512MB RAM, which is quite limited. This guide focuses on lightweight approaches that will work within these constraints.

---

## Part 1: Initial Raspberry Pi Setup

### 1.1 Flash Raspberry Pi OS
1. Download **Raspberry Pi Imager** from https://www.raspberrypi.com/software/
2. Insert your microSD card (minimum 16GB recommended)
3. Open Raspberry Pi Imager:
   - Choose OS: **Raspberry Pi OS Lite (64-bit)** (lighter on resources)
   - Choose Storage: Your SD card
   - Click the gear icon for advanced options:
     - Enable SSH
     - Set username/password
     - Configure WiFi (optional)
4. Click **Write** and wait for completion

### 1.2 First Boot
1. Insert SD card into Pi and power it on
2. Find your Pi's IP address (check router or use `ping raspberrypi.local`)
3. SSH into your Pi:
   ```bash
   ssh pi@<your-pi-ip-address>
   ```

### 1.3 Update System
```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

---

## Part 2: Install Dependencies

### 2.1 Install Python and Essential Tools
```bash
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libopenblas-dev
```

### 2.2 Create Virtual Environment
```bash
cd ~
python3 -m venv yolo-env
source yolo-env/bin/activate
```

### 2.3 Install Ultralytics YOLOv8 (Lightweight)
```bash
pip install --upgrade pip
pip install ultralytics
pip install opencv-python-headless
```

**Note:** This may take 15-30 minutes on Pi 3 A+.

---

## Part 3: Prepare Your YOLO Model

### 3.1 Download Pre-trained Model (Recommended for Pi 3 A+)
The smallest YOLO model that will run on your Pi:

```bash
cd ~
mkdir yolo_project
cd yolo_project
```

Create a Python script to download the model:
```bash
nano download_model.py
```

Paste this content:
```python
from ultralytics import YOLO

# Download the smallest YOLOv8 model
model = YOLO('yolov8n.pt')  # 'n' = nano (smallest)
print("Model downloaded successfully!")
```

Run it:
```bash
python download_model.py
```

### 3.2 Test with a Sample Image
Download a test image:
```bash
wget https://ultralytics.com/images/bus.jpg
```

---

## Part 4: Run YOLO Detection

### 4.1 Create Detection Script
Create a simple detection script:
```bash
nano detect.py
```

Paste this content:
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Run detection on image
results = model('bus.jpg', imgsz=320)  # Smaller image size for Pi

# Save results
for r in results:
    im_array = r.plot()
    cv2.imwrite('output.jpg', im_array)

print("Detection complete! Check output.jpg")
```

### 4.2 Run Detection
```bash
python detect.py
```

**Expected performance:** 5-15 seconds per image on Pi 3 A+.

---

## Part 5: Optimize for Raspberry Pi

### 5.1 Performance Tips
1. **Use smaller input size:**
   ```python
   results = model('image.jpg', imgsz=320)  # Instead of 640
   ```

2. **Use YOLOv8n (nano):** Smallest and fastest model
   - YOLOv8n: ~3MB, fastest
   - YOLOv8s: ~11MB, slower but more accurate

3. **Disable unnecessary features:**
   ```python
   results = model('image.jpg', verbose=False, stream=True)
   ```

### 5.2 Increase Swap Space (for stability)
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
```
Change `CONF_SWAPSIZE=100` to `CONF_SWAPSIZE=1024`

```bash
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## Part 6: Camera Integration (Optional)

### 6.1 If Using Raspberry Pi Camera Module
```bash
sudo apt install -y python3-picamera2
```

### 6.2 Camera Detection Script
```bash
nano camera_detect.py
```

```python
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time

# Initialize camera
picam2 = Picamera2()
picam2.start()
time.sleep(2)

# Load model
model = YOLO('yolov8n.pt')

# Capture and detect
frame = picam2.capture_array()
results = model(frame, imgsz=320)

# Save result
for r in results:
    im_array = r.plot()
    cv2.imwrite('camera_output.jpg', im_array)

picam2.stop()
print("Camera detection complete!")
```

### 6.3 If Using USB Webcam
```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    results = model(frame, imgsz=320)
    for r in results:
        im_array = r.plot()
        cv2.imwrite('webcam_output.jpg', im_array)

cap.release()
print("Webcam detection complete!")
```

---

## Part 7: Using Your Custom YOLOv8 Model

If you've trained your own model:

### 7.1 Transfer Model to Pi
From your computer:
```bash
scp /path/to/your/best.pt pi@<pi-ip>:~/yolo_project/
```

### 7.2 Run Custom Model
```python
from ultralytics import YOLO

# Load your custom model
model = YOLO('best.pt')

# Run detection
results = model('your_image.jpg', imgsz=320)

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected class {cls} with confidence {conf:.2f}")
```

---

## Part 8: Troubleshooting

### Common Issues

**1. Out of Memory Errors**
- Use smaller model (yolov8n)
- Reduce image size: `imgsz=224` or `imgsz=320`
- Increase swap space (see Part 5.2)
- Close other programs

**2. Slow Performance**
- Expected: 5-15 seconds per image
- Use smaller input size
- Consider using TFLite or ONNX export (more advanced)

**3. Import Errors**
- Make sure virtual environment is activated:
  ```bash
  source ~/yolo-env/bin/activate
  ```

**4. Camera Not Working**
- Enable camera in raspi-config:
  ```bash
  sudo raspi-config
  # Interface Options > Camera > Enable
  ```

---

## Quick Reference Commands

### Activate Environment
```bash
source ~/yolo-env/bin/activate
```

### Run Detection
```bash
cd ~/yolo_project
python detect.py
```

### Transfer Files to Pi
```bash
scp file.jpg pi@<pi-ip>:~/yolo_project/
```

### Transfer Files from Pi
```bash
scp pi@<pi-ip>:~/yolo_project/output.jpg ./
```

### View Images Remotely
Use SFTP client like FileZilla or WinSCP to browse and download images.

---

## Performance Expectations

| Model    | Size  | Speed on Pi 3 A+ | Accuracy |
|----------|-------|------------------|----------|
| YOLOv8n  | 3MB   | ~5-10 sec/img    | Good     |
| YOLOv8s  | 11MB  | ~15-30 sec/img   | Better   |
| YOLOv8m+ | 25MB+ | Too slow/crashes | Best     |

**Recommendation:** Stick with YOLOv8n on Raspberry Pi 3 A+.

---

## Next Steps

1. ✅ Set up your Pi and install dependencies
2. ✅ Test with the nano model and sample image
3. ✅ Integrate your custom model if available
4. ✅ Add camera support if needed
5. Consider upgrading to Pi 4 or Pi 5 for real-time detection

---

## Additional Resources

- Ultralytics Documentation: https://docs.ultralytics.com/
- Raspberry Pi Forums: https://forums.raspberrypi.com/
- YOLOv8 GitHub: https://github.com/ultralytics/ultralytics

---

**Good luck with your Raspberry Pi YOLO project! 🚀**

