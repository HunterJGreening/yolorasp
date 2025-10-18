# 🚀 Raspberry Pi YOLO Detection System

A distributed YOLO object detection system that uses a Raspberry Pi 3 A+ as a smart camera and a Windows PC for processing.

## 📋 Overview

This project allows your Raspberry Pi (with limited resources) to capture images and send them to your more powerful PC for YOLO object detection. The PC runs the detection server and sends back annotated results.

**Why this approach?**
- ✅ Works on low-resource Pi (512MB RAM)
- ✅ Fast detection using PC hardware
- ✅ Easy to update models (just change on PC)
- ✅ Can connect multiple Pis to one PC server

## 🎯 Tested & Working

- **Raspberry Pi:** 3 A+ (512MB RAM, 32-bit ARM)
- **PC:** Windows with Python 3.13
- **Model:** YOLOv8n (nano)
- **Detection Speed:** ~3-5 seconds total
- **Network:** Local WiFi

## 📁 Project Structure

```
yolorasp/
├── yolo_server.py              # PC server (runs YOLO detections)
├── START_SERVER.bat            # Easy way to start server
├── pi_webcam_detect.py         # Pi client for USB webcam
├── pi_capture_and_send.py      # Pi client for Pi Camera
├── pi_continuous_detect.py     # Continuous detection mode
├── QUICK_START.md              # Quick setup guide
├── SETUP_COMPLETE.md           # PC setup details
├── pi_to_pc_yolo_guide.md      # Complete guide
└── raspberry_pi_yolo_setup_guide.md  # Alternative on-Pi setup
```

## 🚀 Quick Start

### 1. Set Up PC Server

**Requirements:**
- Windows PC
- Python 3.10+
- On same WiFi network as Pi

**Install & Run:**
```powershell
# Install dependencies
pip install ultralytics opencv-python pillow flask

# Start server
python yolo_server.py
# Or double-click START_SERVER.bat
```

Your PC will start listening on port 5000.

### 2. Set Up Raspberry Pi

**On your Pi (via SSH):**
```bash
# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-opencv
pip3 install requests --break-system-packages

# Create project folder
mkdir ~/yolo_client
cd ~/yolo_client

# Copy script from PC
# (See QUICK_START.md for details)
```

**Transfer script from PC:**
```powershell
scp pi_webcam_detect.py pi@<PI_IP>:~/yolo_client/webcam_detect.py
```

### 3. Run Detection

**On Pi:**
```bash
python3 webcam_detect.py
```

**Get results on PC:**
```powershell
scp pi@<PI_IP>:~/yolo_client/detected.jpg ./
```

## 📸 Example Detection

```
=== Raspberry Pi YOLO Client (USB Webcam) ===
PC Server: 192.168.1.107:5000

✓ Connected to PC server!
Opening webcam...
Capturing image...
✓ Image captured (2304x1536)
Sending to PC...

========================================
DETECTED 3 OBJECTS:
========================================
1. bottle - 34.0%
2. laptop - 31.0%
3. suitcase - 27.0%
========================================

✓ Saved annotated image: detected.jpg
✓ Detection complete!
```

## 🔧 Configuration

### Change PC IP Address

Edit in Pi scripts (line 10):
```python
PC_IP = "192.168.1.107"  # Change to your PC's IP
```

### Use Custom YOLO Model

Edit `yolo_server.py` (line 19):
```python
model = YOLO('best.pt')  # Your custom model
```

### Adjust Detection Interval

In `pi_continuous_detect.py`:
```python
INTERVAL = 5  # Seconds between detections
```

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Fast setup guide
- **[pi_to_pc_yolo_guide.md](pi_to_pc_yolo_guide.md)** - Complete detailed guide
- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - PC setup reference

## 🛠️ Troubleshooting

### "Cannot connect to PC server"
1. Ensure `yolo_server.py` is running on PC
2. Check Windows Firewall (allow port 5000)
3. Both devices must be on same WiFi network
4. Verify PC IP: `ipconfig`

### "Cannot open webcam"
```bash
# Check webcam is detected
ls /dev/video*  # Should show /dev/video0

# Test webcam
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

### "No module named 'requests'"
```bash
pip3 install requests --break-system-packages
```

## 🎨 Features

- ✅ Real-time object detection
- ✅ Supports USB webcam or Pi Camera
- ✅ Returns annotated images with bounding boxes
- ✅ JSON response with detection details
- ✅ Continuous detection mode
- ✅ Web dashboard for server status
- ✅ Compatible with custom YOLO models

## 🔄 Workflow

```
Pi Camera → Capture Image → Send to PC (Flask server)
                                ↓
                          YOLO Detection
                                ↓
                    Annotated Image + Results
                                ↓
                        Back to Pi/Save
```

## 📊 Performance

| Component | Speed | Notes |
|-----------|-------|-------|
| Image Capture (Pi) | ~2-3s | Camera init + capture |
| Network Transfer | ~0.1-0.5s | WiFi dependent |
| YOLO Detection (PC) | ~0.02-0.1s | YOLOv8n on PC |
| **Total** | **~3-5s** | Per image |

## 🌟 Use Cases

- Home security monitoring
- Object counting/tracking
- Wildlife camera
- Inventory management
- Educational projects
- IoT applications

## 📝 Requirements

### PC (Windows)
- Python 3.10+
- 4GB+ RAM
- WiFi connection
- Packages: `ultralytics`, `opencv-python`, `flask`, `pillow`

### Raspberry Pi
- Pi 3 A+ or better
- USB webcam or Pi Camera
- Raspbian OS
- WiFi connection
- Packages: `python3-opencv`, `requests`

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share your use cases

## 📄 License

This project is open source and available for educational and personal use.

## 🙏 Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Designed for Raspberry Pi 3 A+ (512MB RAM)
- Created to solve the challenge of running YOLO on resource-constrained devices

## 📧 Contact

Created by Hunter - [GitHub](https://github.com/HunterJGreening)

---

**⭐ If this helped you, please star the repo!**

