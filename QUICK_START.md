# 🚀 Quick Start Guide - Pi to PC YOLO Detection

Everything is ready! Just follow these simple steps.

---

## STEP 1: Start Your PC Server

### Option A: Double-click `START_SERVER.bat`
- A window will open showing the server status
- **Keep this window open!**
- You should see "Server is running"

### Option B: Or run manually
```powershell
python yolo_server.py
```

**Your PC IP: 192.168.1.107** (already configured in scripts)

---

## STEP 2: Set Up Your Raspberry Pi

### Connect via PuTTY
- Host: 192.168.1.109
- Username: hunte
- Password: [your password]

### Install Dependencies

In PuTTY, run these commands (right-click to paste):

```bash
# Update system
sudo apt update

# Install Python packages
sudo apt install -y python3-pip python3-opencv

# Install requests
pip3 install requests --break-system-packages

# Create project folder
cd ~
mkdir yolo_client
cd yolo_client
```

---

## STEP 3: Copy Script to Your Pi

### Choose ONE based on your camera:

#### Option A: If you have a **USB Webcam**

In PuTTY:
```bash
nano webcam_detect.py
```

Then copy the contents of **`pi_webcam_detect.py`** from your PC:
1. Open `pi_webcam_detect.py` in Cursor
2. Select all (Ctrl+A) and copy (Ctrl+C)
3. Right-click in PuTTY to paste
4. Save: Ctrl+X, Y, Enter

#### Option B: If you have a **Raspberry Pi Camera Module**

In PuTTY:
```bash
# Install Pi Camera support
sudo apt install -y python3-picamera2

# Create script
nano camera_detect.py
```

Then copy contents of **`pi_capture_and_send.py`** (same process as above)

---

## STEP 4: Run Detection!

### For USB Webcam:
```bash
cd ~/yolo_client
python3 webcam_detect.py
```

### For Pi Camera:
```bash
cd ~/yolo_client  
python3 camera_detect.py
```

### What to Expect:
1. ✓ Connects to PC server
2. ✓ Captures image
3. ✓ Sends to PC for detection
4. ✓ Shows results in terminal
5. ✓ Saves `detected.jpg` on Pi

---

## STEP 5: View Your Results

### Get the detected image from Pi to PC:

In Windows PowerShell:
```powershell
scp hunte@192.168.1.109:~/yolo_client/detected.jpg C:\Users\hunte\Desktop\
```

Or use **WinSCP** to browse and download files.

---

## Troubleshooting

### "Cannot connect to PC server"
1. Make sure `START_SERVER.bat` is running on PC
2. Check Windows Firewall:
   - Search "Windows Defender Firewall"
   - Click "Allow an app through firewall"
   - Make sure Python is allowed
3. Or temporarily disable firewall to test

### "Cannot open webcam"
- Check USB webcam is plugged in
- Run: `ls /dev/video*` (should show `/dev/video0`)

### "No module named 'requests'"
```bash
pip3 install requests --break-system-packages
```

### "Camera not found" (Pi Camera)
```bash
sudo raspi-config
# Interface Options → Camera → Enable
sudo reboot
```

---

## Files You Have:

### On PC:
- ✅ `yolo_server.py` - The detection server
- ✅ `START_SERVER.bat` - Easy way to start server
- ✅ `pi_webcam_detect.py` - Script for USB webcam (copy to Pi)
- ✅ `pi_capture_and_send.py` - Script for Pi Camera (copy to Pi)
- ✅ `pi_continuous_detect.py` - Continuous detection (optional)

### Your Network:
- PC IP: **192.168.1.107** ✅
- Pi IP: **192.168.1.109** ✅
- Port: **5000** ✅

---

## Next Steps (Optional)

### Continuous Detection
For continuous monitoring every 5 seconds:

1. Copy `pi_continuous_detect.py` to Pi
2. Run: `python3 continuous_detect.py`
3. Press Ctrl+C to stop

### Use Your Custom YOLO Model
Replace `yolov8n.pt` with your `best.pt` in `yolo_server.py` line 18:
```python
model = YOLO('best.pt')  # Your custom model
```

---

## Summary

1. **Start server on PC** → Run `START_SERVER.bat`
2. **Set up Pi** → Install packages, copy script  
3. **Run detection** → `python3 webcam_detect.py` or `python3 camera_detect.py`
4. **View results** → Transfer `detected.jpg` to PC

**You're all set! 🎉**

