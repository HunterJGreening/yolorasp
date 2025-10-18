# ✅ PC Setup Complete!

Your Windows PC is now ready to process YOLO detections from your Raspberry Pi!

## What's Running:

### ✅ YOLO Server (Running in Background)
- **Status:** Active
- **Address:** http://192.168.1.107:5000
- **Model:** YOLOv8n (downloading on first use)

### ✅ Test the Server
Open your browser and go to: **http://localhost:5000**

You should see the server dashboard!

---

## Next Steps: Set Up Your Raspberry Pi

### 1. Transfer Scripts to Your Pi

In **PuTTY** (connected to your Pi at 192.168.1.109):

```bash
cd ~
mkdir yolo_client
cd yolo_client
nano capture_and_send.py
```

Then **copy the contents** of `pi_capture_and_send.py` from this folder and paste it into nano (right-click to paste in PuTTY).

Save and exit: **Ctrl+X**, **Y**, **Enter**

### 2. Install Dependencies on Pi

In PuTTY:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-opencv python3-picamera2
pip3 install requests pillow
```

### 3. Enable Camera

```bash
sudo raspi-config
```

- Navigate to: **Interface Options** → **Camera** → **Enable**
- Reboot: `sudo reboot`

### 4. Run Detection!

After Pi reboots, reconnect via PuTTY:

```bash
cd ~/yolo_client
python3 capture_and_send.py
```

**It should:**
- ✓ Connect to your PC
- ✓ Capture an image
- ✓ Send it to PC for detection
- ✓ Show you the results!

---

## Files Created for You:

### On Your PC (Already Here):
- ✅ `yolo_server.py` - The main server (running)
- ✅ `pi_capture_and_send.py` - Single shot detection (copy to Pi)
- ✅ `pi_continuous_detect.py` - Continuous detection (copy to Pi)

### Your PC IP: **192.168.1.107**
*(Already configured in all scripts!)*

### Your Pi IP: **192.168.1.109**

---

## Quick Commands Reference:

### Check if Server is Running:
Open browser: http://localhost:5000

### Stop Server (if needed):
Press Ctrl+C in the terminal where it's running

### Restart Server:
```powershell
python yolo_server.py
```

### Transfer Files to Pi:
```powershell
scp pi_capture_and_send.py hunte@192.168.1.109:~/yolo_client/
```

### Get Results from Pi:
```powershell
scp hunte@192.168.1.109:~/yolo_client/detected.jpg C:\Users\hunte\Desktop\
```

---

## Troubleshooting:

### Pi can't connect to PC?
1. Make sure `yolo_server.py` is running on PC
2. Check Windows Firewall (allow port 5000)
3. Both devices must be on same WiFi network

### Camera not working?
1. Run: `sudo raspi-config` → Enable camera
2. Check connection
3. Test: `libcamera-hello`

### Want to use USB webcam instead?
Edit the script and replace `Picamera2` code with:
```python
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
```

---

## What to Do Now:

1. ✅ **Test the server:** Open http://localhost:5000 in your browser
2. 🔄 **Go to your Pi in PuTTY** and follow the "Next Steps" above
3. 🚀 **Run your first detection!**

**Your PC is ready and waiting for images from your Pi! 🎉**

