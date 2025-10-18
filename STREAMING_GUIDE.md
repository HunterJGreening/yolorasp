# 🎥 Live Streaming Setup Guide

Convert your Raspberry Pi into a live YOLO detection camera with real-time video streaming!

## 🌟 New Features

- ✅ **Real-time streaming** - Continuous video feed
- ✅ **Live web viewer** - Watch in your browser
- ✅ **FPS counter** - See performance metrics
- ✅ **Object counting** - Real-time detection stats
- ✅ **MJPEG stream** - Standard video protocol

---

## 🚀 Quick Start

### Step 1: Start Streaming Server on PC

**Option A: Double-click**
```
START_STREAM_SERVER.bat
```

**Option B: Command line**
```powershell
python yolo_server_stream.py
```

### Step 2: Open Web Viewer

In your browser, go to:
```
http://localhost:5000/stream
```

Leave this tab open to view the live stream!

### Step 3: Start Pi Streaming Client

**Transfer script to Pi:**
```powershell
scp pi_stream_to_pc.py hunte@192.168.1.109:~/yolo_client/stream_to_pc.py
```

**On Pi (in PuTTY):**
```bash
cd ~/yolo_client
python3 stream_to_pc.py
```

**That's it!** You should now see live video with YOLO detections in your browser! 🎉

---

## 📊 What You'll See

### In Your Browser:
- Live video feed from Pi camera
- Bounding boxes around detected objects
- FPS counter (top left)
- Object count (top left)
- Real-time updates

### On Your Pi (Terminal):
```
[17:45:23] Frames: 150 | FPS: 9.8 | Server FPS: 10 | Objects: 2 | Total: 45
```

---

## ⚙️ Configuration

### Adjust FPS (Frame Rate)

Edit `pi_stream_to_pc.py` line 13:
```python
FPS_TARGET = 10  # Try 5 for slower, 15 for faster
```

**Recommendations:**
- **5 FPS** - Very smooth on Pi 3 A+, lower bandwidth
- **10 FPS** - Good balance (default)
- **15+ FPS** - May lag on Pi 3 A+

### Change Detection Speed

Edit `yolo_server_stream.py` line 47:
```python
results = model(img, imgsz=416, verbose=False)
```

**Image sizes:**
- `imgsz=320` - Fastest, less accurate
- `imgsz=416` - Balanced (default)
- `imgsz=640` - Slower, more accurate

### Change Camera Resolution

Edit `pi_stream_to_pc.py` lines 67-68:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## 🌐 View from Other Devices

The stream is accessible from any device on your network!

**From another computer/phone on same WiFi:**
```
http://192.168.1.107:5000/stream
```

Just replace with your PC's IP address.

---

## 📱 Mobile Viewing

1. Make sure your phone is on the same WiFi
2. Open browser on phone
3. Go to: `http://192.168.1.107:5000/stream`
4. You can now view the live stream on your phone!

---

## 🔧 Troubleshooting

### Stream is laggy
1. **Lower FPS:** Set `FPS_TARGET = 5` in Pi script
2. **Reduce resolution:** Use 320x240 in camera settings
3. **Smaller detection size:** Use `imgsz=320` in server

### "Cannot connect to server"
- Make sure streaming server is running (not the regular server!)
- Use `yolo_server_stream.py`, not `yolo_server.py`

### No video in browser
- Check that Pi streaming script is running
- Refresh the browser page
- Check firewall allows port 5000

### Low FPS on Pi
- Pi 3 A+ is limited - 5-10 FPS is normal
- Upgrade to Pi 4/5 for higher FPS
- Or reduce camera resolution

---

## 📊 Performance Comparison

| Device | Single Image | Live Stream (10 FPS) |
|--------|--------------|---------------------|
| Pi 3 A+ | ~3-5 sec/image | ~5-10 FPS real-time |
| Pi 4 (4GB) | ~1-2 sec/image | ~15-20 FPS |
| PC Detection | ~0.02-0.1 sec | Limited by Pi camera |

---

## 🎯 Use Cases

### Security Camera
- Monitor entrance/room in real-time
- Get instant alerts when objects detected
- Review detections live

### Object Tracking
- Track moving objects continuously
- Count people/vehicles passing by
- Monitor inventory in real-time

### Demo/Education
- Show YOLO in action live
- Impress with real-time AI
- Great for presentations

---

## 🔄 Switch Between Modes

### Single Image Mode:
```powershell
# PC
python yolo_server.py

# Pi
python3 webcam_detect.py
```

### Live Streaming Mode:
```powershell
# PC
python yolo_server_stream.py

# Pi
python3 stream_to_pc.py
```

---

## 💡 Tips

1. **Start server first** - Always start PC server before Pi client
2. **Browser performance** - Chrome/Edge work best for MJPEG streams
3. **Network** - Use 5GHz WiFi for better performance
4. **Multiple viewers** - Multiple browsers can watch same stream
5. **Recording** - Use browser extensions to record the stream

---

## 🎨 Advanced: Custom Viewer

Want to customize the web viewer? Edit `yolo_server_stream.py` around line 113 (the HTML template).

You can:
- Change colors/styling
- Add detection statistics
- Show detection history
- Add recording buttons

---

## 📝 Files

- `yolo_server_stream.py` - Streaming server (PC)
- `pi_stream_to_pc.py` - Streaming client (Pi)
- `START_STREAM_SERVER.bat` - Easy launcher

**Old files still work:**
- `yolo_server.py` - Single image mode
- `pi_webcam_detect.py` - Single image mode

---

## 🚀 Quick Commands

**Start streaming (PC):**
```powershell
START_STREAM_SERVER.bat
```

**View in browser:**
```
http://localhost:5000/stream
```

**Start streaming (Pi):**
```bash
cd ~/yolo_client
python3 stream_to_pc.py
```

**Stop streaming:**
- Press `Ctrl+C` in both terminals

---

**Enjoy your live YOLO detection stream! 🎉**

