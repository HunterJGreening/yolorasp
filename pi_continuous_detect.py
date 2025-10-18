#!/usr/bin/env python3
import requests
import cv2
from picamera2 import Picamera2
import time

# Configuration - ALREADY SET TO YOUR PC'S IP!
PC_IP = "192.168.1.107"  # Your PC's IP address
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
    print("=== Continuous Detection ===")
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

