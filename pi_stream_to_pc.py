#!/usr/bin/env python3
"""
Live Streaming Version - Continuously sends frames to PC for real-time detection
"""
import requests
import cv2
import sys
import time
from datetime import datetime

# Configuration - ALREADY SET TO YOUR PC'S IP!
PC_IP = "192.168.1.107"  # Your PC's IP address
PC_PORT = 5000
SERVER_URL = f"http://{PC_IP}:{PC_PORT}/detect"

# Streaming settings
FPS_TARGET = 10  # Target frames per second to send
DISPLAY_STATS = True  # Show FPS and detection stats

def test_connection():
    """Test if PC server is reachable"""
    try:
        response = requests.get(f"http://{PC_IP}:{PC_PORT}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Connected to PC server!")
            data = response.json()
            print(f"  Server mode: {data.get('mode', 'standard')}")
            return True
    except:
        print("[ERROR] Cannot connect to PC server!")
        print(f"  Make sure server is running on {PC_IP}:{PC_PORT}")
        return False
    return False

def send_frame(frame):
    """Send frame to PC for detection"""
    try:
        # Encode frame as JPEG
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        # Send to server
        files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        response = requests.post(SERVER_URL, files=files, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

def main():
    print("\n" + "="*50)
    print("Raspberry Pi Live Streaming Client")
    print("="*50)
    print(f"PC Server: {PC_IP}:{PC_PORT}")
    print(f"Target FPS: {FPS_TARGET}")
    print("="*50 + "\n")
    
    # Test connection first
    if not test_connection():
        sys.exit(1)
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    
    print("[OK] Webcam opened")
    print("\nStarting live stream...")
    print("Press Ctrl+C to stop\n")
    
    # Frame timing
    frame_delay = 1.0 / FPS_TARGET
    frame_count = 0
    start_time = time.time()
    last_stats_time = time.time()
    detections_count = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                continue
            
            # Send frame for detection
            result = send_frame(frame)
            
            if result and result.get('success'):
                detections = result['detections']
                server_fps = result.get('fps', 0)
                
                # Count detections
                if detections:
                    detections_count += len(detections)
                
                # Display stats every second
                current_time = time.time()
                if DISPLAY_STATS and (current_time - last_stats_time) >= 1.0:
                    elapsed = current_time - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Frames: {frame_count} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Server FPS: {server_fps} | "
                          f"Objects: {len(detections)} | "
                          f"Total Detected: {detections_count}", 
                          end='', flush=True)
                    
                    last_stats_time = current_time
            
            frame_count += 1
            
            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\nStopping stream...")
    
    finally:
        cap.release()
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*50)
        print("Stream Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Total objects detected: {detections_count}")
        print("="*50)

if __name__ == "__main__":
    main()

