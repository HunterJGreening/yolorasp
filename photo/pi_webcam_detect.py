#!/usr/bin/env python3
"""
USB Webcam version - Use this if you have a USB webcam instead of Pi Camera
"""
import requests
import cv2
import sys
import base64

# Configuration - ALREADY SET TO YOUR PC'S IP!
PC_IP = "192.168.1.107"  # Your PC's IP address
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

def capture_webcam():
    """Capture image from USB webcam"""
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open webcam")
        print("  Make sure USB webcam is connected")
        sys.exit(1)
    
    print("Capturing image...")
    # Capture a few frames to let camera adjust
    for i in range(5):
        ret, frame = cap.read()
    
    # Capture the actual image
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    else:
        print("✗ Error capturing frame")
        sys.exit(1)

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
    print("\n=== Raspberry Pi YOLO Client (USB Webcam) ===")
    print(f"PC Server: {PC_IP}:{PC_PORT}\n")
    
    # Test connection first
    if not test_connection():
        sys.exit(1)
    
    # Capture image
    image = capture_webcam()
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

