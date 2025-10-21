#!/usr/bin/env python3
"""
Script to find the Raspberry Pi's IP address for easy configuration
"""
import socket
import subprocess
import sys

def get_local_ip():
    """Get the local IP address of the Pi"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def get_all_ips():
    """Get all network interface IPs"""
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            return ips
    except:
        pass
    return []

def main():
    print("\n" + "="*50)
    print("🍓 Raspberry Pi IP Address Finder")
    print("="*50)
    
    # Get primary IP
    primary_ip = get_local_ip()
    print(f"Primary IP: {primary_ip}")
    
    # Get all IPs
    all_ips = get_all_ips()
    if all_ips:
        print(f"All IPs: {', '.join(all_ips)}")
    
    print("\n" + "="*50)
    print("📋 Configuration Instructions:")
    print("="*50)
    print("1. Run the Pi server: python3 pi_yolo_server.py")
    print("2. From your PC, open browser and go to:")
    print(f"   http://{primary_ip}:5000")
    print("3. Or access the live stream directly:")
    print(f"   http://{primary_ip}:5000/stream")
    print("="*50)
    
    if primary_ip != "127.0.0.1":
        print(f"\n✅ Pi IP found: {primary_ip}")
        print("✅ Ready to access from PC!")
    else:
        print("\n⚠️  Could not determine IP address")
        print("⚠️  Make sure Pi is connected to network")
    
    print("\nPress Ctrl+C to exit or any key to continue...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
