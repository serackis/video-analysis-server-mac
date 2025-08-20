#!/usr/bin/env python3
"""
Quick Demo: Video Streaming for Video Analysis Server
Simple demonstration of streaming video files on Mac
"""

import os
import subprocess
import time
import sys



def stream_video_simple(video_path):
    """Simple video streaming using FFmpeg"""
    print(f"🎥 Starting simple video stream...")
    print(f"📁 Video: {video_path}")
    print(f"🌐 RTSP URL: rtsp://localhost:8554/test")
    print(f"🔗 For testing: rtsp://admin:password@localhost:8554/test")
    print()
    print("📋 Instructions:")
    print("1. Keep this terminal running")
    print("2. Open http://localhost:5001 in your browser")
    print("3. Add camera with RTSP URL: rtsp://admin:password@localhost:8554/test")
    print("4. Watch the video analysis in action!")
    print("5. The system will process 'Ema_very_short.mp4' for face detection and depersonalization")
    print()
    print("⏹️  Press Ctrl+C to stop")
    
    cmd = [
        'ffmpeg',
        '-re',
        '-stream_loop', '-1',
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        'rtsp://0.0.0.0:8554/test'
    ]
    
    try:
        process = subprocess.Popen(cmd)
        print(f"✅ Stream started! Process ID: {process.pid}")
        
        while True:
            time.sleep(1)
            if process.poll() is not None:
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopping stream...")
        process.terminate()
        process.wait()
        print("✅ Stream stopped")

def main():
    print("🎥 Video Streaming Demo for Video Analysis Server")
    print("=" * 50)
    
    # Check if Ema video exists
    if not os.path.exists('Ema_very_short.mp4'):
        print("❌ Error: Ema_very_short.mp4 not found!")
        print("Please ensure the video file is in the current directory.")
        return
    
    # Start streaming with Ema video
    stream_video_simple('Ema_very_short.mp4')

if __name__ == "__main__":
    main() 