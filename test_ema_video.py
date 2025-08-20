#!/usr/bin/env python3
"""
Test Script: Ema Video Analysis Demo
Uploads and processes the Ema_very_short.mp4 video to demonstrate
the video analysis capabilities of the system.
"""

import requests
import json
import time
import os
from pathlib import Path

def test_video_upload():
    """Test uploading the Ema video to the system"""
    print("🎬 Testing Ema Video Analysis System")
    print("=" * 50)
    
    # Check if Ema video exists
    video_path = "Ema_very_short.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Error: {video_path} not found!")
        return False
    
    print(f"📁 Found video: {video_path}")
    print(f"📊 File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Upload video to the system
    print("\n📤 Uploading video to analysis system...")
    upload_url = "http://localhost:5001/api/upload-video"
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'video': (video_path, video_file, 'video/mp4')}
            response = requests.post(upload_url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Video uploaded successfully!")
            print(f"📋 Upload ID: {result.get('id')}")
            print(f"📁 Stored as: {result.get('filename')}")
            return result.get('filename')
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_video_processing(filename):
    """Test processing the uploaded video"""
    print(f"\n🔍 Processing video: {filename}...")
    process_url = f"http://localhost:5001/api/process-video"
    
    try:
        data = {'filename': filename}
        response = requests.post(process_url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Video processing started!")
            print(f"📊 Processing ID: {result.get('processed_video_id')}")
            return result.get('processed_video_id')
        else:
            print(f"❌ Processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False

def check_processing_status(process_id):
    """Check the status of video processing"""
    print(f"\n📊 Checking processing status (ID: {process_id})...")
    
    # Wait a bit for processing to complete
    time.sleep(3)
    
    try:
        response = requests.get("http://localhost:5001/api/processed-videos")
        if response.status_code == 200:
            videos = response.json()
            for video in videos:
                if video.get('id') == process_id:
                    print("✅ Video processing completed!")
                    print(f"📁 Processed file: {video.get('processed_filename')}")
                    print(f"🔒 Depersonalized: {video.get('depersonalized')}")
                    print(f"⏱️  Processing time: {video.get('processing_duration', 'N/A')} seconds")
                    return True
            
            print("⏳ Video still processing...")
            return False
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def check_video_library():
    """Check the video library for processed videos"""
    print(f"\n📚 Checking video library...")
    
    try:
        response = requests.get("http://localhost:5001/api/videos")
        if response.status_code == 200:
            videos = response.json()
            print(f"📊 Total videos in library: {len(videos)}")
            
            if videos:
                print("\n📋 Video details:")
                for video in videos:
                    print(f"  • {video.get('filename', 'Unknown')}")
                    print(f"    - Duration: {video.get('duration', 'N/A')} seconds")
                    print(f"    - Faces detected: {video.get('faces_detected', 0)}")
                    print(f"    - Plates detected: {video.get('plates_detected', 0)}")
                    print(f"    - Created: {video.get('created_at', 'N/A')}")
                    print()
            
            return True
        else:
            print(f"❌ Library check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Library check error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Ema Video Analysis Test")
    print("=" * 50)
    
    # Test 1: Upload video
    filename = test_video_upload()
    if not filename:
        print("❌ Test failed at upload stage")
        return
    
    # Test 2: Process video
    process_id = test_video_processing(filename)
    if not process_id:
        print("❌ Test failed at processing stage")
        return
    
    # Test 3: Check processing status
    print("\n⏳ Waiting for processing to complete...")
    max_wait = 30  # Wait up to 30 seconds
    wait_time = 0
    
    while wait_time < max_wait:
        if check_processing_status(process_id):
            break
        time.sleep(2)
        wait_time += 2
        print(f"⏱️  Waited {wait_time} seconds...")
    
    if wait_time >= max_wait:
        print("⏰ Processing timeout - checking library anyway...")
    
    # Test 4: Check video library
    check_video_library()
    
    print("\n🎉 Test completed!")
    print("🌐 Open http://localhost:5001/library to view processed videos")
    print("📱 Open http://localhost:5001 to see the main interface")

if __name__ == "__main__":
    main()
