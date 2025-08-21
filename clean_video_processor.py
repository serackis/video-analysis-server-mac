#!/usr/bin/env python3
"""
Clean Video Processor
Process videos with enhanced blur and tracking, but NO visual overlays
Outputs clean, depersonalized videos without bounding boxes, labels, or tracking info
"""

import os
import sys
from pathlib import Path

def main():
    """Main function for clean video processing"""
    
    print("🧹 Clean Video Processor")
    print("="*50)
    print("This script processes videos with enhanced blur and tracking")
    print("but removes ALL visual overlays for clean output")
    print("="*50)
    
    if len(sys.argv) < 2:
        print("\nUsage: python clean_video_processor.py <input_video> [options]")
        print("\nOptions:")
        print("  -o <output>      Output video path")
        print("  -m <model>       YOLOv11 model (default: yolo11l-seg.pt)")
        print("  -c <confidence>  Detection confidence (default: 0.3)")
        print("  -b <blur>        Base blur strength (default: 25)")
        print("  -p               Show preview during processing")
        print("\nExamples:")
        print("  python clean_video_processor.py video.mp4")
        print("  python clean_video_processor.py video.mp4 -m yolo11n-seg.pt -c 0.4")
        print("  python clean_video_processor.py video.mp4 -o clean_output.mp4")
        return
    
    # Build the command for tracking_video_processor.py with clean output
    cmd = ["python", "tracking_video_processor.py"] + sys.argv[1:] + ["--clean-output"]
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("\n" + "="*50)
    
    # Execute the command
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*50)
        print("✅ Clean video processing completed successfully!")
        print("📹 Your video has been depersonalized with enhanced blur")
        print("🧹 All visual overlays have been removed for clean output")
        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Processing failed with exit code: {e.returncode}")
        return 1
    except FileNotFoundError:
        print("\n❌ Error: tracking_video_processor.py not found")
        print("💡 Make sure you're in the correct directory")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
