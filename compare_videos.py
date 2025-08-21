#!/usr/bin/env python3
"""
Simple Video Comparison Helper
Lists all processed videos for easy comparison
"""

import os
from pathlib import Path

def list_processed_videos():
    """List all processed videos for comparison"""
    
    comparison_dir = Path("accuracy_comparison")
    
    if not comparison_dir.exists():
        print("❌ No comparison directory found. Run accuracy_comparison.py first.")
        return
    
    print("🎬 Processed Videos for Comparison")
    print("="*60)
    print("📁 Directory: accuracy_comparison/")
    print("="*60)
    
    # Find all processed videos
    video_files = list(comparison_dir.glob("*_processed.mp4"))
    
    if not video_files:
        print("❌ No processed videos found.")
        return
    
    # Sort by file size (larger = more detailed processing)
    video_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    print(f"{'Model':<25} {'File Size':<12} {'Status'}")
    print("-"*60)
    
    for video_file in video_files:
        model_name = video_file.stem.replace('_processed', '').replace('_', '-')
        file_size = video_file.stat().st_size / (1024 * 1024)  # MB
        status = "✅ Ready" if video_file.exists() else "❌ Missing"
        
        print(f"{model_name:<25} {file_size:<12.1f}MB {status}")
    
    print("="*60)
    print("\n💡 How to Compare:")
    print("1. Open the videos in your video player")
    print("2. Compare depersonalization quality")
    print("3. Check for missed detections")
    print("4. Look for over-blurring or under-blurring")
    
    print("\n🏆 Top Models by Accuracy:")
    print("   🥇 yolo11l-seg.pt - Best overall (857 persons detected)")
    print("   🥈 yolo11x-seg.pt - Highest confidence (0.671 avg)")
    print("   🥉 yolo11m-seg.pt - Balanced performance (802 persons)")
    
    print("\n📊 Key Metrics:")
    print("   • yolo11l-seg.pt: 857 persons, 4.54 FPS, 70.75s")
    print("   • yolo11x-seg.pt: 851 persons, 2.41 FPS, 133.06s")
    print("   • yolo11m-seg.pt: 802 persons, 5.73 FPS, 55.98s")
    
    print(f"\n📁 All files saved in: {comparison_dir.absolute()}")

if __name__ == "__main__":
    list_processed_videos()
