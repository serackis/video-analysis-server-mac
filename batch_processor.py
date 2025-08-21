#!/usr/bin/env python3
"""
Batch Video Processor
Process multiple videos with enhanced blur and tracking
"""

import os
import sys
from pathlib import Path
import subprocess
import time

def process_video_batch(video_directory=".", output_directory="batch_processed", 
                       enable_tracking=True, model="yolo11l-seg.pt", 
                       confidence=0.3, blur_strength=25):
    """Process all videos in a directory with enhanced blur and tracking"""
    
    print("🎬 Batch Video Processing System")
    print("="*60)
    print(f"Input directory: {video_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Tracking: {'✅ Enabled' if enable_tracking else '❌ Disabled'}")
    print(f"Model: {model}")
    print(f"Confidence: {confidence}")
    print(f"Blur strength: {blur_strength}")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(exist_ok=True)
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_directory).glob(f"*{ext}"))
        video_files.extend(Path(video_directory).glob(f"*{ext.upper()}"))
    
    if not video_files:
        print("❌ No video files found in the specified directory")
        return
    
    print(f"📁 Found {len(video_files)} video files:")
    for video in video_files:
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"   • {video.name} ({size_mb:.1f}MB)")
    
    print("\n" + "="*60)
    print("🚀 Starting batch processing...")
    print("="*60)
    
    # Process each video
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n🔄 Processing {i}/{len(video_files)}: {video_file.name}")
        
        try:
            # Build command
            if enable_tracking:
                cmd = [
                    "python", "tracking_video_processor.py",
                    str(video_file),
                    "-o", str(output_path / f"tracked_{video_file.stem}.mp4"),
                    "-m", model,
                    "-c", str(confidence),
                    "-b", str(blur_strength)
                ]
            else:
                # For non-tracking, we need to modify the script temporarily
                print(f"   ⚠️  Non-tracking mode requires script modification")
                print(f"   💡 Use tracking mode or modify test_enhanced_blur.py manually")
                failed += 1
                continue
            
            # Run the command
            print(f"   🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print(f"   ✅ Successfully processed: {video_file.name}")
                successful += 1
                
                # Move output files to batch directory
                if enable_tracking:
                    # Check if tracking output was created
                    tracking_output = Path(f"tracked_{video_file.stem}.mp4")
                    tracking_json = Path(f"tracked_{video_file.stem}.json")
                    
                    if tracking_output.exists():
                        # Move to batch directory
                        final_output = output_path / f"tracked_{video_file.stem}.mp4"
                        tracking_output.rename(final_output)
                        print(f"   📁 Output saved to: {final_output}")
                    
                    if tracking_json.exists():
                        # Move JSON to batch directory
                        final_json = output_path / f"tracked_{video_file.stem}.json"
                        tracking_json.rename(final_json)
                        print(f"   📊 Statistics saved to: {final_json}")
                
            else:
                print(f"   ❌ Failed to process: {video_file.name}")
                print(f"   Error: {result.stderr}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout processing: {video_file.name}")
            failed += 1
        except Exception as e:
            print(f"   ❌ Error processing: {video_file.name}")
            print(f"   Error: {e}")
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per video: {total_time/len(video_files):.2f} seconds")
    
    if successful > 0:
        print(f"\n✅ Successfully processed videos saved to: {output_directory}/")
        print(f"💡 Check the output directory for results")
    
    if failed > 0:
        print(f"\n❌ {failed} videos failed to process")
        print(f"💡 Check error messages above for details")
    
    return successful, failed

def show_help():
    """Show help information"""
    print("🎬 Batch Video Processor - Help")
    print("="*50)
    print("Usage: python batch_processor.py [OPTIONS]")
    print("\nOptions:")
    print("  -d, --directory PATH    Input directory (default: current)")
    print("  -o, --output PATH       Output directory (default: batch_processed)")
    print("  -m, --model MODEL       YOLOv11 model (default: yolo11l-seg.pt)")
    print("  -c, --confidence FLOAT Detection confidence (default: 0.3)")
    print("  -b, --blur INTEGER      Base blur strength (default: 25)")
    print("  --no-tracking           Disable tracking")
    print("  -h, --help              Show this help message")
    print("\nExamples:")
    print("  # Process all videos in current directory")
    print("  python batch_processor.py")
    print("\n  # Process videos in specific directory")
    print("  python batch_processor.py -d /path/to/videos")
    print("\n  # Custom settings")
    print("  python batch_processor.py -m yolo11n-seg.pt -c 0.4 -b 30")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Video Processor')
    parser.add_argument('-d', '--directory', default='.', help='Input directory')
    parser.add_argument('-o', '--output', default='batch_processed', help='Output directory')
    parser.add_argument('-m', '--model', default='yolo11l-seg.pt', help='YOLOv11 model')
    parser.add_argument('-c', '--confidence', type=float, default=0.3, help='Detection confidence')
    parser.add_argument('-b', '--blur', type=int, default=25, help='Base blur strength')
    parser.add_argument('--no-tracking', action='store_true', help='Disable tracking')
    
    args = parser.parse_args()
    
    # Check if required scripts exist
    if args.no_tracking:
        if not Path("test_enhanced_blur.py").exists():
            print("❌ test_enhanced_blur.py not found")
            print("💡 Non-tracking mode requires test_enhanced_blur.py")
            return
    else:
        if not Path("tracking_video_processor.py").exists():
            print("❌ tracking_video_processor.py not found")
            print("💡 Tracking mode requires tracking_video_processor.py")
            return
    
    # Process videos
    successful, failed = process_video_batch(
        video_directory=args.directory,
        output_directory=args.output,
        enable_tracking=not args.no_tracking,
        model=args.model,
        confidence=args.confidence,
        blur_strength=args.blur
    )
    
    if successful > 0:
        print(f"\n🎉 Batch processing completed!")
        print(f"📁 Check {args.output}/ directory for results")
    else:
        print(f"\n❌ No videos were processed successfully")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, show help
        show_help()
    else:
        main()
