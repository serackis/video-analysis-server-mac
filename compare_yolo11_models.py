#!/usr/bin/env python3
"""
Compare different YOLOv11 segmentation models
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

def test_model_performance(model_name, test_video, confidence=0.3):
    """Test a single YOLOv11 model and return performance metrics"""
    
    print(f"\n🔄 Testing {model_name}...")
    
    try:
        # Load model
        start_time = time.time()
        model = YOLO(model_name)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Get model info
        info = model.info()
        params = info.get('parameters', 'Unknown')
        model_size = info.get('model_size', 'Unknown')
        
        # Test on video
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {test_video}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Process first 50 frames for speed test
        test_frames = min(50, total_frames)
        
        start_time = time.time()
        results = model(test_video, conf=confidence, verbose=False, max_det=10)
        process_time = time.time() - start_time
        
        # Calculate metrics
        fps = test_frames / process_time
        avg_time_per_frame = process_time / test_frames
        
        print(f"📊 Processed {test_frames} frames in {process_time:.2f}s")
        print(f"📊 Speed: {fps:.2f} FPS")
        print(f"📊 Time per frame: {avg_time_per_frame*1000:.1f}ms")
        
        return {
            'model': model_name,
            'parameters': params,
            'model_size': model_size,
            'load_time': load_time,
            'process_time': process_time,
            'fps': fps,
            'avg_time_per_frame': avg_time_per_frame
        }
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def main():
    """Compare different YOLOv11 models"""
    
    test_video = "Ema_very_short.mp4"
    
    # Available YOLOv11 models
    models = [
        'yolo11n-seg.pt',  # Nano - fastest
        'yolo11s-seg.pt',  # Small - balanced
        'yolo11m-seg.pt',  # Medium - better
        'yolo11l-seg.pt',  # Large - high accuracy
        'yolo11x-seg.pt',  # XLarge - best accuracy
    ]
    
    print("🏁 YOLOv11 Model Performance Comparison")
    print("="*60)
    print(f"Test video: {test_video}")
    print(f"Confidence threshold: 0.3")
    print("="*60)
    
    results = []
    
    for model_name in models:
        result = test_model_performance(model_name, test_video)
        if result:
            results.append(result)
    
    # Print comparison table
    if results:
        print("\n" + "="*80)
        print("📊 PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(f"{'Model':<15} {'Params':<12} {'Size':<10} {'Load(s)':<8} {'FPS':<8} {'ms/frame':<10}")
        print("-"*80)
        
        for result in results:
            print(f"{result['model']:<15} {result['parameters']:<12} {result['model_size']:<10} "
                  f"{result['load_time']:<8.2f} {result['fps']:<8.2f} {result['avg_time_per_frame']*1000:<10.1f}")
        
        print("="*80)
        
        # Find best models
        fastest = max(results, key=lambda x: x['fps'])
        smallest = min(results, key=lambda x: x['parameters'] if x['parameters'] != 'Unknown' else float('inf'))
        
        print(f"\n🏆 Fastest: {fastest['model']} ({fastest['fps']:.2f} FPS)")
        print(f"🏆 Smallest: {smallest['model']} ({smallest['parameters']} parameters)")
        
        print(f"\n💡 Recommendation: Use {fastest['model']} for speed, {smallest['model']} for efficiency")
    
    print("\n🎯 YOLOv11 comparison completed!")

if __name__ == "__main__":
    main()
