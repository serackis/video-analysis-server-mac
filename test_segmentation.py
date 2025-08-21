#!/usr/bin/env python3
"""
Test YOLO segmentation models
"""

import cv2
import numpy as np
from ultralytics import YOLO

def test_segmentation_models():
    """Test different YOLOv11 segmentation models"""
    
    # Available YOLOv11 segmentation models (latest)
    models = [
        'yolo11n-seg.pt',  # Nano - fastest
        'yolo11s-seg.pt',  # Small - balanced
        'yolo11m-seg.pt',  # Medium - better
        'yolo11l-seg.pt',  # Large - high accuracy
        'yolo11x-seg.pt',  # XLarge - best accuracy
        'yolo11v-seg.pt',  # Vision - optimized
        'yolo11c-seg.pt',  # Compact - efficient
    ]
    
    print("🧪 Testing YOLOv11 Segmentation Models (Latest)")
    print("="*60)
    
    for model_name in models:
        print(f"\n🔄 Testing model: {model_name}")
        
        try:
            # Load model
            model = YOLO(model_name)
            print(f"✅ Model loaded successfully")
            
            # Get model info
            info = model.info()
            print(f"📊 Parameters: {info.get('parameters', 'Unknown')}")
            print(f"📊 Size: {info.get('model_size', 'Unknown')}")
            
            # Test on a simple frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame[200:280, 250:390] = [255, 255, 255]  # White rectangle
            
            # Run inference
            results = model(test_frame, conf=0.1, verbose=False)
            
            print(f"✅ Inference successful")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n" + "="*60)
    print("🎯 Ready to use YOLOv11 segmentation models!")

if __name__ == "__main__":
    test_segmentation_models()
