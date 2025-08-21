#!/usr/bin/env python3
"""
Test YOLO detection on a single frame from the video
"""

import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_on_frame():
    """Test YOLO detection on a single frame"""
    
    # Load YOLO model
    print("🔄 Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("✅ YOLO model loaded")
    
    # Open video and read first frame
    print("📹 Reading video frame...")
    cap = cv2.VideoCapture('Ema_very_short.mp4')
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read video frame")
        return
    
    print(f"📊 Frame shape: {frame.shape}")
    
    # Run YOLO detection with very low confidence
    print("🔍 Running YOLO detection...")
    results = model(frame, conf=0.1, verbose=True)  # Very low confidence
    
    # Process results
    for i, result in enumerate(results):
        print(f"\n📋 Result {i+1}:")
        
        if result.boxes is not None:
            boxes = result.boxes
            print(f"   📦 Detected {len(boxes)} objects")
            
            for j, box in enumerate(boxes):
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy()
                
                # Get class name (COCO dataset)
                class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
                class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                
                print(f"      Object {j+1}: {class_name} (confidence: {conf:.3f})")
                print(f"         Coordinates: {coords}")
                
                # Draw bounding box on frame
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("   ❌ No objects detected")
    
    # Save annotated frame
    output_path = "yolo_test_frame.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n💾 Annotated frame saved to: {output_path}")
    
    # Show frame
    print("🖼️  Displaying frame (press any key to close)...")
    cv2.imshow('YOLO Detection Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_yolo_on_frame()
