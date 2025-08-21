#!/usr/bin/env python3
"""
Create a video with visible YOLO detection results
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def create_detection_video(input_video, output_video, confidence=0.3):
    """Create a video showing YOLO detection results"""
    
    print(f"🎬 Creating detection video: {input_video}")
    
    # Load YOLO model
    print("🔄 Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("✅ YOLO model loaded")
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    total_persons = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"🔄 Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # Run YOLO detection
            results = model(frame, conf=confidence, verbose=False)
            
            # Draw detection results
            annotated_frame = frame.copy()
            persons_in_frame = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if it's a person (class 0 in COCO dataset)
                        if box.cls == 0:  # Person class
                            persons_in_frame += 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0])
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"Person {conf:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            total_persons += persons_in_frame
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Persons: {persons_in_frame}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            # Write frame
            out.write(annotated_frame)
            
            # Show preview (optional)
            cv2.imshow('Detection Preview', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total persons detected: {total_persons}")
        print(f"   Average persons per frame: {total_persons/frame_count:.1f}")
        print(f"✅ Detection video saved to: {output_video}")

def main():
    """Main function"""
    input_video = "Ema_very_short.mp4"
    output_video = "Ema_very_short_with_detections.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    create_detection_video(input_video, output_video, confidence=0.3)

if __name__ == "__main__":
    main()
