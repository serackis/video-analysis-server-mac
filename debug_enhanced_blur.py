#!/usr/bin/env python3
"""
Debug Enhanced Blur on Single Frame
Test the enhanced blur algorithm step by step
"""

import cv2
import numpy as np
from ultralytics import YOLO

def debug_enhanced_blur():
    """Debug enhanced blur on a single frame"""
    
    print("🔍 Debugging Enhanced Blur Algorithm")
    print("="*50)
    
    # Load model
    model = YOLO('yolo11l-seg.pt')
    print("✅ Model loaded")
    
    # Load video and get first frame
    cap = cv2.VideoCapture("Ema_very_short.mp4")
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        return
    
    cap.release()
    
    print(f"📊 Frame shape: {frame.shape}")
    
    # Process frame with YOLO
    print("🔄 Processing frame with YOLO...")
    results = model(frame, conf=0.3, verbose=False)
    
    print(f"📊 Results length: {len(results)}")
    
    if len(results) > 0:
        result = results[0]
        print(f"📊 Result type: {type(result)}")
        
        boxes = result.boxes
        masks = result.masks
        
        print(f"📊 Boxes: {boxes is not None}")
        print(f"📊 Masks: {masks is not None}")
        
        if boxes is not None and masks is not None:
            print(f"📊 Number of boxes: {len(boxes)}")
            print(f"📊 Number of masks: {len(masks)}")
            
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                print(f"\n🔍 Detection {i+1}:")
                print(f"   Class: {box.cls[0]}")
                print(f"   Confidence: {box.conf[0]:.3f}")
                
                if box.cls == 0:  # Person class
                    print(f"   ✅ This is a person!")
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    print(f"   BBox: ({x1}, {y1}) to ({x2}, {y2})")
                    
                    person_area = (x2 - x1) * (y2 - y1)
                    print(f"   Area: {person_area} pixels")
                    
                    # Calculate blur strength
                    blur_strength = calculate_blur_strength(person_area, float(box.conf[0]))
                    print(f"   Blur strength: {blur_strength}")
                    
                    # Test blur application
                    mask_data = mask.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                    
                    print(f"   Mask shape: {mask_resized.shape}")
                    print(f"   Mask range: {mask_resized.min():.3f} to {mask_resized.max():.3f}")
                    
                    # Apply blur
                    processed_frame = apply_enhanced_blur(
                        frame, mask_resized, x1, y1, x2, y2, float(box.conf[0])
                    )
                    
                    print(f"   ✅ Blur applied successfully!")
                    
                    # Save test frame
                    cv2.imwrite("debug_enhanced_blur_frame.jpg", processed_frame)
                    print(f"   📁 Test frame saved as: debug_enhanced_blur_frame.jpg")
                    
                    break  # Just test first person
                else:
                    print(f"   ❌ Not a person (class {box.cls[0]})")
        else:
            print("❌ No boxes or masks found")
    else:
        print("❌ No results from YOLO")
    
    print("\n🔍 Debug completed!")

def calculate_blur_strength(person_area, confidence):
    """Calculate blur strength for given person area and confidence"""
    base_blur_strength = 25
    
    if person_area > 15000:
        blur_strength = base_blur_strength + 15
    elif person_area > 8000:
        blur_strength = base_blur_strength + 10
    elif person_area > 3000:
        blur_strength = base_blur_strength + 5
    else:
        blur_strength = base_blur_strength
    
    if confidence > 0.8:
        blur_strength += 10
    elif confidence > 0.6:
        blur_strength += 5
    
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    return max(15, blur_strength)

def apply_enhanced_blur(frame, mask, x1, y1, x2, y2, conf):
    """Apply enhanced depersonalization blur for better privacy"""
    result = frame.copy()
    
    # Ensure coordinates are within frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    # Extract the region of interest
    roi = frame[y1:y2, x1:x2]
    mask_roi = mask[y1:y2, x1:x2]
    
    if roi.size == 0 or mask_roi.size == 0:
        return result
    
    # Enhanced blur strength based on person size and confidence
    base_blur_strength = 25  # Increased base blur
    
    # Calculate person area and adjust blur based on size
    person_area = (x2 - x1) * (y2 - y1)
    
    # More aggressive blur for bigger/closer persons
    if person_area > 15000:  # Large/close person
        blur_strength = base_blur_strength + 15  # Very strong blur
    elif person_area > 8000:   # Medium person
        blur_strength = base_blur_strength + 10  # Strong blur
    elif person_area > 3000:   # Small person
        blur_strength = base_blur_strength + 5   # Medium blur
    else:                       # Very small person
        blur_strength = base_blur_strength       # Base blur
    
    # Additional blur based on confidence
    if conf > 0.8:
        blur_strength += 10
    elif conf > 0.6:
        blur_strength += 5
    
    # Ensure blur strength is odd and has minimum value
    if blur_strength % 2 == 0:
        blur_strength += 1
    blur_strength = max(15, blur_strength)
    
    # Apply multiple blur passes for stronger effect
    blurred_roi = roi.copy()
    blurred_roi = cv2.GaussianBlur(blurred_roi, (blur_strength, blur_strength), 0)
    
    # Second blur pass for very large persons
    if person_area > 15000:
        # Ensure second blur kernel is also odd and valid
        second_blur = max(15, blur_strength - 10)  # Use a reasonable second blur size
        if second_blur % 2 == 0:
            second_blur += 1
        blurred_roi = cv2.GaussianBlur(blurred_roi, (second_blur, second_blur), 0)
    
    # Create 3-channel mask and blend with high blur dominance
    mask_3d = np.stack([mask_roi] * 3, axis=2)
    blur_alpha = 0.9  # 90% blurred, 10% original
    
    # Ensure mask_roi is the right shape for broadcasting
    mask_roi_3d = mask_roi[:, :, np.newaxis] if len(mask_roi.shape) == 2 else mask_roi
    
    blended_roi = (blurred_roi * mask_roi_3d * blur_alpha + roi * (1 - mask_roi_3d * blur_alpha)).astype(np.uint8)
    
    result[y1:y2, x1:x2] = blended_roi
    return result

if __name__ == "__main__":
    debug_enhanced_blur()
