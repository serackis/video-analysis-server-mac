#!/usr/bin/env python3
"""
Test Best Accuracy YOLOv11 Models
Focuses on the top 3 models for accuracy comparison
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json

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
        blurred_roi = cv2.GaussianBlur(blurred_roi, (blur_strength-5, blur_strength-5), 0)
    
    # Create 3-channel mask and blend with high blur dominance
    mask_3d = np.stack([mask_roi] * 3, axis=2)
    blur_alpha = 0.9  # 90% blurred, 10% original
    blended_roi = (blurred_roi * mask_3d * blur_alpha + roi * (1 - mask_roi * blur_alpha)).astype(np.uint8)
    
    result[y1:y2, x1:x2] = blended_roi
    return result

def test_best_accuracy_models(input_video, confidence=0.3):
    """Test the top 3 YOLOv11 models for accuracy"""
    
    # Top 3 models for accuracy (ordered by expected performance)
    best_models = [
        'yolo11x-seg.pt',  # Best accuracy (62M params)
        'yolo11l-seg.pt',  # High accuracy (28M params) 
        'yolo11m-seg.pt',  # Medium accuracy (22M params)
    ]
    
    print("🎯 Testing Best Accuracy YOLOv11 Models")
    print("="*60)
    print(f"Input video: {input_video}")
    print(f"Confidence threshold: {confidence}")
    print("="*60)
    
    results = {}
    
    for model_name in best_models:
        print(f"\n🔄 Testing {model_name}...")
        
        try:
            # Load model
            start_time = time.time()
            model = YOLO(model_name)
            load_time = time.time() - start_time
            
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Get model info
            try:
                info = model.info()
                if isinstance(info, dict):
                    params = info.get('parameters', 'Unknown')
                else:
                    params = 'Unknown'
            except:
                params = 'Unknown'
            
            # Process video
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {input_video}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Process with model
            start_time = time.time()
            results_yolo = model(input_video, conf=confidence, verbose=False)
            process_time = time.time() - start_time
            
            # Analyze results
            total_persons = 0
            total_confidence = []
            
            for result in results_yolo:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.cls == 0:  # Person class
                            total_persons += 1
                            total_confidence.append(float(box.conf[0]))
            
            # Calculate metrics
            avg_confidence = np.mean(total_confidence) if total_confidence else 0
            fps = total_frames / process_time
            
            model_result = {
                'model': model_name,
                'parameters': params,
                'load_time': load_time,
                'process_time': process_time,
                'total_frames': total_frames,
                'persons_detected': total_persons,
                'avg_confidence': avg_confidence,
                'fps': fps,
                'confidence_list': total_confidence
            }
            
            results[model_name] = model_result
            
            print(f"📊 Results:")
            print(f"   Persons detected: {total_persons}")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Processing time: {process_time:.2f}s")
            print(f"   Processing FPS: {fps:.2f}")
            
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
    
    # Generate comparison
    if results:
        print("\n" + "="*60)
        print("📊 ACCURACY COMPARISON RESULTS")
        print("="*60)
        
        # Sort by persons detected (accuracy metric)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['persons_detected'], reverse=True)
        
        print(f"{'Model':<20} {'Persons':<8} {'Confidence':<12} {'FPS':<8} {'Time(s)':<8}")
        print("-"*60)
        
        for model_name, result in sorted_results:
            print(f"{model_name:<20} {result['persons_detected']:<8} {result['avg_confidence']:<12.3f} "
                  f"{result['fps']:<8.2f} {result['process_time']:<8.2f}")
        
        print("="*60)
        
        # Find best model
        best_model = sorted_results[0]
        print(f"\n🏆 Best Accuracy Model: {best_model[0]}")
        print(f"   Persons detected: {best_model[1]['persons_detected']}")
        print(f"   Average confidence: {best_model[1]['avg_confidence']:.3f}")
        
        # Save results
        output_file = "best_accuracy_results.json"
        try:
            # Convert numpy types for JSON
            results_for_json = {}
            for model_name, result in results.items():
                results_for_json[model_name] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        results_for_json[model_name][key] = value.tolist()
                    elif isinstance(value, np.floating):
                        results_for_json[model_name][key] = float(value)
                    elif isinstance(value, np.integer):
                        results_for_json[model_name][key] = int(value)
                    else:
                        results_for_json[model_name][key] = value
            
            with open(output_file, 'w') as f:
                json.dump(results_for_json, f, indent=2)
            
            print(f"\n📊 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    return results

def main():
    """Main function"""
    
    input_video = "Ema_very_short.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    # Test best accuracy models
    results = test_best_accuracy_models(input_video, confidence=0.3)
    
    if results:
        print(f"\n🎯 Best accuracy testing completed!")
        print(f"💡 Use the top model for production depersonalization")

if __name__ == "__main__":
    main()
