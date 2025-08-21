#!/usr/bin/env python3
"""
Test Enhanced Blur with Best Accuracy Model
Now with optional tracking support
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json

def test_enhanced_blur(input_video, output_dir="enhanced_blur_test", enable_tracking=False):
    """Test enhanced blur with the best accuracy model"""
    
    # Use the best accuracy model from our comparison
    best_model = 'yolo11l-seg.pt'
    
    print("🎯 Testing Enhanced Blur with Best Accuracy Model")
    print("="*60)
    print(f"Model: {best_model}")
    print(f"Input video: {input_video}")
    print(f"Output directory: {output_dir}")
    print(f"Tracking: {'✅ Enabled' if enable_tracking else '❌ Disabled'}")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Load model
        start_time = time.time()
        model = YOLO(best_model)
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Setup video capture and writer
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {input_video}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer
        tracking_suffix = "_tracked" if enable_tracking else ""
        output_video = output_path / f"enhanced_blur{tracking_suffix}_{Path(input_video).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'model': best_model,
            'tracking_enabled': enable_tracking,
            'total_frames': total_frames,
            'processed_frames': 0,
            'persons_detected': 0,
            'total_masks': 0,
            'unique_tracks': set() if enable_tracking else None,
            'processing_time': 0,
            'start_time': time.time(),
            'blur_strengths_used': [],
            'person_sizes': [],
            'confidence_levels': []
        }
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"🔄 Frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
                
                # Process frame with YOLOv11 (with or without tracking)
                if enable_tracking:
                    results = model.track(frame, conf=0.3, verbose=False, persist=10)
                else:
                    results = model(frame, conf=0.3, verbose=False)
                
                processed_frame = frame.copy()
                persons_in_frame = 0
                masks_in_frame = 0
                
                # Debug: Check if results contain detections
                if len(results) > 0:
                    result = results[0]  # Get first result
                    boxes = result.boxes
                    masks = result.masks
                    
                    if boxes is not None and masks is not None:
                        # Get tracking information if enabled
                        track_ids = None
                        if enable_tracking and hasattr(boxes, 'id') and boxes.id is not None:
                            track_ids = boxes.id.cpu().numpy()
                        
                        for i, (box, mask) in enumerate(zip(boxes, masks)):
                            # Check if it's a person (class 0 in COCO dataset)
                            if box.cls == 0:  # Person class
                                persons_in_frame += 1
                                masks_in_frame += 1
                                
                                # Get detection details
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                conf = float(box.conf[0])
                                
                                # Get track ID if tracking is enabled
                                track_id = None
                                if enable_tracking and track_ids is not None:
                                    track_id = int(track_ids[i])
                                    stats['unique_tracks'].add(track_id)
                                
                                # Get segmentation mask
                                mask_data = mask.data[0].cpu().numpy()
                                mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                                
                                # Apply enhanced segmentation-based depersonalization
                                processed_frame = apply_enhanced_blur(
                                    processed_frame, mask_resized, x1, y1, x2, y2, conf
                                )
                                
                                # Draw tracking info if enabled
                                if enable_tracking and track_id is not None:
                                    processed_frame = draw_tracking_info(
                                        processed_frame, x1, y1, x2, y2, conf, track_id
                                    )
                                
                                # Store statistics
                                person_area = (x2 - x1) * (y2 - y1)
                                blur_strength = calculate_blur_strength(person_area, conf)
                                
                                stats['blur_strengths_used'].append(blur_strength)
                                stats['person_sizes'].append(person_area)
                                stats['confidence_levels'].append(conf)
                
                # Update statistics
                stats['persons_detected'] += persons_in_frame
                stats['total_masks'] += masks_in_frame
                
                # Write processed frame
                out.write(processed_frame)
                stats['processed_frames'] += 1
                
        except KeyboardInterrupt:
            print("\n⏹️  Processing interrupted by user")
        finally:
            cap.release()
            out.release()
            
            # Calculate final statistics
            stats['processing_time'] = time.time() - stats['start_time']
            stats['fps_processing'] = stats['processed_frames'] / stats['processing_time']
            
            if enable_tracking:
                stats['unique_tracks'] = len(stats['unique_tracks'])
            
            if stats['blur_strengths_used']:
                stats['avg_blur_strength'] = np.mean(stats['blur_strengths_used'])
                stats['min_blur_strength'] = np.min(stats['blur_strengths_used'])
                stats['max_blur_strength'] = np.max(stats['blur_strengths_used'])
            
            if stats['person_sizes']:
                stats['avg_person_size'] = np.mean(stats['person_sizes'])
                stats['min_person_size'] = np.min(stats['person_sizes'])
                stats['max_person_size'] = np.max(stats['person_sizes'])
            
            if stats['confidence_levels']:
                stats['avg_confidence'] = np.mean(stats['confidence_levels'])
                stats['min_confidence'] = np.min(stats['confidence_levels'])
                stats['max_confidence'] = np.max(stats['confidence_levels'])
            
            # Print summary
            print_summary(stats)
            
            # Save statistics
            stats_file = output_path / f"enhanced_blur{tracking_suffix}_stats.json"
            save_stats(stats, stats_file)
            
            print(f"\n✅ Enhanced blur processing completed!")
            print(f"📁 Output video: {output_video}")
            print(f"📊 Statistics: {stats_file}")
            
            return str(output_video)
            
    except Exception as e:
        print(f"❌ Failed to process with enhanced blur: {e}")
        return None

def draw_tracking_info(frame, x1, y1, x2, y2, conf, track_id):
    """Draw tracking information on the frame"""
    # Draw bounding box
    color = (0, 255, 0)  # Green for tracking
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw track ID and confidence
    label = f"Track {track_id} {conf:.2f}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Background rectangle for label
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                 (x1 + label_size[0], y1), (0, 255, 0), -1)
    
    # Label text
    cv2.putText(frame, label, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

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
        second_blur = max(15, blur_strength - 10)
        if second_blur % 2 == 0:
            second_blur += 1
        blurred_roi = cv2.GaussianBlur(blurred_roi, (second_blur, second_blur), 0)
    
    # Create 3-channel mask and blend with high blur dominance
    mask_roi_3d = mask_roi[:, :, np.newaxis] if len(mask_roi.shape) == 2 else mask_roi
    blur_alpha = 0.9  # 90% blurred, 10% original
    blended_roi = (blurred_roi * mask_roi_3d * blur_alpha + roi * (1 - mask_roi_3d * blur_alpha)).astype(np.uint8)
    
    result[y1:y2, x1:x2] = blended_roi
    return result

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

def print_summary(stats):
    """Print processing summary"""
    print(f"\n📊 Enhanced Blur Processing Summary:")
    print(f"   Model: {stats['model']}")
    print(f"   Tracking: {'✅ Enabled' if stats['tracking_enabled'] else '❌ Disabled'}")
    print(f"   Persons detected: {stats['persons_detected']}")
    print(f"   Total masks: {stats['total_masks']}")
    print(f"   Processing time: {stats['processing_time']:.2f}s")
    print(f"   Processing FPS: {stats['fps_processing']:.2f}")
    
    if stats['tracking_enabled'] and 'unique_tracks' in stats:
        print(f"   Unique tracks: {stats['unique_tracks']}")
    
    if 'avg_blur_strength' in stats:
        print(f"   Blur strength: {stats['avg_blur_strength']:.1f} (min: {stats['min_blur_strength']}, max: {stats['max_blur_strength']})")
    
    if 'avg_person_size' in stats:
        print(f"   Person size: {stats['avg_person_size']:.0f} pixels (min: {stats['min_person_size']}, max: {stats['max_person_size']})")
    
    if 'avg_confidence' in stats:
        print(f"   Confidence: {stats['avg_confidence']:.3f} (min: {stats['min_confidence']:.3f}, max: {stats['max_confidence']:.3f})")

def save_stats(stats, stats_file):
    """Save statistics to JSON file"""
    try:
        # Convert numpy types for JSON
        stats_for_json = {}
        for key, value in stats.items():
            try:
                if isinstance(value, np.ndarray):
                    stats_for_json[key] = value.tolist()
                elif isinstance(value, np.floating):
                    stats_for_json[key] = float(value)
                elif isinstance(value, np.integer):
                    stats_for_json[key] = int(value)
                elif isinstance(value, set):  # Handle set of track IDs
                    stats_for_json[key] = list(value)
                else:
                    stats_for_json[key] = value
            except:
                stats_for_json[key] = str(value)
        
        with open(stats_file, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        
        print(f"📊 Statistics saved to: {stats_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save statistics: {e}")

def main():
    """Main function"""
    
    input_video = "Ema_very_short.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    # Test enhanced blur without tracking
    print("🔄 Testing Enhanced Blur WITHOUT tracking...")
    output_video = test_enhanced_blur(input_video, "enhanced_blur_test", enable_tracking=False)
    
    if output_video:
        print(f"\n🎯 Enhanced blur (no tracking) completed!")
        
        # Test enhanced blur with tracking
        print("\n🔄 Testing Enhanced Blur WITH tracking...")
        output_video_tracked = test_enhanced_blur(input_video, "enhanced_blur_test", enable_tracking=True)
        
        if output_video_tracked:
            print(f"\n🎯 Enhanced blur with tracking completed!")
            print(f"💡 Compare both videos to see tracking benefits")
    
    print(f"\n🎯 Enhanced blur testing completed!")
    print(f"💡 Check the output videos for comparison")

if __name__ == "__main__":
    main()
