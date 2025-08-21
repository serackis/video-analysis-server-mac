#!/usr/bin/env python3
"""
YOLOv11 Model Accuracy Comparison with Video Output
Focuses on accuracy and saves processed videos for visual comparison
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

class AccuracyComparator:
    """Compare YOLOv11 models for accuracy and save processed videos"""
    
    def __init__(self, input_video, output_dir="accuracy_comparison"):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # YOLOv11 models to test (ordered by expected accuracy)
        self.models = [
            'yolo11x-seg.pt',  # Best accuracy
            'yolo11l-seg.pt',  # High accuracy
            'yolo11m-seg.pt',  # Medium accuracy
            'yolo11s-seg.pt',  # Balanced
            'yolo11n-seg.pt',  # Fastest
        ]
        
        self.results = {}
        
    def process_video_with_model(self, model_name, confidence=0.3):
        """Process video with a specific YOLOv11 model and save results"""
        
        print(f"\n🔄 Processing with {model_name}...")
        
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
                    model_size = info.get('model_size', 'Unknown')
                else:
                    params = 'Unknown'
                    model_size = 'Unknown'
            except:
                params = 'Unknown'
                model_size = 'Unknown'
            
            # Setup output paths
            model_name_clean = model_name.replace('.pt', '').replace('-', '_')
            output_video = self.output_dir / f"{model_name_clean}_processed.mp4"
            stats_file = self.output_dir / f"{model_name_clean}_stats.json"
            
            # Process video with detailed tracking
            cap = cv2.VideoCapture(self.input_video)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {self.input_video}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📊 Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            # Processing statistics
            stats = {
                'model': model_name,
                'parameters': params,
                'model_size': model_size,
                'load_time': load_time,
                'total_frames': total_frames,
                'processed_frames': 0,
                'persons_detected': 0,
                'total_masks': 0,
                'processing_time': 0,
                'start_time': time.time(),
                'detection_confidence': [],
                'mask_areas': [],
                'frames_with_detections': 0,
                'max_persons_in_frame': 0,
                'min_persons_in_frame': float('inf'),
                'detection_details': []
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
                    
                    # Process frame with YOLOv11
                    results = model(frame, conf=confidence, verbose=False)
                    
                    processed_frame = frame.copy()
                    persons_in_frame = 0
                    masks_in_frame = 0
                    frame_detections = []
                    
                    for result in results:
                        boxes = result.boxes
                        masks = result.masks
                        
                        if boxes is not None and masks is not None:
                            for i, (box, mask) in enumerate(zip(boxes, masks)):
                                # Check if it's a person (class 0 in COCO dataset)
                                if box.cls == 0:  # Person class
                                    persons_in_frame += 1
                                    masks_in_frame += 1
                                    
                                    # Get detection details
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                    conf = float(box.conf[0])
                                    
                                    # Get segmentation mask
                                    mask_data = mask.data[0].cpu().numpy()
                                    mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                                    
                                    # Apply segmentation-based depersonalization
                                    processed_frame = self._apply_segmentation_blur(
                                        processed_frame, mask_resized, x1, y1, x2, y2, conf
                                    )
                                    
                                    # Store detection details
                                    detection_info = {
                                        'frame': frame_count,
                                        'person_id': i,
                                        'bbox': [x1, y1, x2, y2],
                                        'confidence': conf,
                                        'mask_area': int(np.sum(mask_resized > 0.5))
                                    }
                                    frame_detections.append(detection_info)
                                    
                                    # Update statistics
                                    stats['detection_confidence'].append(conf)
                                    stats['mask_areas'].append(detection_info['mask_area'])
                    
                    # Update frame statistics
                    if persons_in_frame > 0:
                        stats['frames_with_detections'] += 1
                        stats['max_persons_in_frame'] = max(stats['max_persons_in_frame'], persons_in_frame)
                        stats['min_persons_in_frame'] = min(stats['min_persons_in_frame'], persons_in_frame)
                    
                    stats['persons_detected'] += persons_in_frame
                    stats['total_masks'] += masks_in_frame
                    stats['detection_details'].extend(frame_detections)
                    
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
                
                if stats['detection_confidence']:
                    stats['avg_confidence'] = np.mean(stats['detection_confidence'])
                    stats['min_confidence'] = np.min(stats['detection_confidence'])
                    stats['max_confidence'] = np.max(stats['detection_confidence'])
                
                if stats['mask_areas']:
                    stats['avg_mask_area'] = np.mean(stats['mask_areas'])
                    stats['min_mask_area'] = np.min(stats['mask_areas'])
                    stats['max_mask_area'] = np.max(stats['mask_areas'])
                
                # Save statistics
                self._save_stats(stats, stats_file)
                
                # Print summary
                self._print_model_summary(stats)
                
                return {
                    'model': model_name,
                    'output_video': str(output_video),
                    'stats': stats
                }
                
        except Exception as e:
            print(f"❌ Failed to process with {model_name}: {e}")
            return None
    
    def _apply_segmentation_blur(self, frame, mask, x1, y1, x2, y2, conf):
        """Apply precise depersonalization using segmentation mask with enhanced blur"""
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
        # Base blur strength increased for better depersonalization
        base_blur_strength = 25  # Increased from 15
        
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
        
        # Additional blur based on confidence (higher confidence = stronger blur)
        if conf > 0.8:
            blur_strength += 10
        elif conf > 0.6:
            blur_strength += 5
        
        # Ensure blur strength is odd (OpenCV requirement) and has minimum value
        if blur_strength % 2 == 0:
            blur_strength += 1
        blur_strength = max(15, blur_strength)  # Minimum blur of 15
        
        # Apply multiple blur passes for stronger effect
        blurred_roi = roi.copy()
        
        # First pass: Gaussian blur
        blurred_roi = cv2.GaussianBlur(blurred_roi, (blur_strength, blur_strength), 0)
        
        # Second blur pass for very large persons
        if person_area > 15000:
            # Apply second blur pass for extra depersonalization
            second_blur = max(15, blur_strength - 10)  # Use a reasonable second blur size
            if second_blur % 2 == 0:
                second_blur += 1
            blurred_roi = cv2.GaussianBlur(blurred_roi, (second_blur, second_blur), 0)
        
        # Create 3-channel mask for blending
        mask_3d = np.stack([mask_roi] * 3, axis=2)
        alpha = 0.9  # 90% blurred, 10% original
        
        # Ensure mask_roi is the right shape for broadcasting
        mask_roi_3d = mask_roi[:, :, np.newaxis] if len(mask_roi.shape) == 2 else mask_roi
        
        # Blend the blurred region with the original using the mask
        # Increase blur dominance for better depersonalization
        blur_alpha = 0.9  # 90% blurred, 10% original
        blended_roi = (blurred_roi * mask_roi_3d * blur_alpha + roi * (1 - mask_roi_3d * blur_alpha)).astype(np.uint8)
        
        # Put the processed region back
        result[y1:y2, x1:x2] = blended_roi
        
        return result
    
    def _save_stats(self, stats, stats_file):
        """Save statistics to JSON file"""
        try:
            # Convert numpy types to Python native types
            stats_for_json = {}
            for key, value in stats.items():
                try:
                    if hasattr(value, 'item'):  # numpy scalar
                        stats_for_json[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        stats_for_json[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        stats_for_json[key] = float(value) if isinstance(value, np.floating) else int(value)
                    else:
                        stats_for_json[key] = value
                except:
                    stats_for_json[key] = str(value)
            
            with open(stats_file, 'w') as f:
                json.dump(stats_for_json, f, indent=2)
            
            print(f"📊 Statistics saved to: {stats_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save statistics: {e}")
    
    def _print_model_summary(self, stats):
        """Print summary for a single model"""
        print(f"\n📊 {stats['model']} Results:")
        print(f"   Persons detected: {stats['persons_detected']}")
        print(f"   Total masks: {stats['total_masks']}")
        print(f"   Frames with detections: {stats['frames_with_detections']}/{stats['total_frames']}")
        print(f"   Processing time: {stats['processing_time']:.2f}s")
        print(f"   Processing FPS: {stats['fps_processing']:.2f}")
        
        if 'avg_confidence' in stats:
            print(f"   Confidence: {stats['avg_confidence']:.3f} (min: {stats['min_confidence']:.3f}, max: {stats['max_confidence']:.3f})")
        
        if 'avg_mask_area' in stats:
            print(f"   Mask area: {stats['avg_mask_area']:.0f} pixels (min: {stats['min_mask_area']}, max: {stats['max_mask_area']})")
    
    def compare_all_models(self, confidence=0.3):
        """Compare all YOLOv11 models and generate comprehensive report"""
        
        print("🏁 YOLOv11 Model Accuracy Comparison")
        print("="*60)
        print(f"Input video: {self.input_video}")
        print(f"Output directory: {self.output_dir}")
        print(f"Confidence threshold: {confidence}")
        print("="*60)
        
        start_time = time.time()
        
        # Process with each model
        for model_name in self.models:
            result = self.process_video_with_model(model_name, confidence)
            if result:
                self.results[model_name] = result
        
        total_time = time.time() - start_time
        
        # Generate comparison report
        self._generate_comparison_report()
        
        print(f"\n🎯 Comparison completed in {total_time:.2f}s")
        print(f"📁 Processed videos saved to: {self.output_dir}")
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        
        if not self.results:
            print("❌ No results to compare")
            return
        
        report_file = self.output_dir / "accuracy_comparison_report.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("YOLOv11 Model Accuracy Comparison Report\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input video: {self.input_video}\n\n")
                
                # Summary table
                f.write("SUMMARY TABLE\n")
                f.write("-"*60 + "\n")
                f.write(f"{'Model':<20} {'Persons':<8} {'Masks':<8} {'FPS':<8} {'Time(s)':<8} {'Confidence':<12}\n")
                f.write("-"*60 + "\n")
                
                for model_name, result in self.results.items():
                    stats = result['stats']
                    persons = stats['persons_detected']
                    masks = stats['total_masks']
                    fps = stats['fps_processing']
                    time_taken = stats['processing_time']
                    conf = stats.get('avg_confidence', 0)
                    
                    f.write(f"{model_name:<20} {persons:<8} {masks:<8} {fps:<8.2f} {time_taken:<8.2f} {conf:<12.3f}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("DETAILED ANALYSIS\n")
                f.write("="*60 + "\n")
                
                # Detailed analysis for each model
                for model_name, result in self.results.items():
                    stats = result['stats']
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Parameters: {stats.get('parameters', 'Unknown')}\n")
                    f.write(f"  Model size: {stats.get('model_size', 'Unknown')}\n")
                    f.write(f"  Persons detected: {stats['persons_detected']}\n")
                    f.write(f"  Total masks: {stats['total_masks']}\n")
                    f.write(f"  Frames with detections: {stats['frames_with_detections']}/{stats['total_frames']}\n")
                    f.write(f"  Processing FPS: {stats['fps_processing']:.2f}\n")
                    f.write(f"  Processing time: {stats['processing_time']:.2f}s\n")
                    
                    if 'avg_confidence' in stats:
                        f.write(f"  Average confidence: {stats['avg_confidence']:.3f}\n")
                        f.write(f"  Confidence range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}\n")
                    
                    if 'avg_mask_area' in stats:
                        f.write(f"  Average mask area: {stats['avg_mask_area']:.0f} pixels\n")
                        f.write(f"  Mask area range: {stats['min_mask_area']} - {stats['max_mask_area']} pixels\n")
                    
                    f.write(f"  Output video: {result['output_video']}\n")
                    f.write(f"  Statistics: {result['output_video'].replace('.mp4', '_stats.json')}\n")
                
                # Recommendations
                f.write("\n" + "="*60 + "\n")
                f.write("RECOMMENDATIONS\n")
                f.write("="*60 + "\n")
                
                # Find best models by different criteria
                if self.results:
                    best_accuracy = max(self.results.values(), key=lambda x: x['stats']['persons_detected'])
                    best_confidence = max(self.results.values(), key=lambda x: x['stats'].get('avg_confidence', 0))
                    fastest = max(self.results.values(), key=lambda x: x['stats']['fps_processing'])
                    
                    f.write(f"Best Accuracy (Most Detections): {best_accuracy['stats']['model']}\n")
                    f.write(f"Best Confidence: {best_confidence['stats']['model']}\n")
                    f.write(f"Fastest Processing: {fastest['stats']['model']}\n")
                    
                    f.write(f"\nFor production use with focus on accuracy:\n")
                    f.write(f"  Primary: {best_accuracy['stats']['model']}\n")
                    f.write(f"  Secondary: {best_confidence['stats']['model']}\n")
                    f.write(f"  Development/Testing: {fastest['stats']['model']}\n")
            
            print(f"📊 Comparison report saved to: {report_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save comparison report: {e}")

def main():
    """Main function"""
    
    input_video = "Ema_very_short.mp4"
    
    if not Path(input_video).exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    # Create comparator and run comparison
    comparator = AccuracyComparator(input_video)
    comparator.compare_all_models(confidence=0.3)

if __name__ == "__main__":
    main()
