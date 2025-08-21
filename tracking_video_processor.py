#!/usr/bin/env python3
"""
Tracking-Enabled Video Processor with YOLOv11
Uses YOLOv11 segmentation models with tracking for consistent person depersonalization
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import json
from ultralytics import YOLO

class TrackingVideoProcessor:
    """Advanced video processor using YOLOv11 with tracking for consistent depersonalization"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.model = None
        self.device = 'cuda' if self.config.get('use_gpu', False) else 'cpu'
        print(f"🚀 Using device: {self.device}")
        self._load_model()
        
    def _default_config(self) -> Dict:
        return {
            'yolo_model': 'yolo11l-seg.pt',  # Best accuracy model
            'confidence_threshold': 0.3,
            'iou_threshold': 0.45,
            'tracking': True,
            'tracking_persistence': 10,  # Frames to keep tracking without detection
            'tracking_smooth': True,     # Smooth tracking predictions
            'person_only': True,
            'blur_strength': 25,        # Enhanced base blur
            'output_format': 'mp4',
            'save_intermediate': True,
            'show_preview': False,
            'save_masks': True,
            'mask_alpha': 0.3,
            'draw_boxes': False,        # Disable bounding boxes
            'draw_masks': False,        # Disable mask visualization
            'draw_labels': False,       # Disable labels
            'draw_tracks': False,       # Disable tracking visualization
            'enhanced_blur': True,
            'preserve_edges': True,
            'adaptive_blur': True,
            'tracking_visualization': False,  # Disable tracking visualization
            'save_tracking_data': True,
        }
    
    def _load_model(self):
        print(f"📥 Loading YOLOv11 tracking model: {self.config['yolo_model']}")
        try:
            self.model = YOLO(self.config['yolo_model'])
            print(f"✅ Loaded YOLOv11 model: {self.config['yolo_model']}")
        except Exception as e:
            print(f"❌ Failed to load YOLOv11 model: {e}")
            exit(1)
    
    def process_video(self, input_path: str, output_path: str = None) -> str:
        print(f"🎬 Processing video with tracking: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        if output_path is None:
            input_dir = Path(input_path).parent
            input_name = Path(input_path).stem
            output_path = input_dir / f"tracked_{input_name}.{self.config['output_format']}"
        
        # Setup video capture and writer
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Tracking statistics
        self.stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'persons_detected': 0,
            'total_masks': 0,
            'unique_tracks': 0,
            'tracking_switches': 0,
            'processing_time': 0,
            'start_time': time.time(),
            'detection_confidence': [],
            'mask_areas': [],
            'track_ids': set(),
            'tracking_data': []
        }
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"🔄 Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
                
                # Process frame with tracking
                processed_frame = self._process_frame_with_tracking(frame, frame_count)
                out.write(processed_frame)
                self.stats['processed_frames'] += 1
                
                if self.config['show_preview']:
                    cv2.imshow('Tracking Processing Preview', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\n⏹️  Processing interrupted by user")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            self.stats['processing_time'] = time.time() - self.stats['start_time']
            self.stats['fps_processing'] = self.stats['processed_frames'] / self.stats['processing_time']
            self.stats['unique_tracks'] = len(self.stats['track_ids'])
            
            self._print_statistics()
            
            # Save tracking data
            if self.config['save_tracking_data']:
                tracking_file = Path(output_path).with_suffix('.json')
                self._save_tracking_data(tracking_file)
            
            return str(output_path)
    
    def _process_frame_with_tracking(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        try:
            # Process with tracking enabled
            results = self.model.track(
                frame,
                conf=self.config['confidence_threshold'],
                persist=self.config['tracking_persistence'],
                verbose=False
            )
            
            processed_frame = frame.copy()
            persons_in_frame = 0
            masks_in_frame = 0
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                masks = result.masks
                
                if boxes is not None and masks is not None:
                    # Get tracking information
                    track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
                    
                    for i, (box, mask) in enumerate(zip(boxes, masks)):
                        if box.cls == 0:  # Person class
                            persons_in_frame += 1
                            masks_in_frame += 1
                            
                            # Get detection details
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0])
                            
                            # Get track ID
                            track_id = int(track_ids[i]) if track_ids is not None else i
                            self.stats['track_ids'].add(track_id)
                            
                            # Get segmentation mask
                            mask_data = mask.data[0].cpu().numpy()
                            mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                            
                            # Apply enhanced segmentation-based depersonalization
                            processed_frame = self._apply_enhanced_blur(
                                processed_frame, mask_resized, x1, y1, x2, y2, conf
                            )
                            
                            # Draw tracking information
                            if self.config['draw_tracks']:
                                processed_frame = self._draw_tracking_info(
                                    processed_frame, x1, y1, x2, y2, conf, track_id, i
                                )
                            
                            # Store tracking data
                            if self.config['save_tracking_data']:
                                tracking_info = {
                                    'frame': frame_number,
                                    'track_id': track_id,
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': conf,
                                    'mask_area': int(np.sum(mask_resized > 0.5))
                                }
                                self.stats['tracking_data'].append(tracking_info)
                            
                            # Update statistics
                            self.stats['detection_confidence'].append(conf)
                            mask_area = np.sum(mask_resized > 0.5)
                            self.stats['mask_areas'].append(mask_area)
            
            self.stats['persons_detected'] += persons_in_frame
            self.stats['total_masks'] += masks_in_frame
            
            if persons_in_frame > 0:
                print(f"👥 Frame {frame_number}: {persons_in_frame} person(s) with {len(self.stats['track_ids'])} unique tracks")
            
            return processed_frame
            
        except Exception as e:
            print(f"❌ Tracking processing error: {e}")
            return frame
    
    def _apply_enhanced_blur(self, frame: np.ndarray, mask: np.ndarray,
                             x1: int, y1: int, x2: int, y2: int, conf: float) -> np.ndarray:
        """Apply enhanced depersonalization blur with tracking consistency"""
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
        base_blur_strength = self.config['blur_strength']
        
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
    
    def _draw_tracking_info(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                           conf: float, track_id: int, index: int) -> np.ndarray:
        """Draw tracking information on the frame"""
        if not self.config['draw_tracks']:
            return frame
        
        # Draw bounding box
        if self.config['draw_boxes']:
            color = (0, 255, 0)  # Green for tracking
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw track ID and confidence
        if self.config['draw_labels']:
            label = f"Track {track_id} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for label
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def _save_tracking_data(self, tracking_file: Path):
        """Save tracking data to JSON file"""
        try:
            # Convert numpy types for JSON
            tracking_data_for_json = []
            for item in self.stats['tracking_data']:
                item_for_json = {}
                for key, value in item.items():
                    if isinstance(value, np.ndarray):
                        item_for_json[key] = value.tolist()
                    elif isinstance(value, np.floating):
                        item_for_json[key] = float(value)
                    elif isinstance(value, np.integer):
                        item_for_json[key] = int(value)
                    else:
                        item_for_json[key] = value
                tracking_data_for_json.append(item_for_json)
            
            # Prepare final data
            final_data = {
                'video_info': {
                    'model': self.config['yolo_model'],
                    'total_frames': self.stats['total_frames'],
                    'processing_time': self.stats['processing_time'],
                    'fps_processing': self.stats['fps_processing']
                },
                'tracking_stats': {
                    'unique_tracks': self.stats['unique_tracks'],
                    'total_detections': self.stats['persons_detected'],
                    'total_masks': self.stats['total_masks']
                },
                'tracking_data': tracking_data_for_json
            }
            
            with open(tracking_file, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            print(f"📊 Tracking data saved to: {tracking_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save tracking data: {e}")
    
    def _print_statistics(self):
        print("\n" + "="*60)
        print("📊 TRACKING PROCESSING STATISTICS")
        print("="*60)
        print(f"Total frames: {self.stats['total_frames']}")
        print(f"Processed frames: {self.stats['processed_frames']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"Processing FPS: {self.stats['fps_processing']:.2f}")
        print(f"Total persons detected: {self.stats['persons_detected']}")
        print(f"Total masks generated: {self.stats['total_masks']}")
        print(f"Unique tracks: {self.stats['unique_tracks']}")
        
        if self.stats['detection_confidence']:
            avg_conf = np.mean(self.stats['detection_confidence'])
            print(f"Average detection confidence: {avg_conf:.3f}")
        
        if self.stats['mask_areas']:
            avg_mask_area = np.mean(self.stats['mask_areas'])
            print(f"Average mask area: {avg_mask_area:.0f} pixels")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Tracking Video Processor')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('-m', '--model', default='yolo11l-seg.pt',
                       help='YOLOv11 segmentation model (default: yolo11l-seg.pt)')
    parser.add_argument('-c', '--confidence', type=float, default=0.3,
                       help='Detection confidence threshold (default: 0.3)')
    parser.add_argument('-b', '--blur', type=int, default=25,
                       help='Base blur strength (default: 25)')
    parser.add_argument('-p', '--preview', action='store_true',
                       help='Show preview during processing')
    parser.add_argument('--no-tracking', action='store_true',
                       help='Disable tracking')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable tracking visualization')
    parser.add_argument('--clean-output', action='store_true',
                       help='Remove all bounding boxes, labels, and tracking info for clean depersonalized video')
    parser.add_argument('--persistence', type=int, default=10,
                       help='Tracking persistence frames (default: 10)')
    
    args = parser.parse_args()
    
    config = {
        'yolo_model': args.model,
        'confidence_threshold': args.confidence,
        'blur_strength': args.blur,
        'show_preview': args.preview,
        'tracking': not args.no_tracking,
        'tracking_visualization': not args.no_visualization,
        'tracking_persistence': args.persistence,
        'draw_tracks': not args.no_visualization and not args.clean_output,
        'draw_boxes': not args.clean_output,
        'draw_labels': not args.clean_output,
        'draw_masks': False,
        'output_format': 'mp4',
        'save_intermediate': True,
        'save_masks': True,
        'mask_alpha': 0.3,
        'person_only': True,
        'enhanced_blur': True,
        'preserve_edges': True,
        'adaptive_blur': True,
        'save_tracking_data': True,
    }
    
    processor = TrackingVideoProcessor(config)
    
    if args.clean_output:
        print("🧹 Clean output mode: All bounding boxes, labels, and tracking info will be removed")
        print("📹 Output will be a clean, depersonalized video with enhanced blur only")
    
    try:
        output_path = processor.process_video(args.input_video, args.output)
        print(f"\n✅ Tracking processing completed! Output saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
