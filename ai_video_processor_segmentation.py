#!/usr/bin/env python3
"""
Advanced AI Video Processor with YOLO Segmentation
Uses YOLOv8 segmentation models for precise person detection and depersonalization
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import json

# AI/ML imports
try:
    import torch
    import torchvision
    from ultralytics import YOLO
    print("✅ YOLO and PyTorch available")
except ImportError:
    print("❌ YOLO not available. Install with: pip install ultralytics")
    exit(1)

try:
    import supervision as sv
    print("✅ Supervision available for advanced annotations")
except ImportError:
    print("⚠️  Supervision not available. Install with: pip install supervision")

class SegmentationVideoProcessor:
    """Advanced video processor using YOLO segmentation models"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")

        # Initialize segmentation model
        self._load_model()

    def _default_config(self) -> Dict:
        """Default configuration for segmentation processing"""
        return {
            'yolo_model': 'yolo11n-seg.pt',  # YOLOv11 nano with segmentation (latest)
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'person_only': True,  # Focus on person detection
            'blur_strength': 15,  # Gaussian blur kernel size
            'output_format': 'mp4',
            'save_intermediate': True,
            'show_preview': False,
            'save_masks': True,  # Save segmentation masks
            'mask_alpha': 0.3,   # Transparency for mask overlay
            'draw_boxes': True,  # Draw bounding boxes
            'draw_masks': True,  # Draw segmentation masks
            'draw_labels': True,  # Draw confidence labels
            'enhanced_blur': True,  # Use mask-based blurring
            'preserve_edges': True,  # Preserve edges around persons
            'adaptive_blur': True,  # Adjust blur based on person size
        }

    def _load_model(self):
        """Load YOLO segmentation model"""
        print(f"📥 Loading YOLO segmentation model: {self.config['yolo_model']}")
        
        try:
            self.model = YOLO(self.config['yolo_model'])
            print(f"✅ Loaded YOLO segmentation model: {self.config['yolo_model']}")
            
            # Print model info (handle different return types)
            try:
                model_info = self.model.info()
                if isinstance(model_info, dict):
                    print(f"📊 Model parameters: {model_info.get('parameters', 'Unknown')}")
                    print(f"📊 Model size: {model_info.get('model_size', 'Unknown')}")
                else:
                    print(f"📊 Model loaded: {self.config['yolo_model']}")
            except:
                print(f"📊 Model loaded: {self.config['yolo_model']}")
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("💡 Available YOLOv11 segmentation models:")
            print("   - yolo11n-seg.pt (nano, fastest)")
            print("   - yolo11s-seg.pt (small, balanced)")
            print("   - yolo11m-seg.pt (medium, better)")
            print("   - yolo11l-seg.pt (large, high accuracy)")
            print("   - yolo11x-seg.pt (xlarge, best)")
            print("   - yolo11v-seg.pt (vision, optimized)")
            print("   - yolo11c-seg.pt (compact, efficient)")
            exit(1)

    def process_video(self, input_path: str, output_path: str = None) -> str:
        """Process video with advanced segmentation-based depersonalization"""
        print(f"🎬 Processing video: {input_path}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            input_dir = Path(input_path).parent
            input_name = Path(input_path).stem
            output_path = input_dir / f"segmented_{input_name}.{self.config['output_format']}"

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📊 Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Processing statistics
        self.stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'persons_detected': 0,
            'total_masks': 0,
            'processing_time': 0,
            'start_time': time.time(),
            'detection_confidence': [],
            'mask_areas': []
        }

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                print(f"🔄 Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

                # Process frame with segmentation
                processed_frame = self._process_frame_segmentation(frame, frame_count)

                # Write processed frame
                out.write(processed_frame)
                self.stats['processed_frames'] += 1

                # Show preview if enabled
                if self.config['show_preview']:
                    cv2.imshow('Segmentation Processing Preview', processed_frame)
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

            self._print_statistics()

            # Save statistics
            stats_file = Path(output_path).with_suffix('.json')
            
            # Convert numpy types to Python native types for JSON serialization
            stats_for_json = {}
            for key, value in self.stats.items():
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
                    # Fallback: convert to string if all else fails
                    stats_for_json[key] = str(value)
            
            with open(stats_file, 'w') as f:
                json.dump(stats_for_json, f, indent=2)
            print(f"📊 Statistics saved to: {stats_file}")

        return str(output_path)

    def _process_frame_segmentation(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process a single frame with YOLO segmentation"""
        try:
            # Run YOLO segmentation inference
            results = self.model(
                frame,
                conf=self.config['confidence_threshold'],
                iou=self.config['iou_threshold'],
                verbose=False
            )

            processed_frame = frame.copy()
            persons_in_frame = 0
            masks_in_frame = 0

            for result in results:
                boxes = result.boxes
                masks = result.masks

                if boxes is not None and masks is not None:
                    for i, (box, mask) in enumerate(zip(boxes, masks)):
                        # Check if it's a person (class 0 in COCO dataset)
                        if box.cls == 0:  # Person class
                            persons_in_frame += 1
                            masks_in_frame += 1

                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf[0])

                            # Get segmentation mask
                            mask_data = mask.data[0].cpu().numpy()
                            mask_resized = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))

                            # Apply segmentation-based depersonalization
                            if self.config['enhanced_blur']:
                                processed_frame = self._apply_segmentation_blur(
                                    processed_frame, mask_resized, x1, y1, x2, y2, conf
                                )

                            # Draw visual elements if enabled
                            if self.config['draw_boxes']:
                                processed_frame = self._draw_detection_box(
                                    processed_frame, x1, y1, x2, y2, conf
                                )

                            if self.config['draw_masks']:
                                processed_frame = self._draw_segmentation_mask(
                                    processed_frame, mask_resized, x1, y1, x2, y2
                                )

                            if self.config['draw_labels']:
                                processed_frame = self._draw_detection_label(
                                    processed_frame, x1, y1, x2, y2, conf, i
                                )

                            # Update statistics
                            self.stats['detection_confidence'].append(conf)
                            mask_area = np.sum(mask_resized > 0.5)
                            self.stats['mask_areas'].append(mask_area)

            # Update frame statistics
            self.stats['persons_detected'] += persons_in_frame
            self.stats['total_masks'] += masks_in_frame

            if persons_in_frame > 0:
                print(f"👥 Frame {frame_number}: Detected {persons_in_frame} person(s) with {masks_in_frame} mask(s)")

            return processed_frame

        except Exception as e:
            print(f"❌ Segmentation processing error: {e}")
            return frame

    def _apply_segmentation_blur(self, frame: np.ndarray, mask: np.ndarray, 
                                x1: int, y1: int, x2: int, y2: int, conf: float) -> np.ndarray:
        """Apply precise depersonalization using segmentation mask"""
        result = frame.copy()

        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        # Extract the region of interest
        roi = frame[y1:y2, x1:x2]
        mask_roi = mask[y1:y2, x1:x2]

        if roi.size == 0 or mask_roi.size == 0:
            return result

        # Adaptive blur strength based on person size and confidence
        blur_strength = self.config['blur_strength']
        if self.config['adaptive_blur']:
            person_area = (x2 - x1) * (y2 - y1)
            if person_area > 10000:  # Large person
                blur_strength = max(5, blur_strength - 5)
            elif person_area < 2000:  # Small person
                blur_strength = min(25, blur_strength + 5)
        
        # Ensure blur strength is odd (OpenCV requirement)
        if blur_strength % 2 == 0:
            blur_strength += 1
        blur_strength = max(3, blur_strength)  # Minimum kernel size of 3

        # Apply blur to the masked region
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)

        # Create 3-channel mask for blending
        mask_3d = np.stack([mask_roi] * 3, axis=2)

        # Blend the blurred region with the original using the mask
        blended_roi = (blurred_roi * mask_3d + roi * (1 - mask_3d)).astype(np.uint8)

        # Put the processed region back
        result[y1:y2, x1:x2] = blended_roi

        return result

    def _draw_detection_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, conf: float) -> np.ndarray:
        """Draw detection bounding box"""
        # Green box for person detection
        color = (0, 255, 0)
        thickness = 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    def _draw_segmentation_mask(self, frame: np.ndarray, mask: np.ndarray, 
                               x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Draw segmentation mask overlay"""
        if not self.config['draw_masks']:
            return frame

        # Create colored mask overlay
        mask_colored = np.zeros_like(frame)
        mask_colored[mask > 0.5] = [0, 255, 255]  # Cyan color for masks

        # Blend mask with frame
        alpha = self.config['mask_alpha']
        frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)

        return frame

    def _draw_detection_label(self, frame: np.ndarray, x1: int, y1: int, 
                             x2: int, y2: int, conf: float, index: int) -> np.ndarray:
        """Draw detection label with confidence"""
        if not self.config['draw_labels']:
            return frame

        label = f"Person {index+1} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Background rectangle for label
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)

        # Label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def _print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("📊 SEGMENTATION PROCESSING STATISTICS")
        print("="*60)
        print(f"Total frames: {self.stats['total_frames']}")
        print(f"Processed frames: {self.stats['processed_frames']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"Processing FPS: {self.stats['fps_processing']:.2f}")
        print(f"Total persons detected: {self.stats['persons_detected']}")
        print(f"Total masks generated: {self.stats['total_masks']}")
        
        if self.stats['detection_confidence']:
            avg_conf = np.mean(self.stats['detection_confidence'])
            print(f"Average detection confidence: {avg_conf:.3f}")
        
        if self.stats['mask_areas']:
            avg_mask_area = np.mean(self.stats['mask_areas'])
            print(f"Average mask area: {avg_mask_area:.0f} pixels")
        
        print("="*60)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='YOLO Segmentation Video Processor')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('-m', '--model', default='yolo11n-seg.pt', 
                       help='YOLOv11 segmentation model (default: yolo11n-seg.pt)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, 
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('-i', '--iou', type=float, default=0.45, 
                       help='IoU threshold (default: 0.45)')
    parser.add_argument('-b', '--blur', type=int, default=15, 
                       help='Blur strength (default: 15)')
    parser.add_argument('-p', '--preview', action='store_true', 
                       help='Show preview during processing')
    parser.add_argument('--no-masks', action='store_true', 
                       help='Disable mask visualization')
    parser.add_argument('--no-boxes', action='store_true', 
                       help='Disable bounding box visualization')
    parser.add_argument('--no-labels', action='store_true', 
                       help='Disable label visualization')

    args = parser.parse_args()

    # Create configuration
    config = {
        'yolo_model': args.model,
        'confidence_threshold': args.confidence,
        'iou_threshold': args.iou,
        'blur_strength': args.blur,
        'show_preview': args.preview,
        'draw_masks': not args.no_masks,
        'draw_boxes': not args.no_boxes,
        'draw_labels': not args.no_labels,
        'output_format': 'mp4',  # Add missing required config
        'save_intermediate': True,
        'save_masks': True,
        'mask_alpha': 0.3,
        'person_only': True,
        'enhanced_blur': True,
        'preserve_edges': True,
        'adaptive_blur': True,
    }

    # Create processor and process video
    processor = SegmentationVideoProcessor(config)

    try:
        output_path = processor.process_video(args.input_video, args.output)
        print(f"\n✅ Segmentation processing completed! Output saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
