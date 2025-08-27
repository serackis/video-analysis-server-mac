#!/usr/bin/env python3
"""
Face Detection-Based Video Depersonalization
Uses multiple face detection algorithms to identify and blur faces in videos
"""

import os
import cv2
import numpy as np
import time
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDetectionDepersonalizer:
    """Face detection-based video depersonalization using multiple algorithms"""
    
    def __init__(self, method: str = "mediapipe", config: Dict = None):
        self.method = method
        self.config = config or self._default_config()
        self.detector = None
        self.device = 'cpu'  # Face detection typically runs on CPU
        
        # Initialize the selected face detection method
        self._initialize_detector()
    
    def _default_config(self) -> Dict:
        """Default configuration for face detection and depersonalization"""
        return {
            'confidence_threshold': 0.5,
            'blur_strength': 25,  # Gaussian blur kernel size
            'blur_method': 'gaussian',  # 'gaussian', 'pixelate', 'black'
            'expand_bbox': 0.2,  # Expand bounding box by 20%
            'output_format': 'mp4',
            'save_intermediate': True,
            'show_preview': False,
            'fps': 30,
            'codec': 'mp4v'
        }
    
    def _initialize_detector(self):
        """Initialize the selected face detection method"""
        logger.info(f"Initializing {self.method} face detector...")
        
        try:
            if self.method == "haar_cascade":
                self._init_haar_cascade()
            elif self.method == "mediapipe":
                self._init_mediapipe()
            elif self.method == "face_detection":
                self._init_face_detection()
            elif self.method == "opencv_dnn":
                self._init_opencv_dnn()
            elif self.method == "retinaface":
                self._init_retinaface()
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            
            logger.info(f"✅ {self.method} detector initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.method} detector: {e}")
            raise
    
    def _init_haar_cascade(self):
        """Initialize Haar Cascade face detector"""
        cascade_path = "haarcascade_frontalface_default.xml"
        
        # Download cascade file if not exists
        if not os.path.exists(cascade_path):
            logger.info("Downloading Haar Cascade classifier...")
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, cascade_path)
        
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face detector"""
        import mediapipe as mp
        
        mp_face_detection = mp.solutions.face_detection
        self.detector = mp_face_detection.FaceDetection(
            min_detection_confidence=self.config['confidence_threshold']
        )
    
    def _init_face_detection(self):
        """Initialize face-detection library (RetinaNet)"""
        import face_detection
        
        try:
            # Try ResNet50 first
            self.detector = face_detection.build_detector(
                "RetinaNetResNet50", 
                confidence_threshold=self.config['confidence_threshold'],
                nms_iou_threshold=0.3
            )
        except Exception as e:
            logger.warning(f"RetinaNetResNet50 failed: {e}")
            try:
                # Fallback to MobileNet
                logger.info("Trying RetinaNetMobileNetV1 as fallback...")
                self.detector = face_detection.build_detector(
                    "RetinaNetMobileNetV1", 
                    confidence_threshold=self.config['confidence_threshold'],
                    nms_iou_threshold=0.3
                )
            except Exception as e2:
                logger.error(f"All RetinaNet models failed: {e2}")
                raise RuntimeError("RetinaNet models are currently unavailable. Please use MediaPipe or Haar Cascade instead.")
    
    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN face detector (YuNet)"""
        model_path = "face_detection_yunet_2023mar.onnx"
        
        # Download model if not exists
        if not os.path.exists(model_path):
            logger.info("Downloading YuNet face detection model...")
            import urllib.request
            # Use the working URL from media.githubusercontent.com
            url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, model_path)
        
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))
    
    def _init_retinaface(self):
        """Initialize RetinaFace detector"""
        from retinaface import RetinaFace
        self.detector = RetinaFace
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces in a frame using the selected method"""
        if self.method == "haar_cascade":
            return self._detect_haar_cascade(frame)
        elif self.method == "mediapipe":
            return self._detect_mediapipe(frame)
        elif self.method == "face_detection":
            return self._detect_face_detection(frame)
        elif self.method == "opencv_dnn":
            return self._detect_opencv_dnn(frame)
        elif self.method == "retinaface":
            return self._detect_retinaface(frame)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _detect_haar_cascade(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Convert to (x, y, w, h, confidence) format
        detections = []
        for (x, y, w, h) in faces:
            detections.append((x, y, w, h, 1.0))  # Haar doesn't provide confidence
        
        return detections
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]
                
                detections.append((x, y, width, height, confidence))
        
        return detections
    
    def _detect_face_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using face-detection library"""
        detections = self.detector.detect(frame)
        
        # Convert to (x, y, w, h, confidence) format
        results = []
        for detection in detections:
            x, y, x2, y2, confidence = detection
            width = x2 - x
            height = y2 - y
            results.append((int(x), int(y), int(width), int(height), confidence))
        
        return results
    
    def _detect_opencv_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using OpenCV DNN (YuNet)"""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        
        _, detections = self.detector.detect(frame)
        
        results = []
        if detections is not None:
            for detection in detections:
                x, y, w, h, confidence = detection[:5]
                results.append((int(x), int(y), int(w), int(h), confidence))
        
        return results
    
    def _detect_retinaface(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using RetinaFace"""
        # RetinaFace expects a file path, so we need to save the frame temporarily
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        try:
            # Detect faces using RetinaFace
            faces = self.detector.detect_faces(temp_path)
            
            results = []
            for face_key, face_data in faces.items():
                bbox = face_data['facial_area']
                confidence = face_data['score']
                
                x, y, x2, y2 = bbox
                width = x2 - x
                height = y2 - y
                
                results.append((int(x), int(y), int(width), int(height), confidence))
            
            return results
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def apply_depersonalization(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """Apply depersonalization to detected faces"""
        result = frame.copy()
        
        for (x, y, w, h, confidence) in faces:
            if confidence < self.config['confidence_threshold']:
                continue
            
            # Expand bounding box
            expand_x = int(w * self.config['expand_bbox'])
            expand_y = int(h * self.config['expand_bbox'])
            
            x1 = max(0, x - expand_x)
            y1 = max(0, y - expand_y)
            x2 = min(frame.shape[1], x + w + expand_x)
            y2 = min(frame.shape[0], y + h + expand_y)
            
            # Apply depersonalization
            if self.config['blur_method'] == 'gaussian':
                result = self._apply_gaussian_blur(result, x1, y1, x2, y2)
            elif self.config['blur_method'] == 'pixelate':
                result = self._apply_pixelation(result, x1, y1, x2, y2)
            elif self.config['blur_method'] == 'black':
                result = self._apply_black_box(result, x1, y1, x2, y2)
        
        return result
    
    def _apply_gaussian_blur(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Apply Gaussian blur to face region"""
        result = frame.copy()
        roi = frame[y1:y2, x1:x2]
        
        # Apply strong blur
        blurred_roi = cv2.GaussianBlur(
            roi, 
            (self.config['blur_strength'], self.config['blur_strength']), 
            0
        )
        
        result[y1:y2, x1:x2] = blurred_roi
        return result
    
    def _apply_pixelation(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Apply pixelation to face region"""
        result = frame.copy()
        roi = frame[y1:y2, x1:x2]
        
        # Downsample and upsample for pixelation effect
        small = cv2.resize(roi, (8, 8))
        pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        
        result[y1:y2, x1:x2] = pixelated
        return result
    
    def _apply_black_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Apply black box to face region"""
        result = frame.copy()
        result[y1:y2, x1:x2] = 0
        return result
    
    def process_video(self, input_path: str, output_path: str = None) -> str:
        """Process video with face detection-based depersonalization"""
        logger.info(f"🎬 Processing video: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            input_dir = Path(input_path).parent
            input_name = Path(input_path).stem
            output_path = str(input_dir / f"{input_name}_face_depersonalized_{self.method}.mp4")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'faces_detected': 0,
            'processing_time': 0,
            'start_time': time.time()
        }
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect faces
                faces = self.detect_faces(frame)
                stats['faces_detected'] += len(faces)
                
                # Apply depersonalization
                processed_frame = self.apply_depersonalization(frame, faces)
                
                # Write frame
                out.write(processed_frame)
                stats['processed_frames'] += 1
                
                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"🔄 Frame {frame_count}/{total_frames} ({progress:.1f}%) - Detected {len(faces)} face(s)")
                
                # Show preview if enabled
                if self.config['show_preview']:
                    cv2.imshow('Preview', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            out.release()
            if self.config['show_preview']:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        stats['processing_time'] = time.time() - stats['start_time']
        stats['fps_processing'] = stats['processed_frames'] / stats['processing_time']
        
        # Save statistics
        stats_path = output_path.replace('.mp4', '.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self._print_statistics(stats)
        logger.info(f"📊 Statistics saved to: {stats_path}")
        
        return output_path
    
    def _print_statistics(self, stats: Dict):
        """Print processing statistics"""
        logger.info("\n" + "="*60)
        logger.info("📊 FACE DETECTION DEPERSONALIZATION STATISTICS")
        logger.info("="*60)
        logger.info(f"Detection method: {self.method}")
        logger.info(f"Total frames: {stats['total_frames']}")
        logger.info(f"Processed frames: {stats['processed_frames']}")
        logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
        logger.info(f"Processing FPS: {stats['fps_processing']:.2f}")
        logger.info(f"Total faces detected: {stats['faces_detected']}")
        logger.info(f"Average faces per frame: {stats['faces_detected']/stats['processed_frames']:.2f}")
        logger.info("="*60)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Face Detection-Based Video Depersonalization')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('-m', '--method', default='mediapipe', 
                       choices=['haar_cascade', 'mediapipe', 'face_detection', 'opencv_dnn', 'retinaface'],
                       help='Face detection method to use')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, 
                       help='Detection confidence threshold')
    parser.add_argument('-b', '--blur', type=int, default=25, 
                       help='Blur strength (for Gaussian blur)')
    parser.add_argument('--blur-method', default='gaussian',
                       choices=['gaussian', 'pixelate', 'black'],
                       help='Depersonalization method')
    parser.add_argument('--expand-bbox', type=float, default=0.2,
                       help='Expand bounding box by factor (e.g., 0.2 = 20%%)')
    parser.add_argument('--preview', action='store_true', 
                       help='Show preview during processing')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'confidence_threshold': args.confidence,
        'blur_strength': args.blur,
        'blur_method': args.blur_method,
        'expand_bbox': args.expand_bbox,
        'show_preview': args.preview
    }
    
    try:
        # Create depersonalizer
        depersonalizer = FaceDetectionDepersonalizer(args.method, config)
        
        # Process video
        output_path = depersonalizer.process_video(args.input_video, args.output)
        logger.info(f"\n✅ Processing completed! Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"\n❌ Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
