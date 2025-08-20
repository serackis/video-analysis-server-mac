# 🤖 Advanced AI Video Processing Development Environment

This directory contains tools for developing and testing advanced AI algorithms for video depersonalization using state-of-the-art models like YOLOv8 and SAM (Segment Anything Model).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Activate your virtual environment
source venv/bin/activate

# Install AI requirements
pip install -r requirements_ai.txt
```

### 2. Test the System
```bash
# Test with a sample video
python test_ai_processor.py

# Or process a video directly
python ai_video_processor.py static/uploads/your_video.mp4 --preview
```

## 📁 Files Overview

- **`ai_video_processor.py`** - Main AI processing engine
- **`requirements_ai.txt`** - AI/ML library dependencies
- **`ai_config.json`** - Configuration template
- **`test_ai_processor.py`** - Simple test script
- **`README_AI_Development.md`** - This file

## 🎯 Features

### Current Capabilities
- **YOLOv8 Segmentation**: Person detection with precise masks
- **Mask-based Blurring**: Apply depersonalization only to detected persons
- **Real-time Preview**: See processing results as they happen
- **Performance Metrics**: Track processing speed and accuracy

### Planned Enhancements
- **SAM Integration**: Meta's Segment Anything Model for refinement
- **Multi-person Tracking**: Consistent person identification across frames
- **Adaptive Blurring**: Smart blur strength based on person size/distance
- **Edge Preservation**: Maintain video quality in non-person areas

## ⚙️ Configuration

### Basic Settings
```json
{
  "yolo_model": "yolov8x-seg.pt",
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45,
  "blur_strength": 15
}
```

### Advanced Options
```json
{
  "advanced_options": {
    "use_sam_refinement": false,
    "mask_dilation": 2,
    "adaptive_blur": true
  }
}
```

## 🔧 Usage Examples

### Command Line
```bash
# Basic processing
python ai_video_processor.py input.mp4

# With custom settings
python ai_video_processor.py input.mp4 --confidence 0.7 --blur 25

# With preview
python ai_video_processor.py input.mp4 --preview

# Custom output
python ai_video_processor.py input.mp4 -o processed_output.mp4
```

### Python API
```python
from ai_video_processor import AdvancedVideoProcessor

# Create processor
processor = AdvancedVideoProcessor({
    'confidence_threshold': 0.6,
    'blur_strength': 20
})

# Process video
output_path = processor.process_video('input.mp4')
```

## 📊 Performance Optimization

### GPU Acceleration
- Install PyTorch with CUDA support for GPU acceleration
- Use `torch.cuda.is_available()` to check GPU availability

### Memory Management
- Set `batch_size: 1` for memory-constrained systems
- Enable `optimize_memory: true` for large videos

### Model Selection
- **YOLOv8n**: Fastest, lowest accuracy
- **YOLOv8s**: Balanced speed/accuracy
- **YOLOv8m**: Higher accuracy, slower
- **YOLOv8l/x**: Highest accuracy, slowest

## 🧪 Testing and Development

### Test Different Models
```python
# Test different YOLO models
models = ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt']

for model in models:
    processor = AdvancedVideoProcessor({'yolo_model': model})
    # Test processing...
```

### Parameter Tuning
```python
# Test different confidence thresholds
for conf in [0.3, 0.5, 0.7, 0.9]:
    processor = AdvancedVideoProcessor({'confidence_threshold': conf})
    # Compare results...
```

## 🔍 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model download fails**: Check internet connection
3. **Video not processing**: Verify video format and codec

### Debug Mode
```python
# Enable verbose logging
processor = AdvancedVideoProcessor({'verbose': True})
```

## 🚀 Integration with Main System

Once you're satisfied with the AI algorithms:

1. **Copy the processing logic** to `app.py`
2. **Replace the current depersonalization** with your advanced version
3. **Add configuration options** to the web interface
4. **Test thoroughly** before deploying

## 📚 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [Supervision Library](https://supervision.roboflow.com/)

## 🤝 Contributing

When developing new features:

1. **Test thoroughly** with different video types
2. **Document parameters** and their effects
3. **Benchmark performance** against current system
4. **Create unit tests** for critical functions

---

**Happy AI Development! 🚀**
