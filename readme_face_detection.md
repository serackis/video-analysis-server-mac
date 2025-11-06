# Face Detection-Based Video Depersonalization

This script provides video depersonalization using multiple state-of-the-art face detection algorithms. Unlike person detection, this approach specifically targets faces for more precise privacy protection.

## 🚀 Features

- **Multiple Face Detection Methods**: Choose from 4 different algorithms
- **Flexible Depersonalization**: Gaussian blur, pixelation, or black box
- **Configurable Parameters**: Adjust confidence thresholds, blur strength, and bounding box expansion
- **Real-time Preview**: Optional live preview during processing
- **Comprehensive Statistics**: Detailed processing metrics and face detection counts
- **Automatic Model Download**: Downloads required models automatically

## 🔍 Available Face Detection Methods

### 1. **MediaPipe** (Default)
- **Speed**: Fast ⚡
- **Accuracy**: High ✅
- **Requirements**: CPU only
- **Best for**: Real-time processing, general use

### 2. **Haar Cascade**
- **Speed**: Very Fast ⚡⚡
- **Accuracy**: Medium ⚖️
- **Requirements**: CPU only
- **Best for**: Quick processing, resource-constrained environments

### 3. **Face Detection (RetinaNet)**
- **Speed**: Medium 🐌
- **Accuracy**: Very High ✅✅
- **Requirements**: CPU/GPU
- **Best for**: High-accuracy requirements

### 4. **OpenCV DNN (YuNet)**
- **Speed**: Fast ⚡
- **Accuracy**: High ✅
- **Requirements**: CPU only
- **Best for**: Balanced speed/accuracy

## 📦 Installation

1. **Create Virtual Environment** (Recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements_face_detection.txt
```

## 🎯 Usage

### Basic Usage

```bash
# Use default MediaPipe method
python face_detection_depersonalizer.py input_video.mp4

# Specify output file
python face_detection_depersonalizer.py input_video.mp4 -o output_video.mp4

# Use different detection method
python face_detection_depersonalizer.py input_video.mp4 -m haar_cascade
```

### Advanced Options

```bash
# Adjust confidence threshold
python face_detection_depersonalizer.py input_video.mp4 -c 0.7

# Change blur strength
python face_detection_depersonalizer.py input_video.mp4 -b 35

# Use pixelation instead of blur
python face_detection_depersonalizer.py input_video.mp4 --blur-method pixelate

# Expand bounding box by 30%
python face_detection_depersonalizer.py input_video.mp4 --expand-bbox 0.3

# Show live preview
python face_detection_depersonalizer.py input_video.mp4 --preview
```

### Command Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `input_video` | Input video file path | Required | - |
| `-o, --output` | Output video file path | Auto-generated | - |
| `-m, --method` | Face detection method | `mediapipe` | `haar_cascade`, `mediapipe`, `face_detection`, `opencv_dnn` |
| `-c, --confidence` | Detection confidence threshold | `0.5` | `0.0` to `1.0` |
| `-b, --blur` | Blur strength (Gaussian) | `25` | `1` to `99` |
| `--blur-method` | Depersonalization method | `gaussian` | `gaussian`, `pixelate`, `black` |
| `--expand-bbox` | Bounding box expansion factor | `0.2` | `0.0` to `1.0` |
| `--preview` | Show live preview | `False` | Flag |

## 📊 Output

The script generates:

1. **Depersonalized Video**: `{input_name}_face_depersonalized_{method}.mp4`
2. **Statistics JSON**: `{input_name}_face_depersonalized_{method}.json`

### Statistics Include:
- Total frames processed
- Processing time and FPS
- Total faces detected
- Average faces per frame
- Detection method used

## 🔧 Configuration

### Default Settings
```python
config = {
    'confidence_threshold': 0.5,      # Minimum confidence for detection
    'blur_strength': 25,             # Gaussian blur kernel size
    'blur_method': 'gaussian',       # Depersonalization method
    'expand_bbox': 0.2,              # Expand bounding box by 20%
    'show_preview': False,           # Show live preview
}
```

### Custom Configuration
```python
from face_detection_depersonalizer import FaceDetectionDepersonalizer

config = {
    'confidence_threshold': 0.7,
    'blur_strength': 35,
    'blur_method': 'pixelate',
    'expand_bbox': 0.3,
    'show_preview': True
}

depersonalizer = FaceDetectionDepersonalizer('mediapipe', config)
output_path = depersonalizer.process_video('input.mp4')
```

## 🎨 Depersonalization Methods

### 1. **Gaussian Blur** (Default)
- Applies smooth blur to face regions
- Configurable blur strength
- Natural-looking results

### 2. **Pixelation**
- Creates blocky, pixelated effect
- Consistent privacy protection
- Retro aesthetic

### 3. **Black Box**
- Replaces faces with black rectangles
- Maximum privacy protection
- Simple and fast

## 📈 Performance Comparison

| Method | Speed | Accuracy | Memory | Best Use Case |
|--------|-------|----------|---------|---------------|
| **MediaPipe** | ⚡⚡⚡ | ✅✅✅ | 🟢 | General purpose, real-time |
| **Haar Cascade** | ⚡⚡⚡⚡ | ✅✅ | 🟢 | Fast processing, basic accuracy |
| **Face Detection** | ⚡ | ✅✅✅✅ | 🟡 | High accuracy, offline processing |
| **OpenCV DNN** | ⚡⚡ | ✅✅✅ | 🟢 | Balanced performance |

## 🚨 Troubleshooting

### Common Issues

1. **"No module named 'mediapipe'"**
   ```bash
   pip install mediapipe
   ```

2. **"Failed to initialize detector"**
   - Check internet connection for model downloads
   - Verify OpenCV installation

3. **Poor face detection**
   - Lower confidence threshold: `-c 0.3`
   - Try different detection method
   - Check video quality and lighting

4. **Slow processing**
   - Use faster method: `-m haar_cascade`
   - Reduce blur strength: `-b 15`
   - Disable preview

### Performance Tips

- **For speed**: Use `haar_cascade` method
- **For accuracy**: Use `face_detection` method
- **For balance**: Use `mediapipe` method
- **For real-time**: Lower confidence threshold and blur strength

## 🔒 Privacy Considerations

- **Face-only targeting**: Unlike person detection, only faces are processed
- **Configurable expansion**: Adjust bounding box to include hair, ears, etc.
- **Multiple blur methods**: Choose appropriate privacy level
- **No data storage**: Processed frames are not saved permanently

## 📝 Examples

### Example 1: Quick Processing
```bash
python face_detection_depersonalizer.py video.mp4 -m haar_cascade -b 15
```

### Example 2: High Accuracy
```bash
python face_detection_depersonalizer.py video.mp4 -m face_detection -c 0.7 -b 35
```

### Example 3: Live Preview
```bash
python face_detection_depersonalizer.py video.mp4 --preview --blur-method pixelate
```

### Example 4: Custom Output
```bash
python face_detection_depersonalizer.py video.mp4 -o private_video.mp4 -m mediapipe
```

## 🤝 Contributing

Feel free to contribute improvements:
- Add new face detection methods
- Implement additional depersonalization techniques
- Optimize performance
- Enhance documentation

## 📄 License

This project is part of the video analysis server repository.
