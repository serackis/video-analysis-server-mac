# 🎬 Video Processing System - User Guide

This guide explains how to use the complete video depersonalization system with your own videos.

## 📋 Prerequisites

- Python 3.8+ installed
- Virtual environment activated (if using one)
- Required packages installed (see `requirements_ai.txt`)
- YOLOv11 models downloaded (will auto-download on first use)

## 🚀 Quick Start

### 1. Basic Enhanced Blur Processing

Process a video with enhanced blur (no tracking):

```bash
python test_enhanced_blur.py
```

**Default video**: `Ema_very_short.mp4` (hardcoded)

### 2. Enhanced Blur with Tracking

Process a video with enhanced blur + tracking:

```bash
python tracking_video_processor.py your_video.mp4
```

**Features**:
- Enhanced blur algorithm
- Person tracking across frames
- Visual tracking information
- JSON tracking data export

## 📁 Script Reference

### Core Processing Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `test_enhanced_blur.py` | Enhanced blur (no tracking) | `python test_enhanced_blur.py` |
| `tracking_video_processor.py` | Enhanced blur + tracking | `python tracking_video_processor.py <video>` |
| `clean_video_processor.py` | Clean output (no overlays) | `python clean_video_processor.py <video>` |
| `accuracy_comparison.py` | Compare all YOLOv11 models | `python accuracy_comparison.py` |
| `best_accuracy_test.py` | Test top 3 models | `python best_accuracy_test.py` |

### Analysis & Comparison Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `compare_blur_results.py` | Compare blur algorithms | `python compare_blur_results.py` |
| `tracking_comparison.py` | Show tracking benefits | `python tracking_comparison.py` |
| `complete_system_summary.py` | Complete system overview | `python complete_system_summary.py` |

### Batch Processing Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `batch_processor.py` | Process multiple videos | `python batch_processor.py [OPTIONS]` |

## 🎯 Processing Your Own Videos

### Option 1: Enhanced Blur Only

**Best for**: Fast processing, maximum privacy protection

```bash
# Edit the script to change input video
nano test_enhanced_blur.py

# Change this line:
input_video = "your_video.mp4"

# Run the script
python test_enhanced_blur.py
```

**Output**: `enhanced_blur_test/enhanced_blur_your_video.mp4`

### Option 2: Enhanced Blur + Tracking

**Best for**: Production use, consistent depersonalization

```bash
# Process any video file
python tracking_video_processor.py your_video.mp4

# With custom options
python tracking_video_processor.py your_video.mp4 --confidence 0.4 --blur 30 --persistence 20
```

**Output**: `tracked_your_video.mp4` + `tracked_your_video.json`

### Option 3: Clean Output (No Visual Overlays)

**Best for**: Production use, clean depersonalized videos

```bash
# Process video with clean output (no bounding boxes, labels, or tracking info)
python clean_video_processor.py your_video.mp4

# With custom settings
python clean_video_processor.py your_video.mp4 -m yolo11m-seg.pt -c 0.4 -b 30

# Using the main processor with clean output flag
python tracking_video_processor.py your_video.mp4 --clean-output
```

**Output**: Clean depersonalized video with enhanced blur only

### Option 4: Model Comparison

**Best for**: Finding the best model for your use case

```bash
# Edit the script to change input video
nano accuracy_comparison.py

# Change this line:
input_video = "your_video.mp4"

# Run comparison
python accuracy_comparison.py
```

**Output**: `accuracy_comparison/` directory with all model results

### Option 5: Batch Processing

**Best for**: Processing multiple videos at once

```bash
# Process all videos in current directory
python batch_processor.py

# Process videos in specific directory
python batch_processor.py -d /path/to/videos

# Custom settings for batch processing
python batch_processor.py -m yolo11n-seg.pt -c 0.4 -b 30 -o my_results

# Disable tracking for faster processing
python batch_processor.py --no-tracking
```

**Output**: `batch_processed/` directory with all processed videos

## ⚙️ Configuration Options

### Tracking Processor Options

```bash
python tracking_video_processor.py your_video.mp4 [OPTIONS]

Options:
  -o, --output PATH          Output video path
  -m, --model MODEL          YOLOv11 model (default: yolo11l-seg.pt)
  -c, --confidence FLOAT     Detection confidence (default: 0.3)
  -b, --blur INTEGER         Base blur strength (default: 25)
  -p, --preview              Show preview during processing
  --no-tracking              Disable tracking
  --no-visualization         Disable tracking visualization
  --clean-output             Remove all visual overlays for clean output
  --persistence INTEGER      Tracking persistence frames (default: 10)
```

### Available YOLOv11 Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolo11n-seg.pt` | 5.9MB | Fastest | Good | Development/Testing |
| `yolo11s-seg.pt` | 20MB | Fast | Better | Balanced |
| `yolo11m-seg.pt` | 43MB | Medium | High | Production |
| `yolo11l-seg.pt` | 53MB | Slower | Highest | Best Quality |
| `yolo11x-seg.pt` | 119MB | Slowest | Best | Maximum Accuracy |

## 🔧 Customization

### Changing Input Video in Scripts

For scripts that have hardcoded video paths:

1. **Open the script**:
   ```bash
   nano script_name.py
   ```

2. **Find and change the video path**:
   ```python
   # Change this line:
   input_video = "your_video.mp4"
   ```

3. **Save and run**:
   ```bash
   python script_name.py
   ```

### Custom Blur Settings

Modify blur parameters in the scripts:

```python
# In tracking_video_processor.py or test_enhanced_blur.py
base_blur_strength = 30  # Increase from 25
confidence_enhancement = 15  # Increase from 10
size_enhancement = 20  # Increase from 15
```

### Tracking Configuration

Adjust tracking behavior:

```python
# In tracking_video_processor.py
tracking_persistence = 20  # Frames to keep tracking without detection
tracking_smooth = True     # Smooth tracking predictions
```

## 📊 Understanding Output

### Video Files

- **Enhanced blur**: `enhanced_blur_your_video.mp4`
- **With tracking**: `tracked_your_video.mp4`
- **Model comparison**: `yolo11x_seg_processed.mp4` (etc.)

### Statistics Files

- **Enhanced blur**: `enhanced_blur_stats.json`
- **Tracking data**: `tracked_your_video.json`
- **Model comparison**: `accuracy_comparison_report.txt`

### Key Metrics

- **Persons detected**: Total detections across all frames
- **Unique tracks**: Number of distinct persons tracked
- **Processing FPS**: Speed of processing
- **Blur strength**: Average blur applied
- **Confidence**: Detection confidence scores

## 🎬 Video Format Support

### Supported Input Formats

- MP4 (recommended)
- AVI
- MOV
- MKV
- Most common video formats

### Output Format

- **Default**: MP4 with H.264 codec
- **Quality**: Same as input (no quality loss from blur)
- **Size**: Similar to input (blur doesn't significantly increase size)

## 🚨 Troubleshooting

### Common Issues

1. **"No such file or directory"**
   - Check video file path
   - Ensure video file exists in current directory

2. **"Model not found"**
   - Models auto-download on first use
   - Check internet connection
   - Wait for download to complete

3. **"CUDA not available"**
   - System will use CPU instead
   - Processing will be slower but functional

4. **Memory errors**
   - Reduce video resolution
   - Use smaller YOLOv11 model
   - Close other applications

### Performance Tips

- **For speed**: Use `yolo11n-seg.pt` or `yolo11s-seg.pt`
- **For accuracy**: Use `yolo11l-seg.pt` or `yolo11x-seg.pt`
- **For tracking**: Use persistence 10-20 frames
- **For privacy**: Use confidence threshold 0.3-0.5

## 📱 Example Workflows

### Workflow 1: Quick Privacy Protection

```bash
# Process video with best privacy settings
python tracking_video_processor.py my_video.mp4 --confidence 0.3 --blur 30

# Check results
python tracking_comparison.py
```

### Workflow 1b: Clean Privacy Protection

```bash
# Process video with clean output (no visual overlays)
python clean_video_processor.py my_video.mp4 --confidence 0.3 --blur 30

# Or use the main processor with clean flag
python tracking_video_processor.py my_video.mp4 --clean-output --confidence 0.3 --blur 30
```

### Workflow 2: Model Testing

```bash
# Test different models
python best_accuracy_test.py

# Compare all models
python accuracy_comparison.py

# View comparison
python compare_videos.py
```

### Workflow 3: Development/Testing

```bash
# Quick test with small model
python tracking_video_processor.py test_video.mp4 -m yolo11n-seg.pt

# Full test with best model
python tracking_video_processor.py test_video.mp4 -m yolo11l-seg.pt
```

### Workflow 4: Batch Processing

```bash
# Process all videos in a folder
python batch_processor.py -d /path/to/video/folder

# Batch process with custom settings
python batch_processor.py -d /path/to/videos -m yolo11m-seg.pt -c 0.4 -b 30

# Quick batch processing (no tracking)
python batch_processor.py --no-tracking
```

## 🔒 Privacy Features

### Enhanced Blur Algorithm

- **Base blur**: 25 (67% stronger than standard)
- **Adaptive blur**: +5 to +25 based on person size
- **Confidence enhancement**: +5 to +10 based on detection confidence
- **Multi-pass blur**: Additional blur for very large persons
- **High dominance**: 90% blurred, 10% original content

### Tracking Benefits

- **Consistent IDs**: Same person keeps same ID across frames
- **Stable blur**: No flickering or gaps in depersonalization
- **Persistent tracking**: Maintains tracking through occlusions
- **Visual feedback**: See tracking paths and person IDs

## 📈 Performance Benchmarks

### Processing Speed (CPU)

| Model | FPS | Time per frame |
|-------|-----|----------------|
| yolo11n-seg.pt | 15.1 | 66ms |
| yolo11s-seg.pt | 10.5 | 95ms |
| yolo11m-seg.pt | 5.7 | 175ms |
| yolo11l-seg.pt | 4.5 | 222ms |
| yolo11x-seg.pt | 2.4 | 417ms |

### Enhanced Blur Performance

- **Base processing**: ~3.9 FPS
- **With tracking**: ~3.7 FPS
- **Memory usage**: ~2-4GB RAM
- **Output quality**: Same as input

## 🎯 Best Practices

### For Production Use

1. **Use enhanced blur + tracking** for maximum privacy
2. **Choose yolo11l-seg.pt** for best balance of speed/accuracy
3. **Set confidence threshold** to 0.3-0.4
4. **Use persistence** of 15-20 frames
5. **Test on sample videos** before processing large files

### For Development

1. **Start with yolo11n-seg.pt** for fast iteration
2. **Use smaller test videos** for quick testing
3. **Enable preview mode** to see results in real-time
4. **Save tracking data** for analysis

### For Analysis

1. **Compare multiple models** to find best fit
2. **Analyze tracking data** for insights
3. **Check blur statistics** for quality assessment
4. **Use comparison tools** for evaluation

## 📞 Support

### Getting Help

1. **Check script output** for error messages
2. **Verify video file** exists and is readable
3. **Ensure dependencies** are installed
4. **Check system resources** (RAM, disk space)

### Script Locations

All scripts are in the root directory:
- `tracking_video_processor.py` - Main tracking processor
- `test_enhanced_blur.py` - Enhanced blur processor
- `clean_video_processor.py` - Clean output processor
- `accuracy_comparison.py` - Model comparison
- `best_accuracy_test.py` - Top model testing
- `batch_processor.py` - Batch video processor

### Output Locations

- **Enhanced blur**: `enhanced_blur_test/` directory
- **Tracking results**: Root directory
- **Model comparison**: `accuracy_comparison/` directory

---

## 🎉 Ready to Process!

You now have a complete video depersonalization system. Start with:

```bash
# Process your first video with tracking
python tracking_video_processor.py your_video.mp4

# Or test enhanced blur
python test_enhanced_blur.py

# Or get clean output without visual overlays
python clean_video_processor.py your_video.mp4
```

**Happy processing!** 🚀
