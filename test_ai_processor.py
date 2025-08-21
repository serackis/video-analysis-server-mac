#!/usr/bin/env python3
"""
Test script for the Advanced AI Video Processor
"""

import os
from ai_video_processor import AdvancedVideoProcessor

def test_processor():
    """Test the AI processor with a sample video"""
    
    # Test video path (use the original Ema video)
    test_video = "Ema_very_short.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Please ensure Ema_very_short.mp4 is in the current directory")
        return False
    
    print("🧪 Testing AI Video Processor...")
    
    # Create processor with default config
    processor = AdvancedVideoProcessor()
    
    # Test processing
    try:
        output_path = processor.process_video(test_video)
        print(f"✅ Test successful! Output: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_processor()
    exit(0 if success else 1)
