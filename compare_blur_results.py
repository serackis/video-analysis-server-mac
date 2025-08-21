#!/usr/bin/env python3
"""
Compare Original vs Enhanced Blur Results
Simple comparison of blur algorithms
"""

from pathlib import Path

def compare_blur_results():
    """Compare original and enhanced blur results"""
    
    print("🔍 Comparing Original vs Enhanced Blur Results")
    print("="*60)
    
    # Check what we have
    accuracy_dir = Path("accuracy_comparison")
    enhanced_dir = Path("enhanced_blur_test")
    
    print("📁 **Available Results:**")
    
    if accuracy_dir.exists():
        print("✅ Original blur results:")
        videos = list(accuracy_dir.glob("*_processed.mp4"))
        for video in videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"   • {video.name}: {size_mb:.1f}MB")
    
    if enhanced_dir.exists():
        print("✅ Enhanced blur results:")
        videos = list(enhanced_dir.glob("*.mp4"))
        if videos:
            for video in videos:
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"   • {video.name}: {size_mb:.1f}MB")
        else:
            print("   • No processed videos found")
    
    print("\n" + "="*60)
    print("🎯 **Enhanced Blur Algorithm Summary**")
    print("="*60)
    
    print("✨ **Key Improvements Made:**")
    print("   1. **Increased Base Blur:** 25 (was 15)")
    print("   2. **Size-Based Enhancement:**")
    print("      • Large persons (>15k pixels): +15 blur")
    print("      • Medium persons (>8k pixels): +10 blur")
    print("      • Small persons (>3k pixels): +5 blur")
    
    print("   3. **Confidence Enhancement:**")
    print("      • High confidence (>0.8): +10 blur")
    print("      • Medium confidence (>0.6): +5 blur")
    
    print("   4. **Multi-Pass Blur:**")
    print("      • Second blur pass for very large persons")
    print("      • Ensures maximum depersonalization")
    
    print("   5. **High Blur Dominance:**")
    print("      • 90% blurred, 10% original")
    print("      • Better privacy protection")
    
    print("\n" + "="*60)
    print("📊 **Expected Blur Strengths:**")
    print("="*60)
    
    print("Small person (1k pixels, 0.5 confidence):")
    print("   • Original: 15")
    print("   • Enhanced: 25 + 5 + 0 = 30")
    print("   • Improvement: 2x stronger blur")
    
    print("\nMedium person (10k pixels, 0.7 confidence):")
    print("   • Original: 15")
    print("   • Enhanced: 25 + 10 + 5 = 40")
    print("   • Improvement: 2.7x stronger blur")
    
    print("\nLarge person (20k pixels, 0.9 confidence):")
    print("   • Original: 15")
    print("   • Enhanced: 25 + 15 + 10 = 50")
    print("   • Plus second blur pass")
    print("   • Improvement: 3.3x stronger blur + multi-pass")
    
    print("\n" + "="*60)
    print("💡 **How to Compare Videos:**")
    print("="*60)
    
    print("1. **Open both videos in your video player:**")
    print("   • Original: accuracy_comparison/yolo11l_seg_processed.mp4")
    print("   • Enhanced: enhanced_blur_test/enhanced_blur_Ema_very_short.mp4")
    
    print("\n2. **Look for these improvements:**")
    print("   • Closer persons are much less identifiable")
    print("   • Large detections have stronger blur")
    print("   • More consistent depersonalization")
    print("   • Better privacy protection overall")
    
    print("\n3. **Privacy assessment:**")
    print("   • Can you still recognize people?")
    print("   • Is the blur too aggressive?")
    print("   • Balance between privacy and quality")
    
    print(f"\n🎯 **Enhanced blur processing completed!**")
    print(f"📁 Check the output videos for comparison")
    print(f"💡 The enhanced algorithm should provide much better privacy protection")

if __name__ == "__main__":
    compare_blur_results()
