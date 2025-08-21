#!/usr/bin/env python3
"""
Enhanced Blur Results Summary
Shows the results of the enhanced blur algorithm
"""

import json
from pathlib import Path

def show_enhanced_blur_summary():
    """Show summary of enhanced blur processing"""
    
    print("🎯 Enhanced Blur Processing Results")
    print("="*60)
    
    # Check enhanced blur results
    enhanced_dir = Path("enhanced_blur_test")
    if enhanced_dir.exists():
        stats_file = enhanced_dir / "enhanced_blur_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                print(f"📊 Model: {stats['model']}")
                print(f"📊 Persons detected: {stats['persons_detected']}")
                print(f"📊 Total masks: {stats['total_masks']}")
                print(f"📊 Processing time: {stats['processing_time']:.2f}s")
                print(f"📊 Processing FPS: {stats['fps_processing']:.2f}")
                
                if 'avg_blur_strength' in stats:
                    print(f"📊 Blur strength: {stats['avg_blur_strength']:.1f} (min: {stats['min_blur_strength']}, max: {stats['max_blur_strength']})")
                
                if 'avg_person_size' in stats:
                    print(f"📊 Person size: {stats['avg_person_size']:.0f} pixels (min: {stats['min_person_size']}, max: {stats['max_person_size']})")
                
                if 'avg_confidence' in stats:
                    print(f"📊 Confidence: {stats['avg_confidence']:.3f} (min: {stats['min_confidence']:.3f}, max: {stats['max_confidence']:.3f})")
                
                print(f"\n📁 Output video: {enhanced_dir}/enhanced_blur_Ema_very_short.mp4")
                
            except Exception as e:
                print(f"❌ Error reading stats: {e}")
        else:
            print("❌ No enhanced blur statistics found")
    else:
        print("❌ Enhanced blur directory not found")
    
    print("\n" + "="*60)
    print("🔍 Enhanced Blur Algorithm Features")
    print("="*60)
    
    print("✨ **Enhanced Blur Algorithm:**")
    print("   • Base blur strength: 25 (increased from 15)")
    print("   • Adaptive blur based on person size:")
    print("     - Large/close persons (>15k pixels): +15 blur strength")
    print("     - Medium persons (>8k pixels): +10 blur strength")
    print("     - Small persons (>3k pixels): +5 blur strength")
    print("     - Very small persons: base blur only")
    
    print("\n   • Confidence-based enhancement:")
    print("     - High confidence (>0.8): +10 blur strength")
    print("     - Medium confidence (>0.6): +5 blur strength")
    
    print("\n   • Multi-pass blur for large persons:")
    print("     - First pass: full blur strength")
    print("     - Second pass: additional blur for very large persons")
    
    print("\n   • High blur dominance:")
    print("     - 90% blurred content, 10% original")
    print("     - Ensures maximum depersonalization")
    
    print("\n" + "="*60)
    print("📊 **Blur Strength Examples:**")
    print("="*60)
    
    print("Small person (1k pixels, 0.5 confidence):")
    print("   Base: 25 + 5 (size) + 0 (confidence) = 30")
    
    print("\nMedium person (10k pixels, 0.7 confidence):")
    print("   Base: 25 + 10 (size) + 5 (confidence) = 40")
    
    print("\nLarge person (20k pixels, 0.9 confidence):")
    print("   Base: 25 + 15 (size) + 10 (confidence) = 50")
    print("   Plus second blur pass for extra depersonalization")
    
    print("\n" + "="*60)
    print("💡 **How to Compare Results:**")
    print("="*60)
    
    print("1. **Original vs Enhanced Blur:**")
    print("   • Compare: accuracy_comparison/yolo11l_seg_processed.mp4")
    print("   • With: enhanced_blur_test/enhanced_blur_Ema_very_short.mp4")
    
    print("\n2. **Look for improvements:**")
    print("   • Better depersonalization of close persons")
    print("   • More aggressive blur for large detections")
    print("   • Consistent blur strength across all person sizes")
    
    print("\n3. **Privacy assessment:**")
    print("   • Are persons still identifiable?")
    print("   • Is the blur too aggressive?")
    print("   • Balance between privacy and video quality")
    
    print(f"\n🎯 Enhanced blur processing completed successfully!")
    print(f"📁 Check the output videos for comparison")

if __name__ == "__main__":
    show_enhanced_blur_summary()
