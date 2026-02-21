#!/usr/bin/env python3
"""
Test product name normalization.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.ingestion.processors.text_processor import TextProcessor, ProcessorConfig

def test_normalization():
    """Test product name normalization."""
    
    processor = TextProcessor(ProcessorConfig())
    
    test_cases = [
        ("Apple iPhone 12, 128GB, Green - AT&T (Renewed)", "apple iphone 12"),
        ("Apple iPhone 12 Pro 5G, US Version, 512GB, Graphite - Unlocked (Renewed)", "apple iphone 12 pro 5g"),
        ("Samsung Galaxy S21 Ultra 5G, 256GB, Phantom Black", "samsung galaxy s21 ultra 5g"),
        ("Google Pixel 6 Pro - 5G Android Phone", "google pixel 6 pro"),
        ("Apple iPhone 8, US Version, 256GB, Space Gray - T-Mobile (Renewed)", "apple iphone 8"),
    ]
    
    print("Testing Product Name Normalization")
    print("="*80)
    print()
    
    all_passed = True
    for original, expected in test_cases:
        normalized = processor._normalize_product_name(original)
        passed = normalized == expected
        all_passed = all_passed and passed
        
        status = "✓" if passed else "✗"
        print(f"{status} Original:   {original}")
        print(f"  Expected:   {expected}")
        print(f"  Got:        {normalized}")
        if not passed:
            print(f"  ❌ MISMATCH!")
        print()
    
    print("="*80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = test_normalization()
    sys.exit(0 if success else 1)
