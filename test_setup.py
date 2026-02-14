"""
Test script to verify that all dependencies are installed correctly.

Run this script after completing setup to ensure R, Python packages,
and the eurlex R package are working properly.
"""

import sys
from typing import Tuple


def test_python_packages() -> Tuple[bool, str]:
    """Test that required Python packages are installed."""
    print("Testing Python packages...")

    try:
        import torch
        import transformers
        import datasets
        import pandas
        import numpy
        import tqdm
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ Transformers {transformers.__version__}")
        print(f"  ✓ Datasets {datasets.__version__}")
        print(f"  ✓ pandas, numpy, tqdm")
        return True, "All Python packages installed"
    except ImportError as e:
        return False, f"Missing Python package: {e}"


def test_rpy2() -> Tuple[bool, str]:
    """Test rpy2 installation and R connection."""
    print("\nTesting rpy2 (Python-R bridge)...")

    try:
        import rpy2.robjects as ro
        r_version = ro.r('R.version.string')[0]
        print(f"  ✓ rpy2 connected to R")
        print(f"  ✓ R version: {r_version}")
        return True, "rpy2 working correctly"
    except Exception as e:
        return False, f"rpy2 error: {e}"


def test_eurlex() -> Tuple[bool, str]:
    """Test eurlex R package installation."""
    print("\nTesting eurlex R package...")

    try:
        from rpy2.robjects.packages import importr

        # Try to import eurlex
        eurlex = importr('eurlex')
        print("  ✓ eurlex package loaded")

        # Test basic functionality - create a small query
        query = eurlex.elx_make_query(resource_type='directive', limit=5)
        print("  ✓ elx_make_query() works")

        return True, "eurlex package working correctly"
    except Exception as e:
        return False, f"eurlex error: {e}"


def test_pytorch_gpu() -> Tuple[bool, str]:
    """Test PyTorch GPU availability (optional)."""
    print("\nTesting GPU support (optional)...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {gpu_name}")
            return True, f"GPU available: {gpu_name}"
        elif torch.backends.mps.is_available():
            print("  ✓ MPS (Apple Silicon) available")
            return True, "Apple MPS available"
        else:
            print("  ⚠ No GPU detected (CPU only)")
            return True, "CPU only (GPU optional)"
    except Exception as e:
        return False, f"GPU test error: {e}"


def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("FRENCH-GERMAN LEGAL TRANSLATION PROJECT")
    print("Setup Verification Test")
    print("=" * 60)

    tests = [
        ("Python Packages", test_python_packages),
        ("rpy2 (Python-R Bridge)", test_rpy2),
        ("eurlex R Package", test_eurlex),
        ("GPU Support", test_pytorch_gpu),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            results.append((test_name, success, message))
        except Exception as e:
            results.append((test_name, False, str(e)))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_critical_passed = True
    for test_name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name}: {message}")

        # GPU is optional, others are critical
        if not success and test_name != "GPU Support":
            all_critical_passed = False

    print("=" * 60)

    if all_critical_passed:
        print("\n🎉 SUCCESS! All critical components are working.")
        print("\nYou can now run:")
        print("  python src/data/download.py --limit 50 --max-documents 5")
        print("\nThis will download 5 sample documents to test the full pipeline.")
        return 0
    else:
        print("\n⚠ SETUP INCOMPLETE. Please fix the errors above.")
        print("\nFor help, see:")
        print("  - SETUP.md for installation instructions")
        print("  - data/download.log for detailed error messages")
        print("  - https://michalovadek.github.io/eurlex/ for eurlex documentation")
        return 1


if __name__ == '__main__':
    sys.exit(main())
