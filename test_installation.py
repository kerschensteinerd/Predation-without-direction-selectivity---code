#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Python environment setup for hunting behavior analysis.

This script verifies that all required packages can be imported
and that the configuration system works correctly.

Author: KerschensteinerLab
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('scipy', None),
        ('matplotlib', None),
        ('cv2', None),
        ('PyQt5', None),
        ('tables', None),
        ('h5py', None),
    ]
    
    print("Testing package imports...")
    print("-" * 50)
    
    failed = []
    for package_info in packages:
        if isinstance(package_info, tuple):
            package, alias = package_info
        else:
            package = package_info
            alias = None
        
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                __import__(package)
            print(f"✓ {package:20s} OK")
        except ImportError as e:
            print(f"✗ {package:20s} FAILED: {e}")
            failed.append(package)
    
    print("-" * 50)
    
    if failed:
        print(f"\n✗ Failed to import: {', '.join(failed)}")
        print("\nTo fix, run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True


def test_config():
    """Test configuration system."""
    print("\nTesting configuration system...")
    print("-" * 50)
    
    try:
        from predation.config import AnalysisConfig
        
        # Test default configuration
        config = AnalysisConfig()
        print("✓ Configuration loaded")
        
        # Test validation
        config.validate()
        print("✓ Configuration validation passed")
        
        # Test parameter access
        assert config.frame_rate == 30.0
        assert config.arena_width == 45.0
        assert config.arena_height == 38.0
        print("✓ Configuration parameters accessible")
        
        # Test string representation
        config_str = str(config)
        assert "Frame rate" in config_str
        print("✓ Configuration string representation works")
        
        print("-" * 50)
        print("\n✓ Configuration system working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 50)
        return False


def test_script_help():
    """Test that the main script can show help."""
    print("\nTesting main analysis script...")
    print("-" * 50)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'predation/hunting_analysis_script.py', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and 'usage:' in result.stdout.lower():
            print("✓ Main script --help works")
            print("-" * 50)
            return True
        else:
            print(f"✗ Script help failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr}")
            print("-" * 50)
            return False
            
    except Exception as e:
        print(f"✗ Script test failed: {e}")
        print("-" * 50)
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Hunting Behavior Analysis Environment")
    print("=" * 50)
    print()
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test configuration (only if imports succeeded)
    if results[0]:
        results.append(test_config())
        results.append(test_script_help())
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    if all(results):
        print("✓ All tests passed!")
        print("\nYour environment is ready for analysis.")
        sys.exit(0)
    else:
        print("✗ Some tests failed.")
        print("\nPlease fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
