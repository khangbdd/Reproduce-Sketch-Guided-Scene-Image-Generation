#!/usr/bin/env python3
"""
Simple test script to validate the installation and check if all dependencies are working.
"""

import sys
import importlib

def test_import(module_name, description=""):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} {description} - Error: {e}")
        return False

def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def main():
    print("=== Testing Sketch-Guided Scene Image Generation Dependencies ===")
    print("")
    
    success = True
    
    # Test core dependencies
    success &= test_import("torch", "(PyTorch)")
    success &= test_import("torchvision", "(PyTorch Vision)")
    success &= test_import("transformers", "(Hugging Face Transformers)")
    success &= test_import("diffusers", "(Hugging Face Diffusers)")
    success &= test_import("PIL", "(Pillow - Image processing)")
    success &= test_import("numpy", "(NumPy)")
    success &= test_import("matplotlib", "(Matplotlib)")
    success &= test_import("tqdm", "(Progress bars)")
    
    print("")
    
    # Test PyTorch CUDA
    test_torch_cuda()
    
    print("")
    
    # Test local modules
    try:
        sys.path.append('.')
        success &= test_import("finetuning_stable", "(Local module)")
        success &= test_import("controlnet_scribble", "(Local module)")
        success &= test_import("grounded_sam", "(Local module)")
        success &= test_import("utils", "(Local module)")
    except Exception as e:
        print(f"❌ Error testing local modules: {e}")
        success = False
    
    print("")
    
    if success:
        print("🎉 All dependencies are working correctly!")
        print("You can now run the generation pipeline.")
    else:
        print("⚠️  Some dependencies are missing or not working.")
        print("Please run: ./setup.sh to install missing dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
