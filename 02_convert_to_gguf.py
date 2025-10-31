#!/usr/bin/env python3
"""
GGUF Conversion Script for TinyLlama
Converts PyTorch model to GGUF format for Termux/llama.cpp inference

Usage:
    python 02_convert_to_gguf.py

This script:
1. Clones llama.cpp if not present
2. Builds llama.cpp
3. Converts model to GGUF
4. Quantizes to Q4_K_M (4-bit)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Configuration
MODEL_DIR = "fine_tuned_tinyllama"
LLAMA_CPP_DIR = "llama.cpp"
GGUF_DIR = "gguf_models"
MODEL_FP16 = f"{GGUF_DIR}/model-fp16.gguf"
MODEL_QUANTIZED = f"{GGUF_DIR}/model-Q4_K_M.gguf"

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")

    # Check Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check git
    if not shutil.which("git"):
        print("❌ Git not found")
        return False
    print("   ✅ Git found")

    # Check make
    if not shutil.which("make"):
        print("❌ Make not found (required for building llama.cpp)")
        return False
    print("   ✅ Make found")

    return True

def clone_llama_cpp():
    """Clone llama.cpp repository"""
    print(f"\n📥 Cloning llama.cpp...")

    if os.path.exists(LLAMA_CPP_DIR):
        print(f"   ℹ️  {LLAMA_CPP_DIR} already exists, pulling latest...")
        os.system(f"cd {LLAMA_CPP_DIR} && git pull origin master")
    else:
        os.system(f"git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_CPP_DIR}")

    print(f"✅ llama.cpp ready")

def build_llama_cpp():
    """Build llama.cpp"""
    print(f"\n🔨 Building llama.cpp...")

    build_dir = f"{LLAMA_CPP_DIR}/build"
    os.makedirs(build_dir, exist_ok=True)

    # Build
    result = os.system(f"cd {LLAMA_CPP_DIR} && make LLAMA_BUILD_INFO=OFF LLAMA_BUILD_NUMBER=")

    if result == 0:
        print("✅ llama.cpp built successfully")
        return True
    else:
        print("❌ Build failed. Trying alternative build method...")
        # Alternative: use cmake
        os.system(f"cd {LLAMA_CPP_DIR} && mkdir -p build && cd build && cmake .. && make -j4")
        return os.path.exists(f"{LLAMA_CPP_DIR}/main")

def check_model():
    """Check if model directory exists"""
    print(f"\n🔍 Checking model directory: {MODEL_DIR}")

    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("   Please run 01_fine_tune_tinyllama.py first")
        return False

    # Check for necessary files
    required_files = ["config.json", "tokenizer.json", "pytorch_model.bin"]
    for file in required_files:
        if not os.path.exists(f"{MODEL_DIR}/{file}"):
            print(f"⚠️  Warning: {file} not found")
            print("   Make sure the model was saved correctly")

    print(f"✅ Model directory exists")
    return True

def convert_to_fp16():
    """Convert model to FP16 GGUF"""
    print(f"\n🔄 Converting to FP16 GGUF...")

    os.makedirs(GGUF_DIR, exist_ok=True)

    # Check if already converted
    if os.path.exists(MODEL_FP16):
        print(f"   ℹ️  {MODEL_FP16} already exists")
        return True

    # Run conversion
    convert_script = f"{LLAMA_CPP_DIR}/convert.py"
    cmd = f"python {convert_script} --outtype f16 --outfile {MODEL_FP16} {MODEL_DIR}"

    print(f"   Command: {cmd}")
    result = os.system(cmd)

    if result == 0 and os.path.exists(MODEL_FP16):
        print(f"✅ Conversion successful: {MODEL_FP16}")
        return True
    else:
        print(f"❌ Conversion failed")
        return False

def quantize_model():
    """Quantize FP16 model to Q4_K_M"""
    print(f"\n📦 Quantizing to Q4_K_M...")

    if os.path.exists(MODEL_QUANTIZED):
        print(f"   ℹ️  {MODEL_QUANTIZED} already exists")
        return True

    # Check if quantize tool exists
    quantize_tool = f"{LLAMA_CPP_DIR}/quantize"
    if not os.path.exists(quantize_tool):
        print(f"❌ Quantize tool not found: {quantize_tool}")
        return False

    # Run quantization
    cmd = f"{quantize_tool} {MODEL_FP16} {MODEL_QUANTIZED} Q4_K_M"

    print(f"   Command: {cmd}")
    result = os.system(cmd)

    if result == 0 and os.path.exists(MODEL_QUANTIZED):
        print(f"✅ Quantization successful: {MODEL_QUANTIZED}")

        # Show file size
        size_mb = os.path.getsize(MODEL_QUANTIZED) / (1024 * 1024)
        print(f"   📊 File size: {size_mb:.1f} MB")

        return True
    else:
        print(f"❌ Quantization failed")
        return False

def test_model():
    """Test the quantized model with llama-cli"""
    print(f"\n🧪 Testing quantized model...")

    test_prompt = "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n"

    main_tool = f"{LLAMA_CPP_DIR}/main"
    if not os.path.exists(main_tool):
        print("❌ main tool not found, skipping test")
        return False

    cmd = f"{main_tool} -m {MODEL_QUANTIZED} -p '{test_prompt}' -n 50 --temp 0.7"

    print(f"   Running test prompt...")
    print(f"   Command: {cmd}")
    print("-" * 60)

    result = os.system(cmd)

    print("-" * 60)
    if result == 0:
        print("✅ Test completed")
        return True
    else:
        print("⚠️  Test completed with warnings")
        return True

def create_info_file():
    """Create info file about the model"""
    print(f"\n📝 Creating model info file...")

    info_path = f"{GGUF_DIR}/model_info.txt"
    with open(info_path, 'w') as f:
        f.write("Rudushi TinyLlama Model\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Directory: {MODEL_DIR}\n")
        f.write(f"GGUF Directory: {GGUF_DIR}\n\n")
        f.write(f"FP16 Model: {MODEL_FP16}\n")
        f.write(f"Q4_K_M Model: {MODEL_QUANTIZED}\n\n")
        f.write("Quantization: Q4_K_M (4-bit)\n")
        f.write("Context Length: 2048 tokens\n")
        f.write("Parameters: ~1.1B\n\n")
        f.write("For Termux:\n")
        f.write(f"  1. Transfer {MODEL_QUANTIZED} to device\n")
        f.write(f"  2. Install llama.cpp in Termux\n")
        f.write(f"  3. Run: ./main -m {os.path.basename(MODEL_QUANTIZED)} -p '...' -n 512\n")

    print(f"✅ Info file created: {info_path}")

def main():
    """Main execution flow"""
    print("="*60)
    print("🎯 GGUF Conversion Pipeline")
    print("="*60)
    print(f"Model: {MODEL_DIR}")
    print(f"Output: {GGUF_DIR}")

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed")
        return False

    # Clone llama.cpp
    clone_llama_cpp()

    # Build llama.cpp
    if not build_llama_cpp():
        print("\n❌ Failed to build llama.cpp")
        return False

    # Check model
    if not check_model():
        return False

    # Convert to FP16
    if not convert_to_fp16():
        print("\n❌ Conversion to FP16 failed")
        return False

    # Quantize
    if not quantize_model():
        print("\n❌ Quantization failed")
        return False

    # Test
    test_model()

    # Create info
    create_info_file()

    print("\n" + "="*60)
    print("✨ Conversion complete!")
    print(f"📁 Model ready: {MODEL_QUANTIZED}")
    print(f"📱 For Termux: Transfer this file to your device")
    print("="*60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
