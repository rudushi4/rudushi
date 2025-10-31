#!/usr/bin/env python3
"""
Quick Start Guide for Rudushi TinyLlama

This script provides an interactive guide through the entire pipeline
from fine-tuning to deployment.
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_step(step, description):
    """Print a step"""
    print(f"\n{step}. {description}")
    print("-" * 60)

def wait_for_user():
    """Wait for user to press Enter"""
    input("\nPress Enter to continue...")

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n✅ GPU detected: {gpu_name}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("\n⚠️  No GPU detected. Fine-tuning will be very slow or fail.")
            return False
    except:
        print("\n⚠️  Could not check GPU status")
        return False

def check_huggingface_token():
    """Check if HF token is configured"""
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"\n✅ HF_TOKEN found in environment")
        return True
    else:
        print("\n⚠️  HF_TOKEN not found in environment")
        print("   You can set it with: export HF_TOKEN=your_token")
        return False

def main():
    """Main interactive guide"""
    print_header("Rudushi TinyLlama - Quick Start Guide")

    print("\nWelcome to Rudushi TinyLlama!")
    print("\nThis guide will walk you through:")
    print("  1. Fine-tuning TinyLlama on Alpaca dataset")
    print("  2. Converting to GGUF format for CPU inference")
    print("  3. Uploading to Hugging Face Hub")
    print("  4. Deploying in Termux for mobile inference")

    wait_for_user()

    # Step 1: Environment Check
    print_step(1, "Environment Check")

    print("\nChecking your environment...")

    # Check Python
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("   ❌ Python 3.8+ required")
        sys.exit(1)

    # Check GPU
    has_gpu = check_gpu()

    # Check HF token
    has_token = check_huggingface_token()

    print("\n📋 Summary:")
    print(f"   GPU: {'✅ Yes' if has_gpu else '❌ No'}")
    print(f"   HF Token: {'✅ Yes' if has_token else '⚠️  Not configured'}")

    if not has_gpu:
        print("\n⚠️  Warning: GPU strongly recommended for fine-tuning")
        print("   Consider using Google Colab (free T4 GPU)")

    wait_for_user()

    # Step 2: Fine-tuning
    print_step(2, "Fine-tuning TinyLlama")

    print("\nWe'll now fine-tune TinyLlama-1.1B on the Alpaca dataset")
    print("\nThis will:")
    print("  - Load TinyLlama-1.1B (4-bit quantized)")
    print("  - Add LoRA adapters (rank 16)")
    print("  - Train on 52K Alpaca instructions")
    print("  - Save the fine-tuned model")

    print("\n⏱️  Estimated time: 10-30 minutes (with GPU)")

    if has_gpu:
        print("\nRun the fine-tuning script:")
        print("   python 01_fine_tune_tinyllama.py")
    else:
        print("\n⚠️  Skipping fine-tuning (no GPU)")

    wait_for_user()

    # Step 3: Convert to GGUF
    print_step(3, "Convert to GGUF")

    print("\nConverting the model to GGUF format...")
    print("\nThis will:")
    print("  - Clone and build llama.cpp")
    print("  - Convert PyTorch to FP16 GGUF")
    print("  - Quantize to Q4_K_M (4-bit)")
    print("  - Create a model optimized for CPU inference")

    print("\n⏱️  Estimated time: 5-15 minutes")

    print("\nRun the conversion script:")
    print("   python 02_convert_to_gguf.py")

    print("\n📁 Output:")
    print("   gguf_models/model-Q4_K_M.gguf (~550MB)")

    wait_for_user()

    # Step 4: Upload to HF
    print_step(4, "Upload to Hugging Face")

    print("\nUploading the model to Hugging Face Hub...")
    print("\nThis will:")
    print("  - Upload model files to megharudushi/Rudushi")
    print("  - Create a model card with documentation")
    print("  - Make the model publicly available")

    print("\nRequirements:")
    print("  - Hugging Face account")
    print("  - Write-enabled API token")

    if has_token:
        print("\nRun the upload script:")
        print("   python 03_upload_to_hf.py")
    else:
        print("\n⚠️  Please set HF_TOKEN first:")
        print("   export HF_TOKEN=your_token")
        print("   # Or get token from: https://huggingface.co/settings/tokens")

    wait_for_user()

    # Step 5: Deploy in Termux
    print_step(5, "Deploy in Termux")

    print("\nDeploying on mobile device via Termux...")
    print("\nThis will:")
    print("  - Install llama.cpp in Termux")
    print("  - Download the GGUF model")
    print("  - Set up interactive chat interface")
    print("  - Test the model")

    print("\nOn your Android device:")
    print("1. Install Termux from F-Droid")
    print("2. Install git: pkg install git")
    print("3. Clone this repository")
    print("4. Run: ./04_termux_deploy.sh")

    print("\n📱 Recommended device specs:")
    print("   - RAM: 4GB+ (6GB+ recommended)")
    print("   - Storage: 2GB+ free")
    print("   - Android: 7.0+")

    wait_for_user()

    # Step 6: Hugging Face Spaces
    print_step(6, "Deploy on Hugging Face Spaces")

    print("\nCreating a web UI on Hugging Face Spaces...")
    print("\nThis will:")
    print("  - Create a Gradio chat interface")
    print("  - Host the model on free infrastructure")
    print("  - Make it accessible via web browser")

    print("\nSteps:")
    print("1. Go to: https://huggingface.co/new-space")
    print("2. Select 'Gradio' SDK")
    print("3. Upload app.py and requirements.txt")
    print("4. Set model: megharudushi/Rudushi")
    print("5. Deploy!")

    print("\n🌐 Your Spaces app will be at:")
    print("   https://huggingface.co/spaces/YOUR_USERNAME/Rudushi")

    wait_for_user()

    # Completion
    print_header("Complete!")

    print("\n🎉 All steps completed!")
    print("\nYou now have:")
    print("  ✅ Fine-tuned TinyLlama model")
    print("  ✅ GGUF model for mobile inference")
    print("  ✅ Model uploaded to Hugging Face")
    print("  ✅ Termux deployment ready")
    print("  ✅ Spaces web UI option")

    print("\n📚 Next Steps:")
    print("  1. Test the model in Termux")
    print("  2. Try the Spaces demo")
    print("  3. Fine-tune on your own dataset")
    print("  4. Build applications on top of the model")

    print("\n📖 Documentation:")
    print("  README.md - Full documentation")
    print("  app.py - Spaces app code")
    print("  04_termux_deploy.sh - Termux deployment")

    print("\n💬 Need help?")
    print("  Issues: https://github.com/your-username/rudushi/issues")
    print("  Model: https://huggingface.co/megharudushi/Rudushi")

    print("\n" + "="*60)
    print("Thank you for using Rudushi TinyLlama! 🚀")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
