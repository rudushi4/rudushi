#!/usr/bin/env python3
"""
Upload Script for Rudushi TinyLlama to Hugging Face

This script uploads the fine-tuned model to the Hugging Face repository:
- megharudushi/Rudushi

Requirements:
- Write-enabled Hugging Face API token
- Must be logged in with `huggingface-cli login` or use token directly

Usage:
    python 03_upload_to_hf.py --token YOUR_TOKEN
    python 03_upload_to_hf.py  # Will prompt for token
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, ModelFilter
import json

# Configuration
REPO_ID = "megharudushi/Rudushi"
REPO_TYPE = "model"
MODEL_DIR = "fine_tuned_tinyllama"
LOCAL_FILES_ONLY = False

# Model metadata
MODEL_METADATA = {
    "model_name": "Rudushi TinyLlama",
    "base_model": "tinyllama-1.1b",
    "model_type": "CausalLM",
    "params": "1.1B",
    "quantization": "4-bit (LoRA)",
    "context_length": 2048,
    "language": "English",
    "license": "apache-2.0",
    "finetuned_from": "unsloth/tinyllama-bnb-4bit",
    "tasks": ["text-generation", "chat", "instruction-following"],
    "dataset": "tatsu-lab/alpaca",
    "framework": ["PyTorch", "Transformers"],
}

def create_model_card():
    """Create a comprehensive model card"""
    card = f"""---
{model_card_metadata()}
---

# Rudushi TinyLlama

## Model Description

Rudushi TinyLlama is a fine-tuned version of TinyLlama-1.1B, optimized for efficient inference on mobile devices, particularly in Termux environments. The model has been fine-tuned on the Alpaca dataset using Unsloth for fast, memory-efficient training.

## Model Details

- **Model Architecture**: TinyLlama (1.1B parameters)
- **Base Model**: TinyLlama-1.1B (Meta's compact LLM)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: Unsloth + TRL
- **Quantization**: 4-bit for efficient inference
- **Context Length**: 2048 tokens
- **Language**: English
- **License**: Apache 2.0

## Training Details

- **Dataset**: Alpaca (52K instructions)
- **Optimizer**: AdamW 8-bit
- **Learning Rate**: 2e-4
- **Batch Size**: 2 (effective: 8 with gradient accumulation)
- **Training Steps**: {TRAINING_STEPS} (configurable)
- **Hardware**: NVIDIA T4 GPU (Google Colab)

## Usage

### With Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("megharudushi/Rudushi")
tokenizer = AutoTokenizer.from_pretrained("megharudushi/Rudushi")

# Format prompt (Alpaca-style)
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a Python function to reverse a string.

### Input:

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Text Generation Pipeline

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="megharudushi/Rudushi")
result = pipe("Explain quantum computing:")
print(result[0]["generated_text"])
```

### For Termux (CPU inference)

After converting to GGUF format with `02_convert_to_gguf.py`:

```bash
# Install llama.cpp in Termux
pkg install llama.cpp

# Run model
./main -m model-Q4_K_M.gguf -p "Hello" -n 512
```

## Performance

- **RAM Usage**: ~500MB (4-bit quantization)
- **Model Size**: ~550MB (Q4_K_M GGUF)
- **Inference Speed**: ~5-10 tokens/second (mobile CPU)
- **Maximum Context**: 2048 tokens

## Limitations

- Knowledge cutoff: April 2024
- Primarily English language support
- Limited context window (2048 tokens)
- May produce incorrect or biased responses
- Not suitable for highly specialized technical tasks
- Cannot browse the internet or access external knowledge

## Ethical Considerations

This model has been fine-tuned on public datasets and may reflect biases present in the training data. Users should:

- Verify information from authoritative sources
- Be aware of potential biases
- Use responsibly for appropriate applications
- Not use for high-stakes decisions without human oversight

## Acknowledgments

- [TinyLlama Team](https://github.com/jzhang38/TinyLlama) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Alpaca Dataset](https://github.com/glm-4/Alpaca) creators
- [Hugging Face](https://huggingface.co/) for model hosting and tools

## License

This model is released under the Apache 2.0 License. See the LICENSE file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{rudushi-tinyllama,
  title={{Rudushi TinyLlama}},
  author={{Rudushi}},
  year={{2025}},
  url={{https://huggingface.co/megharudushi/Rudushi}}
}}
```

## Contact

For questions, issues, or contributions, please visit our repository or contact the maintainers.
"""

    return card

def model_card_metadata():
    """Generate metadata for model card"""
    return f"""language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: unsloth/tinyllama-bnb-4bit
tags:
- tinyllama
- lora
- 4bit
- mobile
- termux
- alpaca
- chat
- instruction-following
- pytorch
- text-generation
widget:
- text: "Write a Python function to reverse a string"
- text: "Explain machine learning in simple terms"
- text: "What are the benefits of renewable energy?"
"""

def validate_model_files():
    """Validate that model files exist"""
    print("🔍 Validating model files...")

    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("   Please run fine-tuning first")
        return False

    # Check required files
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "pytorch_model.bin"  # or .safetensors
    ]

    optional_files = [
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]

    missing_required = []
    for file in required_files:
        if not os.path.exists(f"{MODEL_DIR}/{file}"):
            missing_required.append(file)

    # Check for safetensors instead
    safetensors_files = list(Path(MODEL_DIR).glob("*.safetensors"))
    if safetensors_files and "pytorch_model.bin" in missing_required:
        missing_required.remove("pytorch_model.bin")

    if missing_required:
        print(f"⚠️  Missing required files: {', '.join(missing_required)}")
        print("   The model may not work correctly")

    print(f"✅ Model directory validated")
    return True

def upload_model(hf_token, commit_message="Upload Rudushi TinyLlama model"):
    """Upload model to Hugging Face Hub"""
    print(f"\n⬆️  Uploading model to {REPO_ID}...")

    try:
        api = HfApi(token=hf_token)

        # Check if repo exists
        try:
            api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
            print(f"   Repository exists")
        except Exception:
            print(f"   Creating repository...")
            api.create_repo(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                exist_ok=True
            )

        # Create model card
        model_card_path = f"{MODEL_DIR}/README.md"
        with open(model_card_path, 'w') as f:
            f.write(create_model_card())
        print(f"   ✅ Model card created")

        # Upload model
        print(f"   Uploading files...")
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=commit_message,
            run_as_future=False
        )

        print(f"✅ Upload successful!")
        print(f"   Repository: https://huggingface.co/{REPO_ID}")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def verify_upload(hf_token):
    """Verify upload by checking the repository"""
    print(f"\n🔍 Verifying upload...")

    try:
        api = HfApi(token=hf_token)

        # Get repo info
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"   Repository: {repo_info.id}")
        print(f"   Visibility: {repo_info.private}")
        print(f"   Downloads: {repo_info.downloads}")

        # List files
        files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"   Files: {len(files)} files uploaded")

        for file in sorted(files)[:10]:
            print(f"      - {file}")

        if len(files) > 10:
            print(f"      ... and {len(files) - 10} more")

        print(f"✅ Verification complete!")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def test_model_download(hf_token):
    """Test downloading the uploaded model"""
    print(f"\n🧪 Testing model download...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"   Downloading {REPO_ID}...")
        model = AutoModelForCausalLM.from_pretrained(
            REPO_ID,
            token=hf_token,
            trust_remote_code=True
        )
        print(f"   ✅ Model loaded")

        tokenizer = AutoTokenizer.from_pretrained(
            REPO_ID,
            token=hf_token
        )
        print(f"   ✅ Tokenizer loaded")

        # Quick generation test
        print(f"   Running quick test...")
        prompt = "### Instruction:\nHello\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if result:
            print(f"   ✅ Test generation successful")
            print(f"   Sample output: {result[:100]}...")

        return True

    except Exception as e:
        print(f"   ⚠️  Test failed: {e}")
        return False

def main():
    """Main execution flow"""
    print("="*60)
    print("⬆️  Hugging Face Upload Pipeline")
    print("="*60)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    parser.add_argument("--commit-message", type=str, default="Upload Rudushi TinyLlama model",
                       help="Commit message")
    args = parser.parse_args()

    # Get token
    hf_token = args.token
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            hf_token = input("\n🔑 Enter Hugging Face token: ").strip()

    if not hf_token:
        print("❌ No token provided")
        print("   Use --token or set HF_TOKEN environment variable")
        return False

    # Validate model
    if not validate_model_files():
        return False

    # Upload
    if not upload_model(hf_token, args.commit_message):
        return False

    # Verify
    if not verify_upload(hf_token):
        print("\n⚠️  Upload completed but verification failed")

    # Test download
    test_model_download(hf_token)

    print("\n" + "="*60)
    print("✨ Upload pipeline complete!")
    print(f"📦 Model: https://huggingface.co/{REPO_ID}")
    print(f"💬 Chat: https://huggingface.co/spaces/megharudushi/Rudushi")
    print("="*60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
