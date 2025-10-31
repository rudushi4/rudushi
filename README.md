# 🎯 Rudushi: Lightweight Language Model for Termux

A compact, efficient language model optimized for mobile devices, particularly Termux on Android. Built on TinyLlama-1.1B and fine-tuned with Unsloth for fast, memory-efficient inference on mobile hardware.

[![Hugging Face Model](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/megharudushi/Rudushi)
[![Spaces Demo](https://img.shields.io/badge/🌐-Spaces%20Demo-blue)](https://huggingface.co/spaces/megharudushi/Rudushi)
[![License](https://img.shields.io/badge/📄-Apache%202.0-green)](LICENSE)

## 🌟 Features

- **Compact Size**: 1.1B parameters, optimized for mobile devices
- **Efficient Inference**: 4-bit quantization, ~500MB RAM usage
- **Mobile-Optimized**: Runs smoothly in Termux on Android
- **Fast Training**: Unsloth-accelerated fine-tuning
- **Multiple Deployment Options**: Hugging Face Spaces, Termux, API
- **Open Source**: Apache 2.0 licensed

## 📊 Model Specifications

| Attribute | Value |
|-----------|-------|
| **Parameters** | 1.1B |
| **Context Length** | 2048 tokens |
| **Quantization** | 4-bit (Q4_K_M) |
| **Model Size** | ~550MB (GGUF) |
| **RAM Usage** | ~500MB |
| **Inference Speed** | 5-10 tokens/sec (mobile CPU) |
| **Language** | English |
| **License** | Apache 2.0 |

## 🚀 Quick Start

### Option 1: Hugging Face Spaces (Easiest)

Visit the live demo: [https://huggingface.co/spaces/megharudushi/Rudushi](https://huggingface.co/spaces/megharface/spaces/megharudushi/Rudushi)

No installation required - just open in your browser!

### Option 2: Termux (Mobile)

```bash
# Install Termux from F-Droid or Google Play
# Then run:

pkg update && pkg install -y git
git clone https://github.com/your-username/rudushi.git
cd rudushi
chmod +x 04_termux_deploy.sh
./04_termux_deploy.sh
```

Select option 11 for full automatic setup.

### Option 3: Direct Inference

```bash
# Download the GGUF model
wget https://huggingface.co/megharudushi/Rudushi/resolve/main/model-Q4_K_M.gguf

# Run with llama.cpp
./main -m model-Q4_K_M.gguf -p "Explain quantum computing" -n 100
```

## 📁 Project Structure

```
rudushi/
├── 01_fine_tune_tinyllama.py    # Fine-tuning script
├── 02_convert_to_gguf.py       # GGUF conversion script
├── 03_upload_to_hf.py          # Hugging Face upload script
├── 04_termux_deploy.sh         # Termux deployment script
├── app.py                      # Spaces app (Gradio)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Full Pipeline

### 1. Fine-tuning

Fine-tune TinyLlama on your dataset using Unsloth (requires GPU):

```bash
python 01_fine_tune_tinyllama.py
```

**Configuration**:
- Model: TinyLlama-1.1B (4-bit)
- Dataset: Alpaca (52K instructions)
- LoRA: Rank 16, Alpha 16
- Steps: 60 (adjust as needed)

### 2. Convert to GGUF

Convert the PyTorch model to GGUF format for CPU inference:

```bash
python 02_convert_to_gguf.py
```

**Output**:
- `gguf_models/model-fp16.gguf` - Full precision
- `gguf_models/model-Q4_K_M.gguf` - 4-bit quantized

### 3. Upload to Hugging Face

Upload to your HF repository:

```bash
python 03_upload_to_hf.py --token YOUR_HF_TOKEN
```

**Requirements**:
- Write-enabled Hugging Face API token
- Repository: `megharudushi/Rudushi`

### 4. Deploy in Termux

Deploy on mobile device:

```bash
chmod +x 04_termux_deploy.sh
./04_termux_deploy.sh
```

**What it does**:
1. Checks system requirements
2. Installs dependencies
3. Downloads the model
4. Builds llama.cpp
5. Tests the model
6. Sets up interactive chat

## 💻 Usage Examples

### Basic Text Generation

```bash
./main -m model-Q4_K_M.gguf \
  -p "Write a Python function to reverse a string" \
  -n 100 \
  --temp 0.7
```

### Interactive Chat

```bash
./main -m model-Q4_K_M.gguf --interactive --mlock
```

### Python Wrapper

```bash
# Interactive mode
python run_rudushi.py --interactive

# Single prompt
python run_rudushi.py --prompt "Explain machine learning" --tokens 200
```

### API Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("megharudushi/Rudushi")
tokenizer = AutoTokenizer.from_pretrained("megharudushi/Rudushi")

prompt = "### Instruction:\nWrite a haiku about AI\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## ⚙️ Performance Optimization

### For Mobile Devices

```bash
# Optimize for speed
./main -m model-Q4_K_M.gguf \
  -p "Your prompt" \
  -b 512 \          # Batch size
  -c 1024 \         # Reduce context
  --mlock           # Lock in memory
```

### For Better Quality

```bash
# Optimize for quality
./main -m model-Q4_K_M.gguf \
  -p "Your prompt" \
  -n 512 \          # More tokens
  --temp 0.5 \      # Lower temperature
  --top-p 0.95 \    # Higher top-p
  --repeat_penalty 1.15
```

## 📈 Benchmarks

| Device | RAM | Tokens/sec | Notes |
|--------|-----|------------|-------|
| Pixel 6 (Termux) | 8GB | 7-10 | 4-bit, Q4_K_M |
| Galaxy S22 | 6GB | 5-8 | 4-bit, optimized |
| Desktop (CPU) | 16GB | 20-30 | 4-bit |

*Benchmarks run with context size 2048, generating 100 tokens*

## 🐛 Troubleshooting

### Model won't download

```bash
# Check network
ping huggingface.co

# Manual download
wget https://huggingface.co/megharudushi/Rudushi/resolve/main/model-Q4_K_M.gguf
```

### Out of memory

```bash
# Reduce context
./main -m model-Q4_K_M.gguf -c 1024

# Use mlock
./main -m model-Q4_K_M.gguf --mlock
```

### Slow inference

- Use Q4_K_M quantization
- Reduce batch size (`-b`)
- Lock memory (`--mlock`)
- Close other apps

### Build errors in Termux

```bash
# Update packages
pkg update -y

# Install build tools
pkg install -y clang make cmake

# Clean rebuild
cd llama.cpp
make clean
make
```

## 📚 Resources

- **Hugging Face Model**: [megharudushi/Rudushi](https://huggingface.co/megharudushi/Rudushi)
- **Spaces Demo**: [Spaces/megharudushi/Rudushi](https://huggingface.co/spaces/megharudushi/Rudushi)
- **TinyLlama**: [GitHub](https://github.com/jzhang38/TinyLlama)
- **Unsloth**: [GitHub](https://github.com/unslothai/unsloth)
- **llama.cpp**: [GitHub](https://github.com/ggerganov/llama.cpp)
- **Termux**: [GitHub](https://github.com/termux/termux-app)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Limitations

- Knowledge cutoff: April 2024
- Primarily English language support
- Limited to 2048 token context
- May produce incorrect or biased responses
- Not suitable for specialized technical tasks
- Cannot browse the internet or access real-time information

## 🙏 Acknowledgments

- [TinyLlama Team](https://github.com/jzhang38/TinyLlama) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca) creators
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for inference engine
- [Hugging Face](https://huggingface.co/) for model hosting
- [Termux](https://termux.dev/) for mobile environment

## 📧 Contact

- **Model**: [megharudushi/Rudushi](https://huggingface.co/megharudushi/Rudushi)
- **Issues**: [GitHub Issues](https://github.com/your-username/rudushi/issues)
- **Email**: your-email@example.com

---

**Rudushi** - *Efficient AI for Mobile Devices* 🤖📱
