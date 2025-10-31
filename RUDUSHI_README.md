# 🤖 Rudushi - Your AI Assistant

[![License](https://img.shields.io/badge/📄-Apache%202.0-green)](LICENSE)
[![Model](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/megharudushi/Rudushi)
[![Platform](https://img.shields.io/badge/📱-Android%20%7C%20Termux-blue)](#)

**Rudushi** is a lightweight language model (1.1B parameters) specifically optimized for mobile devices running Android with Termux. Experience the power of AI on your phone!

---

## ✨ What Makes Rudushi Special

🎯 **Mobile-First**: Designed from the ground up for Android + Termux
⚡ **Fast & Efficient**: 4-bit quantization for optimal performance
💾 **Memory Optimized**: Runs in just 500MB RAM
🔒 **Privacy-First**: Runs entirely on your device
🌐 **Open Source**: Apache 2.0 licensed

---

## 🚀 Quick Start

### Method 1: Use the Rudushi Menu (Easiest)

```bash
cd /data/data/com.termux/files/home/rudushi
./rudushi
```

Choose option **1** to start chatting immediately!

---

### Method 2: Direct Chat

```bash
# List available models
ls /data/data/com.termux/files/home/rudushi_model/*.gguf

# Start chatting
python3 rudushi_chat.py --model TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
```

---

### Method 3: One-Time Generation

```bash
cd /data/data/com.termux/files/home/rudushi
./rudushi
```

Choose option **2** for quick text generation!

---

## 📦 Download a Model

### Option 1: Interactive Downloader

```bash
cd /data/data/com.termux/files/home/rudushi
./05_download_model.sh
```

### Option 2: Browser Download

1. Visit: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
2. Download: `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` (~550MB)
3. Save to: `/data/data/com.termux/files/home/rudushi_model/`

### Recommended Models

| Model | Size | RAM | Quality | Speed |
|-------|------|-----|---------|-------|
| **TinyLlama-1.1B** | 550MB | 500MB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Qwen2-1.5B** | 940MB | 800MB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Phi-3-Mini-4K** | 2.3GB | 2GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 💬 Chat Examples

### Example 1: Q&A

```
You: Explain quantum computing in simple terms

Rudushi: Quantum computing is like having a super-powered calculator that can explore
multiple possibilities at once, unlike regular computers that go step by step. It's
based on quantum mechanics - the science of very tiny particles that can exist in
multiple states simultaneously. This lets quantum computers solve certain problems
much faster than classical computers.
```

### Example 2: Code Help

```
You: Write a Python function to sort a list

Rudushi:
```python
def sort_list(arr):
    # Using Python's built-in sort
    return sorted(arr)

# Or implement bubble sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Example usage
my_list = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(my_list))
```
```

### Example 3: Creative Writing

```
You: Write a haiku about AI

Rudushi:
```
Code dreams in silicon,
Neural networks hum softly—
Future wakes up now.
```
```

---

## 🎛️ Rudushi Commands

### In the Chat Interface

- **`/help`** - Show help menu
- **`/history`** - View conversation history
- **`/clear`** - Clear conversation history
- **`/quit`** or **`/exit`** - Exit chat

### Command Line Options

```bash
# Chat with custom settings
python3 rudushi_chat.py \
  --model TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
  --tokens 512 \
  --temperature 0.5

# Chat without saving history
python3 rudushi_chat.py --model MODEL.gguf --no-history
```

---

## 🛠️ Fine-Tuning Rudushi (Advanced)

### Prerequisites

- GPU with 8GB+ VRAM (T4, V100, A100, etc.)
- Google Colab account (recommended for beginners)

### Steps

1. **Prepare Environment**
   ```bash
   cd /data/data/com.termux/files/home/rudushi
   ./rudushi
   # Select option 5 → Fine-tuning setup
   ```

2. **Run Fine-tuning**
   ```bash
   # In Google Colab or GPU machine
   python3 01_fine_tune_tinyllama.py
   ```

3. **Convert to GGUF**
   ```bash
   python3 02_convert_to_gguf.py
   ```

4. **Upload to Hugging Face**
   ```bash
   python3 03_upload_to_hf.py --token YOUR_HF_TOKEN
   ```

### Fine-tuning Configuration

See `rudushi_config.yaml` for all available options:

- **Model**: TinyLlama-1.1B
- **Method**: LoRA (Rank 16)
- **Dataset**: Alpaca (52K instructions)
- **Steps**: 100 (adjustable)
- **Hardware**: NVIDIA T4 (free in Colab)

---

## 📊 Performance Benchmarks

### On Your Device (7.5GB RAM)

| Metric | TinyLlama-1.1B | Qwen2-1.5B | Phi-3-Mini-4K |
|--------|----------------|------------|---------------|
| Model Load Time | 5-8s | 8-12s | 15-20s |
| First Token | 2-3s | 3-5s | 5-8s |
| Tokens/sec | 7-10 | 5-8 | 3-5 |
| Memory Usage | 500MB | 800MB | 2GB |
| Context Window | 2048 | 32768 | 4096 |

### Optimization Tips

1. **Close other apps** before running
2. **Use `--mlock`** flag (included by default)
3. **Lower context** if needed: `-c 1024`
4. **Reduce tokens**: Use `--tokens 128` instead of 512

---

## 🐛 Troubleshooting

### Problem: Model won't start

**Solution**:
```bash
# Check model file exists
ls -lh /data/data/com.termux/files/home/rudushi_model/

# Check llama-cli works
/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli --help
```

### Problem: Out of memory

**Solution**:
```bash
# Reduce context size
python3 rudushi_chat.py --model MODEL.gguf

# Or edit config to use smaller model
# Download TinyLlama-1.1B (550MB) instead of larger models
```

### Problem: Slow performance

**Solution**:
```bash
# Check available RAM
free -h

# Use performance flags
/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli \
  -m MODEL.gguf \
  -t 4 \
  -b 512 \
  --mlock
```

### Problem: Can't download model

**Solution**:
1. Get free Hugging Face account: https://huggingface.co/join
2. Generate token: https://huggingface.co/settings/tokens
3. Use token for download:
   ```bash
   ./05_download_model.sh
   # Enter token when prompted
   ```

---

## 📁 Project Structure

```
/data/data/com.termux/files/home/rudushi/
├── 🤖 Main Interface
│   ├── rudushi                     # Main menu launcher
│   ├── rudushi_chat.py             # Chatbot interface
│   └── rudushi_config.yaml         # Model configuration
│
├── 🔧 Scripts
│   ├── 00_quickstart.py            # Interactive guide
│   ├── 01_fine_tune_tinyllama.py   # Fine-tuning
│   ├── 02_convert_to_gguf.py       # Model conversion
│   ├── 03_upload_to_hf.py          # Hugging Face upload
│   ├── 04_termux_deploy.sh         # Deployment menu
│   └── 05_download_model.sh        # Model downloader
│
├── 🌐 Web Interface
│   └── app.py                      # Gradio Spaces app
│
└── 📖 Documentation
    ├── README.md                   # Full documentation
    ├── RUDUSHI_README.md           # This file
    ├── SETUP_STATUS.md             # Setup guide
    ├── PROJECT_SUMMARY.md          # Technical details
    └── QUICK_REFERENCE.txt         # Command reference
```

---

## 🎓 Learning Resources

### For Beginners

1. **Start Here**: `QUICK_REFERENCE.txt`
2. **Your Device Setup**: `SETUP_STATUS.md`
3. **Quick Start**: `./rudushi` → Option 1

### For Intermediate Users

1. **Full Documentation**: `README.md`
2. **Chat Interface**: `rudushi_chat.py --help`
3. **Model Download**: `./05_download_model.sh`

### For Advanced Users

1. **Technical Details**: `PROJECT_SUMMARY.md`
2. **Fine-tuning Config**: `rudushi_config.yaml`
3. **Source Scripts**: All `.py` files are documented

---

## 🤝 Community & Support

### Get Help

- 📖 **Documentation**: All files in `/data/data/com.termux/files/home/rudushi/`
- 🐛 **Issues**: All scripts have `--help` option
- 💬 **Chat**: Use `/help` in chat interface

### Share Your Experience

- Try different models
- Share your fine-tuned models
- Build applications with Rudushi
- Contribute improvements

---

## 🔐 Privacy & Security

✅ **Privacy-First**: All inference runs locally on your device
✅ **No Telemetry**: No data sent to external servers
✅ **Open Source**: All code is auditable
✅ **Local Storage**: Conversations saved only on your device
✅ **Token-Based**: Only Hugging Face model downloads require tokens

---

## 📈 Roadmap

### Near Term (Next 30 days)
- [ ] More model support (Qwen, Gemma, etc.)
- [ ] Performance optimizations
- [ ] Mobile app wrapper

### Future (Next 90 days)
- [ ] Fine-tuned Rudushi variants
- [ ] Multi-language support
- [ ] Web interface optimization
- [ ] API server mode

---

## 🙏 Acknowledgments

- **TinyLlama Team** - Base model architecture
- **Unsloth** - Efficient fine-tuning framework
- **llama.cpp** - Inference engine
- **Hugging Face** - Model hosting
- **Alpaca Dataset** - Training data

---

## 📄 License

**Apache 2.0** - See [LICENSE](LICENSE) for details.

You are free to:
- ✅ Use for personal or commercial purposes
- ✅ Modify and distribute
- ✅ Include in proprietary software

---

## ⚡ Quick Commands Cheat Sheet

```bash
# Start Rudushi
./rudushi

# Chat directly
python3 rudushi_chat.py --model MODEL.gguf

# Quick generation
./rudushi → Option 2

# Download model
./05_download_model.sh

# List models
ls /data/data/com.termux/files/home/rudushi_model/*.gguf

# View history
cat rudushi_history.json

# Run full setup
./04_termux_deploy.sh
```

---

## 🎉 Get Started Now!

```bash
cd /data/data/com.termux/files/home/rudushi
./rudushi
```

**Just run `./rudushi` and start chatting! Your AI assistant awaits!** 🤖✨

---

*Rudushi - Making AI accessible on mobile devices* 📱🤖
