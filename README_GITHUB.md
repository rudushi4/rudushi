# 🤖 Rudushi - Automated AI Assistant via GitHub Actions

[![License](https://img.shields.io/badge/📄-Apache%202.0-green)](LICENSE)
[![Model](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/megharudushi/Rudushi)
[![Actions](https://img.shields.io/badge/⚙️-GitHub%20Actions-blue)](./.github/workflows/)

**Rudushi** is a lightweight AI assistant that fine-tunes itself automatically using GitHub Actions! No GPU needed on your device - everything runs in the cloud! ☁️✨

---

## 🎯 What is Rudushi?

Rudushi is a **1.1B parameter language model** (TinyLlama-based) that:
- ✅ **Fine-tunes itself** in the cloud using GitHub Actions
- ✅ **No GPU required** on your device
- ✅ **Runs on Android** (Termux optimized)
- ✅ **Optimized for coding** and agentic tasks
- ✅ **Deploys automatically** to Hugging Face

---

## 🚀 How It Works (Fully Automated!)

### 1️⃣ **Trigger Fine-tuning**
Just click a button in GitHub Actions!

### 2️⃣ **GitHub Actions Does The Rest**
- ✅ Downloads TinyLlama-1.1B
- ✅ Fine-tunes on Alpaca dataset (52K instructions)
- ✅ Uses LoRA (Rank 16) for efficiency
- ✅ Runs on **NVIDIA GPU** in the cloud
- ✅ Saves your custom model

### 3️⃣ **Automatic Conversion & Upload**
- ✅ Converts model to GGUF format
- ✅ Uploads to Hugging Face Hub
- ✅ Creates beautiful model card
- ✅ Deploys to Hugging Face Spaces (web UI)

### 4️⃣ **Download & Use**
- ✅ Download from Hugging Face
- ✅ Run in Termux on your Android device
- ✅ Chat with your custom Rudushi!

---

## 🎛️ Automated Workflows

### Workflow 1: Fine-Tune Model
**File**: [.github/workflows/fine_tune.yml](./.github/workflows/fine_tune.yml)

**Triggers**: Manual dispatch

**What it does**:
- Runs on GPU runner (linux-cuda11.8-cudnn8-runtime)
- Installs Unsloth, PyTorch, Transformers
- Fine-tunes TinyLlama-1.1B on Alpaca
- Configurable steps (100/500/1000+)
- Saves model artifact
- Creates GitHub release

**Duration**: 30-90 minutes (depending on steps)

### Workflow 2: Convert & Upload
**File**: [.github/workflows/convert_and_upload.yml](./.github/workflows/convert_and_upload.yml)

**Triggers**: After fine-tuning completes

**What it does**:
- Downloads fine-tuned model
- Converts to GGUF using llama.cpp
- Quantizes to Q4_K_M (4-bit)
- Uploads to Hugging Face Hub
- Creates model card
- Tests the model

**Duration**: 10-15 minutes

### Workflow 3: Deploy Spaces
**File**: [.github/workflows/deploy_spaces.yml](./.github/workflows/deploy_spaces.yml)

**Triggers**: After upload completes

**What it does**:
- Creates Hugging Face Space
- Deploys Gradio web interface
- Connects to your model
- Makes it accessible via web browser

**Duration**: 2-5 minutes

---

## 🔥 Key Advantages

### Traditional Approach ❌
- Need GPU on device
- Use Google Colab (time limits)
- Manual setup
- Manual upload
- Complex workflow

### **Our Approach** ✅
- **GPU in the cloud** (GitHub Actions)
- **No time limits** (up to 6 hours)
- **Fully automated** (one click)
- **Automatic upload** to Hugging Face
- **Simple workflow** (just trigger!)

---

## 🎯 Quick Start

### Step 1: Fork This Repository

```bash
# Fork on GitHub: https://github.com/rudushi4/rudushi/fork
```

### Step 2: Set GitHub Secrets

Go to `Settings → Secrets and variables → Actions`

Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `HF_TOKEN` | Your Hugging Face token (write permission) |
| `HF_REPO` | `megharudushi/Rudushi` |

**Get HF Token**:
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "Rudushi"
4. Role: **Write**
5. Copy token

### Step 3: Trigger Fine-tuning

1. Go to **Actions** tab
2. Select **"Fine-Tune Rudushi Model"**
3. Click **"Run workflow"**
4. Choose steps (100 for testing, 1000+ for production)
5. Click **"Run workflow"**

### Step 4: Wait & Relax! ☕

GitHub Actions will:
- ✅ Fine-tune the model (30-90 min)
- ✅ Convert to GGUF (10 min)
- ✅ Upload to Hugging Face (5 min)
- ✅ Deploy to Spaces (5 min)

### Step 5: Use Your Rudushi!

**Download Model**: https://huggingface.co/megharudushi/Rudushi

**Web UI**: https://huggingface.co/spaces/YOUR_USERNAME/rudushi-spaces-rudushi

**Termux**: Transfer model and run:
```bash
./rudushi
# Select option 1: Chat with Rudushi
```

---

## 📊 Workflow Status

Check the status in the **Actions** tab:

| Status | Description |
|--------|-------------|
| 🟢 Success | All workflows completed |
| 🔴 Failed | Check logs for errors |
| 🟡 In Progress | Currently running |

---

## 🛠️ Manual Workflows

You can also run workflows manually:

### Fine-Tune Model
```bash
# Actions → Fine-Tune Rudushi Model → Run workflow
```

### Convert & Upload
```bash
# Actions → Convert & Upload to Hugging Face → Run workflow
```

### Deploy Spaces
```bash
# Actions → Deploy to Hugging Face Spaces → Run workflow
```

---

## 📁 Repository Structure

```
rudushi/
├── .github/workflows/          # GitHub Actions workflows
│   ├── fine_tune.yml           # Auto fine-tuning
│   ├── convert_and_upload.yml  # Auto conversion & upload
│   └── deploy_spaces.yml       # Auto Spaces deployment
│
├── app.py                      # Gradio web interface
├── rudushi_chat.py             # Termux chat interface
├── requirements.txt            # Python dependencies
│
└── README.md                   # This file
```

---

## 🎓 Customization

### Change Training Steps

Edit `.github/workflows/fine_tune.yml`:

```yaml
inputs:
  steps:
    description: 'Number of training steps'
    required: true
    default: '1000'  # ← Change here
```

### Different Dataset

Edit `fine_tune.yml`, replace dataset:

```python
dataset = load_dataset("your/dataset", split="train")
```

### Custom Model Size

Currently supports:
- TinyLlama-1.1B ✅
- Add your own in `fine_tune.yml`

---

## 📈 Expected Results

### Model Quality vs Steps

| Steps | Time | Quality | Use Case |
|-------|------|---------|----------|
| 100 | 30 min | Basic | Testing |
| 500 | 90 min | Good | Personal use |
| 1000 | 3+ hours | Excellent | Production |
| 2000 | 6+ hours | SOTA | Research |

### Performance on Mobile

| Model Size | RAM | Speed | Quality |
|------------|-----|-------|---------|
| 1.1B | 500MB | 5-10 tok/s | ⭐⭐⭐ |
| 3B | 1.2GB | 2-5 tok/s | ⭐⭐⭐⭐ |
| 7B | 3GB | 1-2 tok/s | ⭐⭐⭐⭐⭐ |

---

## 🐛 Troubleshooting

### Workflow Failed

**Check logs**:
1. Actions tab → Failed workflow
2. Click on job
3. Click on failing step
4. Read error message

**Common issues**:
- `HF_TOKEN` missing or invalid
- Repository permissions
- Out of memory (reduce batch size)

### Upload Failed

**Solutions**:
- Ensure `HF_TOKEN` has **Write** permission
- Check `HF_REPO` secret is correct
- Repository must exist (will be created)

### Model Quality Poor

**Solutions**:
- Increase training steps
- Use better dataset
- Adjust learning rate

---

## 🎯 Use Cases

### 1. Personal AI Assistant
Fine-tune on your conversations and use daily

### 2. Coding Assistant
Optimize for programming tasks

### 3. Chatbot
Deploy in Spaces for public access

### 4. Mobile AI
Run offline on Android with Termux

---

## 📚 Documentation

- **Full Guide**: [README.md](./README.md)
- **Fine-tuning**: [FINETUNE_GUIDE.md](./FINETUNE_GUIDE.md)
- **Rudushi Chat**: [RUDUSHI_README.md](./RUDUSHI_README.md)
- **Technical**: [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

## 📄 License

**Apache 2.0** - See [LICENSE](LICENSE)

---

## 🎉 Try It Now!

1. **Fork**: https://github.com/rudushi4/rudushi/fork
2. **Set Secrets**: Add `HF_TOKEN` and `HF_REPO`
3. **Run Workflow**: Actions → Fine-Tune Rudushi → Run
4. **Wait**: ~50 minutes total
5. **Use**: Chat with your custom Rudushi!

---

**No GPU? No problem! GitHub Actions has you covered! ☁️✨**

## ⭐ Star This Repo

If you found this useful, ⭐ star the repository!

---

**Made with ❤️ using GitHub Actions & Hugging Face**
