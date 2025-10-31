# 🔬 Rudushi Fine-Tuning & Upload Guide

## 🎯 Complete Workflow

We'll create your **custom Rudushi model** by fine-tuning TinyLlama-1.1B on the Alpaca dataset, then upload it to Hugging Face!

---

## 📋 Step-by-Step Process

### Step 1: Set Up Google Colab (10 minutes)

1. **Open Google Colab**
   - Go to: https://colab.research.google.com
   - Sign in with your Google account

2. **Enable GPU**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU (T4)

3. **Upload the Notebook**
   - File → Upload notebook
   - Upload: `Rudushi_Fine_Tuning.ipynb`

### Step 2: Run Fine-Tuning (30-90 minutes)

**In the Colab notebook, run each cell in order:**

✅ **Cell 1:** Install Dependencies (2 minutes)
✅ **Cell 2:** Import Libraries (1 minute)
✅ **Cell 3:** Load TinyLlama-1.1B (2 minutes)
✅ **Cell 4:** Add LoRA Adapters (1 minute)
✅ **Cell 5:** Load Alpaca Dataset (3 minutes)
✅ **Cell 6:** Configure Training (1 minute)
✅ **Cell 7:** **START FINE-TUNING** (30-90 minutes) ⏱️
✅ **Cell 8:** Save Model (1 minute)
✅ **Cell 9:** Test Model (30 seconds)
✅ **Cell 10:** Download Model (1 minute)

### Step 3: Transfer Model to Your Device (5 minutes)

1. **Download from Colab**
   - Click the download button in Cell 10
   - Get: `rudushi_model.zip`

2. **Transfer to Android**
   - Use Google Drive, USB cable, or any method
   - Save to your device

3. **Extract on Your Device**
   ```bash
   cd /data/data/com.termux/files/home/rudushi
   unzip rudushi_model.zip
   ```

### Step 4: Convert to GGUF (10 minutes)

```bash
cd /data/data/com.termux/files/home/rudushi
python3 02_convert_to_gguf.py
```

**Output:**
- `gguf_models/model-Q4_K_M.gguf` (~550MB)

### Step 5: Upload to Hugging Face (5 minutes)

```bash
cd /data/data/com.termux/files/home/rudushi
python3 03_upload_to_hf.py --token YOUR_HF_TOKEN
```

**Requirements:**
- Hugging Face account: https://huggingface.co/join
- Write-enabled token: https://huggingface.co/settings/tokens
- Repository: `megharudushi/Rudushi`

### Step 6: Test Your Rudushi Model (1 minute)

**Option A: Use Rudushi Menu**
```bash
cd /data/data/com.termux/files/home/rudushi
./rudushi
# Select option 1: Chat with Rudushi
```

**Option B: Direct Chat**
```bash
python3 rudushi_chat.py --model gguf_models/model-Q4_K_M.gguf
```

---

## 🎛️ Fine-Tuning Configuration

**Current Settings** (in the notebook):
- **Base Model**: TinyLlama-1.1B
- **Method**: LoRA (Rank 16, Alpha 16)
- **Dataset**: Alpaca (52K instructions)
- **Steps**: 100 (30 min) / 500 (90 min) / 1000 (3+ hours)
- **Batch Size**: 2 (effective: 8 with accumulation)
- **Learning Rate**: 2e-4
- **Hardware**: NVIDIA T4 (free in Colab)

**Adjustable Parameters:**
- Increase `max_steps` for better quality
- Adjust `learning_rate` (1e-4 to 5e-4)
- Modify `per_device_train_batch_size` (1-4)

---

## 📊 Expected Results

**After Fine-Tuning:**
- Model learns Rudushi's personality
- Understands instruction-following
- Optimized for mobile chat
- Ready for deployment

**Upload Location:**
- **Hugging Face**: https://huggingface.co/megharudushi/Rudushi
- **Local**: `/data/data/com.termux/files/home/rudushi/fine_tuned_rudushi/`
- **GGUF**: `/data/data/com.termux/files/home/rudushi/gguf_models/model-Q4_K_M.gguf`

---

## 🚀 Quick Commands

**Run Fine-Tuning Studio:**
```bash
cd /data/data/com.termux/files/home/rudushi
./finetune_rudushi.sh
# Select option 2: Google Colab
```

**Convert Model:**
```bash
cd /data/data/com.termux/files/home/rudushi
python3 02_convert_to_gguf.py
```

**Upload Model:**
```bash
cd /data/data/com.termux/files/home/rudushi
python3 03_upload_to_hf.py --token YOUR_TOKEN
```

**Test Model:**
```bash
cd /data/data/com.termux/files/home/rudushi
./rudushi
# Select option 1
```

---

## 📝 Model Card (Auto-Generated)

The upload script will create a model card with:

```markdown
# Rudushi TinyLlama

**Description**: A fine-tuned version of TinyLlama-1.1B for mobile AI applications

**Model Details**:
- Architecture: TinyLlama-1.1B (LoRA fine-tuned)
- Parameters: 1.1B
- Context Length: 2048 tokens
- Quantization: Q4_K_M (4-bit)
- Dataset: Alpaca (52K instructions)
- Steps: 100
- License: Apache 2.0

**Usage**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("megharudushi/Rudushi")
tokenizer = AutoTokenizer.from_pretrained("megharudushi/Rudushi")
```

**Performance**:
- Speed: 5-10 tokens/sec (mobile CPU)
- Memory: 500MB RAM
- Model Size: 550MB (GGUF)
```

---

## 🔐 Hugging Face Token Setup

### Get Your Free Token:

1. **Create Account** (if needed)
   - Go to: https://huggingface.co/join
   - Sign up with email

2. **Generate Token**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: "Rudushi Fine-tuning"
   - Role: Write
   - Copy the token (starts with `hf_`)

3. **Use Token**
   ```bash
   python3 03_upload_to_hf.py --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

---

## 🎓 Advanced Options

### More Training Steps

**For better quality (takes longer):**

In the notebook, change:
```python
max_steps=500,  # 90 minutes
# or
max_steps=1000,  # 3+ hours
```

### Custom Dataset

**To fine-tune on your own data:**

1. Prepare JSON file with:
   ```json
   [
     {
       "instruction": "Your instruction here",
       "input": "Optional input",
       "output": "Desired response"
     }
   ]
   ```

2. Upload to Colab and replace dataset loading:
   ```python
   dataset = load_dataset("json", data_files="your_data.json", split="train")
   ```

### Multi-GPU Training

For faster training:
- Use Pro account ($$)
- Or AWS/GCP instances
- Add `ddp_find_unused_parameters=False` to TrainingArguments

---

## 🐛 Troubleshooting

### Error: Out of Memory
**Solution:**
```python
# Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=8

# Or use gradient checkpointing (already enabled)
```

### Error: Colab Disconnect
**Solution:**
- Save checkpoints frequently
- Use Google Drive to persist files
- Lower max_steps for testing

### Error: Upload Failed
**Solution:**
```bash
# Check token
echo $HF_TOKEN
# Should start with hf_

# Check repository exists
# Go to: https://huggingface.co/megharudushi/Rudushi
# If empty, that's expected until first upload
```

### Slow Training
**Solution:**
- Check GPU is enabled (should show T4/V100)
- Reduce max_steps for testing
- Close other Colab sessions

---

## 📈 Performance Optimization

### For Speed (Testing)
```python
max_steps=20,  # 10 minutes
per_device_train_batch_size=4,
gradient_accumulation_steps=2,
```

### For Quality (Production)
```python
max_steps=1000,  # 3+ hours
per_device_train_batch_size=2,
gradient_accumulation_steps=4,
learning_rate=1e-4,
```

---

## 🎯 Success Checklist

- [ ] Colab GPU enabled
- [ ] Notebook uploaded and running
- [ ] Fine-tuning completed (loss decreasing)
- [ ] Model saved and downloaded
- [ ] Model transferred to device
- [ ] GGUF conversion successful
- [ ] Hugging Face token generated
- [ ] Model uploaded to HF
- [ ] Model tested in Termux
- [ ] Rudushi is chatting! 🤖

---

## 🎉 What You'll Have

After completion:

**✅ Your Custom Rudushi Model:**
- Fine-tuned on Alpaca (52K instructions)
- Uploaded to Hugging Face
- Running on your Android device
- Ready for chat!

**✅ Available Everywhere:**
- Android Termux (direct inference)
- Hugging Face Spaces (web UI)
- API access (transformers)

**✅ Performance:**
- 1.1B parameters
- 2048 context window
- 5-10 tokens/sec (mobile CPU)
- 500MB RAM usage

---

## 💡 Pro Tips

1. **Start Small**: Begin with 20 steps to test
2. **Monitor Loss**: Should decrease from ~2.0 to ~0.5
3. **Save Checkpoints**: Model saves every 50 steps
4. **Test Frequently**: Use test cell to check quality
5. **Use Drive**: Mount Google Drive to persist files

---

## 📞 Need Help?

**Documentation:**
- `PROJECT_SUMMARY.md` - Technical details
- `RUDUSHI_README.md` - Complete guide
- `START_HERE.txt` - Quick start

**Resources:**
- Unsloth docs: https://github.com/unslothai/unsloth
- Hugging Face: https://huggingface.co/docs
- Alpaca dataset: https://github.com/tatsu-lab/stanford_alpaca

---

**Ready to create your custom Rudushi? Start the Colab notebook!** 🚀

```bash
# Quick access
./finetune_rudushi.sh
# Then select: 2) Google Colab
```
