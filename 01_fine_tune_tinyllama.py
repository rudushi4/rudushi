#!/usr/bin/env python3
"""
Fine-tuning script for TinyLlama-1.1B using Unsloth
Optimized for Google Colab T4 GPU
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os

# Configuration
MODEL_NAME = "unsloth/tinyllama-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # Auto-detect
LOAD_IN_4BIT = True
OUTPUT_DIR = "fine_tuned_tinyllama"

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": True,
    "random_state": 3407
}

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 60,  # Increase for full training (e.g., 1000)
    "learning_rate": 2e-4,
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": OUTPUT_DIR,
    "evaluation_strategy": "steps",
    "eval_steps": 10,
    "logging_steps": 10,
    "save_steps": 50,
    "report_to": "none"
}

# Alpaca prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def setup_environment():
    """Install required packages"""
    print("🔧 Setting up environment...")
    os.system("pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
    os.system("pip install -q --no-deps \"xformers<0.0.27\" \"trl<0.9.0\" peft accelerate bitsandbytes")
    print("✅ Environment setup complete")

def load_model_and_tokenizer():
    """Load TinyLlama model and tokenizer"""
    print(f"\n📦 Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"✅ Model loaded: {model.config.num_parameters / 1e6:.1f}M parameters")
    return model, tokenizer

def prepare_model(model):
    """Add LoRA adapters to the model"""
    print("\n🔗 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    print("✅ LoRA adapters added")
    return model

def format_prompts(examples):
    """Format examples for Alpaca-style training"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input_, output in zip(instructions, inputs, outputs):
        # Add EOS token
        text = ALPACA_PROMPT.format(instruction, input_, output) + tokenizer.eos_token
        texts.append(text)

    return {"text": texts}

def prepare_dataset():
    """Load and prepare training dataset"""
    print("\n📚 Loading dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"✅ Loaded {len(dataset)} examples")

    # Format prompts
    dataset = dataset.map(format_prompts, batched=True)

    # Split into train and eval
    dataset_dict = dataset.train_test_split(test_size=0.05)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    print(f"   Training set: {len(train_dataset)} examples")
    print(f"   Evaluation set: {len(eval_dataset)} examples")

    return train_dataset, eval_dataset

def setup_trainer(model, tokenizer, train_dataset, eval_dataset):
    """Setup SFTTrainer"""
    print("\n🏋️ Setting up trainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        args=TrainingArguments(**TRAINING_CONFIG),
    )

    print("✅ Trainer configured")
    return trainer

def train_model(trainer):
    """Start training"""
    print("\n🚀 Starting training...")
    print(f"   Steps: {TRAINING_CONFIG['max_steps']}")
    print(f"   Batch size: {TRAINING_CONFIG['per_device_train_batch_size']} (effective: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']})")

    trainer.train()
    print("✅ Training complete!")

def save_model(model, tokenizer):
    """Save fine-tuned model"""
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Model saved!")

def test_model(model, tokenizer):
    """Test the fine-tuned model"""
    print("\n🧪 Testing model...")

    FastLanguageModel.for_inference(model)

    test_prompt = ALPACA_PROMPT.format(
        "Write a Python function to reverse a string.",
        "",
        ""
    )

    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.batch_decode(outputs)[0]
    print("\n📝 Test output:")
    print(response)
    print("✅ Test complete!")

def main():
    """Main execution flow"""
    print("="*60)
    print("🎯 TinyLlama Fine-tuning Pipeline")
    print("="*60)

    # Setup
    setup_environment()

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Prepare model
    model = prepare_model(model)

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset()

    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset)

    # Train
    train_model(trainer)

    # Save
    save_model(model, tokenizer)

    # Test
    test_model(model, tokenizer)

    print("\n" + "="*60)
    print("✨ Pipeline complete!")
    print(f"📁 Fine-tuned model saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
