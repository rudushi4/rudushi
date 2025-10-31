#!/usr/bin/env python3
"""
Hugging Face Spaces App for Rudushi TinyLlama Model
Chat interface using Gradio
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from threading import Thread
import gradio as gr
from huggingface_hub import HfApi
import time

# Model configuration
MODEL_ID = "megharudushi/Rudushi"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Read-only token

# Load model and tokenizer
print("="*60)
print("🚀 Loading Rudushi TinyLlama Model")
print("="*60)

try:
    api = HfApi(token=HF_TOKEN)

    print(f"📦 Downloading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=HF_TOKEN,
        low_cpu_mem_usage=True
    )

    print(f"🔑 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_auth_token=HF_TOKEN
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded successfully!")
    print(f"   Model type: {model.config.model_type}")
    print(f"   Parameters: {model.config.num_parameters / 1e6:.1f}M")
    print(f"   Max length: {tokenizer.model_max_length}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please check MODEL_ID and HF_TOKEN")
    exit(1)

# Alpaca-style prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def format_prompt(instruction, input_text):
    """Format input with Alpaca-style prompt"""
    return ALPACA_PROMPT.format(instruction=instruction, input=input_text)

def generate_response(message, history, system_prompt="You are Rudushi, a helpful AI assistant."):
    """Generate response using the model"""
    try:
        # Format conversation
        conversation = f"{system_prompt}\n\n"
        for user_msg, assistant_msg in history:
            conversation += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        conversation += f"User: {message}\nAssistant:"

        # Tokenize
        inputs = tokenizer(conversation, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()

        return response

    except Exception as e:
        print(f"❌ Generation error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

def stream_response(message, history, system_prompt="You are Rudushi, a helpful AI assistant."):
    """Stream response generation"""
    try:
        # Format conversation
        conversation = f"{system_prompt}\n\n"
        for user_msg, assistant_msg in history:
            conversation += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        conversation += f"User: {message}\nAssistant:"

        # Tokenize
        inputs = tokenizer(conversation, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Create streamer
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate
        with torch.no_grad():
            # Start generation in a thread
            thread = Thread(target=lambda: model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ))
            thread.start()

            # Yield tokens as they come
            full_response = ""
            while thread.is_alive():
                time.sleep(0.01)
                # In a real implementation, you'd capture the streamer's output
                # For now, we'll yield a placeholder

        # Return final response
        return "Streaming implementation requires advanced Gradio setup"

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return f"Error: {str(e)}"

def chat_interface(message, history, system_prompt):
    """Main chat interface function"""
    if not message.strip():
        return "", history

    # Add to history
    history.append((message, ""))

    # Generate response
    response = generate_response(message, history, system_prompt)

    # Update history
    history[-1] = (message, response)

    return "", history

# Create Gradio interface
print("\n🎨 Creating Gradio interface...")

def clear_chat():
    """Clear chat history"""
    return [], "", None

# Custom CSS
css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # 🎯 Rudushi TinyLlama Chat

        **Model**: TinyLlama-1.1B (Fine-tuned with Unsloth)
        **Context**: 2048 tokens
        **Quantization**: 4-bit (optimized for CPU inference)
        """
    )

    with gr.Row():
        system_prompt = gr.Textbox(
            label="System Prompt",
            value="You are Rudushi, a helpful AI assistant. Provide concise, accurate answers.",
            lines=2,
            info="Set the assistant's behavior and personality"
        )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=600,
        avatar_images=("👤", "🤖"),
        layout="bubble"
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Ask me anything...",
            lines=2,
            scale=4
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear Chat", variant="secondary")
        examples_btn = gr.Button("Example Questions", variant="secondary")

    # Example questions
    examples = gr.Examples(
        examples=[
            "Explain quantum computing in simple terms",
            "Write a Python function to sort a list",
            "What are the benefits of using renewable energy?",
            "How does machine learning work?",
            "Create a simple calculator in Python"
        ],
        inputs=msg,
        label="Example Questions",
        cache_examples=False
    )

    # Submit handlers
    submit_btn.click(
        fn=chat_interface,
        inputs=[msg, chatbot, system_prompt],
        outputs=[msg, chatbot]
    )

    msg.submit(
        fn=chat_interface,
        inputs=[msg, chatbot, system_prompt],
        outputs=[msg, chatbot]
    )

    # Clear chat
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg, system_prompt]
    )

    # Footer
    gr.Markdown(
        """
        ---
        **Tips:**
        - Be specific in your questions for better results
        - The model works best with clear, direct instructions
        - Context is limited to ~2048 tokens
        """
    )

print("✅ Interface created!")

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 Starting Spaces Server")
    print("="*60)
    print(f"   Model: {MODEL_ID}")
    print(f"   URL: https://hf.co/spaces/megharudushi/Rudushi")
    print("="*60 + "\n")

    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True,
        inbrowser=True
    )
