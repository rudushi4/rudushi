#!/bin/bash
#
# Termux Deployment Script for Rudushi TinyLlama
# This script sets up and runs the model in Termux
#
# Usage:
#   chmod +x 04_termux_deploy.sh
#   ./04_termux_deploy.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_URL="https://huggingface.co/megharudushi/Rudushi/resolve/main/model-Q4_K_M.gguf"
MODEL_FILE="model-Q4_K_M.gguf"
MODEL_DIR="$HOME/rudushi_model"
LLAMA_DIR="$HOME/llama.cpp"
MIN_MEMORY_GB=2

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check system requirements
check_requirements() {
    print_info "Checking system requirements..."

    # Check if running in Termux
    if [ ! -d "$PREFIX" ]; then
        print_warning "Not running in Termux. This script is optimized for Termux."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check available memory
    if command -v free >/dev/null 2>&1; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt "$MIN_MEMORY_GB" ]; then
            print_warning "Low memory: ${MEMORY_GB}GB available (recommended: ${MIN_MEMORY_GB}GB+)"
        else
            print_success "Memory check passed: ${MEMORY_GB}GB available"
        fi
    fi

    # Check storage
    AVAILABLE_SPACE=$(df -BG "$HOME" | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 2 ]; then
        print_error "Insufficient storage: ${AVAILABLE_SPACE}GB available (need: 2GB+)"
        exit 1
    fi
    print_success "Storage check passed: ${AVAILABLE_SPACE}GB available"
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing dependencies in Termux..."

    # Update package list
    pkg update -y

    # Install required packages
    pkg install -y \
        git \
        cmake \
        clang \
        make \
        wget \
        python \
        python-pip \
        openblas \
        openmp

    # Install Python packages
    pip install --upgrade pip
    pip install transformers huggingface_hub accelerate

    print_success "Dependencies installed"
}

# Function to download model
download_model() {
    print_info "Downloading model..."

    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    if [ -f "$MODEL_FILE" ]; then
        print_success "Model already exists"
        return
    fi

    print_info "Downloading from Hugging Face..."
    wget -O "$MODEL_FILE" "$MODEL_URL"

    if [ $? -eq 0 ]; then
        print_success "Model downloaded successfully"
        ls -lh "$MODEL_FILE"
    else
        print_error "Download failed"
        exit 1
    fi
}

# Function to build llama.cpp
build_llama_cpp() {
    print_info "Building llama.cpp..."

    cd "$HOME"
    rm -rf "$LLAMA_DIR"
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"

    cd "$LLAMA_DIR"
    make LLAMA_BUILD_INFO=OFF LLAMA_BUILD_NUMBER=

    if [ -f "main" ]; then
        print_success "llama.cpp built successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Function to test model
test_model() {
    print_info "Testing model..."

    cd "$LLAMA_DIR"

    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

    print_info "Running quick test..."
    echo "This test should complete in a few seconds..."

    ./main \
        -m "$MODEL_PATH" \
        -p "### Instruction:\nWrite hello\n\n### Response:\n" \
        -n 20 \
        --temp 0.7 \
        --mlock 2>&1 | head -20

    print_success "Test completed"
}

# Function to run interactive chat
run_chat() {
    print_info "Starting interactive chat..."
    echo ""
    echo "Type your messages. Type 'exit' or 'quit' to stop."
    echo "----------------------------------------"

    cd "$LLAMA_DIR"

    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

    ./main \
        -m "$MODEL_PATH" \
        -c 2048 \
        -b 512 \
        --temp 0.7 \
        --top_p 0.9 \
        --top_k 40 \
        --repeat_penalty 1.1 \
        --mlock \
        --interactive \
        --prompt-cache "$HOME/.rudushi_cache" \
        -f prompts/alpaca.txt
}

# Function to create interactive prompt template
create_prompt_template() {
    print_info "Creating prompt template..."

    mkdir -p "$LLAMA_DIR/prompts"

    cat > "$LLAMA_DIR/prompts/alpaca.txt" << 'EOF'
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
EOF

    print_success "Prompt template created"
}

# Function to create a simple Python wrapper
create_python_wrapper() {
    print_info "Creating Python wrapper..."

    cat > "$HOME/run_rudushi.py" << 'EOFPY'
#!/usr/bin/env python3
"""
Simple Python wrapper for Rudushi TinyLlama in Termux
"""

import subprocess
import sys
import argparse

def run_inference(prompt, max_tokens=512, temperature=0.7):
    """Run inference using llama.cpp"""
    cmd = [
        "main",
        "-m", "model-Q4_K_M.gguf",
        "-c", "2048",
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top_p", "0.9",
        "--mlock",
        "-p", prompt
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def interactive_mode():
    """Run in interactive mode"""
    print("Rudushi TinyLlama - Interactive Mode")
    print("Type 'exit' to quit\n")

    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ['exit', 'quit', 'q']:
                break

            full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

            print("Rudushi: ", end="", flush=True)
            output = run_inference(full_prompt, max_tokens=256)
            # Extract only the response part
            response = output.split("### Response:")[-1].strip()
            print(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rudushi TinyLlama Runner")
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument("--tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.prompt:
        output = run_inference(args.prompt, args.tokens, args.temperature)
        print(output)
    else:
        print("Use --help for usage information")
EOFPY

    chmod +x "$HOME/run_rudushi.py"
    print_success "Python wrapper created: run_rudushi.py"
}

# Function to create benchmarks
run_benchmark() {
    print_info "Running benchmark..."

    cd "$LLAMA_DIR"

    MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

    print_info "Generating 100 tokens..."
    time ./main \
        -m "$MODEL_PATH" \
        -p "Explain artificial intelligence:" \
        -n 100 \
        --temp 0.7 \
        --mlock 2>&1 | tee "$HOME/benchmark.log"

    print_success "Benchmark saved to $HOME/benchmark.log"
}

# Function to show usage info
show_usage() {
    echo ""
    echo "=========================================="
    echo "🎯 Rudushi TinyLlama - Usage Guide"
    echo "=========================================="
    echo ""
    echo "Quick Start:"
    echo "  1. Run: ./main -m $MODEL_FILE -p 'Hello' -n 50"
    echo "  2. Or use Python: python $HOME/run_rudushi.py --interactive"
    echo ""
    echo "Available Commands:"
    echo "  ./main -m $MODEL_FILE [OPTIONS]"
    echo ""
    echo "Common Options:"
    echo "  -p, --prompt        Input prompt"
    echo "  -n, --n-predict     Number of tokens to generate (default: 128)"
    echo "  -c, --ctx-size      Context size (default: 2048)"
    echo "  --temp              Temperature (0.0-1.0, default: 0.7)"
    echo "  --top-p             Top-p sampling (default: 0.9)"
    echo "  --top-k             Top-k sampling (default: 40)"
    echo "  --repeat-penalty    Repetition penalty (default: 1.1)"
    echo "  -b, --batch-size    Batch size (default: 512)"
    echo "  --mlock             Lock model in memory"
    echo "  --interactive       Interactive mode"
    echo ""
    echo "Examples:"
    echo "  ./main -m $MODEL_FILE -p 'Write a Python function:' -n 100 --temp 0.7"
    echo "  ./main -m $MODEL_FILE -p 'Hello' --interactive --mlock"
    echo "  ./main -m $MODEL_FILE -p 'Explain quantum computing' -c 1024"
    echo ""
    echo "Python Wrapper:"
    echo "  python $HOME/run_rudushi.py --interactive"
    echo "  python $HOME/run_rudushi.py --prompt 'Hello' --tokens 100"
    echo ""
    echo "Files:"
    echo "  Model: $MODEL_DIR/$MODEL_FILE"
    echo "  LLaMA: $LLAMA_DIR"
    echo "  Cache: $HOME/.rudushi_cache"
    echo ""
    echo "=========================================="
}

# Main menu
show_menu() {
    while true; do
        echo ""
        echo "=========================================="
        echo "🎯 Rudushi TinyLlama - Termux Setup"
        echo "=========================================="
        echo ""
        echo "1) Check requirements"
        echo "2) Install dependencies"
        echo "3) Download model"
        echo "4) Build llama.cpp"
        echo "5) Test model"
        echo "6) Run interactive chat"
        echo "7) Create prompt template"
        echo "8) Create Python wrapper"
        echo "9) Run benchmark"
        echo "10) Show usage guide"
        echo "11) Full setup (all of the above)"
        echo "0) Exit"
        echo ""
        read -p "Select option [0-11]: " choice

        case $choice in
            1) check_requirements ;;
            2) install_dependencies ;;
            3) download_model ;;
            4) build_llama_cpp ;;
            5) test_model ;;
            6) run_chat ;;
            7) create_prompt_template ;;
            8) create_python_wrapper ;;
            9) run_benchmark ;;
            10) show_usage ;;
            11)
                check_requirements
                install_dependencies
                download_model
                build_llama_cpp
                create_prompt_template
                create_python_wrapper
                test_model
                print_success "Full setup complete!"
                show_usage
                ;;
            0) exit 0 ;;
            *) print_error "Invalid option" ;;
        esac
    done
}

# Check if running interactively
if [ -t 0 ]; then
    # Interactive mode
    show_menu
else
    # Non-interactive: run full setup
    print_info "Running in non-interactive mode..."
    check_requirements
    install_dependencies
    download_model
    build_llama_cpp
    create_prompt_template
    create_python_wrapper
    test_model
    print_success "Setup complete!"
    show_usage
fi
