#!/bin/bash
#
# Fine-tune Rudushi - Launch fine-tuning process
# This script helps set up and launch fine-tuning for Rudushi
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

clear

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🔬 RUDUSHI FINE-TUNING STUDIO 🔬                                    ║
║                                                                      ║
║   Create your custom Rudushi model with your own data!              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF

echo -e "${CYAN}${BOLD}"
echo "  Fine-tuning Rudushi TinyLlama"
echo -e "${NC}\n"

# Check GPU availability
check_gpu() {
    echo -e "${BLUE}Checking GPU availability...${NC}"

    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ GPU detected:${NC}"
            echo "$gpu_info" | while read gpu_mem; do
                echo "  • $gpu_mem"
            done

            # Check VRAM
            vram=$(echo "$gpu_info" | grep -o '[0-9]*' | head -1)
            if [ "$vram" -ge 8000 ]; then
                echo -e "${GREEN}✅ Sufficient VRAM (${vram}MB) for fine-tuning${NC}"
                return 0
            else
                echo -e "${YELLOW}⚠️  Warning: Only ${vram}MB VRAM (recommended: 8000MB+)${NC}"
                return 1
            fi
        fi
    fi

    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected${NC}"
    echo ""
    echo -e "${CYAN}Fine-tuning requires a GPU with 8GB+ VRAM${NC}"
    echo -e "${CYAN}Recommended: Google Colab (free T4 GPU)${NC}"
    echo ""

    return 1
}

# Show options
show_menu() {
    echo -e "${CYAN}${BOLD}Select an option:${NC}"
    echo ""
    echo "  1) Check GPU availability"
    echo "  2) Run fine-tuning in Google Colab (recommended)"
    echo "  3) Run fine-tuning locally (if GPU available)"
    echo "  4) View fine-tuning configuration"
    echo "  5) Prepare training environment"
    echo "  6) Convert fine-tuned model to GGUF"
    echo "  7) Upload model to Hugging Face"
    echo "  8) Test fine-tuned model"
    echo "  0) Exit"
    echo ""
}

# Google Colab setup
colab_setup() {
    echo -e "${GREEN}Google Colab Setup${NC}"
    echo ""
    echo -e "${CYAN}Steps to fine-tune Rudushi in Google Colab:${NC}"
    echo ""
    echo "1. ${BOLD}Open Google Colab${NC}"
    echo "   → Go to: https://colab.research.google.com"
    echo "   → Sign in with your Google account"
    echo ""
    echo "2. ${BOLD}Enable GPU${NC}"
    echo "   → Runtime → Change runtime type"
    echo "   → Hardware accelerator: GPU (T4 recommended)"
    echo ""
    echo "3. ${BOLD}Upload the fine-tuning script${NC}"
    echo "   → Upload: 01_fine_tune_tinyllama.py"
    echo "   → Run all cells in the notebook"
    echo ""
    echo "4. ${BOLD}Monitor training${NC}"
    echo "   → Training takes 1-4 hours depending on steps"
    echo "   → Check progress in output logs"
    echo ""
    echo "5. ${BOLD}Download your model${NC}"
    echo "   → Model saved to: fine_tuned_tinyllama/"
    echo "   → Download the folder to your device"
    echo ""
    echo -e "${GREEN}Ready to start? Press Enter to open Colab${NC}"
    read -r

    # Open in browser (if possible)
    if command -v xdg-open &> /dev/null; then
        xdg-open "https://colab.research.google.com" 2>/dev/null
    elif command -v open &> /dev/null; then
        open "https://colab.research.google.com" 2>/dev/null
    fi

    echo -e "${CYAN}Colab opened in your browser${NC}"
    echo -e "${CYAN}Follow the steps above to start fine-tuning!${NC}"
}

# Local fine-tuning
local_finetune() {
    echo -e "${GREEN}Local Fine-tuning${NC}"
    echo ""

    if ! check_gpu; then
        echo ""
        echo -e "${YELLOW}No suitable GPU found${NC}"
        echo "Please use Google Colab or ensure you have an NVIDIA GPU with 8GB+ VRAM"
        echo ""
        read -p "Press Enter to continue..."
        return
    fi

    echo ""
    echo -e "${CYAN}Fine-tuning configuration:${NC}"
    echo "  • Model: TinyLlama-1.1B"
    echo "  • Method: LoRA (Rank 16)"
    echo "  • Dataset: Alpaca (52K instructions)"
    echo "  • Steps: 100 (adjustable)"
    echo "  • Batch size: 2 (effective: 8)"
    echo ""

    read -p "Continue with fine-tuning? [y/N]: " confirm

    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo ""
        echo -e "${GREEN}Starting fine-tuning...${NC}"
        echo -e "${YELLOW}This will take 1-4 hours depending on your GPU${NC}"
        echo ""

        # Run fine-tuning
        python3 01_fine_tune_tinyllama.py

        echo ""
        echo -e "${GREEN}Fine-tuning completed!${NC}"
        echo "Your model is saved in: fine_tuned_tinyllama/"
        echo ""
        read -p "Press Enter to continue..."
    fi
}

# View configuration
view_config() {
    echo -e "${GREEN}Fine-tuning Configuration${NC}"
    echo ""

    if [ -f "rudushi_config.yaml" ]; then
        cat rudushi_config.yaml
    else
        echo -e "${YELLOW}Configuration file not found${NC}"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Prepare environment
prepare_env() {
    echo -e "${GREEN}Preparing Training Environment${NC}"
    echo ""

    echo -e "${CYAN}Installing dependencies...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 not found${NC}"
        return
    fi

    # Install required packages
    pip install --quiet \
        unsloth \
        xformers \
        trl \
        peft \
        accelerate \
        bitsandbytes \
        datasets \
        transformers \
        huggingface_hub \
        safetensors

    echo -e "${GREEN}✅ Environment prepared${NC}"
    echo ""
    read -p "Press Enter to continue..."
}

# Convert model
convert_model() {
    echo -e "${GREEN}Converting Model to GGUF${NC}"
    echo ""

    if [ -d "fine_tuned_tinyllama" ]; then
        echo -e "${CYAN}Converting fine-tuned model...${NC}"
        python3 02_convert_to_gguf.py

        if [ -f "gguf_models/model-Q4_K_M.gguf" ]; then
            echo ""
            echo -e "${GREEN}✅ Conversion successful!${NC}"
            echo "Model saved to: gguf_models/model-Q4_K_M.gguf"
        else
            echo -e "${RED}❌ Conversion failed${NC}"
        fi
    else
        echo -e "${YELLOW}No fine-tuned model found${NC}"
        echo "Run fine-tuning first (option 2 or 3)"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Upload model
upload_model() {
    echo -e "${GREEN}Uploading Model to Hugging Face${NC}"
    echo ""

    if [ -d "fine_tuned_tinyllama" ]; then
        read -p "Enter your Hugging Face token: " token

        if [ -n "$token" ]; then
            echo ""
            echo -e "${CYAN}Uploading to: megharudushi/Rudushi${NC}"
            python3 03_upload_to_hf.py --token "$token"
        else
            echo -e "${YELLOW}No token provided${NC}"
        fi
    else
        echo -e "${YELLOW}No fine-tuned model found${NC}"
        echo "Run fine-tuning first"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Test model
test_model() {
    echo -e "${GREEN}Testing Fine-tuned Model${NC}"
    echo ""

    # Check for GGUF model
    if [ -f "gguf_models/model-Q4_K_M.gguf" ]; then
        echo -e "${CYAN}Testing with sample prompt...${NC}"
        echo ""

        /data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli \
            -m gguf_models/model-Q4_K_M.gguf \
            -p "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n" \
            -n 100 \
            --temp 0.7 \
            --mlock
    elif [ -f "rudushi_model/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf" ]; then
        echo -e "${CYAN}Testing with default model...${NC}"
        echo ""

        /data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli \
            -m rudushi_model/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
            -p "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n" \
            -n 100 \
            --temp 0.7 \
            --mlock
    else
        echo -e "${YELLOW}No model found for testing${NC}"
        echo "Download or fine-tune a model first"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice [0-8]: " choice
    echo ""

    case $choice in
        1) check_gpu; echo ""; read -p "Press Enter to continue..." ;;
        2) colab_setup ;;
        3) local_finetune ;;
        4) view_config ;;
        5) prepare_env ;;
        6) convert_model ;;
        7) upload_model ;;
        8) test_model ;;
        0)
            echo -e "${GREEN}👋 Happy fine-tuning!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
    echo ""
done
