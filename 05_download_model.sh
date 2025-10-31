#!/bin/bash
#
# Model Download Script for Rudushi TinyLlama
# This script helps download GGUF models for testing
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🎯 Rudushi - Model Download Helper                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

MODEL_DIR="$HOME/rudushi_model"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo ""
echo -e "${YELLOW}Choose a model to download:${NC}"
echo ""
echo "1) TinyLlama-1.1B-Chat-v1.0 (Q4_K_M) ~ 550MB"
echo "2) Phi-3-Mini-4K-Instruct (Q4_K_M) ~ 2.3GB"
echo "3) Qwen2-1.5B-Instruct (Q4_K_M) ~ 940MB"
echo "4) Custom model URL"
echo "5) Exit"
echo ""

read -p "Select option [1-5]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Downloading TinyLlama-1.1B-Chat-v1.0 (Q4_K_M)...${NC}"
        echo "This model requires a Hugging Face token for download"
        echo ""
        read -p "Do you have a Hugging Face token? (y/n): " has_token

        if [ "$has_token" = "y" ] || [ "$has_token" = "Y" ]; then
            read -p "Enter your HF token: " token
            wget -O TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
                --header="Authorization: Bearer $token" \
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Download successful!${NC}"
                ls -lh TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
            else
                echo -e "${RED}❌ Download failed${NC}"
            fi
        else
            echo -e "\n${YELLOW}You can download models manually:${NC}"
            echo ""
            echo "1. Go to: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
            echo "2. Download: TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
            echo "3. Place it in: $MODEL_DIR/"
            echo ""
            echo "Or use the browser download and transfer to your device"
        fi
        ;;

    2)
        echo -e "\n${YELLOW}Downloading Phi-3-Mini-4K-Instruct (Q4_K_M)...${NC}"
        read -p "Do you have a Hugging Face token? (y/n): " has_token

        if [ "$has_token" = "y" ] || [ "$has_token" = "Y" ]; then
            read -p "Enter your HF token: " token
            wget -O Phi-3-mini-4k-instruct.Q4_K_M.gguf \
                --header="Authorization: Bearer $token" \
                "https://huggingface.co/mradermacher/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct.Q4_K_M.gguf"

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Download successful!${NC}"
                ls -lh Phi-3-mini-4k-instruct.Q4_K_M.gguf
            else
                echo -e "${RED}❌ Download failed${NC}"
            fi
        else
            echo -e "\nManual download required"
            echo "https://huggingface.co/mradermacher/Phi-3-mini-4k-instruct-gguf"
        fi
        ;;

    3)
        echo -e "\n${YELLOW}Downloading Qwen2-1.5B-Instruct (Q4_K_M)...${NC}"
        read -p "Do you have a Hugging Face token? (y/n): " has_token

        if [ "$has_token" = "y" ] || [ "$has_token" = "Y" ]; then
            read -p "Enter your HF token: " token
            wget -O Qwen2-1.5B-Instruct.Q4_K_M.gguf \
                --header="Authorization: Bearer $token" \
                "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf"

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Download successful!${NC}"
                ls -lh Qwen2-1.5B-Instruct.Q4_K_M.gguf
            else
                echo -e "${RED}❌ Download failed${NC}"
            fi
        else
            echo -e "\nManual download required"
            echo "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF"
        fi
        ;;

    4)
        echo -e "\n${YELLOW}Custom Model Download${NC}"
        read -p "Enter model repository (e.g., username/model-name): " repo
        read -p "Enter filename (exact): " filename
        read -p "Enter output filename: " output

        read -p "Do you have a Hugging Face token? (y/n): " has_token

        if [ "$has_token" = "y" ] || [ "$has_token" = "Y" ]; then
            read -p "Enter your HF token: " token
            wget -O "$output" \
                --header="Authorization: Bearer $token" \
                "https://huggingface.co/$repo/resolve/main/$filename"

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Download successful!${NC}"
                ls -lh "$output"
            else
                echo -e "${RED}❌ Download failed${NC}"
            fi
        else
            echo -e "\nManual download required"
            echo "https://huggingface.co/$repo"
        fi
        ;;

    5)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Ensure you have a model file in: $MODEL_DIR/"
echo "2. Run: ./04_termux_deploy.sh"
echo "3. Select option 6 to run the model!"
echo ""
