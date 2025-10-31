#!/bin/bash
#
# Upload Rudushi to Hugging Face - Automated Script
# This script automates the upload process
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

clear

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🚀 RUDUSHI HUGGING FACE UPLOADER 🚀                                 ║
║                                                                      ║
║   Upload your custom Rudushi model to Hugging Face Hub              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF

echo -e "${CYAN}${BOLD}"
echo "  Uploading Rudushi to Hugging Face"
echo -e "${NC}\n"

# Check if model exists
check_model() {
    echo -e "${BLUE}Checking for fine-tuned model...${NC}"

    if [ -d "fine_tuned_rudushi" ]; then
        echo -e "${GREEN}✅ Fine-tuned model found${NC}"
        return 0
    elif [ -d "fine_tuned_tinyllama" ]; then
        echo -e "${GREEN}✅ Fine-tuned model found${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  No fine-tuned model found${NC}"
        echo ""
        echo "Please run fine-tuning first:"
        echo "  1. Google Colab: ./finetune_rudushi.sh → Option 2"
        echo "  2. Or manually upload your model to: /data/data/com.termux/files/home/rudushi/"
        return 1
    fi
}

# Get Hugging Face token
get_token() {
    echo ""
    echo -e "${CYAN}Hugging Face Token Setup${NC}"
    echo ""

    # Check if token is provided
    if [ -n "$1" ]; then
        HF_TOKEN="$1"
        echo -e "${GREEN}✅ Token provided via command line${NC}"
        return 0
    fi

    # Check environment variable
    if [ -n "$HF_TOKEN" ]; then
        echo -e "${GREEN}✅ Token found in environment${NC}"
        echo "   (Set with: export HF_TOKEN=your_token)"
        return 0
    fi

    # Ask user
    echo -e "${YELLOW}No token found${NC}"
    echo ""
    echo "Get your free token:"
    echo "1. Go to: https://huggingface.co/settings/tokens"
    echo "2. Click 'New token'"
    echo "3. Name: 'Rudushi'"
    echo "4. Role: Write"
    echo "5. Copy the token"
    echo ""
    read -p "Enter your Hugging Face token: " HF_TOKEN

    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}❌ No token provided${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Token received${NC}"
    return 0
}

# Convert to GGUF if needed
convert_model() {
    echo ""
    echo -e "${BLUE}Checking GGUF model...${NC}"

    if [ -f "gguf_models/model-Q4_K_M.gguf" ]; then
        echo -e "${GREEN}✅ GGUF model already exists${NC}"
        return 0
    fi

    echo -e "${YELLOW}GGUF model not found${NC}"
    echo ""
    read -p "Convert to GGUF now? [y/N]: " convert

    if [ "$convert" = "y" ] || [ "$convert" = "Y" ]; then
        echo ""
        echo -e "${GREEN}Running conversion...${NC}"
        python3 02_convert_to_gguf.py

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Conversion successful${NC}"
            return 0
        else
            echo -e "${RED}❌ Conversion failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}Skipping conversion${NC}"
        echo "You can convert later with: python3 02_convert_to_gguf.py"
        return 0
    fi
}

# Upload to Hugging Face
upload_model() {
    echo ""
    echo -e "${BLUE}Uploading to Hugging Face...${NC}"
    echo ""
    echo "Repository: megharudushi/Rudushi"
    echo ""

    # Check which model directory to use
    if [ -d "fine_tuned_rudushi" ]; then
        MODEL_DIR="fine_tuned_rudushi"
    elif [ -d "fine_tuned_tinyllama" ]; then
        MODEL_DIR="fine_tuned_tinyllama"
    else
        echo -e "${RED}❌ No model directory found${NC}"
        return 1
    fi

    echo -e "${CYAN}Uploading from: $MODEL_DIR${NC}"
    echo ""

    # Run upload script
    python3 03_upload_to_hf.py --token "$HF_TOKEN" --commit-message "Upload custom Rudushi model"

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Upload successful!${NC}"
        echo ""
        echo "Your Rudushi model is now available at:"
        echo -e "${BOLD}https://huggingface.co/megharudushi/Rudushi${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Upload failed${NC}"
        echo ""
        echo "Common issues:"
        echo "1. Token is read-only (need write permissions)"
        echo "2. Repository doesn't exist (will be created automatically)"
        echo "3. Network error (check connection)"
        return 1
    fi
}

# Verify upload
verify_upload() {
    echo ""
    echo -e "${BLUE}Verifying upload...${NC}"

    read -p "Visit https://huggingface.co/megharudushi/Rudushi? [y/N]: " verify

    if [ "$verify" = "y" ] || [ "$verify" = "Y" ]; then
        # Try to open in browser
        if command -v xdg-open &> /dev/null; then
            xdg-open "https://huggingface.co/megharudushi/Rudushi" 2>/dev/null
        elif command -v open &> /dev/null; then
            open "https://huggingface.co/megharudushi/Rudushi" 2>/dev/null
        fi
        echo ""
        echo "Check your model page!"
    fi
}

# Test the model
test_model() {
    echo ""
    echo -e "${BLUE}Test your Rudushi model?${NC}"
    echo ""
    echo "After upload, you can test Rudushi with:"
    echo -e "${CYAN}  ./rudushi${NC} (select option 1)"
    echo ""
    echo "Or directly:"
    echo -e "${CYAN}  python3 rudushi_chat.py --model TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf${NC}"
    echo ""
    echo "Note: First test might take a few minutes while model downloads"
}

# Main menu
show_menu() {
    echo ""
    echo -e "${CYAN}${BOLD}Upload Options:${NC}"
    echo ""
    echo "  1) Check model"
    echo "  2) Get token"
    echo "  3) Convert to GGUF"
    echo "  4) Upload to Hugging Face"
    echo "  5) Verify upload"
    echo "  6) Show test commands"
    echo "  0) Exit"
    echo ""
}

# Main execution
main() {
    # Check model
    if ! check_model; then
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi

    # Get token
    if ! get_token "$@"; then
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi

    # Show menu
    while true; do
        show_menu
        read -p "Select option [0-6]: " choice
        echo ""

        case $choice in
            1) check_model ;;
            2) get_token ;;
            3) convert_model ;;
            4) upload_model ;;
            5) verify_upload ;;
            6) test_model ;;
            0)
                echo -e "${GREEN}👋 Happy Rudushi chatting!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main
main "$@"
