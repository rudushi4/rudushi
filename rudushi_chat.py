#!/usr/bin/env python3
"""
Rudushi - Interactive Chatbot
A beautiful terminal-based chat interface for language models

Usage:
    python rudushi_chat.py --model PATH_TO_MODEL.gguf
    python rudushi_chat.py --model TinyLlama --interactive
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Banner
BANNER = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════╗{Colors.ENDC}
{Colors.CYAN}║{Colors.ENDC}                                                                      {Colors.CYAN}║{Colors.ENDC}
{Colors.CYAN}║{Colors.ENDC}   {Colors.BOLD}🤖 RUDUSHI - Your AI Assistant 🤖{Colors.ENDC}{Colors.CYAN}                                 ║{Colors.ENDC}
{Colors.CYAN}║{Colors.ENDC}                                                                      {Colors.CYAN}║{Colors.ENDC}
{Colors.CYAN}║{Colors}║   {Colors.GREEN}Lightweight Language Model for Mobile Devices{Colors.ENDC}{Colors.CYAN}            ║{Colors.ENDC}
{Colors.CYAN}║{Colors.ENDC}                                                                      {Colors.CYAN}║{Colors.ENDC}
{Colors.CYAN}╚══════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""

class RudushiChat:
    def __init__(self, model_path, history_file="rudushi_history.json"):
        self.model_path = model_path
        self.history_file = Path(history_file)
        self.llama_bin = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"
        self.conversation_history = []
        self.load_history()

    def load_history(self):
        """Load conversation history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.conversation_history = json.load(f)
            except:
                self.conversation_history = []
        else:
            self.conversation_history = []

    def save_history(self):
        """Save conversation history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

    def print_banner(self):
        """Print the Rudushi banner"""
        print(BANNER)
        print(f"{Colors.GREEN}Model: {Colors.BOLD}{self.model_path}{Colors.ENDC}")
        print(f"{Colors.GREEN}History: {Colors.BOLD}{self.history_file}{Colors.ENDC}")
        print(f"{Colors.GREEN}Commands: {Colors.BOLD}/help{Colors.ENDC} - Help, {Colors.BOLD}/quit{Colors.ENDC} - Exit, {Colors.BOLD}/clear{Colors.ENDC} - Clear history")
        print(f"{Colors.GREEN}Type: {Colors.BOLD}/exit{Colors.ENDC} or {Colors.BOLD}/quit{Colors.ENDC} to stop, {Colors.BOLD}/clear{Colors.ENDC} to clear chat")
        print("")

    def format_history(self):
        """Format conversation history for the model"""
        if not self.conversation_history:
            return ""

        history = "Conversation history:\n\n"
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            if msg['role'] == 'user':
                history += f"User: {msg['content']}\n"
            else:
                history += f"Rudushi: {msg['content']}\n"
        history += "\nCurrent conversation:\n"
        return history

    def generate_response(self, prompt, max_tokens=256, temperature=0.7):
        """Generate response using llama.cpp"""
        # Prepare the full prompt
        history_context = self.format_history()

        # Create Alpaca-style prompt
        full_prompt = f"""{history_context}User: {prompt}

Rudushi:"""

        try:
            # Run llama.cpp
            cmd = [
                self.llama_bin,
                "-m", self.model_path,
                "-p", full_prompt,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--top_p", "0.9",
                "--top_k", "40",
                "--repeat_penalty", "1.1",
                "--mlock",
                "-c", "2048"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return f"{Colors.FAIL}Error: {result.stderr}{Colors.ENDC}"

            # Extract response
            response = result.stdout.strip()

            # Clean up the response (remove the prompt from output)
            if "Rudushi:" in response:
                response = response.split("Rudushi:")[-1].strip()

            return response

        except subprocess.TimeoutExpired:
            return f"{Colors.FAIL}Error: Generation timed out{Colors.ENDC}"
        except Exception as e:
            return f"{Colors.FAIL}Error: {str(e)}{Colors.ENDC}"

    def print_message(self, role, content):
        """Print a formatted message"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if role == "user":
            print(f"{Colors.CYAN}[{timestamp}] You:{Colors.ENDC} {content}")
        else:
            print(f"{Colors.GREEN}[{timestamp}] Rudushi:{Colors.ENDC} ", end="")
            print(f"{Colors.ENDC}{content}")
        print("")

    def chat_loop(self):
        """Main chat loop"""
        self.print_banner()

        try:
            while True:
                try:
                    # Get user input
                    user_input = input(f"{Colors.BOLD}{Colors.BLUE}You>{Colors.ENDC} ").strip()

                    # Handle commands
                    if user_input.lower() in ['/quit', '/exit', '/q']:
                        print(f"\n{Colors.GREEN}👋 Thanks for chatting with Rudushi!{Colors.ENDC}\n")
                        self.save_history()
                        break
                    elif user_input.lower() == '/clear':
                        self.conversation_history = []
                        self.save_history()
                        print(f"{Colors.WARNING}🧹 Conversation history cleared{Colors.ENDC}\n")
                        continue
                    elif user_input.lower() == '/help':
                        self.show_help()
                        continue
                    elif user_input.lower() == '/history':
                        self.show_history()
                        continue
                    elif user_input == '':
                        continue

                    # Save user message
                    self.conversation_history.append({
                        'role': 'user',
                        'content': user_input,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Print user message
                    self.print_message("user", user_input)

                    # Generate response
                    print(f"{Colors.CYAN}🤔 Rudushi is thinking...{Colors.ENDC}")
                    response = self.generate_response(user_input)

                    # Remove thinking indicator
                    print("\033[1A\033[K", end="")

                    # Save and print response
                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now().isoformat()
                    })

                    self.print_message("assistant", response)

                    # Save periodically
                    if len(self.conversation_history) % 5 == 0:
                        self.save_history()

                except KeyboardInterrupt:
                    print(f"\n\n{Colors.GREEN}👋 Goodbye!{Colors.ENDC}\n")
                    self.save_history()
                    break
                except Exception as e:
                    print(f"\n{Colors.FAIL}❌ Error: {str(e)}{Colors.ENDC}\n")

        except EOFError:
            print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.ENDC}\n")
            self.save_history()

    def show_help(self):
        """Show help message"""
        help_text = f"""
{Colors.HEADER}{Colors.BOLD}╔══════════════════════════════════════════════════════╗{Colors.ENDC}
{Colors.HEADER}{Colors.BOLD}║                    HELP MENU                          ║{Colors.ENDC}
{Colors.HEADER}{Colors.BOLD}╚══════════════════════════════════════════════════════╝{Colors.ENDC}

{Colors.CYAN}Commands:{Colors.ENDC}
  {Colors.BOLD}/help{Colors.ENDC}     - Show this help menu
  {Colors.BOLD}/history{Colors.ENDC} - Show conversation history
  {Colors.BOLD}/clear{Colors.ENDC}   - Clear conversation history
  {Colors.BOLD}/quit{Colors.ENDC}    - Exit the chat
  {Colors.BOLD}/exit{Colors.ENDC}    - Exit the chat

{Colors.CYAN}Tips:{Colors.ENDC}
  • Be specific in your questions for better answers
  • The model has a 2048 token context window
  • History is automatically saved to: {self.history_file}
  • Use Ctrl+C to exit anytime

{Colors.CYAN}Examples:{Colors.ENDC}
  • "Explain quantum computing in simple terms"
  • "Write a Python function to sort a list"
  • "What are the benefits of renewable energy?"
  • "How does machine learning work?"

"""
        print(help_text)

    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print(f"{Colors.WARNING}No conversation history{Colors.ENDC}\n")
            return

        print(f"\n{Colors.HEADER}{Colors.BOLD}╔══════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}║                CONVERSATION HISTORY                 ║{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}╚══════════════════════════════════════════════════════╝{Colors.ENDC}\n")

        for i, msg in enumerate(self.conversation_history[-20:], 1):  # Last 20 messages
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%H:%M:%S")
            if msg['role'] == 'user':
                print(f"{Colors.CYAN}[{timestamp}] You: {Colors.ENDC}{msg['content']}")
            else:
                print(f"{Colors.GREEN}[{timestamp}] Rudushi: {Colors.ENDC}{msg['content']}")
            print()

def find_model(model_name):
    """Find model by name or path"""
    # If it's a path, return as is
    if os.path.exists(model_name):
        return model_name

    # Search in common locations
    search_paths = [
        "/data/data/com.termux/files/home/rudushi_model",
        "/data/data/com.termux/files/home",
        "."
    ]

    # Try different extensions
    extensions = ["", ".gguf"]

    for path in search_paths:
        for ext in extensions:
            full_path = os.path.join(path, model_name + ext)
            if os.path.exists(full_path):
                return full_path

    return None

def main():
    parser = argparse.ArgumentParser(
        description="Rudushi - Interactive AI Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rudushi_chat.py --model TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf
  python rudushi_chat.py --model /data/data/com.termux/files/home/rudushi_model/model.gguf
  python rudushi_chat.py --model TinyLlama
        """
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to model or model name (without .gguf extension)"
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature 0.0-1.0 (default: 0.7)"
    )

    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't save conversation history"
    )

    args = parser.parse_args()

    # Find model
    model_path = find_model(args.model)

    if not model_path:
        print(f"{Colors.FAIL}❌ Model not found: {args.model}{Colors.ENDC}")
        print(f"\n{Colors.CYAN}Searched in:{Colors.ENDC}")
        print(f"  • /data/data/com.termux/files/home/rudushi_model/")
        print(f"  • /data/data/com.termux/files/home/")
        print(f"  • Current directory")
        print(f"\n{Colors.CYAN}Try:{Colors.ENDC}")
        print(f"  python rudushi_chat.py --model TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf")
        print(f"  python rudushi_chat.py --model /full/path/to/model.gguf")
        sys.exit(1)

    # Check if llama-cli exists
    if not os.path.exists("/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"):
        print(f"{Colors.FAIL}❌ llama-cli not found{Colors.ENDC}")
        print(f"   Make sure llama.cpp is built")
        sys.exit(1)

    # Create chat instance
    chat = RudushiChat(
        model_path=model_path,
        history_file="rudushi_history.json" if not args.no_history else None
    )

    # Start chat
    chat.chat_loop()

if __name__ == "__main__":
    main()
