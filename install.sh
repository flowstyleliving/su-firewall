#!/bin/bash
# Semantic Collapse Auditor Installation Script
# The first zero-shot collapse audit tool for foundation model safety

set -e

echo "🔬 SEMANTIC COLLAPSE AUDITOR INSTALLER"
echo "============================================================"
echo "⚡ Zero-shot collapse detection for foundation models"
echo "🎯 Enterprise-grade semantic uncertainty analysis"
echo "📊 ℏₛ(C) = √(Δμ × Δσ)"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    print_error "This script should not be run as root. Please run as a regular user."
    exit 1
fi

# Detect operating system
OS="unknown"
case "$(uname -s)" in
    Darwin*)    OS="macos";;
    Linux*)     OS="linux";;
    CYGWIN*)    OS="windows";;
    MINGW*)     OS="windows";;
    *)          OS="unknown";;
esac

print_status "Detected OS: $OS"

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $PYTHON_VERSION"

# Check if version is >= 3.8
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" || {
    print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
}

# Get installation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
print_status "Installation directory: $SCRIPT_DIR"

# Create or activate virtual environment
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

print_status "Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install required packages
print_status "Installing required packages..."
pip install -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || {
    print_warning "requirements.txt not found. Installing core dependencies..."
    pip install requests pandas numpy matplotlib seaborn scikit-learn aiohttp python-dotenv
}

# Install Rust if not present (for core-engine)
if ! command -v cargo &> /dev/null; then
    print_warning "Rust not found. Installing Rust for core-engine..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Build core-engine
if [ -d "$SCRIPT_DIR/core-engine" ]; then
    print_status "Building Rust core-engine..."
    cd "$SCRIPT_DIR/core-engine"
    if cargo build --release; then
        print_status "Core-engine built successfully"
    else
        print_warning "Core-engine build failed - continuing with mock responses only"
    fi
    cd "$SCRIPT_DIR"
fi

# Make semantic-auditor executable
print_status "Making semantic-auditor executable..."
chmod +x "$SCRIPT_DIR/semantic-auditor"

# Add to PATH
print_status "Setting up PATH..."

# Detect shell
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    bash)
        SHELL_RC="$HOME/.bashrc"
        if [[ "$OS" == "macos" ]]; then
            SHELL_RC="$HOME/.bash_profile"
        fi
        ;;
    zsh)
        SHELL_RC="$HOME/.zshrc"
        ;;
    fish)
        SHELL_RC="$HOME/.config/fish/config.fish"
        ;;
    *)
        SHELL_RC="$HOME/.profile"
        ;;
esac

print_status "Detected shell: $SHELL_NAME ($SHELL_RC)"

# Create symlink or add to PATH
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

if [ -L "$BIN_DIR/semantic-auditor" ]; then
    rm "$BIN_DIR/semantic-auditor"
fi

ln -s "$SCRIPT_DIR/semantic-auditor" "$BIN_DIR/semantic-auditor"

# Add to shell configuration if not already present
if [[ "$OS" != "windows" ]]; then
    if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$SHELL_RC" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        print_status "Added $HOME/.local/bin to PATH in $SHELL_RC"
    fi
fi

# Create environment file if it doesn't exist
if [ ! -f "$SCRIPT_DIR/core-engine/.env" ]; then
    print_status "Creating environment configuration..."
    cat > "$SCRIPT_DIR/core-engine/.env" << EOF
# Semantic Collapse Auditor Configuration
# Set to false to use real API calls (requires valid API keys)
ENABLE_MOCK_RESPONSES=true

# API Keys (optional, used when ENABLE_MOCK_RESPONSES=false)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
MISTRAL_API_KEY=your_mistral_key_here
GROQ_API_KEY=your_groq_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Semantic API Configuration
SEMANTIC_API_URL=http://localhost:3000
EOF
    print_status "Created core-engine/.env with default configuration"
fi

# Test installation
print_status "Testing installation..."
"$SCRIPT_DIR/semantic-auditor" --version || {
    print_error "Installation test failed"
    exit 1
}

echo ""
echo "🎉 INSTALLATION COMPLETE!"
echo "============================================================"
echo "🔬 Semantic Collapse Auditor is now installed and ready to use!"
echo ""
echo "📋 QUICK START:"
echo "  semantic-auditor --prompt \"Your prompt here\""
echo "  semantic-auditor --file prompts.txt"
echo "  semantic-auditor --benchmark quick"
echo ""
echo "🔧 CONFIGURATION:"
echo "  Edit core-engine/.env to configure API keys"
echo "  Set ENABLE_MOCK_RESPONSES=false for real API calls"
echo ""
echo "📚 HELP:"
echo "  semantic-auditor --help"
echo ""
echo "🚀 RESTART YOUR SHELL or run:"
echo "  source $SHELL_RC"
echo "============================================================"

if [[ "$OS" == "macos" ]]; then
    print_status "On macOS, you may need to restart Terminal or run: source $SHELL_RC"
fi

print_status "Installation completed successfully!" 