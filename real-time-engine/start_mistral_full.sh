#!/bin/bash

# 🧠 Mistral 7B Live Uncertainty Auditing System
# Complete system startup script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Define repo root and common paths
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
REALTIME_DIR="$REPO_ROOT/real-time-engine"
FRONTEND_DIR="$REPO_ROOT/dashboard/frontend/uncertainty-dashboard"
RUST_DIR="$REPO_ROOT/core-engine"
LOGS_DIR="$REPO_ROOT/logs"
VENV_DIR="$REALTIME_DIR/venv"

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}🚀 $1${NC}"
}

# Configuration
RUST_API_PORT=8080
PYTHON_WS_PORT=8765
PYTHON_HTTP_PORT=5001
FRONTEND_PORT=3000

# PID files for cleanup
RUST_PID=""
PYTHON_PID=""
FRONTEND_PID=""

# Cleanup function
cleanup() {
    log_warning "Cleaning up processes..."
    
    if [ ! -z "$RUST_PID" ]; then
        kill $RUST_PID 2>/dev/null || true
        log_info "Stopped Rust API server"
    fi
    
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null || true
        log_info "Stopped Python Mistral server"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        log_info "Stopped Frontend server"
    fi
    
    # Kill any remaining processes on our ports
    lsof -ti:$RUST_API_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$PYTHON_WS_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$PYTHON_HTTP_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    
    log_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        log_error "Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for $name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            log_success "$name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$name failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# System requirements check
check_requirements() {
    log_header "Checking System Requirements"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is required but not installed"
        exit 1
    fi
    
    node_version=$(node --version)
    log_info "Node.js version: $node_version"
    
    # Check Rust/Cargo
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo is required but not installed"
        exit 1
    fi
    
    rust_version=$(rustc --version)
    log_info "Rust version: $rust_version"
    
    # Check available ports
    log_info "Checking port availability..."
    check_port $RUST_API_PORT || exit 1
    check_port $PYTHON_WS_PORT || exit 1  
    check_port $PYTHON_HTTP_PORT || exit 1
    check_port $FRONTEND_PORT || exit 1
    
    log_success "All requirements satisfied"
}

# Setup Python environment
setup_python_env() {
    log_header "Setting up Python Environment"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log_info "Activated virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip > /dev/null 2>&1
    
    # Skip PyTorch for demo version
    log_info "Using demo version (no PyTorch required)..."
    
    # Install other requirements
    log_info "Installing Python dependencies..."
    pip install -r "$REALTIME_DIR/requirements/requirements_demo.txt"
    
    log_success "Python environment ready"
}

# Build Rust components
build_rust() {
    log_header "Building Rust Core Engine"
    
    cd "$RUST_DIR"
    
    # Build with optimizations
    log_info "Compiling Rust code (this may take a few minutes)..."
    cargo build --release --features "http-api"
    
    cd "$REPO_ROOT"
    log_success "Rust build completed"
}

# Setup React frontend
setup_frontend() {
    log_header "Setting up React Frontend"
    
    cd "$FRONTEND_DIR"
    
    # Install dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install
    fi
    
    # Create Next.js config if it doesn't exist
    if [ ! -f "next.config.js" ]; then
        log_info "Creating Next.js configuration..."
        cat > next.config.js << EOF
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
}

module.exports = nextConfig
EOF
    fi
    
    # Create Tailwind config if it doesn't exist
    if [ ! -f "tailwind.config.js" ]; then
        log_info "Creating Tailwind CSS configuration..."
        cat > tailwind.config.js << EOF
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
EOF
    fi
    
    # Create global CSS if it doesn't exist
    if [ ! -f "src/app/globals.css" ]; then
        mkdir -p src/app
        log_info "Creating global CSS..."
        cat > src/app/globals.css << EOF
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF
    fi
    
    # Create layout file if it doesn't exist
    if [ ! -f "src/app/layout.tsx" ]; then
        log_info "Creating app layout..."
        cat > src/app/layout.tsx << EOF
import './globals.css'

export const metadata = {
  title: 'Mistral 7B Uncertainty Dashboard',
  description: 'Real-time semantic uncertainty monitoring',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
EOF
    fi
    
    cd "$REPO_ROOT"
    log_success "Frontend setup completed"
}

# Start all services
start_services() {
    log_header "Starting All Services"
    
    # Start Python Mistral demo server
    log_info "Starting Mistral 7B demo server..."
    source "$VENV_DIR/bin/activate"
    python "$REALTIME_DIR/mistral_demo_server.py" &
    PYTHON_PID=$!
    log_info "Mistral server started (PID: $PYTHON_PID)"
    
    # Wait for Python server to be ready
    wait_for_service "http://localhost:$PYTHON_HTTP_PORT/health" "Mistral server"
    
    # Start Rust API server (optional, for additional features)
    log_info "Starting Rust uncertainty engine..."
    cd "$RUST_DIR"
    cargo run --release --features "http-api" &
    RUST_PID=$!
    cd "$REPO_ROOT"
    log_info "Rust engine started (PID: $RUST_PID)"
    
    # Start React frontend
    log_info "Starting React frontend..."
    cd "$FRONTEND_DIR"
    npm run dev &
    FRONTEND_PID=$!
    cd "$REPO_ROOT"
    log_info "Frontend started (PID: $FRONTEND_PID)"
    
    # Wait for frontend to be ready
    wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend"
    
    log_success "All services started successfully!"
}

# Display system information
show_system_info() {
    log_header "System Information"
    
    echo -e "${CYAN}🖥️  System: $(uname -s) $(uname -r)${NC}"
    echo -e "${CYAN}🧠 CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')${NC}"
    echo -e "${CYAN}💾 Memory: $(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc 2>/dev/null || echo 'Unknown') GB${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${CYAN}🚀 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)${NC}"
    else
        echo -e "${CYAN}🚀 GPU: Not available (using CPU)${NC}"
    fi
}

# Main execution
main() {
    echo -e "${PURPLE}"
    cat << "EOF"
   🧠 Mistral 7B Live Uncertainty Auditing System
   ================================================
   
   Real-time semantic uncertainty quantification
   ℏₛ = √(Δμ × Δσ) with live logit analysis
   
EOF
    echo -e "${NC}"
    
    show_system_info
    check_requirements
    setup_python_env
    build_rust
    setup_frontend
    start_services
    
    echo
    log_success "🎉 System startup completed!"
    echo
    echo -e "${GREEN}📋 Access Points:${NC}"
    echo -e "   🌐 Frontend Dashboard: ${CYAN}http://localhost:$FRONTEND_PORT${NC}"
    echo -e "   📡 WebSocket API: ${CYAN}ws://localhost:$PYTHON_WS_PORT${NC}"
    echo -e "   🔗 HTTP API: ${CYAN}http://localhost:$PYTHON_HTTP_PORT${NC}"
    echo -e "   ⚙️  Rust Engine: ${CYAN}http://localhost:$RUST_API_PORT${NC}"
    echo
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo -e "   • Open the dashboard in your browser"
    echo -e "   • Try prompts like 'Explain quantum computing'"
    echo -e "   • Watch real-time uncertainty metrics"
    echo -e "   • Monitor ℏₛ values and risk levels"
    echo
    echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"
    
    # Keep script running and monitor services
    while true; do
        sleep 5
        
        # Check if services are still running
        if ! kill -0 $PYTHON_PID 2>/dev/null; then
            log_error "Python server died, restarting..."
            source "$VENV_DIR/bin/activate"
            python "$REALTIME_DIR/mistral_demo_server.py" &
            PYTHON_PID=$!
        fi
        
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            log_error "Frontend server died, restarting..."
            cd "$FRONTEND_DIR"
            npm run dev &
            FRONTEND_PID=$!
            cd "$REPO_ROOT"
        fi
    done
}

# Run main function
main "$@" 