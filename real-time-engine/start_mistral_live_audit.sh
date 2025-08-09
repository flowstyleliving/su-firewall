#!/bin/bash

# 🚀 Mistral 7B + Live Auditing System Startup Script
# Launches the complete system: Rust audit API + Python Mistral server + React frontend

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define repo root and common paths
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
REALTIME_DIR="$REPO_ROOT/real-time-engine"
FRONTEND_DIR="$REPO_ROOT/dashboard/frontend/uncertainty-dashboard"
RUST_DIR="$REPO_ROOT/core-engine"
LOGS_DIR="$REPO_ROOT/logs"

mkdir -p "$LOGS_DIR"

# Configuration
RUST_API_PORT=8080
PYTHON_SERVER_PORT=5000
WEBSOCKET_PORT=8765
FRONTEND_PORT=3000

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if port is available
check_port() {
    local port=$1
    local service=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_error "Port $port is already in use (needed for $service)"
        print_status "Please stop the service using port $port or change the configuration"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - waiting for $service_name..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Function to cleanup background processes
cleanup() {
    print_status "Shutting down services..."
    
    # Kill background processes
    for pid in ${BACKGROUND_PIDS[@]}; do
        if kill -0 $pid 2>/dev/null; then
            print_status "Terminating process $pid"
            kill $pid 2>/dev/null || true
        fi
    done
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill if still running
    for pid in ${BACKGROUND_PIDS[@]}; do
        if kill -0 $pid 2>/dev/null; then
            print_warning "Force killing process $pid"
            kill -9 $pid 2>/dev/null || true
        fi
    done
    
    print_status "Cleanup complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Array to track background process PIDs
declare -a BACKGROUND_PIDS

print_status "🚀 Starting Mistral 7B Live Uncertainty Auditing System"
print_status "=================================================="

# Check prerequisites
print_status "Checking prerequisites..."

# Check if required commands exist
command -v cargo >/dev/null 2>&1 || { print_error "cargo is required but not installed. Please install Rust."; exit 1; }
command -v python3 >/dev/null 2>&1 || { print_error "python3 is required but not installed."; exit 1; }
command -v npm >/dev/null 2>&1 || { print_error "npm is required but not installed."; exit 1; }
command -v curl >/dev/null 2>&1 || { print_error "curl is required but not installed."; exit 1; }

print_success "All required commands found"

# Check if ports are available
print_status "Checking port availability..."
check_port $RUST_API_PORT "Rust Audit API" || exit 1
check_port $PYTHON_SERVER_PORT "Python Mistral Server" || exit 1
check_port $WEBSOCKET_PORT "WebSocket Server" || exit 1
check_port $FRONTEND_PORT "React Frontend" || exit 1

print_success "All ports are available"

# Build Rust audit API
print_status "Building Rust audit API..."
cd "$RUST_DIR"
if cargo build --release --features="http-api,websocket"; then
    print_success "Rust audit API built successfully"
else
    print_error "Failed to build Rust audit API"
    exit 1
fi
cd "$REPO_ROOT"

# Install Python dependencies
print_status "Installing Python dependencies..."
if python3 -m pip install -r "$REALTIME_DIR/requirements/requirements.txt"; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    exit 1
fi

# Install frontend dependencies
print_status "Installing frontend dependencies..."
cd "$FRONTEND_DIR"
if npm install; then
    print_success "Frontend dependencies installed"
else
    print_error "Failed to install frontend dependencies"
    exit 1
fi
cd "$REPO_ROOT"

print_status "🎯 Starting services..."

# Start Rust audit API server
print_status "Starting Rust audit API server on port $RUST_API_PORT..."
cd "$RUST_DIR"
cargo run --release --features="http-api,websocket" --bin audit_server -- --port $RUST_API_PORT > "$LOGS_DIR/rust_api.log" 2>&1 &
RUST_PID=$!
BACKGROUND_PIDS+=($RUST_PID)
cd "$REPO_ROOT"

# Wait for Rust API to be ready
if ! wait_for_service "http://localhost:$RUST_API_PORT/health" "Rust Audit API"; then
    print_error "Rust Audit API failed to start"
    exit 1
fi

# Start Python Mistral server
print_status "Starting Python Mistral server..."
print_warning "⚠️  This will download Mistral 7B model if not already cached (~13GB)"
print_status "Model download may take 10-30 minutes depending on internet speed"

python3 "$REALTIME_DIR/mistral_audit_server.py" > "$LOGS_DIR/mistral_server.log" 2>&1 &
MISTRAL_PID=$!
BACKGROUND_PIDS+=($MISTRAL_PID)

# Give Mistral server more time to load the model
print_status "Waiting for Mistral model to load (this may take several minutes)..."
if ! wait_for_service "http://localhost:$PYTHON_SERVER_PORT/health" "Mistral Server"; then
    print_error "Mistral Server failed to start"
    print_status "Check $LOGS_DIR/mistral_server.log for details"
    exit 1
fi

# Start React frontend
print_status "Starting React frontend on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
npm run dev -- --port $FRONTEND_PORT > "$LOGS_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
BACKGROUND_PIDS+=($FRONTEND_PID)
cd "$REPO_ROOT"

# Wait for frontend to be ready
if ! wait_for_service "http://localhost:$FRONTEND_PORT" "React Frontend"; then
    print_error "React Frontend failed to start"
    exit 1
fi

print_success "🎉 All services are running successfully!"
print_status "=========================================="
print_status "Service URLs:"
print_status "  📊 Live Dashboard:     http://localhost:$FRONTEND_PORT"
print_status "  🔍 Rust Audit API:     http://localhost:$RUST_API_PORT"
print_status "  🤖 Mistral Server:     http://localhost:$PYTHON_SERVER_PORT"
print_status "  📡 WebSocket Stream:    ws://localhost:$WEBSOCKET_PORT"
print_status ""
print_status "📋 Quick Start:"
print_status "  1. Open http://localhost:$FRONTEND_PORT in your browser"
print_status "  2. Enter a prompt in the text area"
print_status "  3. Click 'Start Generation' to see live uncertainty analysis"
print_status "  4. Watch real-time uncertainty metrics and alerts"
print_status ""
print_status "📁 Logs are available in the 'logs/' directory"
print_status "🛑 Press Ctrl+C to stop all services"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Monitor services and keep script running
print_status "Monitoring services... (Press Ctrl+C to stop)"

while true; do
    # Check if all background processes are still running
    all_running=true
    
    for pid in ${BACKGROUND_PIDS[@]}; do
        if ! kill -0 $pid 2>/dev/null; then
            all_running=false
            print_error "Process $pid has stopped unexpectedly"
            break
        fi
    done
    
    if [ "$all_running" = false ]; then
        print_error "One or more services have stopped. Shutting down..."
        break
    fi
    
    sleep 5
done

# Cleanup will be called automatically due to trap 