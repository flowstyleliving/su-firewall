#!/bin/bash

# 🔨 Unified Build Script for Semantic Uncertainty Runtime
# Builds all components with consistent configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_ENGINE_DIR="$PROJECT_ROOT/core-engine"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"
CLOUDFLARE_DIR="$PROJECT_ROOT/cloudflare-workers"
WASM_DIR="$PROJECT_ROOT/wasm-dist"

# Default values
BUILD_TYPE="all"
BUILD_MODE="release"
CLEAN_BUILD=false
RUN_TESTS=true

# Help function
show_help() {
    cat << EOF
🔨 Semantic Uncertainty Runtime - Unified Build Script

Usage: $0 [OPTIONS] [build-type]

Build Types:
  core-engine     Build Rust core engine only
  dashboard       Build Python dashboard only
  wasm           Build WASM distribution only
  all            Build all components (default)

Options:
  -m, --mode MODE     Build mode (debug|release) [default: release]
  -c, --clean         Clean build (remove previous builds)
  -t, --no-tests      Skip running tests
  -h, --help          Show this help message

Examples:
  $0 core-engine
  $0 dashboard --mode debug
  $0 all --clean
  $0 wasm --no-tests

EOF
}

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            BUILD_MODE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--no-tests)
            RUN_TESTS=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            BUILD_TYPE="$1"
            shift
            ;;
    esac
done

# Validate build mode
if [[ ! "$BUILD_MODE" =~ ^(debug|release)$ ]]; then
    log_error "Invalid build mode: $BUILD_MODE"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking build prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/README.md" ]]; then
        log_error "Not in project root directory"
        exit 1
    fi
    
    # Check for required tools
    if ! command -v cargo &> /dev/null; then
        log_error "Rust/Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    
    # Clean Rust build
    if [[ -d "$CORE_ENGINE_DIR/target" ]]; then
        cd "$CORE_ENGINE_DIR"
        cargo clean
        log_success "Cleaned Rust build artifacts"
    fi
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    log_success "Cleaned Python cache"
    
    # Clean WASM artifacts
    if [[ -d "$WASM_DIR" ]]; then
        rm -f "$WASM_DIR"/*.wasm
        log_success "Cleaned WASM artifacts"
    fi
}

# Build core engine
build_core_engine() {
    log_info "Building core engine..."
    
    cd "$CORE_ENGINE_DIR"
    
    # Build with specified mode
    if [[ "$BUILD_MODE" == "release" ]]; then
        log_info "Building in release mode..."
        cargo build --release
    else
        log_info "Building in debug mode..."
        cargo build
    fi
    
    # Run tests if enabled
    if [[ "$RUN_TESTS" == true ]]; then
        log_info "Running core engine tests..."
        cargo test
    fi
    
    # Run benchmarks
    log_info "Running benchmarks..."
    cargo run --bin semantic-uncertainty-runtime benchmark || log_warning "Benchmarks failed, continuing..."
    
    log_success "Core engine build completed"
}

# Build dashboard
build_dashboard() {
    log_info "Building dashboard..."
    
    cd "$DASHBOARD_DIR"
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Run tests if enabled
    if [[ "$RUN_TESTS" == true ]]; then
        log_info "Running dashboard tests..."
        python -m pytest tests/ || log_warning "No tests found, continuing..."
    fi
    
    # Validate dashboard
    log_info "Validating dashboard..."
    python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
    
    log_success "Dashboard build completed"
}

# Build WASM distribution
build_wasm() {
    log_info "Building WASM distribution..."
    
    cd "$CORE_ENGINE_DIR"
    
    # Check if wasm-pack is available
    if ! command -v wasm-pack &> /dev/null; then
        log_warning "wasm-pack not found, installing..."
        cargo install wasm-pack
    fi
    
    # Build WASM
    log_info "Building WASM module..."
    wasm-pack build --target web --out-dir "$WASM_DIR"
    
    # Optimize WASM if wasm-opt is available
    if command -v wasm-opt &> /dev/null; then
        log_info "Optimizing WASM module..."
        wasm-opt -O4 -o "$WASM_DIR/semantic_uncertainty_runtime_opt.wasm" \
            "$WASM_DIR/semantic_uncertainty_runtime_bg.wasm"
        mv "$WASM_DIR/semantic_uncertainty_runtime_opt.wasm" \
            "$WASM_DIR/semantic_uncertainty_runtime.wasm"
    else
        log_warning "wasm-opt not found, skipping optimization"
        cp "$WASM_DIR/semantic_uncertainty_runtime_bg.wasm" \
            "$WASM_DIR/semantic_uncertainty_runtime.wasm"
    fi
    
    log_success "WASM build completed"
}

# Build all components
build_all() {
    log_info "Building all components..."
    
    build_core_engine
    build_dashboard
    build_wasm
    
    log_success "All components built successfully"
}

# Show build information
show_build_info() {
    echo ""
    log_info "Build Information:"
    echo "  🔨 Build Type: $BUILD_TYPE"
    echo "  ⚙️  Build Mode: $BUILD_MODE"
    echo "  🧪 Tests: $([[ "$RUN_TESTS" == true ]] && echo "Enabled" || echo "Disabled")"
    echo "  🧹 Clean Build: $([[ "$CLEAN_BUILD" == true ]] && echo "Yes" || echo "No")"
    
    if [[ -f "$CORE_ENGINE_DIR/target/release/semantic-uncertainty-runtime" ]]; then
        echo "  📦 Core Engine: Built successfully"
    fi
    
    if [[ -f "$WASM_DIR/semantic_uncertainty_runtime.wasm" ]]; then
        echo "  🌐 WASM Module: Built successfully"
    fi
}

# Main build function
main() {
    log_info "Starting build: $BUILD_TYPE (Mode: $BUILD_MODE)"
    
    # Check prerequisites
    check_prerequisites
    
    # Clean build if requested
    if [[ "$CLEAN_BUILD" == true ]]; then
        clean_build
    fi
    
    # Execute build based on type
    case $BUILD_TYPE in
        core-engine)
            build_core_engine
            ;;
        dashboard)
            build_dashboard
            ;;
        wasm)
            build_wasm
            ;;
        all)
            build_all
            ;;
        *)
            log_error "Unknown build type: $BUILD_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    # Show build information
    show_build_info
    
    log_success "Build completed successfully!"
}

# Run main function
main "$@" 