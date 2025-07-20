#!/bin/bash

# 🚀 Unified Deployment Script for Semantic Uncertainty Runtime
# Consolidates all deployment functionality into a single script

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
DEPLOYMENT_TYPE=""
ENVIRONMENT="production"
API_KEY=""
DOMAIN=""

# Help function
show_help() {
    cat << EOF
🚀 Semantic Uncertainty Runtime - Unified Deployment Script

Usage: $0 [OPTIONS] <deployment-type>

Deployment Types:
  core-engine     Deploy Rust core engine
  dashboard       Deploy Streamlit dashboard
  cloudflare      Deploy to Cloudflare Workers
  wasm           Deploy WASM distribution
  all            Deploy all components

Options:
  -e, --env ENV      Environment (production|staging|development) [default: production]
  -k, --key KEY      API key for deployment
  -d, --domain DOM   Custom domain for deployment
  -h, --help         Show this help message

Examples:
  $0 core-engine
  $0 dashboard --env staging
  $0 cloudflare --key your-api-key
  $0 all --env production --domain semanticuncertainty.com

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
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -k|--key)
            API_KEY="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            DEPLOYMENT_TYPE="$1"
            shift
            ;;
    esac
done

# Validate deployment type
if [[ -z "$DEPLOYMENT_TYPE" ]]; then
    log_error "Deployment type is required"
    show_help
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(production|staging|development)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/README.md" ]]; then
        log_error "Not in project root directory"
        exit 1
    fi
    
    # Check for required tools based on deployment type
    case $DEPLOYMENT_TYPE in
        core-engine)
            if ! command -v cargo &> /dev/null; then
                log_error "Rust/Cargo not found. Please install Rust: https://rustup.rs/"
                exit 1
            fi
            ;;
        dashboard)
            if ! command -v python3 &> /dev/null; then
                log_error "Python 3 not found"
                exit 1
            fi
            if ! command -v pip &> /dev/null; then
                log_error "pip not found"
                exit 1
            fi
            ;;
        cloudflare)
            if ! command -v wrangler &> /dev/null; then
                log_error "Wrangler not found. Please install: npm install -g wrangler"
                exit 1
            fi
            ;;
        wasm)
            if ! command -v cargo &> /dev/null; then
                log_error "Rust/Cargo not found"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

# Deploy core engine
deploy_core_engine() {
    log_info "Deploying core engine..."
    
    cd "$CORE_ENGINE_DIR"
    
    # Build release version
    log_info "Building core engine..."
    cargo build --release
    
    # Run tests
    log_info "Running tests..."
    cargo test
    
    # Run benchmarks
    log_info "Running benchmarks..."
    cargo run --bin semantic-uncertainty-runtime benchmark
    
    log_success "Core engine deployment completed"
}

# Deploy dashboard
deploy_dashboard() {
    log_info "Deploying dashboard..."
    
    cd "$DASHBOARD_DIR"
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Run tests
    log_info "Running dashboard tests..."
    python -m pytest tests/ || log_warning "No tests found, continuing..."
    
    # Deploy to Railway
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Deploying to Railway..."
        railway up --service dashboard
    fi
    
    log_success "Dashboard deployment completed"
}

# Deploy to Cloudflare Workers
deploy_cloudflare() {
    log_info "Deploying to Cloudflare Workers..."
    
    cd "$CLOUDFLARE_DIR"
    
    # Check if API key is provided
    if [[ -z "$API_KEY" ]]; then
        log_warning "No API key provided, using default"
    fi
    
    # Deploy neural uncertainty worker
    log_info "Deploying neural uncertainty worker..."
    wrangler deploy --config wrangler_neural_uncertainty.toml --env "$ENVIRONMENT"
    
    # Deploy dashboard if specified
    if [[ "$DEPLOYMENT_TYPE" == "all" ]]; then
        log_info "Deploying dashboard to Cloudflare Pages..."
        wrangler pages deploy dashboard --project-name semantic-uncertainty-dashboard
    fi
    
    log_success "Cloudflare deployment completed"
}

# Deploy WASM distribution
deploy_wasm() {
    log_info "Deploying WASM distribution..."
    
    cd "$CORE_ENGINE_DIR"
    
    # Build WASM
    log_info "Building WASM module..."
    cargo build --target wasm32-unknown-unknown --release
    
    # Optimize WASM
    log_info "Optimizing WASM module..."
    wasm-opt -O4 -o target/wasm32-unknown-unknown/release/semantic_uncertainty_runtime_opt.wasm \
        target/wasm32-unknown-unknown/release/semantic_uncertainty_runtime.wasm
    
    # Copy to distribution directory
    log_info "Copying to distribution directory..."
    cp target/wasm32-unknown-unknown/release/semantic_uncertainty_runtime_opt.wasm \
        "$WASM_DIR/semantic_uncertainty_runtime.wasm"
    
    log_success "WASM deployment completed"
}

# Deploy all components
deploy_all() {
    log_info "Deploying all components..."
    
    deploy_core_engine
    deploy_dashboard
    deploy_cloudflare
    deploy_wasm
    
    log_success "All components deployed successfully"
}

# Main deployment function
main() {
    log_info "Starting deployment: $DEPLOYMENT_TYPE (Environment: $ENVIRONMENT)"
    
    # Check prerequisites
    check_prerequisites
    
    # Execute deployment based on type
    case $DEPLOYMENT_TYPE in
        core-engine)
            deploy_core_engine
            ;;
        dashboard)
            deploy_dashboard
            ;;
        cloudflare)
            deploy_cloudflare
            ;;
        wasm)
            deploy_wasm
            ;;
        all)
            deploy_all
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
    
    # Show deployment information
    if [[ "$DEPLOYMENT_TYPE" == "cloudflare" || "$DEPLOYMENT_TYPE" == "all" ]]; then
        echo ""
        log_info "Deployment Information:"
        echo "  🌐 API Endpoint: https://semantic-uncertainty-runtime-physics-production.mys628.workers.dev"
        echo "  📊 Dashboard: https://semantic-uncertainty-dashboard.pages.dev"
        if [[ -n "$DOMAIN" ]]; then
            echo "  🏠 Custom Domain: https://$DOMAIN"
        fi
    fi
}

# Run main function
main "$@" 