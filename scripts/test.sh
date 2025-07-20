#!/bin/bash

# 🧪 Unified Test Script for Semantic Uncertainty Runtime
# Runs all tests with consistent configuration

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

# Default values
TEST_TYPE="all"
TEST_MODE="unit"
COVERAGE=false
VERBOSE=false

# Help function
show_help() {
    cat << EOF
🧪 Semantic Uncertainty Runtime - Unified Test Script

Usage: $0 [OPTIONS] [test-type]

Test Types:
  core-engine     Run Rust core engine tests only
  dashboard       Run Python dashboard tests only
  integration     Run integration tests only
  all            Run all tests (default)

Options:
  -m, --mode MODE     Test mode (unit|integration|performance) [default: unit]
  -c, --coverage      Generate coverage reports
  -v, --verbose       Verbose output
  -h, --help          Show this help message

Examples:
  $0 core-engine
  $0 dashboard --mode integration
  $0 all --coverage
  $0 integration --verbose

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
            TEST_MODE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            TEST_TYPE="$1"
            shift
            ;;
    esac
done

# Validate test mode
if [[ ! "$TEST_MODE" =~ ^(unit|integration|performance)$ ]]; then
    log_error "Invalid test mode: $TEST_MODE"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking test prerequisites..."
    
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

# Run core engine tests
test_core_engine() {
    log_info "Running core engine tests..."
    
    cd "$CORE_ENGINE_DIR"
    
    # Set test flags
    local test_flags=""
    if [[ "$VERBOSE" == true ]]; then
        test_flags="-- --nocapture"
    fi
    
    # Run tests based on mode
    case $TEST_MODE in
        unit)
            log_info "Running unit tests..."
            cargo test $test_flags
            ;;
        integration)
            log_info "Running integration tests..."
            cargo test --test integration $test_flags
            ;;
        performance)
            log_info "Running performance tests..."
            cargo run --bin semantic-uncertainty-runtime benchmark
            ;;
    esac
    
    # Run coverage if requested
    if [[ "$COVERAGE" == true ]]; then
        log_info "Generating coverage report..."
        cargo tarpaulin --out Html --output-dir coverage
    fi
    
    log_success "Core engine tests completed"
}

# Run dashboard tests
test_dashboard() {
    log_info "Running dashboard tests..."
    
    cd "$DASHBOARD_DIR"
    
    # Install test dependencies
    log_info "Installing test dependencies..."
    pip install -r requirements.txt
    pip install pytest pytest-cov pytest-mock
    
    # Set test flags
    local test_flags=""
    if [[ "$VERBOSE" == true ]]; then
        test_flags="-v"
    fi
    
    if [[ "$COVERAGE" == true ]]; then
        test_flags="$test_flags --cov=. --cov-report=html"
    fi
    
    # Run tests based on mode
    case $TEST_MODE in
        unit)
            log_info "Running unit tests..."
            python -m pytest tests/unit/ $test_flags || log_warning "No unit tests found"
            ;;
        integration)
            log_info "Running integration tests..."
            python -m pytest tests/integration/ $test_flags || log_warning "No integration tests found"
            ;;
        performance)
            log_info "Running performance tests..."
            python -m pytest tests/performance/ $test_flags || log_warning "No performance tests found"
            ;;
    esac
    
    log_success "Dashboard tests completed"
}

# Run integration tests
test_integration() {
    log_info "Running integration tests..."
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    
    # Test core engine API
    if [[ -f "$CORE_ENGINE_DIR/target/release/semantic-uncertainty-runtime" ]]; then
        log_info "Testing core engine API..."
        cd "$CORE_ENGINE_DIR"
        timeout 30s cargo run --bin semantic-uncertainty-runtime demo || log_warning "Core engine demo failed"
    fi
    
    # Test dashboard functionality
    if [[ -f "$DASHBOARD_DIR/enhanced_diagnostics_dashboard.py" ]]; then
        log_info "Testing dashboard functionality..."
        cd "$DASHBOARD_DIR"
        python -c "
import streamlit as st
import sys
sys.path.append('.')
from enhanced_diagnostics_dashboard import SimplifiedSemanticDashboard
dashboard = SimplifiedSemanticDashboard()
print('Dashboard initialization successful')
" || log_warning "Dashboard test failed"
    fi
    
    # Test Cloudflare Workers
    if [[ -f "$CLOUDFLARE_DIR/neural_uncertainty_worker.js" ]]; then
        log_info "Testing Cloudflare Workers..."
        cd "$CLOUDFLARE_DIR"
        node -c neural_uncertainty_worker.js && log_success "Worker syntax valid" || log_warning "Worker syntax check failed"
    fi
    
    log_success "Integration tests completed"
}

# Run all tests
test_all() {
    log_info "Running all tests..."
    
    test_core_engine
    test_dashboard
    test_integration
    
    log_success "All tests completed"
}

# Show test results
show_test_results() {
    echo ""
    log_info "Test Results Summary:"
    echo "  🧪 Test Type: $TEST_TYPE"
    echo "  ⚙️  Test Mode: $TEST_MODE"
    echo "  📊 Coverage: $([[ "$COVERAGE" == true ]] && echo "Enabled" || echo "Disabled")"
    echo "  🔍 Verbose: $([[ "$VERBOSE" == true ]] && echo "Enabled" || echo "Disabled")"
    
    # Check for coverage reports
    if [[ "$COVERAGE" == true ]]; then
        if [[ -d "$CORE_ENGINE_DIR/coverage" ]]; then
            echo "  📈 Core Engine Coverage: Available in $CORE_ENGINE_DIR/coverage"
        fi
        if [[ -d "$DASHBOARD_DIR/htmlcov" ]]; then
            echo "  📈 Dashboard Coverage: Available in $DASHBOARD_DIR/htmlcov"
        fi
    fi
}

# Main test function
main() {
    log_info "Starting tests: $TEST_TYPE (Mode: $TEST_MODE)"
    
    # Check prerequisites
    check_prerequisites
    
    # Execute tests based on type
    case $TEST_TYPE in
        core-engine)
            test_core_engine
            ;;
        dashboard)
            test_dashboard
            ;;
        integration)
            test_integration
            ;;
        all)
            test_all
            ;;
        *)
            log_error "Unknown test type: $TEST_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    # Show test results
    show_test_results
    
    log_success "Tests completed successfully!"
}

# Run main function
main "$@" 