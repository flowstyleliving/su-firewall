#!/bin/bash

# 🚀 Semantic Uncertainty Runtime - Complete Deployment Script
# Deploys both Cloudflare Worker and Streamlit Dashboard

set -e

echo "🚀 Starting Complete Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're in the right directory
if [ ! -f "wrangler.toml" ]; then
    print_error "Please run this script from the cloudflare-workers directory"
    exit 1
fi

# Step 1: Deploy Cloudflare Worker
print_status "Deploying Cloudflare Worker..."
cd ../cloudflare-workers

print_status "Deploying to production environment..."
wrangler deploy --env production
if [ $? -eq 0 ]; then
    print_success "Production deployment successful"
else
    print_error "Production deployment failed"
    exit 1
fi

print_status "Deploying to staging environment..."
wrangler deploy --env staging
if [ $? -eq 0 ]; then
    print_success "Staging deployment successful"
else
    print_error "Staging deployment failed"
    exit 1
fi

# Step 2: Test Worker Deployment
print_status "Testing worker deployment..."
sleep 5

# Test health endpoint
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "https://semanticuncertainty.com/health")
if [ "$HEALTH_RESPONSE" = "200" ]; then
    print_success "Health check passed"
else
    print_error "Health check failed: $HEALTH_RESPONSE"
    exit 1
fi

# Test analysis endpoint
ANALYSIS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://semanticuncertainty.com/api/v1/analyze" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "test", "output": "test"}')
if [ "$ANALYSIS_RESPONSE" = "200" ]; then
    print_success "Analysis endpoint test passed"
else
    print_error "Analysis endpoint test failed: $ANALYSIS_RESPONSE"
    exit 1
fi

# Step 3: Setup Dashboard Environment
print_status "Setting up dashboard environment..."
cd ../dashboard

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
print_status "Installing dashboard dependencies..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install streamlit plotly pandas numpy scikit-learn requests

# Step 4: Test Dashboard Integration
print_status "Testing dashboard integration..."
python3 test_integration.py
if [ $? -eq 0 ]; then
    print_success "Dashboard integration test passed"
else
    print_error "Dashboard integration test failed"
    exit 1
fi

# Step 5: Start Dashboard (Optional)
print_status "Dashboard setup complete!"
print_warning "To start the dashboard, run:"
echo "cd dashboard"
echo "source venv/bin/activate"
echo "streamlit run enhanced_diagnostics_dashboard.py"

# Step 6: Display Deployment Summary
echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo "🌐 Production Worker: https://semanticuncertainty.com"
echo "🧪 Staging Worker: https://semantic-uncertainty-runtime-staging.mys628.workers.dev"
echo "📊 Health Check: https://semanticuncertainty.com/health"
echo "🔍 API Endpoint: https://semanticuncertainty.com/api/v1/analyze"
echo ""
echo "📋 Available Commands:"
echo "  • Test Integration: python3 test_integration.py"
echo "  • Start Dashboard: streamlit run enhanced_diagnostics_dashboard.py"
echo "  • Deploy Worker: wrangler deploy --env production"
echo ""

print_success "All systems operational! 🚀" 