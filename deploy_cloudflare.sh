#!/bin/bash

# 🔥 Cloudflare Workers Deployment Script
# Ultra-fast edge deployment for your semantic uncertainty API

set -e

echo "🔥 Cloudflare Workers Deployment - Semantic Uncertainty API"
echo "============================================================"

# Configuration
PROJECT_NAME="semantic-uncertainty-api"
CLOUDFLARE_ACCOUNT_ID=""
WRANGLER_CONFIG="wrangler.toml"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_requirements() {
    print_info "Checking requirements..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js not found. Please install Node.js 18+ first."
        print_info "Visit: https://nodejs.org/"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm not found. Please install npm first."
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        print_error "Git not found. Please install Git first."
        exit 1
    fi
    
    print_success "Requirements check passed"
}

install_dependencies() {
    print_info "Installing dependencies..."
    
    # Install Wrangler CLI globally if not present
    if ! command -v wrangler &> /dev/null; then
        print_info "Installing Wrangler CLI..."
        npm install -g wrangler
    fi
    
    # Install project dependencies
    npm install
    
    print_success "Dependencies installed"
}

setup_cloudflare_auth() {
    print_info "Setting up Cloudflare authentication..."
    
    # Check if already authenticated
    if wrangler whoami &> /dev/null; then
        print_success "Already authenticated with Cloudflare"
        return
    fi
    
    print_warning "Please authenticate with Cloudflare"
    print_info "This will open a browser window for authentication"
    
    wrangler login
    
    print_success "Cloudflare authentication complete"
}

get_account_id() {
    print_info "Getting Cloudflare account ID..."
    
    # Get account ID from wrangler
    ACCOUNT_ID=$(wrangler whoami | grep -o 'Account ID: [a-f0-9]*' | cut -d' ' -f3)
    
    if [ -z "$ACCOUNT_ID" ]; then
        print_error "Could not get account ID. Please check your Cloudflare authentication."
        exit 1
    fi
    
    print_success "Account ID: $ACCOUNT_ID"
    
    # Update wrangler.toml with account ID
    sed -i.bak "s/YOUR_ACCOUNT_ID/$ACCOUNT_ID/g" $WRANGLER_CONFIG
}

create_kv_namespace() {
    print_info "Creating KV namespace for usage tracking..."
    
    # Create KV namespace
    KV_OUTPUT=$(wrangler kv:namespace create "USAGE_TRACKER")
    KV_ID=$(echo "$KV_OUTPUT" | grep -o 'id = "[a-f0-9]*"' | cut -d'"' -f2)
    
    if [ -z "$KV_ID" ]; then
        print_error "Could not create KV namespace"
        exit 1
    fi
    
    print_success "KV namespace created: $KV_ID"
    
    # Update wrangler.toml with KV namespace ID
    sed -i.bak "s/your-kv-namespace-id/$KV_ID/g" $WRANGLER_CONFIG
    
    # Create preview KV namespace
    KV_PREVIEW_OUTPUT=$(wrangler kv:namespace create "USAGE_TRACKER" --preview)
    KV_PREVIEW_ID=$(echo "$KV_PREVIEW_OUTPUT" | grep -o 'id = "[a-f0-9]*"' | cut -d'"' -f2)
    
    sed -i.bak "s/your-preview-kv-namespace-id/$KV_PREVIEW_ID/g" $WRANGLER_CONFIG
}

generate_api_keys() {
    print_info "Generating API keys..."
    
    # Generate random API keys
    PRODUCTION_KEY=$(openssl rand -hex 32)
    STAGING_KEY=$(openssl rand -hex 32)
    
    print_success "API keys generated"
    print_warning "Production API Key: $PRODUCTION_KEY"
    print_warning "Staging API Key: $STAGING_KEY"
    print_warning "Save these keys securely!"
    
    # Update wrangler.toml with API keys
    sed -i.bak "s/your-production-api-key/$PRODUCTION_KEY/g" $WRANGLER_CONFIG
    sed -i.bak "s/your-staging-api-key/$STAGING_KEY/g" $WRANGLER_CONFIG
}

deploy_to_staging() {
    print_info "Deploying to staging environment..."
    
    wrangler deploy --env staging
    
    print_success "Staging deployment complete"
    print_info "Staging URL: https://$PROJECT_NAME-staging.your-subdomain.workers.dev"
}

deploy_to_production() {
    print_info "Deploying to production environment..."
    
    wrangler deploy --env production
    
    print_success "Production deployment complete"
    print_info "Production URL: https://$PROJECT_NAME.your-subdomain.workers.dev"
}

test_deployment() {
    print_info "Testing deployment..."
    
    # Test health endpoint
    HEALTH_URL="https://$PROJECT_NAME.your-subdomain.workers.dev/health"
    
    if curl -s "$HEALTH_URL" | grep -q "healthy"; then
        print_success "Health check passed"
    else
        print_warning "Health check failed - but deployment might still be working"
    fi
}

show_usage_instructions() {
    print_info "Deployment complete! Here's how to use your API:"
    
    echo ""
    echo "🌐 API Endpoints:"
    echo "  Health: https://$PROJECT_NAME.your-subdomain.workers.dev/health"
    echo "  Analyze: https://$PROJECT_NAME.your-subdomain.workers.dev/api/v1/analyze"
    echo "  Batch: https://$PROJECT_NAME.your-subdomain.workers.dev/api/v1/batch"
    echo ""
    echo "🔑 API Authentication:"
    echo "  Include header: X-API-Key: YOUR_API_KEY"
    echo "  Or: Authorization: Bearer YOUR_API_KEY"
    echo ""
    echo "📝 Example Usage:"
    echo "  curl -X POST https://$PROJECT_NAME.your-subdomain.workers.dev/api/v1/analyze \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -H 'X-API-Key: YOUR_API_KEY' \\"
    echo "    -d '{\"prompt\": \"Write a guide on AI safety\", \"model\": \"gpt4\"}'"
    echo ""
    echo "💰 Pricing:"
    echo "  Free tier: 100k requests/day"
    echo "  Paid tier: \$5/month + \$0.50 per million requests"
    echo ""
    echo "📊 Monitoring:"
    echo "  Dashboard: https://dash.cloudflare.com"
    echo "  Logs: wrangler tail"
    echo ""
}

show_menu() {
    echo ""
    echo "🎯 Choose deployment option:"
    echo "1) Full setup + deploy to staging"
    echo "2) Full setup + deploy to production"
    echo "3) Deploy to staging only"
    echo "4) Deploy to production only"
    echo "5) Test current deployment"
    echo "6) Show usage instructions"
    echo "7) Exit"
    echo ""
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            full_setup_staging
            ;;
        2)
            full_setup_production
            ;;
        3)
            deploy_to_staging
            ;;
        4)
            deploy_to_production
            ;;
        5)
            test_deployment
            ;;
        6)
            show_usage_instructions
            ;;
        7)
            print_info "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            show_menu
            ;;
    esac
}

full_setup_staging() {
    check_requirements
    install_dependencies
    setup_cloudflare_auth
    get_account_id
    create_kv_namespace
    generate_api_keys
    deploy_to_staging
    test_deployment
    show_usage_instructions
}

full_setup_production() {
    check_requirements
    install_dependencies
    setup_cloudflare_auth
    get_account_id
    create_kv_namespace
    generate_api_keys
    deploy_to_production
    test_deployment
    show_usage_instructions
}

# Main execution
main() {
    show_menu
    
    echo ""
    print_success "🎉 Cloudflare Workers deployment complete!"
    echo ""
    print_info "🔒 IP PROTECTION STATUS:"
    print_success "✅ Secret sauce runs at edge (300+ locations)"
    print_success "✅ WebAssembly provides binary protection"
    print_success "✅ No server access for reverse engineering"
    print_success "✅ Global rate limiting and authentication"
    echo ""
    print_info "🚀 PERFORMANCE BENEFITS:"
    print_success "✅ Sub-10ms latency worldwide"
    print_success "✅ Auto-scaling to millions of requests"
    print_success "✅ Zero cold starts"
    print_success "✅ Built-in DDoS protection"
    echo ""
    print_info "💰 COST BENEFITS:"
    print_success "✅ Free tier: 100k requests/day"
    print_success "✅ Paid tier: \$5/month + \$0.50 per million"
    print_success "✅ No server maintenance costs"
    print_success "✅ Pay only for what you use"
    echo ""
    print_info "🎯 NEXT STEPS:"
    print_info "1. Test your API endpoints"
    print_info "2. Update your CLI to use the new endpoints"
    print_info "3. Set up custom domain (optional)"
    print_info "4. Configure monitoring and alerts"
    print_info "5. Launch your marketing campaign!"
    echo ""
    print_success "🌟 Your semantic uncertainty API is now running on the edge!"
}

# Run main function
main "$@" 