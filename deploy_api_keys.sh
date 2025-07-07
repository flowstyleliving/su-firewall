#!/bin/bash

# 🔑 SEMANTIC UNCERTAINTY API KEY DEPLOYMENT SCRIPT
# Deploy and configure API keys for production use

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev"
KEY_MANAGER="python3 api_key_manager.py"

echo -e "${BLUE}🔑 SEMANTIC UNCERTAINTY API KEY DEPLOYMENT${NC}"
echo "=================================================="

# Check if key manager exists
if [ ! -f "api_key_manager.py" ]; then
    echo -e "${RED}❌ api_key_manager.py not found!${NC}"
    echo "Please ensure you're in the correct directory."
    exit 1
fi

# Function to generate a new API key
generate_key() {
    local name="$1"
    local email="$2"
    local tier="$3"
    
    echo -e "${YELLOW}🔐 Generating $tier tier key for: $name${NC}"
    
    if [ -n "$email" ]; then
        $KEY_MANAGER generate --name "$name" --email "$email" --tier "$tier"
    else
        $KEY_MANAGER generate --name "$name" --tier "$tier"
    fi
}

# Function to test API key
test_key() {
    local key="$1"
    local name="$2"
    
    echo -e "${YELLOW}🧪 Testing API key for: $name${NC}"
    
    # Test health endpoint
    echo "Testing health endpoint..."
    health_response=$(curl -s -w "%{http_code}" "$API_URL/health")
    health_status="${health_response: -3}"
    health_body="${health_response%???}"
    
    if [ "$health_status" = "200" ]; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${RED}❌ Health check failed (Status: $health_status)${NC}"
        echo "Response: $health_body"
        return 1
    fi
    
    # Test analysis endpoint
    echo "Testing analysis endpoint..."
    analysis_response=$(curl -s -w "%{http_code}" \
        -X POST "$API_URL/api/v1/analyze" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $key" \
        -d '{"prompt": "Test prompt for API validation", "model": "gpt4"}')
    
    analysis_status="${analysis_response: -3}"
    analysis_body="${analysis_response%???}"
    
    if [ "$analysis_status" = "200" ]; then
        echo -e "${GREEN}✅ Analysis endpoint working${NC}"
        echo "Response preview: ${analysis_body:0:100}..."
    else
        echo -e "${RED}❌ Analysis endpoint failed (Status: $analysis_status)${NC}"
        echo "Response: $analysis_body"
        return 1
    fi
}

# Function to display usage statistics
show_stats() {
    echo -e "${YELLOW}📊 API Usage Statistics${NC}"
    echo "=========================="
    $KEY_MANAGER stats --days 7
}

# Function to list all keys
list_keys() {
    echo -e "${YELLOW}📋 Current API Keys${NC}"
    echo "====================="
    $KEY_MANAGER list
}

# Function to set up Cloudflare Workers secrets
setup_cloudflare_secrets() {
    echo -e "${YELLOW}☁️ Setting up Cloudflare Workers secrets${NC}"
    
    # Check if wrangler is installed
    if ! command -v wrangler &> /dev/null; then
        echo -e "${RED}❌ Wrangler CLI not found!${NC}"
        echo "Install with: npm install -g wrangler"
        return 1
    fi
    
    # Get the latest generated key
    echo "Getting latest API key..."
    latest_key=$($KEY_MANAGER list | grep "su_" | head -1 | awk '{print $NF}')
    
    if [ -z "$latest_key" ]; then
        echo -e "${RED}❌ No API keys found!${NC}"
        echo "Generate a key first with: $KEY_MANAGER generate --name 'Production Key' --tier enterprise"
        return 1
    fi
    
    echo -e "${GREEN}🔑 Using key: ${latest_key:0:20}...${NC}"
    
    # Set the secret in Cloudflare Workers
    echo "Setting API key secret in Cloudflare Workers..."
    echo "$latest_key" | wrangler secret put API_KEY_SECRET --env production
    
    echo -e "${GREEN}✅ Cloudflare Workers secret updated${NC}"
}

# Function to create production environment
create_production_env() {
    echo -e "${YELLOW}🏭 Creating production environment${NC}"
    
    # Create .env.production file
    cat > .env.production << EOF
# Production Environment Variables
API_URL=$API_URL
API_KEY_SECRET=your_generated_key_here
RATE_LIMIT_PER_MINUTE=1000
ALLOWED_ORIGINS=https://semanticuncertainty.com,https://inference.ai

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=info

# Security
ENABLE_RATE_LIMITING=true
ENABLE_IP_WHITELIST=false
EOF
    
    echo -e "${GREEN}✅ Production environment file created: .env.production${NC}"
    echo "Remember to update API_KEY_SECRET with your actual key!"
}

# Function to run security audit
security_audit() {
    echo -e "${YELLOW}🔒 Running security audit${NC}"
    
    # Check for exposed keys in files
    echo "Checking for exposed API keys..."
    if grep -r "su_[a-z]*_[A-Za-z0-9+/=]*" . --exclude-dir=.git --exclude=api_key_manager.py 2>/dev/null; then
        echo -e "${RED}⚠️  Potential API key exposure detected!${NC}"
        echo "Check the files above for exposed keys."
    else
        echo -e "${GREEN}✅ No exposed API keys found${NC}"
    fi
    
    # Check file permissions
    echo "Checking file permissions..."
    if [ -f "api_keys.db" ]; then
        perms=$(stat -c "%a" api_keys.db)
        if [ "$perms" != "600" ]; then
            echo -e "${YELLOW}⚠️  Database file permissions should be 600 (current: $perms)${NC}"
            chmod 600 api_keys.db
            echo -e "${GREEN}✅ Fixed database permissions${NC}"
        else
            echo -e "${GREEN}✅ Database permissions are secure${NC}"
        fi
    fi
}

# Main deployment menu
main_menu() {
    echo ""
    echo -e "${BLUE}🎯 DEPLOYMENT OPTIONS${NC}"
    echo "====================="
    echo "1. Generate new API key"
    echo "2. Test existing API key"
    echo "3. Setup Cloudflare Workers secrets"
    echo "4. Create production environment"
    echo "5. Show usage statistics"
    echo "6. List all keys"
    echo "7. Run security audit"
    echo "8. Full deployment (all steps)"
    echo "9. Exit"
    echo ""
    read -p "Select option (1-9): " choice
    
    case $choice in
        1)
            read -p "Enter key name: " key_name
            read -p "Enter email (optional): " key_email
            read -p "Enter tier (free/pro/enterprise/unlimited): " key_tier
            generate_key "$key_name" "$key_email" "$key_tier"
            ;;
        2)
            read -p "Enter API key to test: " test_key_value
            read -p "Enter key name for reference: " test_key_name
            test_key "$test_key_value" "$test_key_name"
            ;;
        3)
            setup_cloudflare_secrets
            ;;
        4)
            create_production_env
            ;;
        5)
            show_stats
            ;;
        6)
            list_keys
            ;;
        7)
            security_audit
            ;;
        8)
            echo -e "${BLUE}🚀 Running full deployment...${NC}"
            
            # Generate production key
            generate_key "Production Key" "admin@semanticuncertainty.com" "enterprise"
            
            # Setup Cloudflare secrets
            setup_cloudflare_secrets
            
            # Create production environment
            create_production_env
            
            # Run security audit
            security_audit
            
            # Show final status
            echo ""
            echo -e "${GREEN}🎉 Full deployment completed!${NC}"
            echo "Next steps:"
            echo "1. Update .env.production with your API key"
            echo "2. Deploy to your production environment"
            echo "3. Test the API endpoints"
            echo "4. Monitor usage with: $KEY_MANAGER stats"
            ;;
        9)
            echo -e "${GREEN}👋 Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid option${NC}"
            ;;
    esac
}

# Check if running interactively
if [ -t 0 ]; then
    # Interactive mode
    while true; do
        main_menu
        echo ""
        read -p "Press Enter to continue..."
    done
else
    # Non-interactive mode - run full deployment
    echo -e "${BLUE}🚀 Running automated deployment...${NC}"
    
    # Generate production key
    generate_key "Production Key" "admin@semanticuncertainty.com" "enterprise"
    
    # Setup Cloudflare secrets
    setup_cloudflare_secrets
    
    # Create production environment
    create_production_env
    
    # Run security audit
    security_audit
    
    echo -e "${GREEN}🎉 Deployment completed!${NC}"
fi 