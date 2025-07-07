#!/bin/bash
# 🌐 Global Edge Deployment Script
# Deploy semantic uncertainty API to all regions with optimization

set -e

echo "🚀 Starting Global Edge Deployment"
echo "=================================="

# 📊 Pre-deployment checks
echo "📋 Running pre-deployment checks..."

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler CLI not found. Installing..."
    npm install -g wrangler@latest
fi

# Check if we're in the right directory
if [ ! -f "cloudflare-global-config.toml" ]; then
    echo "❌ Global config not found. Please run from edge-optimization directory."
    exit 1
fi

# Check authentication
echo "🔐 Checking Cloudflare authentication..."
if ! wrangler whoami &> /dev/null; then
    echo "❌ Not authenticated with Cloudflare. Please run 'wrangler login'"
    exit 1
fi

echo "✅ Pre-deployment checks passed"

# 🏗️ Build optimized bundle
echo ""
echo "🏗️ Building optimized edge bundle..."
npm run build:edge

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Aborting deployment."
    exit 1
fi

echo "✅ Build completed successfully"

# 🔧 Deploy to staging first
echo ""
echo "🧪 Deploying to staging environment..."
wrangler deploy --env staging --config cloudflare-global-config.toml

if [ $? -ne 0 ]; then
    echo "❌ Staging deployment failed. Aborting."
    exit 1
fi

echo "✅ Staging deployment successful"

# 🧪 Run staging tests
echo ""
echo "🧪 Running staging validation tests..."
sleep 5  # Wait for deployment to propagate

# Test staging endpoint
STAGING_URL="https://semantic-uncertainty-staging.semantic-uncertainty.workers.dev"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$STAGING_URL/health")

if [ "$RESPONSE" != "200" ]; then
    echo "❌ Staging health check failed (HTTP $RESPONSE)"
    exit 1
fi

echo "✅ Staging validation passed"

# 🚀 Deploy to production regions
echo ""
echo "🌍 Deploying to production regions..."

# Deploy to Americas
echo "🇺🇸 Deploying to Americas region..."
wrangler deploy --env americas --config cloudflare-global-config.toml

if [ $? -eq 0 ]; then
    echo "✅ Americas deployment successful"
else
    echo "❌ Americas deployment failed"
    exit 1
fi

# Deploy to Europe
echo "🇪🇺 Deploying to Europe region..."
wrangler deploy --env europe --config cloudflare-global-config.toml

if [ $? -eq 0 ]; then
    echo "✅ Europe deployment successful"
else
    echo "❌ Europe deployment failed"
    exit 1
fi

# Deploy to Asia
echo "🇦🇺 Deploying to Asia region..."
wrangler deploy --env asia --config cloudflare-global-config.toml

if [ $? -eq 0 ]; then
    echo "✅ Asia deployment successful"
else
    echo "❌ Asia deployment failed"
    exit 1
fi

# 📊 Set up monitoring and secrets
echo ""
echo "🔐 Configuring secrets and monitoring..."

# Set API keys for each environment
echo "Setting API secrets..."
echo "your-production-api-key" | wrangler secret put API_KEY_SECRET --env americas
echo "your-production-api-key" | wrangler secret put API_KEY_SECRET --env europe  
echo "your-production-api-key" | wrangler secret put API_KEY_SECRET --env asia

echo "✅ Secrets configured"

# 🏥 Post-deployment health checks
echo ""
echo "🏥 Running post-deployment health checks..."

# Regional endpoints to test
declare -a ENDPOINTS=(
    "https://api-us.semanticuncertainty.com"
    "https://api-eu.semanticuncertainty.com" 
    "https://api-asia.semanticuncertainty.com"
)

ALL_HEALTHY=true

for endpoint in "${ENDPOINTS[@]}"; do
    echo "Testing $endpoint..."
    
    # Health check
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint/health")
    
    if [ "$HEALTH_RESPONSE" = "200" ]; then
        echo "  ✅ Health check passed"
    else
        echo "  ❌ Health check failed (HTTP $HEALTH_RESPONSE)"
        ALL_HEALTHY=false
    fi
    
    # API functionality test
    API_RESPONSE=$(curl -s -w "%{http_code}" -X POST "$endpoint/api/v1/analyze" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: your-production-api-key" \
        -d '{"prompt": "Test deployment", "model": "gpt4"}' | tail -c 3)
    
    if [ "$API_RESPONSE" = "200" ]; then
        echo "  ✅ API test passed"
    else
        echo "  ❌ API test failed (HTTP $API_RESPONSE)"
        ALL_HEALTHY=false
    fi
    
    echo ""
done

# 📊 Performance benchmark
echo "⚡ Running performance benchmark..."
if command -v artillery &> /dev/null; then
    artillery run benchmarks/edge-performance.yml --output benchmark-results.json
    echo "✅ Performance benchmark completed"
else
    echo "⚠️ Artillery not found. Skipping performance benchmark."
fi

# 🎯 Final status
echo ""
echo "=================================="
if [ "$ALL_HEALTHY" = true ]; then
    echo "🎉 GLOBAL DEPLOYMENT SUCCESSFUL!"
    echo ""
    echo "🌍 Regional Endpoints:"
    echo "  🇺🇸 Americas: https://api-us.semanticuncertainty.com"
    echo "  🇪🇺 Europe:   https://api-eu.semanticuncertainty.com"
    echo "  🇦🇺 Asia:     https://api-asia.semanticuncertainty.com"
    echo ""
    echo "📊 Monitoring:"
    echo "  Dashboard: npm run monitor"
    echo "  Performance: npm run test:performance"
    echo ""
    echo "🚀 Your global edge network is ready for John Yue's demo!"
else
    echo "⚠️ DEPLOYMENT COMPLETED WITH ISSUES"
    echo "Some regional endpoints may not be fully operational."
    echo "Check the health status above and review deployment logs."
fi

echo ""
echo "📝 Next steps:"
echo "  1. Run 'npm run monitor' to start real-time monitoring"
echo "  2. Test the API with the demo script"
echo "  3. Review performance metrics"
echo ""
echo "🎯 Happy demonstrating to John Yue!"