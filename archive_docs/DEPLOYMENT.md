# 🚀 Deployment Guide

**Comprehensive deployment guide for Semantic Uncertainty Runtime**

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Cloudflare Workers](#-cloudflare-workers)
- [Self-Hosting](#-self-hosting)
- [Production Deployment](#-production-deployment)
- [Custom Domain Setup](#-custom-domain-setup)
- [Configuration](#-configuration)

## 🚀 Quick Start

### Prerequisites
- **Rust**: 1.70+ (for building from source)
- **Python**: 3.8+ (for dashboard)
- **Node.js**: 16+ (for Cloudflare Workers)

### Unified Deployment
```bash
# Deploy all components
./scripts/deploy.sh all --env production

# Deploy specific component
./scripts/deploy.sh cloudflare --env production
./scripts/deploy.sh dashboard --env staging
```

## 🌐 Cloudflare Workers

### Setup
```bash
# Install Wrangler
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Deploy to Cloudflare Workers
cd cloudflare-workers
wrangler deploy --config wrangler_neural_uncertainty.toml --env production
```

### Configuration
```toml
# wrangler_neural_uncertainty.toml
name = "semantic-uncertainty-runtime-physics-production"
main = "neural_uncertainty_worker.js"
compatibility_date = "2024-01-01"

[env.production]
name = "semantic-uncertainty-runtime-physics-production"

[env.staging]
name = "semantic-uncertainty-runtime-physics-staging"
```

### API Endpoints
- **Production**: `https://semantic-uncertainty-runtime-physics-production.mys628.workers.dev`
- **Staging**: `https://semantic-uncertainty-runtime-physics-staging.mys628.workers.dev`

## 🏠 Self-Hosting

### Local Development
```bash
# Build core engine
cd core-engine
cargo build --release

# Run dashboard
cd dashboard
streamlit run enhanced_diagnostics_dashboard.py

# Run API server
cd core-engine
cargo run --bin semantic-uncertainty-runtime server 3000
```

### Docker Deployment
```bash
# Build Docker image
docker build -t semantic-uncertainty-runtime .

# Run container
docker run -p 3000:3000 semantic-uncertainty-runtime
```

### System Requirements
- **OS**: Linux, macOS, Windows
- **Memory**: 1GB+ RAM recommended
- **Storage**: 500MB available space
- **Network**: Internet access for API calls

## 🏭 Production Deployment

### Railway (Dashboard)
```bash
# Deploy dashboard to Railway
cd dashboard
railway login
railway up --service dashboard
```

### Vercel (Static)
```bash
# Deploy static assets to Vercel
vercel --prod
```

### Environment Variables
```bash
# Required environment variables
export API_KEY="your-api-key"
export ENVIRONMENT="production"
export DOMAIN="your-domain.com"
```

## 🌍 Custom Domain Setup

### Cloudflare Pages
1. **Add Custom Domain**:
   - Go to Cloudflare Pages dashboard
   - Select your project
   - Click "Custom domains"
   - Add your domain

2. **DNS Configuration**:
   ```bash
   # Add CNAME record
   your-domain.com CNAME your-project.pages.dev
   ```

3. **SSL Certificate**:
   - Cloudflare automatically provisions SSL
   - Verify certificate is active

### DNS Records
```bash
# Required DNS records
your-domain.com          CNAME  your-project.pages.dev
api.your-domain.com      CNAME  semantic-uncertainty-runtime-physics-production.mys628.workers.dev
www.your-domain.com      CNAME  your-project.pages.dev
```

### Verification
```bash
# Test domain resolution
nslookup your-domain.com
curl -I https://your-domain.com
```

## ⚙️ Configuration

### Core Engine Configuration
```toml
# semantic-config.toml
[thresholds]
critical = 0.8
warning = 1.0
safe = 1.2

[performance]
max_processing_time_ms = 10
enable_jsd_kl = true

[security]
deterministic_mode = true
zero_dependencies = true
```

### Dashboard Configuration
```python
# config.py
DASHBOARD_CONFIG = {
    "api_endpoint": "https://semantic-uncertainty-runtime-physics-production.mys628.workers.dev",
    "api_key": "your-api-key",
    "theme": "light",
    "debug": False
}
```

### Environment Variables
```bash
# .env
API_KEY=your-api-key
ENVIRONMENT=production
DOMAIN=your-domain.com
DEBUG=false
LOG_LEVEL=info
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Clean and rebuild
./scripts/build.sh all --clean

# Check Rust toolchain
rustup show
cargo --version
```

#### 2. Deployment Failures
```bash
# Check Wrangler configuration
wrangler whoami
wrangler config

# Verify API keys
echo $CLOUDFLARE_API_TOKEN
```

#### 3. Domain Issues
```bash
# Check DNS propagation
dig your-domain.com
nslookup your-domain.com

# Verify SSL certificate
openssl s_client -connect your-domain.com:443
```

### Logs and Monitoring
```bash
# View Cloudflare Workers logs
wrangler tail --config wrangler_neural_uncertainty.toml

# Monitor dashboard logs
railway logs --service dashboard
```

## 📊 Performance Optimization

### Build Optimization
```bash
# Optimize Rust build
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Optimize WASM
wasm-opt -O4 -o optimized.wasm semantic_uncertainty_runtime.wasm
```

### Runtime Optimization
- **Memory**: Monitor memory usage with `htop`
- **CPU**: Use performance profiling tools
- **Network**: Optimize API response times
- **Cache**: Implement appropriate caching strategies

## 🔒 Security Considerations

### API Key Management
- Store API keys in environment variables
- Rotate keys regularly
- Use least privilege principle
- Monitor API usage

### SSL/TLS
- Always use HTTPS in production
- Verify SSL certificate validity
- Implement proper CORS policies
- Use secure headers

### Access Control
- Implement rate limiting
- Add authentication where needed
- Monitor for suspicious activity
- Regular security audits

---

**For additional help, see the [API Reference](../api/README.md) or [Technical Documentation](../technical/README.md).** 