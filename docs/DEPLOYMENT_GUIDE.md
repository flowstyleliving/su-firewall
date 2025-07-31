# 🚀 Deployment Guide: Optimized Rust WASM Runtime

## Overview

This guide covers deploying the optimized Semantic Uncertainty Runtime using Rust WASM to Cloudflare Workers. The runtime implements the core equation **ℏₛ = √(Δμ × Δσ)** with only essential fields for maximum performance.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │  Cloudflare      │    │  Rust WASM      │
│   Request       │───▶│  Worker          │───▶│  Core Engine    │
│                 │    │  (index.js)      │    │  (Simple)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Response        │
                       │  (8/10 fields)   │
                       └──────────────────┘
```

## 📋 Prerequisites

- Rust toolchain (latest stable)
- `wasm-pack` or `wasm-bindgen-cli`
- `wrangler` CLI (Cloudflare Workers)
- Node.js 18+ (for build tools)

## 🔧 Build Process

### 1. Build Simple WASM Module

```bash
# Navigate to core engine
cd core-engine

# Build simple WASM module
cd wasm-simple
cargo build --target wasm32-unknown-unknown --release

# Generate JavaScript bindings
wasm-bindgen target/wasm32-unknown-unknown/release/semantic_uncertainty_runtime_wasm.wasm \
  --out-dir ../../cloudflare-workers \
  --target web
```

### 2. Verify WASM Output

```bash
# Check generated files
ls -la cloudflare-workers/semantic_uncertainty_runtime_wasm*

# Expected output:
# - semantic_uncertainty_runtime_wasm.js
# - semantic_uncertainty_runtime_wasm_bg.wasm
# - semantic_uncertainty_runtime_wasm.d.ts
```

### 3. Clean Old WASM Files

```bash
# Remove complex WASM files
cd cloudflare-workers
rm -f semantic_uncertainty_runtime.wasm semantic_uncertainty_runtime.js
rm -f semantic_uncertainty_runtime_bg.wasm semantic_uncertainty_runtime_bg.js
rm -f semantic-uncertainty-runtime.wasm
```

## 🚀 Deployment

### Staging Environment

```bash
# Deploy to staging
wrangler deploy --env staging

# Test staging endpoint
curl -X POST "https://semantic-uncertainty-runtime-staging.mys628.workers.dev/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}'
```

### Production Environment

```bash
# Deploy to production
wrangler deploy --env production

# Test production endpoint
curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}'
```

## 🔍 Verification

### Expected Response Format

```json
{
  "method": "wasm",
  "core_equation": "ℏₛ = √(Δμ × Δσ)",
  "precision": 0.156,
  "flexibility": 0.156,
  "semantic_uncertainty": 0.884,
  "raw_hbar": 0.884,
  "risk_level": "Safe",
  "processing_time_ms": 0,
  "request_id": "wasm-uuid",
  "timestamp": "2025-07-31T01:32:37.105Z"
}
```

### Validation Script

```bash
#!/bin/bash
# validate_deployment.sh

ENDPOINT="https://semanticuncertainty.com/api/v1/analyze"

echo "Testing deployment..."

# Test basic functionality
RESPONSE=$(curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}')

# Check response structure
FIELD_COUNT=$(echo "$RESPONSE" | jq 'keys | length')
echo "Field count: $FIELD_COUNT (expected: 10)"

# Check for unwanted fields
UNWANTED_FIELDS=$(echo "$RESPONSE" | jq 'keys | map(select(test("neural_physics|architecture|predictive_uncertainty"))) | length')
echo "Unwanted fields: $UNWANTED_FIELDS (expected: 0)"

# Check response time
RESPONSE_TIME=$(curl -w "%{time_total}" -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}' -o /dev/null)
echo "Response time: ${RESPONSE_TIME}s (expected: < 0.2s)"

# Validate core equation
CORE_EQ=$(echo "$RESPONSE" | jq -r '.core_equation')
if [ "$CORE_EQ" = "ℏₛ = √(Δμ × Δσ)" ]; then
    echo "✅ Core equation correct"
else
    echo "❌ Core equation incorrect: $CORE_EQ"
fi
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. WASM Not Loading
```bash
# Check WASM files exist
ls -la cloudflare-workers/semantic_uncertainty_runtime_wasm*

# Verify import path in index.js
grep -n "semantic_uncertainty_runtime_wasm" cloudflare-workers/index.js
```

#### 2. Complex Response Still Returned
```bash
# Purge Cloudflare cache
# Go to Cloudflare Dashboard > semanticuncertainty.com > Caching > Purge Everything

# Or force redeployment
wrangler delete semantic-uncertainty-runtime-production
wrangler deploy --env production
```

#### 3. JavaScript Fallback Active
```bash
# Check worker logs
wrangler tail semantic-uncertainty-runtime-production

# Verify WASM initialization
# Look for "✅ WASM module initialized successfully" in logs
```

### Debug Mode

```bash
# Deploy with debug logging
wrangler deploy --env production --compatibility-flag debug

# Monitor logs in real-time
wrangler tail semantic-uncertainty-runtime-production --format pretty
```

## 📊 Performance Monitoring

### Health Check

```bash
# Basic health check
curl -s "https://semanticuncertainty.com/health" | jq '.'

# Expected response:
{
  "status": "healthy",
  "runtime": "semantic-uncertainty-runtime",
  "version": "1.0.0",
  "core_equation": "ℏₛ = √(Δμ × Δσ)",
  "timestamp": "2025-07-31T01:32:37.105Z"
}
```

### Performance Metrics

```bash
# Response time test
time curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}' > /dev/null

# Throughput test
for i in {1..10}; do
  curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "test", "output": "test"}' &
done
wait
```

## 🔄 Continuous Deployment

### Automated Build Script

```bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Starting deployment..."

# Build WASM
echo "📦 Building WASM module..."
cd core-engine/wasm-simple
cargo build --target wasm32-unknown-unknown --release

# Generate bindings
echo "🔗 Generating JavaScript bindings..."
wasm-bindgen target/wasm32-unknown-unknown/release/semantic_uncertainty_runtime_wasm.wasm \
  --out-dir ../../cloudflare-workers \
  --target web

# Clean old files
echo "🧹 Cleaning old WASM files..."
cd ../../cloudflare-workers
rm -f semantic_uncertainty_runtime.wasm semantic_uncertainty_runtime.js
rm -f semantic_uncertainty_runtime_bg.wasm semantic_uncertainty_runtime_bg.js
rm -f semantic-uncertainty-runtime.wasm

# Deploy
echo "🚀 Deploying to production..."
wrangler deploy --env production

# Test
echo "🧪 Testing deployment..."
sleep 5
curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}' | jq 'keys | length'

echo "✅ Deployment complete!"
```

### GitHub Actions Workflow

```yaml
name: Deploy Semantic Uncertainty Runtime

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        target: wasm32-unknown-unknown
    
    - name: Build WASM
      run: |
        cd core-engine/wasm-simple
        cargo build --target wasm32-unknown-unknown --release
    
    - name: Generate bindings
      run: |
        wasm-bindgen core-engine/wasm-simple/target/wasm32-unknown-unknown/release/semantic_uncertainty_runtime_wasm.wasm \
          --out-dir cloudflare-workers \
          --target web
    
    - name: Deploy to Cloudflare
      uses: cloudflare/wrangler-action@v3
      with:
        apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
        accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
        command: deploy --env production
        workingDirectory: cloudflare-workers
```

## 🎯 Success Criteria

- [ ] Production returns only 10 fields
- [ ] Response time < 200ms
- [ ] WASM status shows 'active'
- [ ] No complex fields (neural_physics, architecture, etc.)
- [ ] Staging and production responses match
- [ ] Health endpoint returns correct status

## 📚 Additional Resources

- [Cloudflare Workers Documentation](https://developers.cloudflare.com/workers/)
- [WASM-Bindgen Documentation](https://rustwasm.github.io/docs/wasm-bindgen/)
- [Rust WASM Book](https://rustwasm.github.io/docs/book/)
- [Optimization Trailheads](./OPTIMIZATION_TRAILHEADS.md)

---

**Next Steps**: Follow the build process, deploy to staging first, then production, and verify using the validation script. 