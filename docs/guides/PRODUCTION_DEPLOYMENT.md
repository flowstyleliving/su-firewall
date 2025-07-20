# 🚀 Production Deployment Guide

**Enterprise-ready deployment for the streamlined Semantic Uncertainty Runtime**

---

## 📋 **Prerequisites**

### **System Requirements**
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11
- **Rust**: 1.70+ (for building from source)
- **Memory**: 1GB+ RAM (recommended: 2GB+)
- **Storage**: 500MB available space
- **CPU**: x86_64 or ARM64 architecture

### **Network Requirements**
- **Outbound**: None required (zero dependencies)
- **Inbound**: Configure as needed for API access
- **Bandwidth**: Minimal (analysis is local)

---

## 🏗️ **Deployment Options**

### **Option 1: Binary Distribution (Recommended)**

**For most production environments**

```bash
# Download pre-compiled binary
wget https://releases.semantic-uncertainty.com/v1.0/semantic-uncertainty-runtime-linux-x64
chmod +x semantic-uncertainty-runtime-linux-x64

# Test installation
./semantic-uncertainty-runtime-linux-x64 test

# Expected output:
# 🧪 Running Streamlined Engine Tests
# ✅ All tests completed successfully!
```

**Available binaries:**
- `semantic-uncertainty-runtime-linux-x64`
- `semantic-uncertainty-runtime-linux-arm64`
- `semantic-uncertainty-runtime-macos-x64`
- `semantic-uncertainty-runtime-macos-arm64`
- `semantic-uncertainty-runtime-windows-x64.exe`

### **Option 2: Source Build**

**For custom optimization or development**

```bash
# Clone repository
git clone https://github.com/yourusername/semantic-uncertainty-runtime.git
cd semantic-uncertainty-runtime/core-engine

# Build optimized release
cargo build --release

# Binary location
ls target/release/semantic-uncertainty-runtime
```

**Build optimization flags:**
```bash
# Maximum performance build
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release

# Minimal size build
cargo build --release --profile=min-size
```

### **Option 3: Container Deployment**

**For orchestrated environments**

```dockerfile
# Dockerfile
FROM rust:1.70-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/semantic-uncertainty-runtime /usr/local/bin/
EXPOSE 8080
CMD ["semantic-uncertainty-runtime"]
```

```bash
# Build and run container
docker build -t semantic-uncertainty-runtime .
docker run -p 8080:8080 semantic-uncertainty-runtime
```

---

## ⚙️ **Configuration**

### **Configuration File**

Create `semantic-config.toml`:

```toml
[thresholds]
# Risk assessment thresholds
critical = 0.8    # Block immediately
warning = 1.0     # Proceed with caution  
safe = 1.2        # Normal operation

[performance]
# Performance optimization
max_processing_time_ms = 10
enable_mad_tensor = true
jsd_fallback = true
hash_embedding_dims = 64

[security]
# Security settings
deterministic_mode = true
zero_dependencies = true
enable_logging = true

[api]
# API configuration (if using API wrapper)
host = "0.0.0.0"
port = 8080
max_request_size = "10MB"
timeout_seconds = 30

[monitoring]
# Monitoring and metrics
enable_metrics = true
metrics_endpoint = "/metrics"
health_endpoint = "/health"
```

### **Environment Variables**

```bash
# Optional environment overrides
export SEMANTIC_CONFIG_PATH="/etc/semantic-uncertainty/config.toml"
export SEMANTIC_LOG_LEVEL="info"
export SEMANTIC_MAX_THREADS="4"
export SEMANTIC_CACHE_SIZE="1000"
```

### **Runtime Flags**

```bash
# Command line options
./semantic-uncertainty-runtime \
  --config /path/to/config.toml \
  --log-level info \
  --max-threads 4 \
  --enable-metrics
```

---

## 🔧 **Integration Patterns**

### **Direct Binary Integration**

```bash
# Analyze single request
./semantic-uncertainty-runtime streamlined \
  "User prompt here" \
  "Model output here"

# Batch processing
./semantic-uncertainty-runtime batch \
  --input requests.jsonl \
  --output results.jsonl
```

### **API Wrapper Service**

Create a simple REST API wrapper:

```rust
// api_wrapper.rs
use axum::{extract::Json, http::StatusCode, response::Json as ResponseJson, routing::post, Router};
use serde::{Deserialize, Serialize};
use semantic_uncertainty_runtime::StreamlinedEngine;

#[derive(Deserialize)]
struct AnalyzeRequest {
    prompt: String,
    output: String,
}

#[derive(Serialize)]
struct AnalyzeResponse {
    hbar_s: f64,
    risk_level: String,
    processing_time_ms: f64,
    // ... other fields
}

async fn analyze(Json(request): Json<AnalyzeRequest>) -> Result<ResponseJson<AnalyzeResponse>, StatusCode> {
    let engine = StreamlinedEngine::new();
    let result = engine.analyze(&request.prompt, &request.output);
    
    Ok(ResponseJson(AnalyzeResponse {
        hbar_s: result.calibrated_hbar,
        risk_level: format!("{:?}", result.risk_level),
        processing_time_ms: result.processing_time_ns as f64 / 1_000_000.0,
    }))
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/analyze", post(analyze));
    axum::Server::bind(&"0.0.0.0:8080".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### **Kubernetes Deployment**

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-uncertainty-runtime
  labels:
    app: semantic-uncertainty-runtime
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-uncertainty-runtime
  template:
    metadata:
      labels:
        app: semantic-uncertainty-runtime
    spec:
      containers:
      - name: semantic-uncertainty-runtime
        image: semantic-uncertainty-runtime:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: semantic-uncertainty-service
spec:
  selector:
    app: semantic-uncertainty-runtime
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

---

## 📊 **Monitoring & Observability**

### **Health Checks**

```bash
# Basic health check
./semantic-uncertainty-runtime test

# API health endpoint
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "uptime_seconds": 3600,
#   "performance_target_met": true,
#   "last_analysis_time_ms": 2.34
# }
```

### **Metrics Collection**

**Prometheus metrics endpoint:**
```
# HELP semantic_uncertainty_analysis_duration_seconds Time spent analyzing requests
# TYPE semantic_uncertainty_analysis_duration_seconds histogram
semantic_uncertainty_analysis_duration_seconds_bucket{le="0.001"} 0
semantic_uncertainty_analysis_duration_seconds_bucket{le="0.005"} 1200
semantic_uncertainty_analysis_duration_seconds_bucket{le="0.01"} 1200
semantic_uncertainty_analysis_duration_seconds_bucket{le="+Inf"} 1200

# HELP semantic_uncertainty_risk_level_total Number of analyses by risk level
# TYPE semantic_uncertainty_risk_level_total counter
semantic_uncertainty_risk_level_total{level="safe"} 800
semantic_uncertainty_risk_level_total{level="warning"} 350
semantic_uncertainty_risk_level_total{level="critical"} 50
```

### **Logging Configuration**

```bash
# Set log level
export RUST_LOG=semantic_uncertainty_runtime=info

# Structured logging output
{
  "timestamp": "2024-12-10T12:00:00Z",
  "level": "INFO",
  "target": "semantic_uncertainty_runtime::streamlined_engine",
  "message": "Analysis completed",
  "fields": {
    "hbar_s": 1.234,
    "risk_level": "Safe",
    "processing_time_ms": 2.34,
    "deterministic_hash": "960737593379886611"
  }
}
```

---

## 🔒 **Security Considerations**

### **Input Validation**

```rust
// Example input validation
fn validate_input(prompt: &str, output: &str) -> Result<(), ValidationError> {
    if prompt.len() > 10_000 {
        return Err(ValidationError::PromptTooLong);
    }
    if output.len() > 50_000 {
        return Err(ValidationError::OutputTooLong);
    }
    // Additional validation as needed
    Ok(())
}
```

### **Rate Limiting**

```toml
# config.toml
[rate_limiting]
requests_per_second = 100
burst_capacity = 200
enable_per_ip_limiting = true
```

### **Access Control**

```bash
# API key authentication (if using API wrapper)
curl -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "...", "output": "..."}' \
     http://localhost:8080/analyze
```

---

## 📈 **Performance Tuning**

### **Resource Allocation**

**CPU Optimization:**
```bash
# Set CPU affinity (Linux)
taskset -c 0-3 ./semantic-uncertainty-runtime

# Set process priority
nice -n -10 ./semantic-uncertainty-runtime
```

**Memory Configuration:**
```bash
# Set memory limits (systemd)
echo "MemoryLimit=1G" >> /etc/systemd/system/semantic-uncertainty.service
```

### **Concurrent Processing**

```rust
// Example concurrent processing
use tokio::task::JoinSet;

async fn process_batch(requests: Vec<AnalyzeRequest>) -> Vec<AnalyzeResponse> {
    let engine = Arc::new(StreamlinedEngine::new());
    let mut set = JoinSet::new();
    
    for request in requests {
        let engine = engine.clone();
        set.spawn(async move {
            engine.analyze(&request.prompt, &request.output)
        });
    }
    
    let mut results = Vec::new();
    while let Some(result) = set.join_next().await {
        results.push(result.unwrap());
    }
    results
}
```

### **Optimization Flags**

```bash
# Maximum performance
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat" \
cargo build --release

# Profile-guided optimization (PGO)
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" \
cargo build --release
# Run representative workload, then:
RUSTFLAGS="-C profile-use=/tmp/pgo-data -C llvm-args=-pgo-warn-missing-function" \
cargo build --release
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**Performance below target:**
```bash
# Check system resources
htop
iostat -x 1

# Verify configuration
./semantic-uncertainty-runtime test

# Enable debug logging
RUST_LOG=debug ./semantic-uncertainty-runtime
```

**Memory usage concerns:**
```bash
# Monitor memory usage
valgrind --tool=massif ./semantic-uncertainty-runtime test

# Check for memory leaks
valgrind --tool=memcheck ./semantic-uncertainty-runtime test
```

**Compilation issues:**
```bash
# Update Rust toolchain
rustup update

# Clear build cache
cargo clean

# Rebuild with verbose output
cargo build --release --verbose
```

### **Support & Diagnostics**

**Generate diagnostic report:**
```bash
./semantic-uncertainty-runtime --generate-diagnostic-report
```

**Performance profiling:**
```bash
# CPU profiling
perf record -g ./semantic-uncertainty-runtime test
perf report

# Memory profiling  
heaptrack ./semantic-uncertainty-runtime test
```

---

## 📋 **Deployment Checklist**

### **Pre-Deployment**
- [ ] System requirements verified
- [ ] Configuration file prepared
- [ ] Security settings reviewed
- [ ] Performance targets validated
- [ ] Monitoring configured

### **Deployment**
- [ ] Binary deployed and executable
- [ ] Configuration loaded successfully
- [ ] Health checks passing
- [ ] Performance tests completed
- [ ] Security scan completed

### **Post-Deployment**
- [ ] Monitoring dashboards configured
- [ ] Log aggregation setup
- [ ] Alerting rules configured
- [ ] Documentation updated
- [ ] Team training completed

---

## 🔗 **Additional Resources**

- **[API Reference](api-reference.md)** - Complete API documentation
- **[Performance Guide](performance-guide.md)** - Optimization strategies
- **[Security Best Practices](security-guide.md)** - Security hardening
- **[Monitoring Setup](monitoring-guide.md)** - Observability configuration

---

**The streamlined Semantic Uncertainty Runtime is production-ready and optimized for enterprise deployment with sub-3ms performance, zero dependencies, and enterprise-grade reliability.**