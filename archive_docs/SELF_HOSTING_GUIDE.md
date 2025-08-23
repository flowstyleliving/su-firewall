# 🔧 Self-Hosting Guide

**Advanced setup for enterprise deployments requiring data privacy or custom configuration**

⚠️ **Warning**: This is a complex setup requiring significant technical expertise. For 99% of users, we recommend the [hosted API service](https://api.semantic-uncertainty.com).

## 🎯 **When to Self-Host**

### **✅ Good Reasons**
- **Data Privacy**: Keep analysis on your infrastructure
- **Custom Configuration**: Modify thresholds and policies
- **High Volume**: Unlimited analyses without API costs
- **Integration**: Deep integration with existing systems
- **Compliance**: Regulatory requirements for on-premises

### **❌ Bad Reasons**
- "I want to try it out" → Use hosted API instead
- "I don't want to pay" → Free tier available
- "It looks simple" → It's not simple at all

## 📋 **Prerequisites**

### **Technical Expertise Required**
- **Rust Programming**: Understanding of ownership, lifetimes, async/await
- **System Administration**: Linux/Unix administration, process management
- **Security Configuration**: Firewall rules, API key management
- **Database Management**: Configuration and optimization
- **Network Configuration**: Load balancing, SSL/TLS setup

### **Time Investment**
- **Experienced Developer**: 2-4 hours minimum
- **System Administrator**: 4-8 hours with documentation
- **Novice**: 8-16 hours or more (not recommended)
- **Troubleshooting**: Additional 2-8 hours for issues

### **System Requirements**
- **RAM**: 4GB+ for compilation, 2GB+ for runtime
- **Disk**: 2GB+ for dependencies and build artifacts
- **CPU**: Multi-core recommended for reasonable build times
- **Network**: Fast internet for dependency downloads
- **OS**: Linux/Unix (macOS for development)

## 🔧 **Installation Process**

### **Step 1: Development Environment (15-30 minutes)**
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev

# Install system dependencies (CentOS/RHEL)
sudo yum groupinstall -y "Development Tools"
sudo yum install -y openssl-devel

# Install system dependencies (macOS)
brew install openssl pkg-config
```

### **Step 2: Core Engine Compilation (10-30 minutes)**
```bash
# Clone repository
git clone https://github.com/yourusername/semantic-uncertainty-runtime.git
cd semantic-uncertainty-runtime

# Build core engine (this takes time and resources)
cd core-engine
cargo build --release --features api,python,streaming

# Verify build
./target/release/semantic-uncertainty-runtime --version
```

**Common Build Issues:**
- **Out of memory**: Increase swap space or use fewer parallel jobs
- **Missing dependencies**: Install development headers for system libraries
- **Compilation errors**: Check Rust version compatibility

### **Step 3: Configuration Setup (30-60 minutes)**

#### **A. Core Configuration**
```bash
# Create configuration directory
mkdir -p config

# Copy example configurations
cp config/prompt_scalar_recovery.json.example config/prompt_scalar_recovery.json
cp config/threshold_config.toml.example config/threshold_config.toml
cp config/security_policies.json.example config/security_policies.json
```

#### **B. Scalar Firewall Configuration**
Edit `config/prompt_scalar_recovery.json`:
```json
{
  "point": {
    "x": 11,
    "y": 6,
    "modulus": 17
  },
  "task": "scalar_prediction",
  "meta_instruction": "Only proceed with scalar prediction if the semantic bandwidth ℏₛ(C) ≥ 1.0. If ℏₛ is below threshold, halt inference and trigger fallback_action.",
  "thresholds": {
    "hbar_s": {
      "abort_below": 1.0,
      "warn_below": 1.2
    }
  },
  "collapse_policy": {
    "thresholds": {
      "entropy": {"min": 0.1, "soft_warn": 0.3, "max": 1.5},
      "confidence": {"min": 0.0, "soft_warn": 0.9, "max": 0.97},
      "hbar_s": {"soft_warn": 1.1, "abort_below": 0.9}
    },
    "abort_if": "confidence > 0.97 AND entropy < 0.1 AND hbar_s < 0.9",
    "fallback_action": "return_alias_class"
  }
}
```

#### **C. Security Configuration**
Edit `config/security_policies.json`:
```json
{
  "api_key_rotation": {
    "enabled": true,
    "rotation_interval_hours": 24,
    "max_key_age_days": 30
  },
  "rate_limiting": {
    "requests_per_minute": 100,
    "burst_size": 200
  },
  "breach_detection": {
    "enabled": true,
    "threshold_violations": 5,
    "time_window_minutes": 10
  }
}
```

### **Step 4: Database Setup (15-30 minutes)**
```bash
# Install PostgreSQL (recommended)
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres createdb semantic_uncertainty
sudo -u postgres createuser -P semantic_user

# Run migrations
cd core-engine
cargo run --features database -- migrate
```

### **Step 5: Python Environment (15-30 minutes)**
```bash
# Create virtual environment
cd ../dashboard
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fix common dependency issues
pip install --upgrade pip setuptools wheel
pip install streamlit plotly pandas numpy scipy rich
```

### **Step 6: API Key Management (10-20 minutes)**
```bash
# Generate API keys
cd ../core-engine
cargo run --features api -- generate-keys --count 10

# Deploy keys to secure storage
./deploy_api_keys.sh
```

### **Step 7: Testing and Validation (30-60 minutes)**
```bash
# Test core engine
cargo test --features api,python

# Test API endpoints
cargo run --features api -- server 8080 &
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "output": "test"}'

# Test dashboard
cd ../dashboard
python3 -m streamlit run streamlit_app.py --server.port 8501
```

## 🚀 **Deployment Options**

### **Option 1: Cloudflare Workers**
```bash
# Deploy to Cloudflare (using existing scripts)
./deploy_cloudflare.sh

# Configure wrangler.toml
wrangler publish
```

### **Option 2: Docker Deployment**
```dockerfile
# Dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features api

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y libssl1.1 ca-certificates
COPY --from=builder /app/target/release/semantic-uncertainty-runtime /usr/local/bin/
EXPOSE 8080
CMD ["semantic-uncertainty-runtime", "server", "8080"]
```

### **Option 3: Kubernetes Deployment**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-uncertainty
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-uncertainty
  template:
    metadata:
      labels:
        app: semantic-uncertainty
    spec:
      containers:
      - name: semantic-uncertainty
        image: semantic-uncertainty:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

## 🔧 **Configuration Reference**

### **Core Engine Configuration**
```toml
# config/core.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[database]
url = "postgresql://semantic_user:password@localhost/semantic_uncertainty"
max_connections = 10

[analysis]
default_threshold = 1.0
max_batch_size = 100
timeout_seconds = 30

[security]
api_key_header = "X-API-Key"
rate_limit_per_minute = 100
```

### **Threshold Configuration**
```toml
# config/threshold_config.toml
[models.gpt4]
hbar_s_threshold = 1.0
confidence_threshold = 0.9
entropy_threshold = 0.3

[models.claude3]
hbar_s_threshold = 1.1
confidence_threshold = 0.85
entropy_threshold = 0.35

[models.gemini]
hbar_s_threshold = 0.95
confidence_threshold = 0.92
entropy_threshold = 0.28
```

## 🔍 **Monitoring and Maintenance**

### **Health Checks**
```bash
# Check service status
curl http://localhost:8080/health

# Check metrics
curl http://localhost:8080/metrics

# Check logs
tail -f /var/log/semantic-uncertainty/app.log
```

### **Performance Tuning**
```bash
# Optimize database
sudo -u postgres vacuumdb --analyze semantic_uncertainty

# Monitor resource usage
htop
iostat -x 1
```

### **Security Monitoring**
```bash
# Check for security alerts
grep "SECURITY" /var/log/semantic-uncertainty/security.log

# Rotate API keys
cargo run --features api -- rotate-keys

# Update dependencies
cargo update
```

## 🚨 **Common Issues and Solutions**

### **Build Failures**
```bash
# Out of memory during compilation
export CARGO_BUILD_JOBS=1
cargo build --release

# Missing system dependencies
sudo apt-get install build-essential libssl-dev pkg-config
```

### **Runtime Errors**
```bash
# Database connection issues
systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"

# Permission errors
sudo chown -R semantic-user:semantic-user /opt/semantic-uncertainty
```

### **Performance Issues**
```bash
# High memory usage
echo 'vm.swappiness=10' >> /etc/sysctl.conf
sysctl -p

# Slow analysis
# Increase worker threads in config
```

## 🏁 **Production Checklist**

### **Security**
- [ ] API keys properly secured
- [ ] Rate limiting configured
- [ ] HTTPS/TLS enabled
- [ ] Firewall rules in place
- [ ] Regular security updates

### **Performance**
- [ ] Database optimized
- [ ] Caching configured
- [ ] Load balancing set up
- [ ] Monitoring in place
- [ ] Backup strategy implemented

### **Maintenance**
- [ ] Log rotation configured
- [ ] Automated updates planned
- [ ] Backup and recovery tested
- [ ] Documentation updated
- [ ] Team training completed

## 📞 **Support**

### **Self-Service**
- Check logs first: `/var/log/semantic-uncertainty/`
- Review configuration files
- Test with minimal setup
- Check GitHub issues

### **Enterprise Support**
- **Email**: enterprise@yourcompany.com
- **Documentation**: [Enterprise Portal](https://enterprise.semantic-uncertainty.com)
- **SLA**: 4-hour response time
- **Includes**: Setup assistance, configuration review, performance tuning

## 🏁 **Conclusion**

Self-hosting the Semantic Uncertainty Runtime is a complex undertaking that requires significant technical expertise and ongoing maintenance. 

**For most users**, we strongly recommend using the [hosted API service](https://api.semantic-uncertainty.com) for immediate results without the complexity.

**For enterprise users** with specific requirements, self-hosting provides complete control at the cost of significant setup and maintenance overhead. 