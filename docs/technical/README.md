# 🧮 Technical Documentation

**Comprehensive technical reference for Semantic Uncertainty Runtime**

## 📋 Table of Contents

- [Mathematical Framework](#-mathematical-framework)
- [Architecture Overview](#-architecture-overview)
- [Core Engine](#-core-engine)
- [API Reference](#-api-reference)
- [Performance Analysis](#-performance-analysis)
- [Integration Guide](#-integration-guide)

## 🧮 Mathematical Framework

### Semantic Uncertainty Formula

The core mathematical framework uses the formula:

```
ℏₛ = √(Δμ × Δσ)
```

Where:
- **ℏₛ**: Semantic uncertainty metric (0-2+ range)
- **Δμ**: Precision measurement using JSD divergence
- **Δσ**: Flexibility measurement using KL divergence

### Risk Assessment Framework

- **Critical** (ℏₛ ≤ 0.8): Block immediately
- **Warning** (0.8 < ℏₛ ≤ 1.2): Proceed with caution  
- **Safe** (ℏₛ > 1.2): Normal operation

### Component Analysis

#### JSD Divergence (Δμ)
```rust
// Precision measurement
let jsd_divergence = calculate_jsd_divergence(prompt_embedding, output_embedding);
```

#### KL Divergence (Δσ)
```rust
// Flexibility measurement
let kl_divergence = calculate_kl_divergence(prompt_distribution, output_distribution);
```

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Core Engine   │    │    Dashboard    │    │ Cloudflare API  │
│   (Rust)        │    │   (Python)      │    │   (Workers)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   WASM Module   │
                    │   (Edge)        │
                    └─────────────────┘
```

### Data Flow

1. **Input Processing**: Prompt and output text
2. **Embedding Generation**: Hash-based 64-dimensional embeddings
3. **Divergence Calculation**: JSD and KL divergence computation
4. **Uncertainty Assessment**: Combined metric calculation
5. **Risk Classification**: Threshold-based risk assessment
6. **Response Generation**: Structured API response

## 🔧 Core Engine

### StreamlinedEngine

```rust
pub struct StreamlinedEngine {
    // Zero dependencies, deterministic analysis
}

impl StreamlinedEngine {
    pub fn new() -> Self;
    pub fn analyze(&self, prompt: &str, output: &str) -> StreamlinedResult;
}
```

### StreamlinedResult

```rust
pub struct StreamlinedResult {
    pub raw_hbar: f64,           // Raw semantic uncertainty
    pub calibrated_hbar: f64,     // Calibrated for decision-making
    pub delta_mu: f64,           // Precision component
    pub delta_sigma: f64,        // Flexibility component
    pub risk_level: RiskLevel,   // Risk assessment
    pub processing_time_ns: u64,  // Performance metric
}
```

### Key Modules

#### Semantic Metrics (`modules/semantic_metrics.rs`)
- JSD divergence calculation
- KL divergence computation
- Embedding generation
- Calibration functions

#### Token Analyzer (`modules/token_analyzer.rs`)
- Token counting and analysis
- Cost estimation
- Efficiency scoring
- Class-based analysis

#### Prompt Scorer (`modules/prompt_scorer.rs`)
- Prompt classification
- Risk assessment
- Threshold management
- Confidence scoring

## 🌐 API Reference

### REST API Endpoints

#### Analyze Endpoint
```http
POST /api/v1/analyze
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "Your input prompt",
  "output": "Expected or actual output"
}
```

**Response:**
```json
{
  "raw_hbar": 0.25,
  "adjusted_hbar": 0.20687,
  "risk_level": "Warning",
  "laws": {
    "curvature_amplification": { "alpha": 0.822 },
    "metacognitive_priors": { "factors": { "c_balance": 1 } }
  },
  "token_analysis": {
    "prompt_tokens": 2,
    "output_tokens": 2,
    "total_tokens": 4
  }
}
```

#### Health Check
```http
GET /health
```

### WASM API

```javascript
import init, { WasmSemanticAnalyzer } from './semantic_uncertainty_runtime.js';

async function analyze() {
    await init('./semantic_uncertainty_runtime.wasm');
    const analyzer = new WasmSemanticAnalyzer();
    const result = await analyzer.analyze("Your prompt", "Your output");
    console.log('ℏₛ:', result.hbar_s);
}
```

## 📊 Performance Analysis

### Benchmarks

| Metric | Value | Target |
|--------|-------|--------|
| **Runtime** | <3ms | <10ms |
| **Memory Usage** | ~20MB | <50MB |
| **Build Time** | ~30sec | <2min |
| **Bundle Size** | ~366KB | <1MB |

### Optimization Techniques

#### Rust Optimizations
```rust
// Release build with optimizations
cargo build --release

// CPU-specific optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

// Link-time optimization
[profile.release]
lto = true
codegen-units = 1
```

#### WASM Optimizations
```bash
# Optimize WASM size
wasm-opt -O4 -o optimized.wasm semantic_uncertainty_runtime.wasm

# Compress WASM
gzip -9 semantic_uncertainty_runtime.wasm
```

### Performance Monitoring

```rust
// Performance metrics
let start_time = std::time::Instant::now();
let result = engine.analyze(prompt, output);
let processing_time = start_time.elapsed();

println!("Processing time: {:?}", processing_time);
```

## 🔗 Integration Guide

### Rust Integration

```rust
use semantic_uncertainty_runtime::StreamlinedEngine;

let engine = StreamlinedEngine::new();
let result = engine.analyze(
    "Explain AI safety",
    "AI safety involves ensuring AI systems are beneficial..."
);

println!("ℏₛ: {}", result.calibrated_hbar);
println!("Risk: {:?}", result.risk_level);
```

### Python Integration

```python
import requests

response = requests.post(
    "https://semantic-uncertainty-runtime-physics-production.mys628.workers.dev/api/v1/analyze",
    headers={"X-API-Key": "your-api-key"},
    json={
        "prompt": "What is semantic uncertainty?",
        "output": "Semantic uncertainty measures..."
    }
)

result = response.json()
print(f"ℏₛ: {result['raw_hbar']}")
print(f"Risk Level: {result['risk_level']}")
```

### JavaScript Integration

```javascript
import init, { WasmSemanticAnalyzer } from './semantic_uncertainty_runtime.js';

async function analyze() {
    await init('./semantic_uncertainty_runtime.wasm');
    const analyzer = new WasmSemanticAnalyzer();
    const result = await analyzer.analyze("Your prompt", "Your output");
    console.log('ℏₛ:', result.hbar_s);
}
```

### Command Line Interface

```bash
# Basic analysis
cargo run --bin semantic-uncertainty-runtime streamlined <prompt> <output>

# Available commands
cargo run --bin semantic-uncertainty-runtime demo
cargo run --bin semantic-uncertainty-runtime test
cargo run --bin semantic-uncertainty-runtime benchmark
```

## 🔬 Advanced Features

### Deterministic Analysis

```rust
// Hash-based deterministic analysis
let hash = calculate_deterministic_hash(prompt, output);
let result = engine.analyze_with_hash(prompt, output, hash);
```

### Token Economics

```rust
// Token analysis and cost estimation
let token_analysis = analyze_tokens(prompt, output);
let cost_estimate = estimate_cost(token_analysis);
let efficiency_score = calculate_efficiency(token_analysis);
```

### Risk Assessment

```rust
// Dynamic risk assessment
let risk_level = assess_risk(calibrated_hbar);
let confidence_interval = calculate_confidence_interval(result);
let collapse_probability = estimate_collapse_probability(result);
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_semantic_uncertainty_calculation

# Run with output
cargo test -- --nocapture
```

### Integration Tests

```bash
# Run integration tests
cargo test --test integration

# Test API endpoints
cargo test --test api_tests
```

### Performance Tests

```bash
# Run benchmarks
cargo run --bin semantic-uncertainty-runtime benchmark

# Performance profiling
cargo bench
```

## 🔧 Configuration

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

### Environment Variables

```bash
# .env
API_KEY=your-api-key
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
MAX_PROCESSING_TIME_MS=10
```

---

**For deployment information, see the [Deployment Guide](../guides/DEPLOYMENT.md).** 