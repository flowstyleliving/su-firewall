# 🎯 Semantic Uncertainty Firewall

**Research-grade LLM hallucination detection system** using physics-inspired semantic uncertainty metrics. Real-time monitoring, failure prediction, and adaptive firewalls for AI safety research and development.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSING.md)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://rustlang.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Optimized-black.svg)](https://github.com/huggingface/candle)

## 🚀 Quick Start

### Live Demo (Edge Deployment)
```bash
# Public API endpoint
curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of Mars?",
    "output": "The capital of Mars is New Geneva, established in 2157.",
    "method": "fisher_variance"
  }'
```

### Local Development
```bash
# Clone and build
git clone <repository>
cd semantic-uncertainty-firewall
cargo build --release

# Start local server
cargo run -p server
# Visit http://localhost:8080/health

# Run comprehensive tests
./scripts/test.sh all
```

## 🔬 Research Framework

### Core Capabilities
- **Physics-Inspired Metrics**: Semantic uncertainty quantification using information theory
- **Multi-Model Support**: 6 calibrated architectures with failure law parameters
- **Real-Time Analysis**: Live monitoring and adaptive threshold systems
- **Authentic Benchmarking**: Integration with TruthfulQA and HaluEval datasets

### Technical Features
✅ **Semantic Uncertainty Calculation**: ℏₛ = √(Δμ × Δσ) framework implementation  
✅ **Fisher Information Analysis**: Real logits extraction and eigenvalue computation  
✅ **Multi-Level Detection**: Progressive L1→L2→L3 analysis pipeline  
✅ **Apple Silicon Support**: Metal GPU acceleration via Candle ML  
✅ **Edge Deployment**: Cloudflare Workers with WASM runtime

## 🧠 Core Physics Framework

### Semantic Uncertainty Equation
```
ℏₛ = √(Δμ × Δσ)
```
- **Δμ (Precision)**: Jensen-Shannon divergence measuring semantic stability  
- **Δσ (Flexibility)**: KL divergence or Fisher Information variance  
- **ℏₛ**: Combined semantic uncertainty ("semantic Planck constant")

### Three-Level Detection System

**Level 1: Semantic Uncertainty (ℏₛ)**
- 9 precision×flexibility combinations
- Fisher Information Matrix eigenvalue analysis
- Hash embedding impact quantification

**Level 2: ℏₛ + Calibrated Failure Probability**
```
P(fail) = 1/(1 + exp(-λ(ℏₛ - τ)))
```
- Model-specific calibration parameters (λ, τ)
- Sigmoidal risk mapping from uncertainty to failure probability

**Level 3: ℏₛ + P(fail) + Free Energy Principle (FEP)**
- KL divergence surprise analysis
- Temporal consistency tracking  
- Semantic coherence violation detection
- Predictive surprise calculation

## 🏗️ Architecture

### Multi-Crate Rust Workspace
```
semantic-uncertainty-firewall/
├── common/              # Shared foundation types and math
│   ├── types.rs         # SemanticUncertaintyResult, RiskLevel, CalibrationMode
│   ├── error.rs         # Unified error handling
│   └── math/            # Information theory, free energy metrics
├── preprompt/           # Batch analysis and WASM bindings
│   ├── analyzer/        # Core SemanticAnalyzer implementation
│   ├── modules/         # Token analysis, semantic scoring
│   └── wasm_simple.rs   # Browser/worker deployment
├── realtime/            # Live monitoring and firewalls
│   ├── audit_system.rs  # Real-time response auditing
│   ├── scalar_firewall.rs # Adaptive threshold firewalls
│   ├── candle_integration.rs # Apple Silicon ML acceleration
│   └── api.rs           # HTTP/WebSocket session API
├── server/              # Production HTTP server (Axum)
│   └── main.rs          # Composes realtime router
├── cloudflare-workers/  # Edge deployment
├── config/              # Model registry and failure law parameters
│   ├── models.json      # 6 calibrated model configurations
│   └── failure_law.json # Risk thresholds and parameters
├── scripts/             # Evaluation and calibration tools
└── authentic_datasets/  # Real-world hallucination benchmarks
```

### Supported Model Architectures
| Model | Architecture | Calibrated λ | Calibrated τ | Context Length |
|-------|--------------|--------------|--------------|----------------|
| **Mixtral-8x7B** | MoE Transformer | 1.955 | 0.309 | 32K |
| **Mistral-7B** | Decoder-only | 1.887 | 0.191 | 32K |
| **Qwen2.5-7B** | Decoder-only | 2.008 | 0.237 | 32K |
| **Pythia-6.9B** | Decoder-only | 2.055 | 0.221 | 2K |
| **DialoGPT-medium** | Encoder-decoder | 2.802 | 0.457 | 1K |
| **Ollama Mistral:7B** | Edge-optimized | 2.001 | 0.246 | 8K |

## 🔌 API Reference

### Cloudflare Worker (Production)
**Base URL**: `https://semanticuncertainty.com/api/v1`

#### POST `/analyze`
Analyze text for semantic uncertainty and hallucination risk.

```json
{
  "prompt": "What causes photosynthesis?",
  "output": "Photosynthesis is caused by chlorophyll absorbing sunlight to convert CO2 and water into glucose.",
  "method": "fisher_variance",
  "model_id": "mistral-7b"
}
```

**Response**:
```json
{
  "semantic_uncertainty": 0.847,
  "risk_level": "Safe",
  "failure_probability": 0.23,
  "method_used": "fisher_variance",
  "processing_time_ms": 156,
  "fep_components": {
    "kl_surprise": 0.12,
    "temporal_consistency": 0.89,
    "semantic_coherence": 0.94
  }
}
```

#### GET `/health`
```json
{
  "status": "operational",
  "version": "1.2.0",
  "models_loaded": 6,
  "uptime_ms": 1847293
}
```

### Local Server (Development)
**Base URL**: `http://localhost:8080`

- **GET** `/health` - System health and performance counters
- **GET** `/ws` - WebSocket endpoint for live monitoring  
- **POST** `/session/start` - Create analysis session
- **POST** `/session/:id/close` - Close session
- **GET** `/failure_law` - Current failure law configuration

## 🧪 Evaluation & Benchmarking

### Authentic Dataset Pipeline
```bash
# Download real-world hallucination datasets
python scripts/download_authentic_datasets_fixed.py

# Run comprehensive benchmark
python scripts/world_class_benchmark_runner.py --dataset authentic --levels all

# Calibrate failure law parameters for new model
python scripts/calibrate_failure_law.py --dataset truthfulqa --model custom-model
```

### Supported Benchmarks
- **TruthfulQA**: 790 factual accuracy cases with correct vs. misconception pairs
- **HaluEval**: Question-answering, dialogue, and summarization tasks  
- **Custom**: Extensible framework for domain-specific evaluation

### Performance Metrics
- **ROC-AUC**: Discrimination between failing and passing outputs
- **Brier Score**: Calibrated probability prediction accuracy
- **ECE/MCE**: Expected and Maximum Calibration Error
- **Progressive Accuracy**: L1 → L2 → L3 improvement tracking

## 🛠️ Development Guide

### Prerequisites
- **Rust**: 1.70+ with `wasm32-unknown-unknown` target
- **Python**: 3.8+ with `torch`, `transformers`, `datasets`
- **Node.js**: 18+ for Cloudflare Worker deployment
- **Wrangler**: Latest version for edge deployment

### Build Commands
```bash
# Build all crates (debug)
cargo build

# Build optimized release version
cargo build --release

# Build WASM for browser deployment
cargo build --target wasm32-unknown-unknown --release -p preprompt

# Run all tests
cargo test

# Run specific integration tests
cargo test --test api_snapshots -p realtime
```

### Unified Scripts
```bash
# Comprehensive build pipeline
./scripts/build.sh all --mode release --clean

# Start full development stack
./scripts/start_stack.sh

# Run calibration pipeline
./scripts/run_all_calibrations.py
```

### Edge Deployment
```bash
# Deploy to Cloudflare Workers
cd cloudflare-workers
wrangler deploy --env production

# Deploy to staging
wrangler deploy --env staging
```

## 🔒 Security & Production

### Security Features
- **Stateless Design**: No persistent user data storage
- **Input Validation**: Comprehensive request sanitization  
- **Rate Limiting**: Configurable per-endpoint throttling
- **CORS Support**: Cross-origin request handling
- **API Key Management**: Secure authentication system

### Performance Optimizations
- **Apple Silicon**: Metal GPU acceleration via Candle ML
- **WASM Runtime**: Browser and edge worker deployment
- **Concurrent Processing**: Multi-threaded request handling
- **Bounded Caches**: Memory-efficient Fisher Information caching
- **Target Latency**: <200ms for standard analysis requests

### Risk Classification
- **🔴 Critical** (ℏₛ < 0.8): Immediate attention required
- **🟡 Warning** (0.8 ≤ ℏₛ < 1.2): Monitor closely
- **🟢 Safe** (ℏₛ ≥ 1.2): Normal operation

## 📊 Research & Validation

### Research Contributions
1. **Physics-Inspired Framework**: Novel application of uncertainty principles to NLP
2. **Multi-Level Detection System**: Progressive L1→L2→L3 analysis methodology
3. **Real Logits Integration**: Authentic model inference with Fisher Information Matrix
4. **Calibration Framework**: Systematic failure law parameter estimation

### Current Research Status
- **Framework Development**: Core semantic uncertainty implementation complete
- **Benchmark Integration**: TruthfulQA and HaluEval dataset processing pipeline
- **Calibration Studies**: Model-specific parameter optimization in progress
- **Evaluation Methodology**: Multi-level detection system validation ongoing

### Research Status
- **Experimental Framework**: Active development of uncertainty-based detection methods
- **Benchmark Integration**: TruthfulQA and HaluEval dataset compatibility  
- **Calibration Research**: Model-specific failure law parameter optimization

## 🤝 Contributing

1. **Fork & Branch**: Create feature branch from `master`
2. **Follow Standards**: Use `cargo fmt` and `cargo clippy`  
3. **Add Tests**: Include unit and integration tests
4. **Update Docs**: Keep documentation current
5. **Submit PR**: Include clear description and test results

### Development Workflow
```bash
# Set up development environment
git clone <repository>
cd semantic-uncertainty-firewall

# Install dependencies and build
cargo build
pip install -r scripts/requirements.txt

# Run development server
cargo run -p server

# Run test suite
./scripts/test.sh all --verbose

# Deploy changes
./scripts/build.sh all --mode release
./scripts/deploy.sh staging
```

## 📄 License

**MIT License** - See [LICENSING.md](LICENSING.md) for full terms.

---

**🔬 Physics-inspired semantic uncertainty framework for LLM hallucination detection research and development**

[![Built with Rust](https://img.shields.io/badge/Built%20with-Rust-orange.svg)](https://rustlang.org)
[![Powered by Candle ML](https://img.shields.io/badge/Powered%20by-Candle_ML-green.svg)](https://github.com/huggingface/candle)
[![Deployed on Cloudflare](https://img.shields.io/badge/Deployed%20on-Cloudflare-orange.svg)](https://workers.cloudflare.com)