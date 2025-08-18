# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

### Rust Workspace Commands
```bash
# Build all crates (default: debug mode)
cargo build

# Build in release mode (recommended for production)
cargo build --release

# Run tests for all crates
cargo test

# Test specific crate
cargo test -p common
cargo test -p preprompt
cargo test -p realtime
cargo test -p server

# Build and run server (local development)
cargo run -p server
# Runs on http://localhost:8080

# Build WASM for browser/worker deployment
cargo build --target wasm32-unknown-unknown --release -p preprompt

# Run specific test file
cargo test --test api_snapshots -p realtime
```

### Unified Scripts
```bash
# Comprehensive build with all components
./scripts/build.sh all --mode release --clean

# Build specific components
./scripts/build.sh core-engine    # Rust crates only
./scripts/build.sh wasm          # WASM distribution
./scripts/build.sh all           # Everything

# Run all tests
./scripts/test.sh all

# Run specific test suites
./scripts/test.sh core-engine
./scripts/test.sh integration --verbose

# Start full development stack
./scripts/start_stack.sh

# Calibrate failure law parameters
python scripts/calibrate_failure_law.py --dataset default --output_json calibration.json
```

### Python Frontend Commands
```bash
# Frontend dashboard (Streamlit)
cd frontend
pip install -r requirements.txt
python app.py
# or: streamlit run app.py

# Realtime dashboard
cd realtime/web-ui
python app.py
```

### Cloudflare Worker Development
```bash
cd cloudflare-workers
wrangler dev                    # Local development
wrangler deploy --env production # Deploy to production
```

## High-Level Architecture

### Multi-Crate Rust Workspace
This is a physics-inspired semantic uncertainty analysis system with the following structure:

#### Core Crates
- **`common/`**: Shared foundation types and mathematical utilities
  - `SemanticUncertaintyResult`, `RiskLevel`, `CalibrationMode`
  - Information theory calculators (entropy, divergences)
  - Free energy metrics and error types

- **`preprompt/`**: Batch analysis and metrics pipeline
  - `SemanticAnalyzer` for text analysis
  - WASM bindings for browser/worker deployment  
  - Compression, benchmarking, and API security analysis
  - Token analysis and semantic scoring modules

- **`realtime/`**: Live monitoring and firewall systems
  - Real-time auditing and response monitoring
  - Scalar firewalls with adaptive thresholds
  - OSS model adapters (Ollama, Mistral integration)
  - WebSocket and HTTP session APIs
  - Alias-ambiguity defense mechanisms

- **`server/`**: Binary crate that composes the realtime router
  - Axum HTTP server exposing realtime API endpoints
  - Health checks, WebSocket connections, session management

### Core Physics Equation
The system implements semantic uncertainty using:
```
ℏₛ = √(Δμ × Δσ)
```
Where:
- **Δμ (Precision)**: Jensen-Shannon divergence for semantic stability
- **Δσ (Flexibility)**: Kullback-Leibler divergence for adaptability
- **ℏₛ**: Combined semantic uncertainty metric

### Configuration System
- **`config/failure_law.json`**: Risk thresholds and failure law parameters (λ, τ)
- **`config/models.json`**: Model registry with per-model calibration constants and failure law overrides

### Diagnostic Suite Architecture
The system includes a comprehensive 5-step diagnostic protocol:

1. **Prompt Normalization**: Tokenizer-agnostic, semantically equivalent test sets
2. **Calibration Set Construction**: Tiered stress testing (basic facts → logical paradoxes → existential paradoxes)
3. **Information-Aligned Probing**: Detect brittle generalization and perturbation sensitivity
4. **Collapse Heatmaps**: 2D visualization of semantic terrain and failure topology
5. **Semantic Collapse Profiles**: Generate failure fingerprints (not performance rankings)

**Key Philosophy**: This is a diagnostic tool for understanding model cognition under stress, not a leaderboard system.

### API Architecture
#### Local Server (`cargo run -p server`)
- `GET /health`: System health and performance counters
- `GET /ws`: WebSocket endpoint for live monitoring
- `POST /session/start`: Create analysis session
- `POST /session/:id/close`: Close session

#### Cloudflare Worker (Edge deployment)
- `POST /api/v1/analyze`: Core semantic analysis endpoint
- `GET /api/v1/health`: Worker health check
- **Production**: `https://semanticuncertainty.com`
- **Staging**: `https://semantic-uncertainty-runtime-staging.mys628.workers.dev`

### Multi-Model Support
Supports multiple model architectures with calibrated κ constants:
- **encoder_only**: κ = 1.000 ± 0.035
- **decoder_only**: κ = 0.950 ± 0.089  
- **encoder_decoder**: κ = 0.900 ± 0.107
- **unknown**: κ = 1.040 ± 0.120

### Frontend Components
- **Streamlit Dashboard**: Real-time visualization and model comparison
- **Static Frontend**: HTML/JS interface for Cloudflare deployment
- **Multi-model Interface**: Concurrent analysis across different models

## Development Workflow

### Local Development Setup
1. **Prerequisites**: Rust 1.70+, Python 3.8+, Node 18+, wasm-pack
2. **Clone and build**: `cargo build --release`
3. **Start server**: `cargo run -p server` 
4. **Run dashboard**: `cd frontend && python app.py`
5. **Test API**: Visit `http://localhost:8080/health`

### Testing Strategy
- **Unit Tests**: `cargo test` for Rust components
- **Integration Tests**: `./scripts/test.sh integration`
- **API Snapshots**: Automated API response validation
- **WASM Tests**: Browser-based WASM functionality testing

### Deployment Targets
- **Local**: Direct Rust server for development
- **Edge**: Cloudflare Workers with WASM runtime (`wrangler deploy --env production`)
- **Cloud**: Railway/similar for Streamlit dashboards
- **WASM Build**: `cargo build --target wasm32-unknown-unknown --release` + `wasm-bindgen` for browser integration

## Important Implementation Notes

### Risk Classification System
- **Critical** (ℏₛ < 0.8): Immediate attention required
- **Warning** (0.8 ≤ ℏₛ < 1.2): Monitor closely  
- **Safe** (ℏₛ ≥ 1.2): Normal operation

### Performance Considerations
- Use release builds for accurate uncertainty measurements
- WASM targets optimize for edge deployment constraints
- Fisher information calculations can be computationally expensive
- Hash embeddings available as fast approximation for large texts
- **Target Response Time**: <200ms for API endpoints
- **Concurrency Support**: Multi-threaded request processing with rate limiting

### Security Features
- Stateless design with input validation
- API key management in preprompt crate
- CORS support for cross-origin requests
- Bounded caches to prevent memory exhaustion

## Important Dataset and Calibration Information

### Supported Datasets for Calibration
- **TruthfulQA**: Loaded via `datasets` library for generation tasks
- **HaluEval**: Fetched from GitHub with configurable task subsets (qa, dialogue, summarization)
- **Default**: Built-in toy dataset for testing

### Calibration Workflow
```bash
# Basic calibration with default dataset
python scripts/calibrate_failure_law.py --dataset default

# TruthfulQA with custom parameters
python scripts/calibrate_failure_law.py --dataset truthfulqa --max_samples 2000 --concurrency 8

# HaluEval QA subset
python scripts/calibrate_failure_law.py --dataset halueval --halueval_task qa --max_samples 500
```

### Key Calibration Metrics
- **ROC-AUC**: Discrimination between failing and passing outputs
- **Brier Score**: Mean squared error of predicted failure probability
- **ECE/MCE**: Calibration errors with uniform binning
- **Welch's t-test**: Statistical difference between failure groups