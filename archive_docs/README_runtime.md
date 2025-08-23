# 🧠 Semantic Uncertainty Runtime

A high-performance Rust runtime for computing semantic uncertainty metrics in AI outputs. This is the **machine-native firewall** for cognition, providing real-time analysis of semantic coherence and collapse risk.

## ⚡ Features

- **Quantum-Inspired Metrics**: Computes ℏₛ(C) = sqrt(I(C;W) * JS_max) for semantic uncertainty
- **Multi-Modal Analysis**: 
  - Δμ (semantic precision) from embedding entropy
  - Δσ (semantic flexibility) from Jensen-Shannon divergence
- **High Performance**: <100ms analysis time per prompt-output pair
- **Multiple Interfaces**: CLI, REST API, Python bindings
- **ONNX Integration**: Supports sentence embedding models (MiniLM, BGE, etc.)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone <repository-url>
cd semantic-uncertainty-runtime
cargo build --release
```

### 2. Download an ONNX Model

Place your sentence transformer ONNX model in the `models/` directory:

```bash
# Example: Download a lightweight sentence transformer
wget -O models/sentence_transformer.onnx <model-url>
```

### 3. Run Analysis

```bash
# Single analysis
./target/release/semantic-uncertainty-runtime analyze \
  "What is artificial intelligence?" \
  "AI is a branch of computer science that aims to create intelligent machines."

# Interactive mode
./target/release/semantic-uncertainty-runtime interactive

# Start API server
./target/release/semantic-uncertainty-runtime server 8080
```

## 📊 Output Metrics

### Core Measurements

- **ℏₛ (Quantum Uncertainty)**: Combined metric indicating overall semantic uncertainty
- **Δμ (Semantic Precision)**: Inverse of embedding entropy - higher values indicate more focused semantics
- **Δσ (Semantic Flexibility)**: Jensen-Shannon divergence between prompt and output embeddings
- **Collapse Risk**: Boolean flag when ℏₛ falls below threshold (default: 1.0)

### Example Output

```json
{
  "hbar_s": 1.2547,
  "delta_mu": 0.8234,
  "delta_sigma": 0.4521,
  "collapse_risk": false
}
```

## 🔧 Usage

### Command Line Interface

```bash
# Analyze single prompt-output pair
semantic-uncertainty-runtime analyze <prompt> <output> [model_path]

# Start REST API server
semantic-uncertainty-runtime server [port] [model_path]

# Batch process JSONL file
semantic-uncertainty-runtime batch <input.jsonl> [model_path]

# Performance benchmark
semantic-uncertainty-runtime benchmark [model_path]

# Interactive mode
semantic-uncertainty-runtime interactive [model_path]
```

### REST API

Start the server:
```bash
cargo run --features api -- server 8080
```

Make requests:
```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "output": "Quantum computing uses quantum mechanical phenomena...",
    "config": {
      "collapse_threshold": 0.8
    }
  }'
```

### Python Bindings

Build with Python support:
```bash
pip install maturin
maturin develop --features python
```

Use in Python:
```python
import semantic_uncertainty_runtime as sur

# Quick analysis
result = sur.analyze(
    "What is machine learning?",
    "Machine learning is a subset of AI...",
    model_path="models/sentence_transformer.onnx"
)

print(f"Quantum uncertainty: {result.hbar_s}")
print(f"Collapse risk: {result.collapse_risk}")

# Advanced usage
config = sur.SemanticConfig(
    model_path="models/sentence_transformer.onnx",
    collapse_threshold=0.9,
    perturbation_count=15
)

analyzer = sur.SemanticAnalyzer(config)
result = analyzer.analyze(prompt, output)
```

## 🏗️ Architecture

```
src/
├── lib.rs          # Core computation logic and algorithms
├── api.rs          # REST API using Axum (feature: api)
├── ffi.rs          # Python bindings using PyO3 (feature: python)
└── main.rs         # CLI binary with multiple modes

models/
└── sentence_transformer.onnx  # ONNX model file (user-provided)
```

### Core Algorithm

1. **Embedding Generation**: Text → ONNX sentence transformer → dense vectors
2. **Entropy Calculation**: Δμ = 1 - (H(embedding) / H_max)
3. **JS Divergence**: Δσ = sqrt(JS(prompt_emb, output_emb))
4. **Mutual Information**: I(C;W) ≈ correlation(prompt_emb, output_emb)
5. **Perturbation Analysis**: Generate variants, compute max JS divergence
6. **Quantum Metric**: ℏₛ = sqrt(I(C;W) * JS_max)

## ⚙️ Configuration

### SemanticConfig Options

```rust
SemanticConfig {
    model_path: String,           // Path to ONNX model
    collapse_threshold: f32,      // Risk threshold (default: 1.0)
    perturbation_count: usize,    // Number of perturbations (default: 10)
    max_sequence_length: usize,   // Max token length (default: 512)
}
```

### Build Features

- `default`: Core functionality only
- `api`: Include REST API server (Axum)
- `python`: Include Python bindings (PyO3)

```bash
# Build with all features
cargo build --features api,python

# Build API-only
cargo build --features api
```

## 🧪 Testing

```bash
# Run unit tests
cargo test

# Run with specific features
cargo test --features api,python

# Benchmark performance
cargo run --release -- benchmark
```

## 📈 Performance

- **Target**: <100ms per analysis
- **Optimizations**: 
  - Parallel perturbation processing with Rayon
  - Optimized ONNX runtime settings
  - Efficient tensor operations with ndarray
  - Memory-mapped model loading

### Benchmark Results

```bash
$ cargo run --release -- benchmark
🏃 Benchmarking semantic uncertainty runtime...
Iteration 1: 245ms for 5 analyses
Iteration 2: 198ms for 5 analyses
...
📈 Benchmark Results:
━━━━━━━━━━━━━━━━━━━━
Average batch time: 210ms
Average per analysis: 42ms
✅ Performance target achieved (<100ms per analysis)
```

## 🔬 Mathematical Foundation

### Quantum-Inspired Uncertainty

The core metric ℏₛ(C) combines information theory with quantum mechanical principles:

```
ℏₛ(C) = √(I(C;W) × JS_max)
```

Where:
- `I(C;W)`: Mutual information between context C and output W
- `JS_max`: Maximum Jensen-Shannon divergence across perturbations
- The square root provides dimensional consistency similar to quantum uncertainty relations

### Semantic Precision (Δμ)

Measures focused vs. dispersed semantic content:

```
Δμ = 1 - H(embedding) / H_max
```

Higher values indicate more semantically precise outputs.

### Semantic Flexibility (Δσ)

Quantifies semantic distance between prompt and output:

```
Δσ = √JS(P(prompt), P(output))
```

Where JS is Jensen-Shannon divergence between probability distributions.

## 🛠️ Development

### Adding New Models

1. Convert your sentence transformer to ONNX format
2. Place in `models/` directory
3. Update model path in configuration

### Custom Metrics

Extend the `SemanticAnalyzer` to add new uncertainty measurements:

```rust
impl SemanticAnalyzer {
    fn compute_custom_metric(&self, embedding: &Array1<f32>) -> Result<f32> {
        // Your custom calculation here
        Ok(custom_value)
    }
}
```

## 📦 Dependencies

- **onnxruntime**: ONNX model inference
- **ndarray**: Efficient tensor operations  
- **nalgebra**: Linear algebra computations
- **rayon**: Parallel processing
- **serde**: Serialization
- **tokio**: Async runtime
- **axum**: HTTP server (optional)
- **pyo3**: Python bindings (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure benchmarks pass (<100ms requirement)
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 Use Cases

- **LLM Safety**: Real-time monitoring of model outputs
- **Content Filtering**: Detect semantically inconsistent responses
- **Quality Assurance**: Automated analysis of AI-generated content
- **Research**: Study semantic uncertainty patterns in different models
- **Production Monitoring**: Alert on potential model degradation

---

**Built for real-time inference safety in LLM pipelines** 🚀 