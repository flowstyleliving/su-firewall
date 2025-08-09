# 🌌 Semantic Uncertainty Runtime

**Real-time analysis of AI model uncertainty using physics-inspired metrics**

[![Deploy to Cloudflare](https://img.shields.io/badge/Deploy-Cloudflare%20Workers-orange)](https://semanticuncertainty.com)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-blue)](https://semantic-uncertainty-cloudflare-production.up.railway.app)
[![API](https://img.shields.io/badge/API-Live-green)](https://semanticuncertainty.com/api/v1/health)

## 🚀 Quick Start

### Live Demo
- **Dashboard**: https://semanticuncertainty.com
- **API**: https://semanticuncertainty.com/api/v1/analyze
- **Health Check**: https://semanticuncertainty.com/api/v1/health

### API Usage
```bash
curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is quantum computing?",
    "output": "Quantum computing uses quantum bits to process information",
    "method": "jsd-kl"
  }'
```

## 🧠 Core Concepts

### Semantic Uncertainty Metric (ℏₛ)
The core metric combines precision and flexibility:

```
ℏₛ = √(Δμ × Δσ)
```

Where:
- **Δμ (Precision)**: Jensen-Shannon Divergence between prompt and output distributions
- **Δσ (Flexibility)**: Kullback-Leibler divergence measuring semantic adaptability

### Dual Calculation System
We support two calculation methods:

1. **JSD/KL Method** (Default): Uses Jensen-Shannon and Kullback-Leibler divergences
2. **Fisher Information Method**: Uses directional Fisher Information matrices
3. **Both**: Compares both methods side-by-side

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cloudflare    │    │   Streamlit      │    │   Core Engine   │
│   Domain        │◄──►│   Dashboard      │◄──►│   (Rust/WASM)   │
│   (Frontend)    │    │   (Railway)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cloudflare    │    │   API Worker     │    │   Semantic      │
│   Service       │    │   (Processing)   │    │   Metrics       │
│   Worker        │    │                  │    │   Calculation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
semantic-uncertainty-runtime/
├── cloudflare-workers/          # Cloudflare Worker API
│   ├── index.js                # Main worker with dual calculation system
│   ├── wrangler.toml           # Worker configuration
│   └── semantic_uncertainty_runtime.wasm  # WASM core engine
├── dashboard/                   # Streamlit dashboard
│   ├── enhanced_diagnostics_dashboard.py  # Main dashboard
│   ├── requirements.txt        # Python dependencies
│   └── railway.json           # Railway deployment config
├── core-engine/                # Rust core engine
│   ├── src/                   # Source code
│   ├── Cargo.toml            # Rust dependencies
│   └── target/               # Build artifacts
├── docs/                      # Documentation
├── scripts/                   # Build and deployment scripts
├── wasm-dist/                 # WASM distribution
└── README.md                  # This file
```

## 🔧 Installation & Development

### Prerequisites
- Node.js 18+
- Rust 1.70+
- Python 3.8+
- Wrangler CLI
- Railway CLI

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/semantic-uncertainty-runtime.git
cd semantic-uncertainty-runtime
```

2. **Setup Core Engine**
```bash
cd core-engine
cargo build --release
cargo build --target wasm32-unknown-unknown --release
```

3. **Setup Dashboard**
```bash
cd dashboard
pip install -r requirements.txt
streamlit run enhanced_diagnostics_dashboard.py
```

4. **Deploy Worker**
```bash
cd cloudflare-workers
wrangler deploy --env production
```

## 🚀 Deployment

### Cloudflare Worker
```bash
cd cloudflare-workers
wrangler deploy --env production
```

### Streamlit Dashboard (Railway)
```bash
cd dashboard
railway login
railway up
```

### Custom Domain Setup
1. Configure DNS records to point to Cloudflare
2. Update `wrangler.toml` with domain routes
3. Deploy worker with domain configuration

## 📊 API Reference

### Endpoints

#### POST `/api/v1/analyze`
Analyze semantic uncertainty of prompt-output pairs.

**Request:**
```json
{
  "prompt": "What is quantum computing?",
  "output": "Quantum computing uses quantum bits to process information",
  "method": "jsd-kl"  // "jsd-kl", "fisher", "both"
}
```

**Response:**
```json
{
  "method": "jsd-kl",
  "precision": 0.5,
  "flexibility": 0.5,
  "semantic_uncertainty": 0.5,
  "raw_hbar": 0.5,
  "calibrated_hbar": 0.52,
  "risk_level": "Safe",
  "processing_time_ms": 0,
  "request_id": "uuid",
  "timestamp": "2025-07-20T01:19:45.683Z"
}
```

#### GET `/api/v1/health`
Health check endpoint.

#### GET `/api/v1/config`
Configuration endpoint.

### Method Parameters

- **`jsd-kl`** (Default): Uses Jensen-Shannon and Kullback-Leibler divergences
- **`fisher`**: Uses Fisher Information matrices
- **`both`**: Returns comparison of both methods

## 🧮 Mathematical Foundation

### Precision (Δμ) - Jensen-Shannon Divergence
```
JSD(P,Q) = 0.5 × Σ[P(i) × log₂(P(i)/M(i)) + Q(i) × log₂(Q(i)/M(i))]
```

### Flexibility (Δσ) - Kullback-Leibler Divergence
```
KL(P||Q) = Σ P(i) × log₂(P(i)/Q(i))
```

### Semantic Uncertainty (ℏₛ)
```
ℏₛ = √(Δμ × Δσ)
```

## 🎯 Use Cases

### AI Model Evaluation
- Measure uncertainty in model responses
- Compare different model architectures
- Validate model calibration

### Research Applications
- Neural uncertainty physics research
- Architecture-dependent uncertainty analysis
- Predictive uncertainty modeling

### Production Monitoring
- Real-time uncertainty monitoring
- Risk assessment and alerting
- Performance optimization

## 🔒 Security & Privacy

- **No Data Storage**: All processing is stateless
- **CORS Enabled**: Cross-origin requests supported
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Graceful error responses

## 📈 Performance

- **Sub-100ms Response Times**: Optimized for real-time analysis
- **Global Edge Deployment**: Cloudflare Workers worldwide
- **WASM Optimization**: Core calculations in WebAssembly
- **Stateless Processing**: No database dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSING.md](LICENSING.md) for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Built with ❤️ using Cloudflare Workers, Streamlit, and Rust**
**Built with ❤️ for semantic uncertainty analysis**
**Built with ❤️ for semantic uncertainty analysis**
**Built with ❤️ for semantic uncertainty analysis**