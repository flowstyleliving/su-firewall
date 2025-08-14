# 🌌 Semantic Uncertainty Runtime

Real-time analysis of AI model uncertainty using physics-inspired metrics.

## 🚀 Quick Start

### Live (edge)
- Dashboard: https://semanticuncertainty.com
- API (Cloudflare Worker): https://semanticuncertainty.com/api/v1
  - Health: https://semanticuncertainty.com/api/v1/health
  - Analyze: https://semanticuncertainty.com/api/v1/analyze

### Local (Rust server)
```bash
# Run the realtime HTTP server with WS and session routes
cargo run -p server
# Visit http://localhost:8080/health
```

## 🧠 Core Metric

```
ℏₛ = √(Δμ × Δσ)
```
- Δμ (Precision): Jensen–Shannon divergence
- Δσ (Flexibility): KL divergence or Fisher-information derived variance

## 🏗️ Architecture (multi-crate workspace)

```
semantic-uncertainty-runtime/
├── common/         # Shared types, errors, math (information theory, free energy)
├── preprompt/      # Pre-prompt batch analysis, metrics pipeline, WASM/Python FFI
├── realtime/       # Live monitoring, firewalls, adapters, WS/session API
├── server/         # Binary crate: composes realtime router and runs Axum HTTP
├── cloudflare-workers/ # Edge worker for public API
└── docs/, scripts/, frontend/, ...
```

- common: `SemanticUncertaintyResult`, `RiskLevel`, `CalibrationMode`, `RequestId`, `SemanticError`, `InformationTheoryCalculator`, `FreeEnergyMetrics`.
- preprompt: `SemanticAnalyzer`, metrics pipeline, compression, benchmarking, API security analyzer, WASM bindings.
- realtime: live auditors, dashboards, scalar firewalls, OSS/Mistral adapters, alias-ambiguity defense, HTTP/WS session API.
- server: runs an Axum server that exposes the realtime router locally.

## 🔌 APIs

### Cloudflare Worker (edge)
- POST `/api/v1/analyze`: semantic analysis for prompt-output pairs
- GET `/api/v1/health`: worker health

Example:
```bash
curl -X POST "https://semanticuncertainty.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is quantum computing?",
    "output": "Quantum computing uses quantum bits to process information",
    "method": "jsd-kl"
  }'
```

### Local Realtime Server (Axum)
- GET `/health`: JSON health and counters
- GET `/ws`: WebSocket (echo placeholder; attach live hooks as needed)
- POST `/session/start`: start a session, returns `session_id`
- POST `/session/:id/close`: close a session by id

Health response (example):
```json
{
  "status": "ok",
  "uptime_ms": 12345,
  "counters": {
    "requests": 10,
    "ws_connections": 1,
    "sessions_active": 0
  }
}
```

## 📦 Crates

- common
  - serde, thiserror, ndarray, chrono, uuid, anyhow
- preprompt
  - depends on `common`; features: `wasm`
- realtime
  - depends on `common`, `preprompt`; features: `api` (Axum)
  - bounded FIM cache via `lru`
- server
  - depends on `realtime` (with `api`), `preprompt`, `axum`, `tokio`, `tracing-subscriber`

## 🧪 Methods

- JSD/KL: Jensen–Shannon + Kullback–Leibler divergences
- Fisher Information: directional FIM and information geometry estimators
- Both: side-by-side comparison for calibration studies

## 📝 Development

Prereqs: Rust 1.70+, Node 18+, Python 3.8+, Wrangler, Railway

- Local realtime server:
```bash
cargo run -p server
```
- Cloudflare worker:
```bash
cd cloudflare-workers
wrangler deploy --env production
```

## 🔒 Security
- Stateless by default; input validation and CORS
- API key management and security analysis available in `preprompt`

## 📈 Performance
- WASM acceleration for analyzer paths
- Async runtime (`tokio`) and structured logging (`tracing`)

## 🤝 Contributing
1. Fork and branch
2. Implement feature/tests
3. PR welcome

## 📄 License
MIT. See `LICENSING.md`.