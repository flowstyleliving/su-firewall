# 🚀 Mistral 7B + Live Uncertainty Auditing System

Complete real-time uncertainty quantification system for Mistral 7B with live frontend dashboard.

## 🎯 **What This System Provides**

- **🤖 Mistral 7B Integration**: Full logit access for precise uncertainty calculation
- **🔍 Live Auditing**: Real-time uncertainty quantification using advanced information-geometric methods  
- **📊 Live Dashboard**: Beautiful React frontend with real-time charts and alerts
- **🚨 Smart Alerts**: Automatic detection of uncertainty spikes and problematic generations
- **📡 WebSocket Streaming**: Real-time updates as tokens are generated
- **🎛️ Model Agnostic**: The auditing system works with ANY LLM (not just Mistral)

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│   React Frontend│◄──►│  Python Mistral    │◄──►│ Rust Audit API  │
│   (Port 3000)   │    │  Server            │    │ (Port 8080)     │
│                 │    │  (Ports 5000/8765) │    │                 │
│ • Live Charts   │    │ • Mistral 7B       │    │ • UQ Engine     │
│ • Real-time UQ  │    │ • Token Streaming  │    │ • Alert System  │
│ • Alerts        │    │ • WebSocket Server │    │ • Session Mgmt  │
└─────────────────┘    └────────────────────┘    └─────────────────┘
```

## 📋 **Prerequisites**

### System Requirements
- **RAM**: 16GB+ (32GB recommended for Mistral 7B)
- **GPU**: 8GB+ VRAM (optional but recommended)
- **Storage**: 20GB+ free space
- **OS**: Linux/macOS (Windows with WSL2)

### Software Dependencies
- **Rust** (latest stable)
- **Python 3.8+** 
- **Node.js 16+** and npm
- **CUDA** (optional, for GPU acceleration)

## 🚀 **Quick Start (Automated)**

### Option 1: One-Command Setup
```bash
# Make the startup script executable
chmod +x start_mistral_live_audit.sh

# Run the complete system
./start_mistral_live_audit.sh
```

This will:
1. ✅ Check prerequisites and port availability
2. 🔧 Build the Rust audit API
3. 📦 Install Python and Node.js dependencies  
4. 🚀 Start all services in the correct order
5. ⏳ Wait for Mistral 7B model to load
6. 📊 Open the dashboard at `http://localhost:3000`

### What to Expect
- **First run**: 10-30 minutes (downloads Mistral 7B model ~13GB)
- **Subsequent runs**: 2-5 minutes (model cached)
- **Browser opens**: Live dashboard at `http://localhost:3000`

## 🛠️ **Manual Setup (Step by Step)**

### Step 1: Install Dependencies

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend/uncertainty-dashboard
npm install
cd ../..
```

### Step 2: Build Rust Audit API

```bash
cd core-engine
cargo build --release --features="http-api,websocket"
cd ..
```

### Step 3: Start Services

```bash
# Terminal 1: Start Rust audit API
cd core-engine
cargo run --release --features="http-api,websocket" --bin audit_server -- --port 8080

# Terminal 2: Start Python Mistral server  
python3 mistral_audit_server.py

# Terminal 3: Start React frontend
cd frontend/uncertainty-dashboard
npm run dev -- --port 3000
```

## 🎮 **Usage Guide**

### Basic Usage
1. **Open Dashboard**: Navigate to `http://localhost:3000`
2. **Enter Prompt**: Type your prompt in the text area
3. **Configure Settings**: Adjust temperature, top-p, max tokens
4. **Start Generation**: Click "Start Generation" 
5. **Watch Live UQ**: See real-time uncertainty metrics and alerts

### Understanding the Dashboard

#### 📊 **Real-Time Metrics**
- **Current Uncertainty**: Live ℏₛ value for the latest token
- **Average Uncertainty**: Session average uncertainty  
- **Risk Level**: Safe/Warning/Critical/Emergency
- **Tokens Processed**: Number of tokens generated

#### 📈 **Charts**
- **Uncertainty Over Time**: Live area chart showing uncertainty trends
- **Token History**: Recent tokens with probabilities and uncertainty values

#### 🚨 **Alerts**
- **High Uncertainty**: When ℏₛ exceeds threshold (default: 2.5)
- **Uncertainty Spike**: Sudden increases in uncertainty
- **Low Confidence**: Tokens with probability < 0.3
- **Anomalous Patterns**: Statistical outliers

### Example Prompts to Try

```
# High certainty (factual)
"The capital of France is"

# Medium uncertainty (explanatory)  
"Explain quantum computing in simple terms"

# High uncertainty (speculative)
"What will technology look like in 2090?"

# Technical uncertainty (complex reasoning)
"Derive the mathematical relationship between Fisher Information and entropy"
```

## ⚙️ **Configuration**

### Mistral Model Settings
```python
# In mistral_audit_server.py
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Change model
device = "auto"  # "cpu", "cuda", or "auto"
```

### Audit Thresholds
```python
# In mistral_audit_server.py
uncertainty_threshold = 2.5    # Alert threshold
spike_threshold = 1.5          # Spike detection
confidence_threshold = 0.3     # Low confidence alert
```

### Frontend Settings
```typescript
// In frontend/uncertainty-dashboard/src/app/page.tsx
const WS_URL = 'ws://localhost:8765'  // WebSocket URL
const MAX_TOKENS_DEFAULT = 256        // Default generation length
```

## 🔧 **Advanced Features**

### Using Different Models

The system works with any Hugging Face transformers model:

```python
# In mistral_audit_server.py, change:
model_name = "microsoft/DialoGPT-large"
# or
model_name = "meta-llama/Llama-2-7b-chat-hf"
# or any compatible model
```

### API Integration

#### HTTP API
```bash
# Start generation
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain AI", "max_tokens": 100}'

# Health check
curl http://localhost:5000/health
```

#### WebSocket API
```javascript
const ws = new WebSocket('ws://localhost:8765');

// Start generation
ws.send(JSON.stringify({
  type: "generate",
  prompt: "Explain machine learning",
  max_tokens: 200,
  temperature: 0.7
}));

// Receive live updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Live update:', data);
};
```

### Custom Auditing Rules

```python
# Add custom alert logic in mistral_audit_server.py
class CustomAuditRules:
    def check_domain_specific_uncertainty(self, token, uncertainty):
        # Custom rules for your domain
        if "quantum" in token and uncertainty > 3.0:
            return Alert("QUANTUM_UNCERTAINTY", "HIGH", 
                        "High uncertainty in quantum physics term")
        return None
```

## 📊 **Performance & Optimization**

### GPU Acceleration
```bash
# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
python3 mistral_audit_server.py
```

### Memory Optimization
```python
# In mistral_audit_server.py, add:
torch_dtype=torch.float16,     # Use half precision
device_map="auto",             # Auto device mapping
load_in_8bit=True,            # 8-bit quantization (requires bitsandbytes)
```

### Production Deployment
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 mistral_audit_server:app

# Use nginx for static files
# Configure nginx to serve React build files
```

## 🚨 **Troubleshooting**

### Common Issues

#### "CUDA out of memory"
```python
# Reduce model precision or use CPU
device = "cpu"  # Force CPU usage
torch_dtype = torch.float16  # Use half precision
```

#### "Port already in use"
```bash
# Check what's using the port
lsof -i :8080
lsof -i :3000
lsof -i :8765

# Kill processes if needed
kill -9 <PID>
```

#### "Model download fails"
```bash
# Set proxy if needed
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror
export HF_TOKEN=your_token_here           # If model requires auth
```

#### "WebSocket connection fails"
- Check if Python server is running on port 8765
- Verify firewall settings allow WebSocket connections  
- Check browser console for detailed error messages

### Debug Mode
```bash
# Enable debug logging
export RUST_LOG=debug
export PYTHONPATH=.
python3 -u mistral_audit_server.py
```

## 📁 **Project Structure**

```
semantic-uncertainty-runtime/
├── core-engine/                   # Rust uncertainty quantification
│   ├── src/
│   │   ├── live_response_auditor.rs    # Main auditing engine
│   │   ├── audit_interface.rs          # Clean API interface
│   │   └── oss_logit_adapter.rs        # Enhanced FIM analysis
│   └── Cargo.toml
├── frontend/uncertainty-dashboard/     # React frontend
│   ├── src/app/page.tsx               # Main dashboard
│   └── package.json
├── mistral_audit_server.py            # Python Mistral server
├── requirements.txt                   # Python dependencies
├── start_mistral_live_audit.sh       # One-command startup
└── MISTRAL_LIVE_AUDIT_README.md      # This file
```

## 🤝 **Integration with Other Models**

The separated auditing system works with **any** model:

```python
# OpenAI API
from audit_interface import AuditClient
client = AuditClient()
session = client.start_audit("Explain AI", "gpt-4", "OpenAI")

# Local model (any framework)
for token in your_model.generate_stream(prompt):
    result = client.add_token(SimpleToken(
        text=token.text,
        probability=token.prob  # Optional
    ))
    print(f"Uncertainty: {result.current_uncertainty}")
```

## 📈 **Performance Benchmarks**

### Typical Performance
- **Mistral 7B (GPU)**: 15-25 tokens/sec
- **Mistral 7B (CPU)**: 3-8 tokens/sec  
- **Audit Overhead**: <5ms per token
- **WebSocket Latency**: <10ms
- **Frontend Update Rate**: 60 FPS

### Resource Usage
- **GPU Memory**: 6-8GB (Mistral 7B FP16)
- **RAM**: 4-8GB (system + Python)
- **CPU**: 2-4 cores recommended
- **Network**: <1MB/min for live streaming

## 🛡️ **Security Notes**

### Production Considerations
- **API Keys**: Store securely, never commit to git
- **Network**: Use HTTPS/WSS in production
- **Rate Limiting**: Implement request limits
- **Authentication**: Add auth for production deployments

### Data Privacy
- **Model Inference**: Runs locally, no data sent to external APIs
- **Audit Data**: Stored in memory only (not persisted)
- **Logs**: May contain prompts/responses, secure appropriately

## 🆘 **Support & Resources**

### Getting Help
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check docs/ directory for technical details
- **Logs**: Check logs/ directory for debugging information

### Useful Commands
```bash
# Check service status
curl http://localhost:8080/health  # Rust API
curl http://localhost:5000/health  # Python server

# View live logs
tail -f logs/rust_api.log
tail -f logs/mistral_server.log  
tail -f logs/frontend.log

# Monitor resource usage
htop  # CPU/Memory
nvidia-smi  # GPU usage
```

## 🎉 **Success!**

If everything is working, you should see:
- ✅ Dashboard at `http://localhost:3000`
- ✅ Live uncertainty metrics updating as you generate text
- ✅ Real-time alerts for high uncertainty tokens  
- ✅ Beautiful charts showing uncertainty trends
- ✅ Token-by-token analysis with probabilities

**🎯 You now have a complete live uncertainty auditing system for Mistral 7B!**

---

*For technical details about the uncertainty quantification algorithms, see the core-engine documentation and research papers.* 