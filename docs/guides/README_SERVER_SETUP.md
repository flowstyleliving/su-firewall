# 🚀 Real Logits Uncertainty System - Server Setup Guide

## 📋 **Quick Start (Recommended)**

### 🎯 **One-Command Startup**
```bash
./start_real_logits_system.sh
```

### 🛑 **One-Command Shutdown**
```bash
./stop_servers.sh
```

---

## 🔧 **Manual Server Setup**

### **Prerequisites**
- Python 3.11 (for PyTorch compatibility)
- Node.js 18+ (for Next.js frontend)
- 8GB+ RAM (for model loading)

### **Step 1: Python Environment Setup**
```bash
# Install Python 3.11 (if not already installed)
brew install python@3.11

# Create virtual environment
/usr/local/bin/python3.11 -m venv venv_python311

# Activate environment
source venv_python311/bin/activate

# Install PyTorch dependencies
pip install -r requirements_pytorch.txt
```

### **Step 2: Frontend Dependencies**
```bash
# Navigate to frontend directory
cd frontend/uncertainty-dashboard

# Install Node.js dependencies
npm install

# Return to project root
cd ../..
```

---

## 🖥️ **Manual Server Startup**

### **Option A: PyTorch Server (Real Logits)**
```bash
# Terminal 1: Start PyTorch server
source venv_python311/bin/activate
python mistral_pytorch_server.py
```
**Endpoints:**
- WebSocket: `ws://localhost:8766`
- HTTP API: `http://localhost:5002/health`

### **Option B: Ollama Server (Alternative)**
```bash
# Terminal 1: Start Ollama server
source venv_python311/bin/activate
python mistral_ollama_server.py
```
**Endpoints:**
- WebSocket: `ws://localhost:8765`
- HTTP API: `http://localhost:5001/health`

### **Frontend Dashboard**
```bash
# Terminal 2: Start Next.js frontend
cd frontend/uncertainty-dashboard
npm run dev
```
**Endpoint:**
- Dashboard: `http://localhost:3000`

---

## 🔍 **Server Status Check**

### **Check if servers are running:**
```bash
# Check PyTorch server
curl http://localhost:5002/health

# Check frontend
curl http://localhost:3000

# Check WebSocket ports
netstat -an | grep 8766
```

### **Expected Health Response:**
```json
{
  "model_loaded": true,
  "model_type": "DialoGPT-Medium with PyTorch",
  "provider": "Transformers",
  "real_logits": true,
  "status": "healthy"
}
```

---

## 🎛️ **System Architecture**

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Frontend      │ ←──────────────→ │  PyTorch Server │
│  (Next.js)      │                 │  (Real Logits)  │
│ localhost:3000  │                 │ localhost:8766  │
└─────────────────┘                 └─────────────────┘
         │                                    │
         │ HTTP                               │ HTTP
         ▼                                    ▼
┌─────────────────┐                 ┌─────────────────┐
│   Browser       │                 │  Health API     │
│  Dashboard      │                 │ localhost:5002  │
└─────────────────┘                 └─────────────────┘
```

---

## 🚨 **Troubleshooting**

### **Port Already in Use**
```bash
# Find process using port
lsof -i :8766
lsof -i :3000
lsof -i :5002

# Kill process
kill -9 <PID>
```

### **Python Environment Issues**
```bash
# Recreate virtual environment
rm -rf venv_python311
/usr/local/bin/python3.11 -m venv venv_python311
source venv_python311/bin/activate
pip install -r requirements_pytorch.txt
```

### **Frontend Build Issues**
```bash
# Clear Next.js cache
cd frontend/uncertainty-dashboard
rm -rf .next
npm run dev
```

### **Model Loading Issues**
```bash
# Check available memory
free -h

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check transformers installation
python -c "import transformers; print(transformers.__version__)"
```

---

## 📊 **Performance Monitoring**

### **Memory Usage**
```bash
# Monitor Python process
ps aux | grep mistral_pytorch

# Monitor Node.js process
ps aux | grep "next dev"
```

### **Log Monitoring**
```bash
# PyTorch server logs (in terminal where it's running)
# Look for:
# - Model loading progress
# - WebSocket connections
# - Generation requests
# - Error messages
```

---

## 🔄 **Development Workflow**

### **1. Start Development**
```bash
./start_real_logits_system.sh
```

### **2. Make Changes**
- Edit frontend: `frontend/uncertainty-dashboard/src/app/page.tsx`
- Edit backend: `mistral_pytorch_server.py`
- Frontend auto-reloads on changes

### **3. Test Changes**
- Open: `http://localhost:3000`
- Generate text and observe uncertainty values
- Check browser console for errors

### **4. Stop Development**
```bash
./stop_servers.sh
```

---

## 🎯 **Access Points**

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend Dashboard** | `http://localhost:3000` | Main user interface |
| **PyTorch WebSocket** | `ws://localhost:8766` | Real-time communication |
| **Health API** | `http://localhost:5002/health` | Server status |
| **Model Info** | `http://localhost:5002/model/info` | Model details |

---

## 🧠 **Model Options**

### **DialoGPT-Medium (Real Logits) - Default**
- **Provider**: PyTorch Transformers
- **Logits**: Real model output
- **Uncertainty**: Genuine ℏₛ = √(Δμ × Δσ)
- **Performance**: High accuracy, slower inference

### **Mistral 7B (Ollama)**
- **Provider**: Ollama
- **Logits**: Estimated
- **Uncertainty**: Approximated
- **Performance**: Fast inference, lower accuracy

### **GPT-OSS (Simulated)**
- **Provider**: Mock simulation
- **Logits**: Generated
- **Uncertainty**: Simulated
- **Performance**: Instant, for testing only

---

## 🎉 **Success Indicators**

✅ **System is working when you see:**
- Frontend loads at `http://localhost:3000`
- Model dropdown shows "DialoGPT-Medium (Real Logits)"
- Risk tolerance slider is interactive
- Text generation produces real uncertainty values
- WebSocket connection shows "Connected" status
- No error messages in browser console

🚨 **Common Issues:**
- "Connection failed" → Check if PyTorch server is running
- "Model not loaded" → Wait for model to finish loading
- "Port in use" → Run `./stop_servers.sh` first
- "Module not found" → Check Python environment activation 