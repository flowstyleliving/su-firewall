#!/bin/bash

echo "🚀 **STARTING REAL LOGITS UNCERTAINTY SYSTEM**"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
REALTIME_DIR="$REPO_ROOT/real-time-engine"
FRONTEND_DIR="$REPO_ROOT/dashboard/frontend/uncertainty-dashboard"
VENV311_DIR="$REPO_ROOT/venv_python311"
PID_PYTORCH_FILE="$REPO_ROOT/.pytorch_pid"
PID_FRONTEND_FILE="$REPO_ROOT/.frontend_pid"

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $port is available${NC}"
        return 0
    fi
}

# Function to kill existing processes
kill_existing() {
    echo -e "${YELLOW}🔄 Killing existing processes...${NC}"
    pkill -f "mistral_pytorch_server.py" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    sleep 2
}

# Check and kill existing processes
kill_existing

echo ""
echo "🔍 **CHECKING PORTS**"
check_port 8766 || exit 1
check_port 3000 || exit 1
check_port 5002 || exit 1

echo ""
echo "🐍 **STARTING PYTORCH SERVER (REAL LOGITS)**"

# Check if Python 3.11 venv exists
if [ ! -d "$VENV311_DIR" ]; then
    echo -e "${RED}❌ Python 3.11 virtual environment not found!${NC}"
    echo "Please run the setup first:"
    echo "1. Install Python 3.11: brew install python@3.11"
    echo "2. Create venv: /usr/local/bin/python3.11 -m venv venv_python311"
    echo "3. Install dependencies: source venv_python311/bin/activate && pip install -r $REALTIME_DIR/requirements/requirements_pytorch.txt"
    exit 1
fi

# Activate Python environment and start PyTorch server
echo -e "${BLUE}🔧 Activating Python 3.11 environment...${NC}"
source "$VENV311_DIR/bin/activate"

echo -e "${BLUE}🚀 Starting PyTorch server with real logits...${NC}"
python "$REALTIME_DIR/mistral_pytorch_server.py" &
PYTORCH_PID=$!

echo -e "${GREEN}✅ PyTorch server started (PID: $PYTORCH_PID)${NC}"
echo -e "${BLUE}📡 WebSocket: ws://localhost:8766${NC}"
echo -e "${BLUE}🌐 HTTP API: http://localhost:5002${NC}"

# Wait for PyTorch server to be ready
echo ""
echo -e "${YELLOW}⏳ Waiting for PyTorch server to load model...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:5002/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PyTorch server is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 2
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ PyTorch server failed to start within 60 seconds${NC}"
        exit 1
    fi
done

echo ""
echo "🌐 **STARTING FRONTEND DASHBOARD**"

# Check if frontend dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    cd "$FRONTEND_DIR"
    npm install
    cd "$REPO_ROOT"
fi

# Start Next.js frontend
echo -e "${BLUE}🚀 Starting Next.js frontend...${NC}"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
cd "$REPO_ROOT"

echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"
echo -e "${BLUE}🌐 Dashboard: http://localhost:3000${NC}"

# Wait for frontend to be ready
echo ""
echo -e "${YELLOW}⏳ Waiting for frontend to be ready...${NC}"
for i in {1..15}; do
    if curl -s http://localhost:3000 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 2
    if [ $i -eq 15 ]; then
        echo -e "${RED}❌ Frontend failed to start within 30 seconds${NC}"
        exit 1
    fi
done

echo ""
echo "🎉 **ALL SERVERS STARTED SUCCESSFULLY!**"
echo ""
echo -e "${GREEN}📊 **SYSTEM STATUS:**${NC}"
echo -e "${GREEN}✅ PyTorch Server: ws://localhost:8766${NC}"
echo -e "${GREEN}✅ HTTP API: http://localhost:5002${NC}"
echo -e "${GREEN}✅ Frontend Dashboard: http://localhost:3000${NC}"
echo -e "${GREEN}✅ Real Logits: DialoGPT-Medium${NC}"
echo ""
echo -e "${BLUE}🎯 **ACCESS YOUR SYSTEM:**${NC}"
echo -e "${BLUE}🌐 Open browser: http://localhost:3000${NC}"
echo ""
echo -e "${YELLOW}🎛️ **FEATURES AVAILABLE:**${NC}"
echo -e "${YELLOW}• Real uncertainty quantification (ℏₛ = √(Δμ × Δσ))${NC}"
echo -e "${YELLOW}• Actual model probabilities from logits${NC}"
echo -e "${YELLOW}• Risk tolerance slider (0-100%)${NC}"
echo -e "${YELLOW}• Live token-by-token analysis${NC}"
echo -e "${YELLOW}• Dynamic parameter adjustment${NC}"
echo ""
echo -e "${BLUE}🛑 **TO STOP SERVERS:**${NC}"
echo -e "${BLUE}Run: ./stop_servers.sh${NC}"
echo ""

# Save PIDs for easy cleanup
echo $PYTORCH_PID > "$PID_PYTORCH_FILE"
echo $FRONTEND_PID > "$PID_FRONTEND_FILE"

echo -e "${GREEN}🎉 **REAL LOGITS UNCERTAINTY SYSTEM IS RUNNING!**${NC}" 