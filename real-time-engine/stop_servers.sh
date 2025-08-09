#!/bin/bash

echo "🛑 **STOPPING REAL LOGITS UNCERTAINTY SYSTEM**"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
PID_PYTORCH_FILE="$REPO_ROOT/.pytorch_pid"
PID_FRONTEND_FILE="$REPO_ROOT/.frontend_pid"

# Function to kill process by PID file
kill_by_pid_file() {
    local pid_file=$1
    local process_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}🔄 Stopping $process_name (PID: $pid)...${NC}"
            kill $pid
            sleep 2
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${YELLOW}🔄 Force killing $process_name...${NC}"
                kill -9 $pid
            fi
            echo -e "${GREEN}✅ $process_name stopped${NC}"
        else
            echo -e "${BLUE}ℹ️  $process_name not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${BLUE}ℹ️  No PID file found for $process_name${NC}"
    fi
}

# Kill processes by name (fallback)
kill_by_name() {
    local process_name=$1
    local display_name=$2
    
    if pkill -f "$process_name" 2>/dev/null; then
        echo -e "${GREEN}✅ $display_name stopped${NC}"
    else
        echo -e "${BLUE}ℹ️  $display_name not running${NC}"
    fi
}

echo -e "${YELLOW}🔄 Stopping all servers...${NC}"

# Stop by PID files first (cleaner)
kill_by_pid_file "$PID_PYTORCH_FILE" "PyTorch Server"
kill_by_pid_file "$PID_FRONTEND_FILE" "Frontend Dashboard"

# Fallback: kill by process name
kill_by_name "mistral_pytorch_server.py" "PyTorch Server"
kill_by_name "next dev" "Frontend Dashboard"

# Clean up any remaining processes
echo ""
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
pkill -f "python.*mistral_pytorch" 2>/dev/null || true
pkill -f "node.*next" 2>/dev/null || true

# Remove PID files
rm -f "$PID_PYTORCH_FILE" "$PID_FRONTEND_FILE"

echo ""
echo -e "${GREEN}✅ **ALL SERVERS STOPPED!**${NC}"
echo ""
echo -e "${BLUE}📊 **SYSTEM STATUS:**${NC}"
echo -e "${BLUE}• PyTorch Server: Stopped${NC}"
echo -e "${BLUE}• Frontend Dashboard: Stopped${NC}"
echo -e "${BLUE}• WebSocket connections: Closed${NC}"
echo ""
echo -e "${YELLOW}🚀 **TO RESTART:**${NC}"
echo -e "${YELLOW}Run: ./start_real_logits_system.sh${NC}" 