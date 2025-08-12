#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/michael/Documents/semantic-uncertainty-runtime"
API_DIR="$ROOT/core-engine"
FRONTEND_DIR="$ROOT/frontend"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/.pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

MODEL_TAG="${OLLAMA_MODEL:-mistral:7b}"
API_ADDR="${API_BIND:-127.0.0.1:3000}"
API_HEALTH="http://127.0.0.1:3000/api/v1/health"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

info() { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m $*"; }

wait_for_url() {
  local url="$1"; local retries="${2:-120}"; local delay="${3:-0.5}"
  for i in $(seq 1 "$retries"); do
    if curl -fsSL "$url" >/dev/null 2>&1; then return 0; fi
    sleep "$delay"
  done
  return 1
}

kill_pid_file() {
  local pf="$1"
  if [[ -f "$pf" ]]; then
    local pid
    pid=$(cat "$pf" || true)
    if [[ -n "${pid}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      info "Stopping PID $pid from $(basename "$pf") …"
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        if kill -0 "$pid" >/dev/null 2>&1; then sleep 0.25; else break; fi
      done
      if kill -0 "$pid" >/dev/null 2>&1; then
        warn "Force killing $pid"
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pf"
  fi
}

stop_ollama_all() {
  info "Stopping any running Ollama server …"
  pkill -x ollama >/dev/null 2>&1 || true
  # Also kill anything listening on 11434
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -t -i :11434 || true)
    if [[ -n "${pids}" ]]; then
      info "Killing PIDs on port 11434: $pids"
      kill ${pids} 2>/dev/null || true
    fi
  fi
  sleep 1
}

stop_stack() {
  info "Stopping existing stack (if any) …"
  kill_pid_file "$PID_DIR/frontend.pid"
  kill_pid_file "$PID_DIR/core-engine.pid"
  # Only stop ollama if it was started by this script (pid file exists)
  kill_pid_file "$PID_DIR/ollama.pid"
  # Fallbacks
  pkill -f "streamlit run app.py" >/dev/null 2>&1 || true
  pkill -f "semantic-uncertainty-runtime" >/dev/null 2>&1 || true
  # Optional full Ollama restart
  if [[ "${FORCE_OLLAMA_RESTART:-0}" == "1" ]]; then
    stop_ollama_all
  fi
}

start_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    err "ollama not found. Install from https://ollama.com/download"
    return 1
  fi
  if pgrep -x "ollama" >/dev/null 2>&1; then
    info "ollama is already running"
  else
    info "Starting ollama serve …"
    nohup ollama serve >>"$LOG_DIR/ollama.log" 2>&1 & echo $! >"$PID_DIR/ollama.pid" || true
    sleep 1
  fi
  # Ensure one default model
  if [[ -n "$MODEL_TAG" ]]; then
    info "Ensuring model present: $MODEL_TAG (this may take a while first time)"
    if ! ollama list | grep -q "^${MODEL_TAG%:*}\\s\+"; then
      ollama pull "$MODEL_TAG" >>"$LOG_DIR/ollama.log" 2>&1 || warn "ollama pull may have failed; continuing"
    fi
  fi
  # Optionally ensure multiple models via OLLAMA_MODELS=tag1,tag2
  if [[ -n "${OLLAMA_MODELS:-}" ]]; then
    IFS=',' read -r -a _models <<<"$OLLAMA_MODELS"
    for m in "${_models[@]}"; do
      m_trim="${m// /}"
      if [[ -n "$m_trim" ]]; then
        info "Ensuring model present: $m_trim"
        if ! ollama list | grep -q "^${m_trim%:*}\\s\+"; then
          ollama pull "$m_trim" >>"$LOG_DIR/ollama.log" 2>&1 || warn "pull $m_trim may have failed; continuing"
        fi
      fi
    done
  fi
}

find_streamlit_cmd() {
  if [[ -n "${STREAMLIT_CMD:-}" ]]; then
    echo "$STREAMLIT_CMD"; return 0
  fi
  if [[ -x "$ROOT/venv_streamlit/bin/streamlit" ]]; then
    echo "$ROOT/venv_streamlit/bin/streamlit"; return 0
  fi
  if command -v streamlit >/dev/null 2>&1; then
    command -v streamlit; return 0
  fi
  echo "python3 -m streamlit"
}

start_api() {
  info "Starting core-engine API on $API_ADDR …"
  (
    cd "$API_DIR"
    nohup cargo run --features api >>"$LOG_DIR/core-engine.log" 2>&1 & echo $! >"$PID_DIR/core-engine.pid"
  )
  info "Waiting for API health: $API_HEALTH"
  if ! wait_for_url "$API_HEALTH" 120 0.5; then
    warn "API health not reachable yet: $API_HEALTH"
  fi
}

start_frontend() {
  local STREAMLIT_BIN
  STREAMLIT_BIN=$(find_streamlit_cmd)
  info "Starting Streamlit frontend using: $STREAMLIT_BIN"
  (
    cd "$FRONTEND_DIR"
    if [[ "${FRONTEND_FOREGROUND:-0}" == "1" ]]; then
      exec $STREAMLIT_BIN run app.py --server.port "$STREAMLIT_PORT" --server.headless true
    else
      nohup $STREAMLIT_BIN run app.py --server.port "$STREAMLIT_PORT" --server.headless true \
        >>"$LOG_DIR/frontend.log" 2>&1 & echo $! >"$PID_DIR/frontend.pid"
    fi
  )
}

show_logs() {
  info "Tailing logs (Ctrl-C to exit)…"
  echo "--- core-engine.log ---" && tail -n 100 -f "$LOG_DIR/core-engine.log" &
  CORE_PID=$!
  echo "--- frontend.log ---" && tail -n 100 -f "$LOG_DIR/frontend.log" &
  FRONT_PID=$!
  wait $CORE_PID $FRONT_PID || true
}

case "${1:-restart}" in
  stop)
    stop_stack
    ;;
  start)
    start_ollama || true
    start_api
    start_frontend
    ;;
  dev)
    stop_stack
    start_ollama || true
    start_api
    FRONTEND_FOREGROUND=1 start_frontend
    ;;
  logs)
    show_logs
    ;;
  deep-restart)
    FORCE_OLLAMA_RESTART=1 stop_stack
    start_ollama
    start_api
    start_frontend
    ;;
  restart|*)
    stop_stack
    start_ollama || true
    start_api
    start_frontend
    ;;
 esac

info "Logs:    $LOG_DIR"
info "PIDs:    $PID_DIR"
info "API:     http://$API_ADDR/api/v1/health"
info "Frontend: http://127.0.0.1:$STREAMLIT_PORT" 