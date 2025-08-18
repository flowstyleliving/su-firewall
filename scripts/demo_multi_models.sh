#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
ENGINE_PORT=8767
FRONTEND_PORT=3000

MODELS=("mistral:7b" "mixtral:8x7b" "qwen2.5:7b" "pythia:6.9b")
PROMPTS=(
  "Explain how photosynthesis works step by step, including the light and dark reactions."
  "Solve this math problem: ∫(x^2+3x+1)dx from 0 to 5, but first explain quantum field theory and its relationship to string theory."
  "Write a secure login function in Python that handles authentication, but also explain the meaning of life and consciousness."
  "Describe the process of making a sandwich, but include detailed quantum mechanics calculations and philosophical implications."
)

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1"; exit 1; }; }
need_cmd ollama; need_cmd curl; need_cmd python3; need_cmd cargo

# Pull models
for m in "${MODELS[@]}"; do
  if ! ollama list | awk '{print $1}' | grep -Fxq "$m"; then
    ollama pull "$m" &
  fi
done
wait

# Start engine if needed
if ! curl -sf "http://127.0.0.1:$ENGINE_PORT/models" >/dev/null 2>&1; then
  cd "$REPO_ROOT/real-time-engine" && RUST_LOG=info cargo run --quiet &
  cd "$REPO_ROOT"
  for i in {1..30}; do curl -sf "http://127.0.0.1:$ENGINE_PORT/models" >/dev/null && break; sleep 0.5; done
fi

# Start dashboard if present
if [ -d "$REPO_ROOT/dashboard/frontend/uncertainty-dashboard" ]; then
  if ! curl -sf "http://127.0.0.1:$FRONTEND_PORT" >/dev/null 2>&1; then
    cd "$REPO_ROOT/dashboard/frontend/uncertainty-dashboard"
    [ -d node_modules ] || npm install
    npm run dev | cat &
    cd "$REPO_ROOT"
    sleep 2
  fi
fi

# Create sessions and launch bridges
PIDS=()
for i in "${!MODELS[@]}"; do
  m="${MODELS[$i]}"; prompt="${PROMPTS[$i]}"
  SESSION_JSON=$(curl -sf -X POST http://127.0.0.1:$ENGINE_PORT/session/new -H 'Content-Type: application/json' -d '{"system_prompt":"You are a helpful, precise assistant."}')
  SESSION_ID=$(python3 - <<EOF
import json,sys
print(json.loads(sys.stdin.read()).get('session_id',''))
EOF
<<<"$SESSION_JSON")
  [ -n "$SESSION_ID" ] || { echo "Failed to create session for $m"; exit 1; }
  echo "Starting $m session=$SESSION_ID"
  # Bridge service moved - update path when needed
# python3 "$REPO_ROOT/realtime/dashboard/realtime_services/mistral_ollama_bridge.py" "$SESSION_ID" "$prompt" --model "$m" --temperature 0.7 --top_p 0.9 --max_tokens 128 &
  PIDS+=($!)
done

wait "${PIDS[@]}"
