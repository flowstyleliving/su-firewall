#!/usr/bin/env python3
import streamlit as st
import requests
import threading
import time
import json
import subprocess
from typing import Optional, List, Dict
import math
import os
import logging
from logging.handlers import RotatingFileHandler
from models_registry import ModelsRegistry  # local fallback loader

# Logging setup
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, 'frontend.log')
logger = logging.getLogger("frontend")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
logger.info("Frontend starting up")

# Config
ENGINE_HTTP = "http://127.0.0.1:3000/api/v1"
ENGINE_WS = "ws://127.0.0.1:3000/api/v1/ws"
BRIDGE = os.path.join(os.path.dirname(__file__), 'realtime', 'mistral_ollama_bridge.py')

# Quick API health check
def api_health_check() -> Dict:
    status = {"ok": False, "details": ""}
    try:
        r = requests.get(f"{ENGINE_HTTP}/health", timeout=3)
        if r.ok:
            status["ok"] = True
            status["details"] = r.json()
        else:
            status["details"] = {"status_code": r.status_code}
    except Exception as e:
        status["details"] = {"error": str(e)}
    return status

health = api_health_check()
health_col1, health_col2 = st.columns([3, 2])
with health_col1:
    st.caption("Backend API")
    if health.get("ok"):
        st.success("API online at http://127.0.0.1:3000/api/v1")
    else:
        st.error("API unreachable. Start it with: `cd core-engine && cargo run --features api`")
with health_col2:
    if isinstance(health.get("details"), dict):
        st.caption("Health details")
        st.json(health["details"])

# Map engine model IDs to Ollama tags
OLLAMA_TAG_FOR = {
    'mistral-7b': 'mistral:7b',
    'mixtral-8x7b': 'mixtral:8x7b',
    'qwen2.5-7b': 'qwen2.5:7b',
    'pythia-6.9b': 'pythia:6.9b',
    'ollama-mistral-7b': 'mistral:7b',
}

st.set_page_config(page_title="Real-time Uncertainty (Streamlit)", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None
if "ws_stop" not in st.session_state:
    st.session_state.ws_stop = False
if "events" not in st.session_state:
    st.session_state.events = []
if "generated" not in st.session_state:
    st.session_state.generated = ""
if "bridge_proc" not in st.session_state:
    st.session_state.bridge_proc = None
if "session_failure_law" not in st.session_state:
    st.session_state.session_failure_law = None

# Utilities
def fetch_models_info() -> Dict:
    try:
        logger.info("GET /models")
        r = requests.get(f"{ENGINE_HTTP}/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        logger.info("/models ok: %s", json.dumps(data)[:256])
        return data
    except Exception as e:
        logger.exception("/models failed: %s", e)
        st.warning(f"Models fetch failed: {e}")
        return {"default_model_id": None, "models": []}

def fetch_models() -> List[Dict]:
    try:
        logger.info("GET /models")
        r = requests.get(f"{ENGINE_HTTP}/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        logger.info("/models ok: %s", json.dumps(data)[:256])
        return data.get("models", [])
    except Exception as e:
        logger.exception("/models failed: %s", e)
        # Fallback to local models.json reader
        try:
            reg = ModelsRegistry.load_from_file()
            return [m.__dict__ | ({'failure_law': {'lambda': m.failure_law.lambda_, 'tau': m.failure_law.tau}} if m.failure_law else {}) for m in reg.list()]
        except Exception as e2:
            st.warning(f"Models fetch failed: {e}; fallback failed: {e2}")
            return []

def create_session(model_id: Optional[str]) -> Optional[str]:
    try:
        payload = {"model_id": model_id or ""}
        logger.info("POST /session/new %s", payload)
        r = requests.post(f"{ENGINE_HTTP}/session/new", json=payload, timeout=10)
        r.raise_for_status()
        sid = r.json().get("session_id")
        logger.info("session created: %s", sid)
        return sid
    except Exception as e:
        logger.exception("create_session failed: %s", e)
        st.error(f"Create session failed: {e}")
        return None

def fetch_session_failure_law(session_id: str) -> Optional[Dict]:
    try:
        url = f"{ENGINE_HTTP}/session/{session_id}/failure_law"
        logger.info("GET %s", url)
        r = requests.get(url, timeout=5)
        if r.ok:
            logger.info("failure_law: %s", json.dumps(r.json())[:256])
            return r.json()
    except Exception as e:
        logger.exception("fetch_session_failure_law failed: %s", e)
        return None
    return None


def update_session_failure_law(session_id: str, lambda_val: float, tau_val: float) -> Optional[Dict]:
    try:
        url = f"{ENGINE_HTTP}/session/{session_id}/failure_law"
        payload = {"lambda": lambda_val, "tau": tau_val}
        logger.info("POST %s %s", url, payload)
        r = requests.post(url, json=payload, timeout=5)
        if r.ok:
            logger.info("failure_law updated: %s", json.dumps(r.json())[:256])
            return r.json()
    except Exception as e:
        logger.exception("update_session_failure_law failed: %s", e)
        st.error(f"Failed to update failure law: {e}")
        return None
    return None

def update_session_config(session_id: str) -> bool:
    try:
        cfg = {
            "precision_method": st.session_state.get("precision_method"),
            "flexibility_method": st.session_state.get("flexibility_method"),
            "hash_embeddings_precision": st.session_state.get("precision_hash"),
            "hash_embeddings_flexibility": st.session_state.get("flexibility_hash"),
            "domain_tuning": st.session_state.get("domain_tuning"),
            "performance_mode": "balanced",
        }
        # Enforce rule: if Fisher Full Matrix selected, disable hash embeddings
        if cfg["precision_method"] and "Fisher Full Matrix" in cfg["precision_method"]:
            cfg["hash_embeddings_precision"] = False
            cfg["hash_embeddings_flexibility"] = False
        r = requests.post(f"{ENGINE_HTTP}/session/{session_id}/config", json=cfg, timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        st.warning(f"Config update failed: {e}")
        return False

# WebSocket listening in background
def ws_loop():
    import websocket  # websocket-client
    # Ensure session state keys exist when thread starts
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'generated' not in st.session_state:
        st.session_state.generated = ""
    if 'ws_stop' not in st.session_state:
        st.session_state.ws_stop = False
    st.session_state.events.clear()
    st.session_state.generated = ""
    try:
        logger.info("WS connect %s", ENGINE_WS)
        ws = websocket.create_connection(ENGINE_WS, timeout=5)
        logger.info("WS connected")
    except Exception as e:
        logger.exception("WS connect failed: %s", e)
        st.error(f"WebSocket connect failed: {e}")
        return
    ws.settimeout(1.0)
    try:
        while not st.session_state.get('ws_stop', False):
            try:
                msg = ws.recv()
            except Exception:
                continue
            try:
                ev = json.loads(msg)
                logger.info("WS event: %s", json.dumps(ev)[:256])
            except Exception as e:
                logger.exception("WS parse failed: %s", e)
                continue
            if ev.get("type") == "generation_update":
                st.session_state.events.append(ev)
                token = ev.get("token") or ev.get("data", {}).get("token", {}).get("text") or ""
                if token:
                    prev = st.session_state.generated
                    needs_space = prev and not token.startswith((",", ".", ";", ":", "!", "?")) and not prev.endswith(" ")
                    st.session_state.generated = prev + (" " if needs_space else "") + token
            elif ev.get("type") == "hello":
                pass
    finally:
        try:
            ws.close()
        except Exception:
            pass

# Sidebar controls
st.sidebar.header("Generation Settings")
models_info = fetch_models_info()
models = models_info.get("models", [])
default_model_id = models_info.get("default_model_id")
models_by_id = {m.get("id"): m for m in models}
model_options = [m.get("id") for m in models]
try:
    default_index = model_options.index(default_model_id) if default_model_id in model_options else 0
except Exception:
    default_index = 0 if model_options else None
selected_model = st.sidebar.selectbox(
    "Model",
    model_options,
    index=default_index,
    format_func=lambda mid: f"{models_by_id.get(mid, {}).get('display_name', mid)} ({mid})" if mid else "—",
)
ollama_tag = OLLAMA_TAG_FOR.get(selected_model or '', selected_model or 'mistral:7b')
prompt = st.sidebar.text_area("Prompt", "Explain quantum computing to an 8th grader", height=100)
col_temp, col_top_p, col_tokens = st.sidebar.columns(3)
temperature = col_temp.number_input("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
top_p = col_top_p.number_input("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
max_tokens = int(col_tokens.number_input("Max tokens", min_value=16, max_value=2048, value=128, step=16))

# Run mode toggle (intuitive)
is_ollama_model = str(models_by_id.get(selected_model or '', {}).get('hf_repo', '')).startswith('ollama/') or ((selected_model or '') in OLLAMA_TAG_FOR)
run_modes = ["Live (Ollama)", "API Analyze Only"]
default_mode_idx = 0 if is_ollama_model else 1
run_mode = st.sidebar.radio("Run Mode", run_modes, index=default_mode_idx, help="Live streams tokens via Ollama when available; API Analyze runs the undeniable test without live streaming")

# New: Method configuration controls
st.sidebar.markdown("---")
st.sidebar.subheader("Method Configuration")
# Hash embedding toggles (separate for precision/flexibility)
precision_hash = st.sidebar.checkbox("Use Hash Embeddings for Precision ⚡", value=True)
flexibility_hash = st.sidebar.checkbox("Use Hash Embeddings for Flexibility ⚡", value=True)

# Precision methods
precision_methods = [
    "Fisher Diagonal (Fast)",
    "Fisher Full Matrix (Slow)",
    "Gradient Magnitude (Fast)",
    "Hessian Diagonal (Medium)",
    "Information Gain (Medium)"
]
precision_method = st.sidebar.selectbox("Precision Method", precision_methods, index=0,
    help="Fisher Full Matrix disables hash embeddings due to full quadratic form requirements")

# Flexibility methods
flexibility_methods = [
    "Fisher Diagonal + Hash (Fast)",
    "Fisher Diagonal (Full) (Medium)",
    "Max Entropy (Fast)",
    "KL Divergence + Jensen-Shannon (Medium)",
    "Wasserstein (Slow)",
    "Spectral Entropy (Medium)",
    "Ensemble Average (Slow)"
]
flexibility_method = st.sidebar.selectbox("Flexibility Method", flexibility_methods, index=3)

# Domain-specific tuning
domain_option = st.sidebar.selectbox("Domain Tuning", ["text", "code", "image"], index=0,
    help="Different domains have distinct semantic properties and calibration needs")

# Persist selections in session state for downstream use
st.session_state.precision_hash = precision_hash
st.session_state.flexibility_hash = flexibility_hash
st.session_state.precision_method = precision_method
st.session_state.flexibility_method = flexibility_method
st.session_state.domain_tuning = domain_option

run_col, stop_col = st.sidebar.columns(2)

# Failure law controls
st.sidebar.markdown("---")
st.sidebar.subheader("Failure Law (per-session)")
lambda_val = st.sidebar.number_input("lambda", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
tau_val = st.sidebar.number_input("tau", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
apply_fl = st.sidebar.button("Apply", use_container_width=True)

# Start run: create session, start ws listener, launch bridge
if run_col.button("Run", use_container_width=True):
    sid = create_session(selected_model)
    if sid:
        st.session_state.session_id = sid
        # Apply initial failure law values
        st.session_state.session_failure_law = update_session_failure_law(sid, lambda_val, tau_val)
        # Apply analysis configuration from sidebar
        update_session_config(sid)
        if run_mode == "Live (Ollama)":
            # Start WS listener
            st.session_state.ws_stop = False
            t = threading.Thread(target=ws_loop, daemon=True)
            st.session_state.ws_thread = t
            t.start()
            # Launch Ollama bridge only for Ollama-backed models
            model_info = models_by_id.get(selected_model, {})
            is_ollama = str(model_info.get("hf_repo", "")).startswith("ollama/") or (selected_model in OLLAMA_TAG_FOR)
            if is_ollama:
                cmd = [
                    "python3", BRIDGE, sid, prompt,
                    "--model", ollama_tag,
                    "--temperature", str(temperature),
                    "--top_p", str(top_p),
                    "--max_tokens", str(max_tokens)
                ]
                try:
                    logger.info("Launching bridge: %s", " ".join(cmd))
                    proc = subprocess.Popen(cmd)
                    st.session_state.bridge_proc = proc
                    st.success(f"Started session {sid} (Ollama Live)")
                except Exception as e:
                    logger.exception("Bridge launch failed: %s", e)
                    st.error(f"Bridge launch failed: {e}")
            else:
                st.info("Selected model is not Ollama-backed. Switch Run Mode to 'API Analyze Only' to use the Undeniable Test.")
        else:
            # API Analyze Only mode: no live streaming; guide user to Undeniable Test
            st.session_state.ws_stop = True
            st.session_state.ws_thread = None
            st.session_state.bridge_proc = None
            st.info("API Analyze Only: use the 'Undeniable Before/After Test (API)' section below to run evaluations.")

if apply_fl and st.session_state.session_id:
    st.session_state.session_failure_law = update_session_failure_law(st.session_state.session_id, lambda_val, tau_val)

if stop_col.button("Stop", use_container_width=True):
    st.session_state.ws_stop = True
    if st.session_state.ws_thread:
        st.session_state.ws_thread.join(timeout=2)
        st.session_state.ws_thread = None
    if st.session_state.bridge_proc:
        try:
            st.session_state.bridge_proc.terminate()
        except Exception:
            pass
        st.session_state.bridge_proc = None

# Main layout
st.title("Real-time Uncertainty Dashboard (Streamlit)")
status_cols = st.columns(5)
status_cols[0].metric("Session", st.session_state.session_id or "—")
status_cols[1].metric("Events", len(st.session_state.events))
if st.session_state.session_id and not st.session_state.session_failure_law:
    st.session_state.session_failure_law = fetch_session_failure_law(st.session_state.session_id)
if st.session_state.session_failure_law:
    thr = st.session_state.session_failure_law.get("hbar_thresholds", {})
    status_cols[2].metric("supercritical h*", f"{thr.get('supercritical', 0):.3f}")
    status_cols[3].metric("critical h*", f"{thr.get('critical', 0):.3f}")

export_col = status_cols[4]
if st.session_state.events:
    export_col.download_button("Export JSON", data=json.dumps(st.session_state.events).encode("utf-8"), file_name="stream_events.json", mime="application/json")

# --- Mission Critical Real-Time Collapse Monitor ---
st.markdown("---")
st.subheader("🚨 Mission Critical: Real-Time Collapse Monitor")

# Gather recent window
recent_events = st.session_state.events[-30:]
if recent_events:
    # Extract series
    hs = [float(ev.get("hbar_s") or 0.0) for ev in recent_events]
    pf = [float(ev.get("failure_probability") or 0.0) for ev in recent_events]
    regs = [str(ev.get("regime") or "") for ev in recent_events]
    ts = [int(ev.get("timestamp_ms") or 0) for ev in recent_events]
    proc_ms = [float(ev.get("processing_time_ms") or 0.0) for ev in recent_events]
    last = recent_events[-1]

    current_hbar = hs[-1] if hs else 0.0
    current_pfail = pf[-1] if pf else 0.0
    current_regime = regs[-1] if regs else "unknown"

    # Trend via simple slope over last K points
    def slope(vals: List[float]) -> float:
        if len(vals) < 3:
            return 0.0
        x = list(range(len(vals)))
        xm = sum(x) / len(x)
        ym = sum(vals) / len(vals)
        num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, vals))
        den = sum((xi - xm) ** 2 for xi in x) or 1.0
        return num / den
    trend_slope = slope(hs)
    trend_arrow = "📈" if trend_slope > 1e-3 else ("📉" if trend_slope < -1e-3 else "➖")

    # Threshold proximity
    sc_thr = (st.session_state.session_failure_law or {}).get("hbar_thresholds", {}).get("supercritical")
    cr_thr = (st.session_state.session_failure_law or {}).get("hbar_thresholds", {}).get("critical")
    distance_label = "N/A"
    proximity_pct = 0
    if sc_thr is not None and cr_thr is not None:
        if current_hbar < sc_thr:
            distance = sc_thr - current_hbar
            span = max(cr_thr - sc_thr, 1e-6)
            proximity_pct = int(min(100, max(0, (1 - distance / span) * 100)))
            distance_label = f"{distance:.2f} from transitional"
        elif current_hbar < cr_thr:
            distance = cr_thr - current_hbar
            span = max(cr_thr - sc_thr, 1e-6)
            proximity_pct = int(min(100, max(0, (1 - distance / span) * 100)))
            distance_label = f"{distance:.2f} from unstable"
        else:
            proximity_pct = 100
            distance_label = "breached unstable"

    # Recommended action based on risk and trend
    risk_level = str(last.get("risk_level") or "safe")
    def recommend(risk: str, slope_val: float) -> str:
        if risk == "critical":
            return "Halt generation"
        if risk == "high_risk":
            return "Review immediately"
        if risk == "warning":
            return "Monitor closely" if slope_val <= 0 else "Review immediately"
        return "Continue normal"
    recommendation = recommend(risk_level, trend_slope)

    # Primary alert row
    a1, a2, a3, a4 = st.columns([2, 1, 1, 2])
    a1.metric("Pfail", f"{current_pfail*100:.1f}%")
    regime_badge = {
        "stable": "🟢 Stable",
        "transitional": "🟡 Transitional",
        "unstable": "🔴 Unstable",
    }.get(current_regime, current_regime)
    a2.metric("Regime", regime_badge)
    a3.metric("Trend", trend_arrow)
    a4.metric("Action", recommendation)

    # Threshold proximity visuals
    st.caption("Threshold Proximity")
    st.progress(proximity_pct)
    st.text(distance_label)

    # Sparkline
    try:
        import pandas as pd
        spark_df = pd.DataFrame({"hbar": hs})
        st.line_chart(spark_df, height=120)
    except Exception:
        pass

    # Context: model/session and tokens
    c1, c2, c3 = st.columns(3)
    c1.write(f"Model: {selected_model or '—'}   |   Session: {st.session_state.session_id or '—'}")
    c2.write(f"Tokens: {len(st.session_state.events)}   |   Avg latency: {sum(proc_ms)/max(1,len(proc_ms)):.1f} ms")
    last_tokens = "".join([e.get("token") or "" for e in recent_events[-5:]])
    c3.write(f"Recent: {last_tokens}")
else:
    st.info("Start a session to view real-time collapse monitoring.")

st.markdown("---")

# Generated Response
st.subheader("Generated Response")
st.write(st.session_state.generated or "")

# Recent tokens and basic metrics
left, right = st.columns([2, 1])

with left:
    st.subheader("Uncertainty Stream")
    # Build a simple time series from events
    xs: List[int] = []
    hbars: List[float] = []
    regimes: List[str] = []
    pfails: List[float] = []
    for ev in st.session_state.events[-300:]:
        xs.append(ev.get("token_index") or ev.get("data", {}).get("token", {}).get("position") or len(xs))
        hbars.append(float(ev.get("hbar_s") or ev.get("data", {}).get("audit", {}).get("current_uncertainty") or 0))
        regimes.append(ev.get("regime") or "")
        pfails.append(float(ev.get("failure_probability") or 0))
    try:
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame({"step": xs, "hbar_s": hbars, "pfail": pfails})
        fig = px.line(df, x="step", y=["hbar_s", "pfail"], markers=True)
        # Add regime bands if thresholds present
        if st.session_state.session_failure_law:
            thr = st.session_state.session_failure_law.get("hbar_thresholds", {})
            sc = thr.get("supercritical")
            cr = thr.get("critical")
            try:
                if sc is not None:
                    fig.add_hline(y=sc, line_dash="dot", line_color="green")
                if cr is not None:
                    fig.add_hline(y=cr, line_dash="dot", line_color="orange")
            except Exception:
                pass
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Install plotly/pandas to view charts")

with right:
    st.subheader("Recent Tokens")
    for ev in st.session_state.events[-20:][::-1]:
        tok = ev.get("token") or ev.get("data", {}).get("token", {}).get("text") or ""
        p = ev.get("probability") or ev.get("data", {}).get("token", {}).get("probability") or 0
        h = ev.get("hbar_s") or 0
        rl = ev.get("risk_level") or ""
        st.write(f"{tok}  |  p={p:.3f}  ℏₛ={h:.3f}  [{rl}]") 

# --- Undeniable Before/After API Test ---
st.markdown("---")
st.subheader("🧪 Undeniable Before/After Test (API)")

# Helpers for API analyze
def analyze_once(prompt_text: str, output_text: str) -> Optional[Dict]:
    try:
        payload = {"prompt": prompt_text, "output": output_text}
        r = requests.post(f"{ENGINE_HTTP}/analyze", json=payload, timeout=10)
        r.raise_for_status()
        resp = r.json()
        if not resp.get("success"):
            st.error(f"Analyze error: {resp.get('error')}")
            return None
        return resp.get("data") or None
    except Exception as e:
        st.error(f"Analyze request failed: {e}")
        return None

def pfail_from_hbar(hbar: float) -> float:
    # Fallback if API doesn't return p_fail
    # Try session failure law if available
    lam = 5.0
    tau = 1.0
    try:
        fl = st.session_state.get("session_failure_law") or {}
        if isinstance(fl, dict):
            lam = float(fl.get("lambda", lam))
            tau = float(fl.get("tau", tau))
    except Exception:
        pass
    try:
        import math
        # Increasing logistic in hbar: higher hbar => higher P_fail
        return 1.0 / (1.0 + math.exp(-lam * (hbar - tau)))
    except Exception:
        return 0.0

# Presets
presets = {
    "Capital of France": {
        "prompt": "What is the capital of France?",
        "failing": "Lyon is the capital of France.",
        "passing": "Paris is the capital of France.",
    },
    "Water Freezing": {
        "prompt": "At what temperature does water freeze?",
        "failing": "Water freezes at 50 degrees Celsius.",
        "passing": "Water freezes at 0 degrees Celsius.",
    },
    "Mathematics": {
        "prompt": "What is 2 + 2?",
        "failing": "2 + 2 equals 7.",
        "passing": "2 + 2 equals 4.",
    },
    "Custom": {
        "prompt": "",
        "failing": "",
        "passing": "",
    },
}

col_l, col_r = st.columns(2)
with col_l:
    sel = st.selectbox("Choose preset", list(presets.keys()), index=0)
    pr = st.text_area("Prompt", presets[sel]["prompt"], height=80)
    fail_out = st.text_area("❌ Failing Output", presets[sel]["failing"], height=80)
    pass_out = st.text_area("✅ Passing Output", presets[sel]["passing"], height=80)
    b1, b2, b3 = st.columns([1,1,1])
    if b1.button("Analyze Failing", use_container_width=True):
        res = analyze_once(pr, fail_out)
        if res:
            st.session_state["und_fail_res"] = res
            st.success("Failing analyzed")
    if b2.button("Analyze Passing", use_container_width=True):
        res = analyze_once(pr, pass_out)
        if res:
            st.session_state["und_pass_res"] = res
            st.success("Passing analyzed")
    if b3.button("Compare", use_container_width=True):
        st.session_state["und_compare"] = True

with col_r:
    st.markdown("**Instructions**")
    st.markdown("- Run failing and passing analyses, then Compare\n- Expect: failing has higher ℏₛ and higher P_fail; passing has lower ℏₛ and P_fail")

if st.session_state.get("und_compare") and st.session_state.get("und_fail_res") and st.session_state.get("und_pass_res"):
    f = st.session_state["und_fail_res"]
    p = st.session_state["und_pass_res"]
    f_h = float(f.get("hbar_s", 0.0))
    p_h = float(p.get("hbar_s", 0.0))
    f_mu = float(f.get("delta_mu", 0.0))
    p_mu = float(p.get("delta_mu", 0.0))
    f_sigma = float(f.get("delta_sigma", 0.0))
    p_sigma = float(p.get("delta_sigma", 0.0))
    f_pf = float(f.get("p_fail") if f.get("p_fail") is not None else pfail_from_hbar(f_h))
    p_pf = float(p.get("p_fail") if p.get("p_fail") is not None else pfail_from_hbar(p_h))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Failing")
        st.metric("ℏₛ", f"{f_h:.3f}")
        st.metric("P_fail", f"{f_pf*100:.1f}%")
        st.metric("Δμ", f"{f_mu:.3f}")
        st.metric("Δσ", f"{f_sigma:.3f}")
    with c2:
        st.markdown("#### Passing")
        st.metric("ℏₛ", f"{p_h:.3f}")
        st.metric("P_fail", f"{p_pf*100:.1f}%")
        st.metric("Δμ", f"{p_mu:.3f}")
        st.metric("Δσ", f"{p_sigma:.3f}")
    with c3:
        st.markdown("#### Improvement")
        st.metric("Δℏₛ", f"{(f_h - p_h):.3f}")
        st.metric("ΔP_fail", f"{(f_pf - p_pf)*100:.1f}%")

    # Visual comparison
    try:
        import pandas as pd
        import plotly.graph_objects as go
        df = pd.DataFrame({
            "Metric": ["ℏₛ", "P_fail", "Δμ", "Δσ"],
            "Failing": [f_h, f_pf, f_mu, f_sigma],
            "Passing": [p_h, p_pf, p_mu, p_sigma],
        })
        fig = go.Figure()
        fig.add_bar(name="Failing", x=df["Metric"], y=df["Failing"], marker_color="crimson", opacity=0.7)
        fig.add_bar(name="Passing", x=df["Metric"], y=df["Passing"], marker_color="seagreen", opacity=0.7)
        fig.update_layout(barmode='group', height=350)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # Equation check
    eq_ok_f = abs(f_h - (f_mu * f_sigma) ** 0.5) < 1e-3
    eq_ok_p = abs(p_h - (p_mu * p_sigma) ** 0.5) < 1e-3
    st.markdown(f"Equation check (ℏₛ = √(Δμ×Δσ)): Failing: {'✅' if eq_ok_f else '❌'}  |  Passing: {'✅' if eq_ok_p else '❌'}") 