#!/usr/bin/env python3
import os
import json
import time
import threading
import streamlit as st
import requests
from typing import Optional, Dict, List
from models_registry import ModelsRegistry

# Config: base URL for realtime API
REALTIME_BASE = os.environ.get("REALTIME_API_BASE", "http://127.0.0.1:8080")
HTTP = REALTIME_BASE.rstrip("/")
WS = HTTP.replace("http://", "ws://").replace("https://", "wss://")

st.set_page_config(page_title="Realtime Uncertainty", layout="wide")

# Health
def api_health() -> Dict:
	try:
		r = requests.get(f"{HTTP}/health", timeout=3)
		return {"ok": r.ok, "data": r.json() if r.ok else {"status": r.status_code}}
	except Exception as e:
		return {"ok": False, "error": str(e)}

h = api_health()
left, right = st.columns([3,2])
with left:
	st.caption("Realtime API")
	st.success(f"OK {HTTP}") if h.get("ok") else st.error(f"Unreachable {HTTP}")
with right:
	st.json(h.get("data") or h)

# Models from repository config (display only)
try:
	reg = ModelsRegistry.load_from_file()
	models = [m.__dict__ for m in reg.list()]
	default_model_id = reg.default_id
except Exception:
	models, default_model_id = [], None

model_ids = [m.get("id") for m in models]
sel_model = st.sidebar.selectbox("Model", model_ids, index=0 if model_ids else None)
prompt = st.sidebar.text_area("Prompt", "Explain quantum computing to an 8th grader", height=120)

# Session controls
if "session_id" not in st.session_state:
	st.session_state.session_id = None
if "events" not in st.session_state:
	st.session_state.events = []
if "ws_thread" not in st.session_state:
	st.session_state.ws_thread = None
if "ws_stop" not in st.session_state:
	st.session_state.ws_stop = False

# Create/Close session
col1, col2 = st.columns(2)
with col1:
	if st.button("Start Session", use_container_width=True) and not st.session_state.session_id:
		try:
			resp = requests.post(f"{HTTP}/session/start", json={"model_id": sel_model or ""}, timeout=5)
			resp.raise_for_status()
			st.session_state.session_id = resp.json().get("session_id") or resp.json().get("id") or ""
			st.success(f"Session: {st.session_state.session_id}")
		except Exception as e:
			st.error(f"Session start failed: {e}")
with col2:
	if st.button("Close Session", use_container_width=True) and st.session_state.session_id:
		try:
			requests.post(f"{HTTP}/session/{st.session_state.session_id}/close", timeout=5)
			st.session_state.session_id = None
			st.session_state.ws_stop = True
			st.session_state.ws_thread = None
			st.session_state.events.clear()
			st.success("Session closed")
		except Exception as e:
			st.warning(f"Close failed: {e}")

# WS listener
def ws_loop():
	import websocket
	try:
		ws = websocket.create_connection(f"{WS}/ws", timeout=5)
		ws.settimeout(1.0)
		while not st.session_state.ws_stop:
			try:
				msg = ws.recv()
			except Exception:
				continue
			try:
				ev = json.loads(msg)
				st.session_state.events.append(ev)
			except Exception:
				pass
	finally:
		try:
			ws.close()
		except Exception:
			pass

# Start stream
if st.session_state.session_id:
	if st.session_state.ws_thread is None:
		st.session_state.ws_stop = False
		st.session_state.ws_thread = threading.Thread(target=ws_loop, daemon=True)
		st.session_state.ws_thread.start()
		st.info("WS listening...")

# Events view
st.subheader("Events")
ev_left, ev_right = st.columns([2,1])
with ev_left:
	st.json(st.session_state.events[-50:])
with ev_right:
	st.metric("Events", len(st.session_state.events))

st.caption("Set REALTIME_API_BASE to change target (default http://127.0.0.1:8080)") 