#!/usr/bin/env python3
import sys
import json
from typing import List, Dict, Any

import requests

BASE = "http://127.0.0.1:8767"
MODELS = [
    "mistral-7b",
    "mixtral-8x7b",
    "qwen2.5-7b",
    "pythia-6.9b",
]

# Higher uncertainty sequence (3-way probs)
HIGH_UNCERTAINTY_SEQ: List[List[float]] = [
    [0.4, 0.3, 0.3],
    [0.35, 0.325, 0.325],
    [0.34, 0.33, 0.33],
]

# Extreme uncertainty sequence (3-way, stronger push)
EXTREME_UNCERTAINTY_SEQ: List[List[float]] = [
    [0.8, 0.1, 0.1],
    [0.5, 0.25, 0.25],
    [0.4, 0.3, 0.3],
    [0.35, 0.325, 0.325],
    [0.34, 0.33, 0.33],
]

# Very high entropy jump with K classes: peak -> uniform -> semi-peak

def make_jump_sequence(k: int = 10) -> List[List[float]]:
    seq: List[List[float]] = []
    peaked = [1.0] + [0.0] * (k - 1)
    uniform = [1.0 / k] * k
    tail = [0.0] * (k - 1)
    semi = [0.6] + [0.4 / (k - 1)] * (k - 1)
    seq.append(peaked)
    seq.append(uniform)
    seq.append(semi)
    return seq

def health_check() -> None:
    r = requests.get(f"{BASE}/health", timeout=3)
    r.raise_for_status()


def create_session(model_id: str) -> str:
    r = requests.post(f"{BASE}/session/new", json={"model_id": model_id}, timeout=5)
    r.raise_for_status()
    return r.json()["session_id"]


def get_session_failure_law(session_id: str) -> Dict[str, Any]:
    r = requests.get(f"{BASE}/session/{session_id}/failure_law", timeout=5)
    r.raise_for_status()
    return r.json()


def post_token(session_id: str, idx: int, token: str, probs: List[float]) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/session/{session_id}/token",
        json={"token_index": idx, "token_text": token, "probabilities": probs},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


def run_seq(model_id: str, label: str, seq: List[List[float]]) -> None:
    sid = create_session(model_id)
    fl = get_session_failure_law(sid)
    thr = fl.get("hbar_thresholds", {})

    print(f"\n=== {model_id} :: {label} ===")
    print(json.dumps({
        "session_id": sid,
        "lambda": fl.get("lambda"),
        "tau": fl.get("tau"),
        "hbar_thresholds": thr,
    }, indent=2))

    stable = trans = unstable = 0
    for i, probs in enumerate(seq):
        resp = post_token(sid, i, f"S{i}", probs)
        regime = resp.get("regime")
        if regime == "stable":
            stable += 1
        elif regime == "transitional":
            trans += 1
        elif regime == "unstable":
            unstable += 1
        print(json.dumps({
            "i": i,
            "probs": probs,
            "hbar": resp.get("hbar_s"),
            "rolling": resp.get("rolling_hbar_s"),
            "pfail": resp.get("failure_probability"),
            "regime": regime,
        }))

    print(json.dumps({"counts": {"stable": stable, "transitional": trans, "unstable": unstable}}, indent=2))


def main() -> None:
    try:
        health_check()
    except Exception:
        print("Server not reachable at /health. Start the real-time engine first (port 8767).", file=sys.stderr)
        sys.exit(1)

    models = MODELS if len(sys.argv) == 1 else sys.argv[1:]
    for m in models:
        run_seq(m, "3-way-high-uncertainty", HIGH_UNCERTAINTY_SEQ)
        run_seq(m, "3-way-extreme-uncertainty", EXTREME_UNCERTAINTY_SEQ)
        run_seq(m, "K10-entropy-jump", make_jump_sequence(k=10))


if __name__ == "__main__":
    main() 