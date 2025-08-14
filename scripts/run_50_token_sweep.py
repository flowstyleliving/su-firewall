#!/usr/bin/env python3
import sys
import json
import math
from typing import Dict, Any, List

import requests

BASE = "http://127.0.0.1:8767"
MODELS = [
    "mistral-7b",
    "mixtral-8x7b",
    "qwen2.5-7b",
    "pythia-6.9b",
]


def health_check() -> None:
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        print("Server not reachable at /health. Start the real-time engine first (port 8767).", file=sys.stderr)
        sys.exit(1)


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


def run_sweep_for_model(model_id: str, steps: int = 50) -> None:
    sid = create_session(model_id)
    fl = get_session_failure_law(sid)

    stable = trans = unstable = 0
    last = None

    for i in range(steps):
        conf = 0.95 - i * (0.95 - 0.33) / max(1, steps - 1)
        rem = (1.0 - conf) / 2.0
        probs = [conf, rem, rem]
        resp = post_token(sid, i, f"T{i}", probs)
        reg = resp.get("regime")
        if reg == "stable":
            stable += 1
        elif reg == "transitional":
            trans += 1
        elif reg == "unstable":
            unstable += 1
        last = resp

    print(f"\n=== {model_id} ===")
    print(json.dumps({
        "session_id": sid,
        "failure_law": {
            "lambda": fl.get("lambda"),
            "tau": fl.get("tau"),
            "hbar_thresholds": fl.get("hbar_thresholds"),
        },
        "counts": {"stable": stable, "transitional": trans, "unstable": unstable},
    }, indent=2))

    if last:
        summary = {
            "final_token": last.get("token_index"),
            "hbar_s": last.get("hbar_s"),
            "rolling_hbar_s": last.get("rolling_hbar_s"),
            "pfail": last.get("failure_probability"),
            "regime": last.get("regime"),
        }
        print(json.dumps(summary, indent=2))


def main() -> None:
    health_check()

    # Allow overriding models from CLI
    models = MODELS
    if len(sys.argv) > 1:
        models = sys.argv[1:]

    for m in models:
        run_sweep_for_model(m, steps=50)


if __name__ == "__main__":
    main() 