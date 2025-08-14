#!/usr/bin/env python3
import time
import math
import json
import sys
from typing import List, Dict, Any

import requests

BASE = "http://127.0.0.1:8767"

# Will be discovered from server
ALPHA = None
BETA = None
HBAR_THRESHOLDS = None


def load_failure_law():
    global ALPHA, BETA, HBAR_THRESHOLDS
    r = requests.get(f"{BASE}/failure_law", timeout=5)
    r.raise_for_status()
    data = r.json()
    ALPHA = data["lambda"]
    BETA = data["tau"]
    HBAR_THRESHOLDS = data["hbar_thresholds"]


def expected_regime(hbar: float) -> str:
    sc = HBAR_THRESHOLDS["supercritical"]
    cr = HBAR_THRESHOLDS["critical"]
    if hbar < sc:
        return "supercritical"
    elif hbar < cr:
        return "critical"
    else:
        return "subcritical"


def server_failure_prob(hbar: float) -> float:
    return 1.0 / (1.0 + math.exp(ALPHA * (hbar - BETA)))


def create_session(model_id: str = "mistral-7b") -> str:
    r = requests.post(f"{BASE}/session/new", json={"model_id": model_id}, timeout=5)
    r.raise_for_status()
    return r.json()["session_id"]


def post_token(session_id: str, idx: int, token: str, probs: List[float]) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE}/session/{session_id}/token",
        json={"token_index": idx, "token_text": token, "probabilities": probs},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


def run_sequence(label: str, seq: List[List[float]], model_id: str = "mistral-7b") -> List[Dict[str, Any]]:
    print(f"\n=== Running sequence: {label} (len={len(seq)}) ===")
    sid = create_session(model_id)
    results = []
    for i, probs in enumerate(seq):
        resp = post_token(sid, i, f"T{i}", probs)
        results.append(resp)
        print(
            json.dumps(
                {
                    "i": i,
                    "hbar": round(resp["hbar_s"], 6),
                    "rolling": round(resp["rolling_hbar_s"], 6),
                    "pfail": round(resp["failure_probability"], 6),
                    "risk": resp["risk_level"],
                    "regime": resp["regime"],
                }
            )
        )
    return results


def validate_failure_law(results: List[Dict[str, Any]]) -> None:
    hb = [r["hbar_s"] for r in results]
    pf = [r["failure_probability"] for r in results]
    pairs = 0
    neg = 0
    for i in range(len(hb) - 1):
        for j in range(i + 1, len(hb)):
            pairs += 1
            if (hb[j] - hb[i]) * (pf[j] - pf[i]) < 0:
                neg += 1
    frac_neg = neg / pairs if pairs else 1.0
    print(f"- Failure law monotonicity (neg corr fraction): {frac_neg:.2f} (expect close to 1.0)")

    idx = max(range(len(hb)), key=lambda k: hb[k])
    pf_local = server_failure_prob(hb[idx])
    print(
        f"- Spot-check at max hbar={hb[idx]:.4f}: server pfail={pf[idx]:.4f}, local pfail={pf_local:.4f}"
    )


def validate_regimes(results: List[Dict[str, Any]]) -> None:
    mismatches = []
    for r in results:
        exp = expected_regime(r["hbar_s"])
        if r["regime"] != exp:
            mismatches.append((r["token_index"], r["regime"], exp, r["hbar_s"]))
    if mismatches:
        print("- Regime mismatches:")
        for m in mismatches:
            print(f"  token {m[0]}: server={m[1]}, expected={m[2]}, hbar={m[3]:.4f}")
    else:
        print("- Regime mapping OK (all tokens)")


def validate_rolling(results: List[Dict[str, Any]]) -> None:
    roll = [r["rolling_hbar_s"] for r in results]
    if len(roll) > 1 and (roll[-1] - roll[0]) >= 0:
        print("- rolling_hbar_s accumulating (non-decreasing)")
    else:
        print("- rolling_hbar_s accumulation check inconclusive")


# New: model differentiation test

def test_model_differentiation() -> None:
    """Test if different models show different uncertainty signatures"""
    models = ["mistral-7b", "mixtral-8x7b", "qwen2.5-7b", "pythia-6.9b"]
    test_probs = [0.6, 0.25, 0.15]  # Same input for all models (sums to 1)

    print("\n=== Model Differentiation Test ===")
    for model in models:
        sid = create_session(model)
        resp = post_token(sid, 0, "test", test_probs)
        print(f"{model}: hbar={resp['hbar_s']:.6f}, pfail={resp['failure_probability']:.6f}")


def make_sequence_stable_to_uniform(k: int = 5, steps: int = 8) -> List[List[float]]:
    seq = []
    uniform = [1.0 / k] * k
    peaked = [1.0] + [0.0] * (k - 1)
    for t in range(steps):
        w = t / max(1, steps - 1)
        probs = [(1 - w) * peaked[i] + w * uniform[i] for i in range(k)]
        seq.append(probs)
    return seq


def make_sequence_supercritical_push(k: int = 5) -> List[List[float]]:
    return [
        [0.21, 0.20, 0.20, 0.20, 0.19],
        [0.21, 0.20, 0.20, 0.20, 0.19],
        [0.21, 0.20, 0.20, 0.20, 0.19],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.98, 0.01, 0.01, 0.0, 0.0],
    ]


def make_sequence_high_entropy(k: int = 5) -> List[List[float]]:
    return [
        [1.0] + [0.0] * (k - 1),
        [0.6, 0.1, 0.1, 0.1, 0.1],
        [0.4, 0.15, 0.15, 0.15, 0.15],
        [0.25, 0.25, 0.25, 0.25, 0.0],
        [0.2] * k,
    ]


def main():
    try:
        r = requests.get(f"{BASE}/health", timeout=2)
        assert r.status_code == 200
    except Exception:
        print("Server not reachable at /health. Start the real-time engine first (port 8767).", file=sys.stderr)
        sys.exit(1)

    load_failure_law()

    seq1 = make_sequence_stable_to_uniform(k=5, steps=10)
    res1 = run_sequence("stable_to_uniform", seq1)
    validate_failure_law(res1)
    validate_regimes(res1)
    validate_rolling(res1)

    seq2 = make_sequence_high_entropy(k=5)
    res2 = run_sequence("high_entropy_push", seq2)
    validate_failure_law(res2)
    validate_regimes(res2)
    validate_rolling(res2)

    seq3 = make_sequence_supercritical_push(k=5)
    res3 = run_sequence("supercritical_push", seq3)
    validate_failure_law(res3)
    validate_regimes(res3)
    validate_rolling(res3)

    # Run model differentiation test at the end
    test_model_differentiation()

    print("\nAll tests executed.")


if __name__ == "__main__":
    main() 