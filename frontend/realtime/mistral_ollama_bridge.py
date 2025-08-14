#!/usr/bin/env python3
import sys
import argparse
import json
import time
import math
import requests
from typing import Dict, List, Optional

ENGINE_BASE = "http://127.0.0.1:3000/api/v1"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def softmax_from_logprobs(logprobs: Dict[str, float], top_k_floor: float = 1e-9) -> Dict[str, float]:
    if not logprobs:
        return {}
    # logprobs may be in natural log or log10; Ollama returns natural log base e
    # Compute normalized probabilities from provided logprobs (top-k only)
    max_logp = max(logprobs.values())
    exp_vals = {tok: math.exp(lp - max_logp) for tok, lp in logprobs.items()}
    s = sum(exp_vals.values()) or 1.0
    probs = {tok: max(top_k_floor, v / s) for tok, v in exp_vals.items()}
    # renormalize after floor
    s2 = sum(probs.values()) or 1.0
    probs = {tok: v / s2 for tok, v in probs.items()}
    return probs


def shannon_entropy_from_probs(probs: Dict[str, float]) -> float:
    if not probs:
        return 0.0
    # entropy in nats
    return -sum(p * math.log(max(p, 1e-12)) for p in probs.values())


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    # operate on union of keys
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    def _v(d, k):
        return d.get(k, 0.0)
    m = {k: 0.5 * (_v(p,k) + _v(q,k)) for k in keys}
    def _kl(a, b):
        s = 0.0
        for k in keys:
            ak = max(_v(a,k), 1e-12)
            bk = max(_v(b,k), 1e-12)
            s += ak * math.log(ak / bk)
        return s
    return 0.5 * (_kl(p, m) + _kl(q, m))


def post_token_metrics(session_id: str, idx: int, token: str, prob: float, entropy: float, jsd: float) -> None:
    payload = {
        "token_index": idx,
        "token_text": token,
        "probability": float(max(prob, 1e-12)),
        "entropy": float(max(entropy, 0.0)),
        "jsd": float(max(jsd, 0.0)),
    }
    url = f"{ENGINE_BASE}/session/{session_id}/token_metrics"
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[bridge] warn: failed posting metrics: {e}", file=sys.stderr)


def stream_from_ollama(model: str, prompt: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float, repeat_last_n: int):
    # Request streaming generation with logprobs
    req = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "logprobs": 20,
            "repeat_penalty": repetition_penalty,
            "repeat_last_n": repeat_last_n
        }
    }
    with requests.post(OLLAMA_URL, json=req, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def run(session_id: str, prompt: str, model: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float, repeat_last_n: int):
    prev_probs: Optional[Dict[str, float]] = None
    idx = 0
    # Accumulate probs for current token if provided across chunks (Ollama may send per-token)
    for chunk in stream_from_ollama(model, prompt, temperature, top_p, max_tokens, repetition_penalty, repeat_last_n):
        if chunk.get("done"):
            break
        token = chunk.get("response", "")
        # Ollama may include logprobs structure under 'logprobs' or 'top_logprobs'
        # Expect format: { "logprobs": { "<token>": logp, ... } } or a list; handle both gracefully
        lp_map: Dict[str, float] = {}
        if isinstance(chunk.get("logprobs"), dict):
            lp_map = chunk["logprobs"]
        elif isinstance(chunk.get("logprobs"), list) and chunk["logprobs"]:
            # list of {token, logprob}
            for entry in chunk["logprobs"]:
                tok = entry.get("token")
                lp = entry.get("logprob")
                if tok is not None and lp is not None:
                    lp_map[tok] = lp
        elif isinstance(chunk.get("top_logprobs"), list) and chunk["top_logprobs"]:
            for entry in chunk["top_logprobs"]:
                if isinstance(entry, dict):
                    lp_map.update({k: v for k, v in entry.items() if isinstance(v, (int, float))})
        # Build approx distribution from top-k
        probs_map = softmax_from_logprobs(lp_map) if lp_map else {}
        # Observed prob estimate
        observed_prob = probs_map.get(token, None)
        if observed_prob is None:
            # If not available, fallback to a small probability mass for the observed token
            observed_prob = 1e-3
        # Entropy and JSD
        entropy = shannon_entropy_from_probs(probs_map)
        jsd = js_divergence(prev_probs or {}, probs_map or {}) if prev_probs is not None else 0.0
        prev_probs = probs_map or prev_probs
        post_token_metrics(session_id, idx, token, observed_prob, entropy, jsd)
        idx += 1


def main():
    ap = argparse.ArgumentParser(description="Ollama → Engine bridge for real inference")
    ap.add_argument("session_id", help="Existing session id from the engine")
    ap.add_argument("prompt", help="Prompt text")
    ap.add_argument("--model", default="mistral:7b", help="Ollama model name (e.g., mistral:7b)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--repeat_last_n", type=int, default=256)
    args = ap.parse_args()

    print(f"[bridge] starting with session={args.session_id} model={args.model}")
    print("[bridge] note: ensure ollama is running: ollama run mistral")
    try:
        run(args.session_id, args.prompt, args.model, args.temperature, args.top_p, args.max_tokens, args.repetition_penalty, args.repeat_last_n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main() 