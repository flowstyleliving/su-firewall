#!/usr/bin/env python3
import os
import json
import hashlib
from typing import Dict, List, Tuple

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Ollama TopK Service")

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

class OllamaTopkRequest(BaseModel):
	model_id: str
	prompt: str
	top_k: int = 128
	num_samples: int = 64
	temperature: float = 1.0
	forced_output: str | None = None

class OllamaTopkResponse(BaseModel):
	prompt_next_topk_indices: List[int]
	prompt_next_topk_probs: List[float]
	prompt_next_topk_rest_mass: float
	topk_indices: List[int]
	topk_probs: List[float]
	rest_mass: float


def _stable_token_id(token_text: str) -> int:
	# Use sha1 hash to create a stable 31-bit id
	h = hashlib.sha1(token_text.encode("utf-8")).digest()
	return int.from_bytes(h[:4], "big") & 0x7fffffff


def _sample_next_token(client: httpx.Client, model: str, prompt: str, temperature: float, seed: int) -> str:
	payload = {
		"model": model,
		"prompt": prompt,
		"stream": False,
		"raw": True,
		"options": {
			"num_predict": 1,
			"temperature": max(0.0, float(temperature)),
			"seed": int(seed),
		}
	}
	r = client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
	r.raise_for_status()
	data = r.json()
	return data.get("response", "")


def _estimate_topk(tokens: Dict[str, int], k: int) -> Tuple[List[int], List[float], float]:
	total = sum(tokens.values())
	if total <= 0:
		return [], [], 0.0
	items = sorted(tokens.items(), key=lambda x: x[1], reverse=True)
	top = items[: k]
	idx = [_stable_token_id(t) for (t, _) in top]
	probs = [cnt / total for (_, cnt) in top]
	s = float(sum(probs))
	rest = max(0.0, 1.0 - s)
	return idx, probs, rest


@app.post("/ollama_topk", response_model=OllamaTopkResponse)
def ollama_topk(req: OllamaTopkRequest):
	client = httpx.Client()
	try:
		counts_p: Dict[str, int] = {}
		counts_q: Dict[str, int] = {}
		prompt_p = req.prompt
		prompt_q = (req.prompt + " " + req.forced_output) if req.forced_output else req.prompt
		for i in range(req.num_samples):
			seed = 12345 + i
			tok_p = _sample_next_token(client, req.model_id, prompt_p, req.temperature, seed)
			counts_p[tok_p] = counts_p.get(tok_p, 0) + 1
			tok_q = _sample_next_token(client, req.model_id, prompt_q, req.temperature, seed + 9999)
			counts_q[tok_q] = counts_q.get(tok_q, 0) + 1

		pi, pv, prest = _estimate_topk(counts_p, req.top_k)
		qi, qv, qrest = _estimate_topk(counts_q, req.top_k)
		return OllamaTopkResponse(
			prompt_next_topk_indices=pi,
			prompt_next_topk_probs=pv,
			prompt_next_topk_rest_mass=prest,
			topk_indices=qi,
			topk_probs=qv,
			rest_mass=qrest,
		)
	finally:
		client.close()

if __name__ == "__main__":
	import uvicorn
	host = os.environ.get("OLLAMA_SERVICE_HOST", "127.0.0.1")
	port = int(os.environ.get("OLLAMA_SERVICE_PORT", "8898"))
	uvicorn.run("ollama_logits_service:app", host=host, port=port, reload=False, workers=1) 