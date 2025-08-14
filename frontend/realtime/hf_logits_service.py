#!/usr/bin/env python3
import os
import json
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="HF Logits Service")

MODEL_CACHE = {}

def load_model(model_id: str):
	if model_id in MODEL_CACHE:
		return MODEL_CACHE[model_id]
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(model_id)
	model.eval()
	MODEL_CACHE[model_id] = (tokenizer, model)
	return tokenizer, model

class LogitsRequest(BaseModel):
	model_id: str
	prompt: str
	max_tokens: int = 32
	temperature: float = 1.0
	return_gradients: bool = False
	forced_output: Optional[str] = None
	# Sparse top-k controls
	sparse_topk: bool = False
	top_k: int = 256

@app.post("/logits")
def logits(req: LogitsRequest):
	tokenizer, model = load_model(req.model_id)
	prompt_text = req.prompt
	forced = req.forced_output
	enc_prompt = tokenizer(prompt_text, return_tensors="pt").to(model.device)

	with torch.no_grad():
		prompt_logits = model(**enc_prompt).logits  # [1, seq, vocab]
		prompt_next = prompt_logits[:, -1, :].squeeze(0)

	def softmax(x: torch.Tensor) -> torch.Tensor:
		xm = x.max().item()
		exp = torch.exp(x - xm)
		return exp / exp.sum().clamp_min(1e-30)

	def topk_probs(x: torch.Tensor, k: int):
		p = softmax(x)
		val, idx = torch.topk(p, k=min(k, p.shape[-1]))
		rest = (1.0 - val.sum().item())
		return idx.tolist(), val.tolist(), max(0.0, float(rest))

	resp = {
		"temperature": req.temperature,
	}

	# Prompt side outputs
	if req.sparse_topk:
		pi, pv, prest = topk_probs(prompt_next, req.top_k)
		resp["prompt_next_topk_indices"] = pi
		resp["prompt_next_topk_probs"] = pv
		resp["prompt_next_rest_mass"] = prest
	else:
		resp["prompt_next_logits"] = prompt_next.detach().cpu().tolist()

	token_logits: List[List[float]] = []
	topk_indices: List[List[int]] = []
	topk_probs_list: List[List[float]] = []
	rest_masses: List[float] = []

	if forced:
		enc_forced = tokenizer(forced, add_special_tokens=False, return_tensors="pt").to(model.device)
		input_ids = torch.cat([enc_prompt.input_ids, enc_forced.input_ids], dim=1)
		attention_mask = torch.cat([enc_prompt.attention_mask, enc_forced.attention_mask], dim=1)
		with torch.no_grad():
			out = model(input_ids=input_ids, attention_mask=attention_mask)
			logits_full = out.logits
			forced_len = enc_forced.input_ids.shape[1]
			start_idx = logits_full.shape[1] - forced_len
			for pos in range(start_idx, logits_full.shape[1]):
				vec = logits_full[:, pos - 1, :].squeeze(0) if pos > 0 else logits_full[:, 0, :].squeeze(0)
				if req.sparse_topk:
					idx, val, rest = topk_probs(vec, req.top_k)
					topk_indices.append(idx)
					topk_probs_list.append(val)
					rest_masses.append(rest)
				else:
					token_logits.append(vec.detach().cpu().tolist())
	else:
		vec = prompt_next
		if req.sparse_topk:
			idx, val, rest = topk_probs(vec, req.top_k)
			topk_indices.append(idx)
			topk_probs_list.append(val)
			rest_masses.append(rest)
		else:
			token_logits.append(vec.detach().cpu().tolist())

	# Build vocab if dense
	if not req.sparse_topk:
		vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(model.config.vocab_size)]
		resp["vocab"] = vocab
		resp["token_logits"] = token_logits
	else:
		resp["topk_indices"] = topk_indices
		resp["topk_probs"] = topk_probs_list
		resp["rest_mass"] = rest_masses
		resp["vocab_size"] = int(model.config.vocab_size)

	return resp

if __name__ == "__main__":
	import uvicorn
	host = os.environ.get("HF_SERVICE_HOST", "127.0.0.1")
	port = int(os.environ.get("HF_SERVICE_PORT", "8899"))
	uvicorn.run("hf_logits_service:app", host=host, port=port, reload=False, workers=1) 