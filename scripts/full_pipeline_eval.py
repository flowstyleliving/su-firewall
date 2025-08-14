#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import sys

@dataclass
class Pair:
	prompt: str
	failing: str
	passing: str


def default_pairs() -> List[Pair]:
	return [
		Pair("What is the capital of France?", "Lyon is the capital of France.", "Paris is the capital of France."),
		Pair("What is the capital of Japan?", "Seoul is the capital of Japan.", "Tokyo is the capital of Japan."),
		Pair("Which is the largest ocean on Earth?", "The Atlantic Ocean is the largest.", "The Pacific Ocean is the largest."),
		Pair("Who wrote the novel '1984'?", "Aldous Huxley wrote '1984'.", "George Orwell wrote '1984'."),
	]


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Full pipeline eval: models x methods with logits/topk + calibration")
	p.add_argument("--server_base", default="http://127.0.0.1:8080/api/v1", help="Runtime API base")
	p.add_argument("--hf_base", default="http://127.0.0.1:8899/logits", help="HF logits service base")
	p.add_argument("--ollama_base", default="http://127.0.0.1:8898/ollama_topk", help="Ollama TopK compact service endpoint")
	p.add_argument("--provider", choices=["hf", "ollama"], default="hf", help="Logits/topk provider")
	p.add_argument("--models_config", default="config/models.json", help="Path to models.json")
	p.add_argument("--methods", default="full_fim_dir,diag_fim_dir", help="Comma-separated methods")
	p.add_argument("--top_k", type=int, default=128, help="Top-k for sparse probs")
	p.add_argument("--ollama_num_samples", type=int, default=24, help="Sampling count for Ollama estimator (keep modest to avoid timeouts)")
	p.add_argument("--report", default="/tmp/full_eval_report.json", help="Output report path")
	p.add_argument("--write_models", action="store_true", help="Persist calibrated lambda/tau back to models.json")
	p.add_argument("--only_model", default=None, help="If set, only calibrate this model id/name")
	return p.parse_args()


def load_models(path: str) -> List[str]:
	try:
		with open(path, "r") as f:
			cfg = json.load(f)
	except Exception:
		return []
	models = []
	for m in cfg.get("models", []):
		for key in ("id", "name", "repo", "hf_id"):
			if key in m and isinstance(m[key], str):
				models.append(m[key])
				break
	return list(dict.fromkeys(models))


def load_models_config(path: str) -> Dict:
	try:
		with open(path, "r") as f:
			return json.load(f)
	except Exception:
		return {"models": []}


def get_model_entry(cfg: Dict, model_id: str) -> Optional[Dict]:
	for m in cfg.get("models", []):
		if m.get("id") == model_id or m.get("name") == model_id or m.get("repo") == model_id or m.get("hf_id") == model_id:
			return m
	return None


def derive_ollama_model_name(model_entry: Optional[Dict], fallback: str) -> str:
	if not model_entry:
		return fallback
	repo = model_entry.get("hf_repo") or model_entry.get("repo") or ""
	if isinstance(repo, str) and repo.startswith("ollama/"):
		return repo.split("/", 1)[1]
	return fallback


def derive_hf_model_name(model_entry: Optional[Dict], fallback: str) -> str:
	if not model_entry:
		return fallback
	repo = model_entry.get("hf_repo") or model_entry.get("repo") or model_entry.get("hf_id")
	if isinstance(repo, str) and len(repo) > 0:
		return repo
	return fallback


def hf_topk(hf_base: str, model_id: str, prompt: str, forced_output: str, top_k: int) -> Dict:
	r = requests.post(hf_base, json={
		"model_id": model_id,
		"prompt": prompt,
		"forced_output": forced_output,
		"sparse_topk": True,
		"top_k": top_k,
	}, timeout=120)
	r.raise_for_status()
	return r.json()


def ollama_topk(ollama_base: str, model: str, prompt: str, forced_output: str, top_k: int, num_samples: int) -> Dict:
	r = requests.post(ollama_base, json={
		"model_id": model,
		"prompt": prompt,
		"top_k": top_k,
		"num_samples": num_samples,
		"forced_output": forced_output,
	}, timeout=180)
	r.raise_for_status()
	return r.json()


def analyze_topk(server_base: str, model_id: str, j: Dict, method: str) -> Dict:
	body = {
		"model_id": model_id,
		"topk_indices": j["topk_indices"],
		"topk_probs": j["topk_probs"],
		"rest_mass": j["rest_mass"],
		"prompt_next_topk_indices": j.get("prompt_next_topk_indices"),
		"prompt_next_topk_probs": j.get("prompt_next_topk_probs"),
		"prompt_next_rest_mass": j.get("prompt_next_rest_mass"),
		"vocab_size": j.get("vocab_size"),
		"method": method,
	}
	r = requests.post(f"{server_base}/analyze_topk", json=body, timeout=30)
	r.raise_for_status()
	return r.json()


def analyze_topk_compact(server_base: str, model_id: str, j: Dict, method: str) -> Dict:
	body = {
		"model_id": model_id,
		"prompt_next_topk_indices": j.get("prompt_next_topk_indices"),
		"prompt_next_topk_probs": j.get("prompt_next_topk_probs"),
		# Map service field to API expected name
		"prompt_next_rest_mass": j.get("prompt_next_topk_rest_mass", j.get("prompt_next_rest_mass", 0.0)),
		"topk_indices": j["topk_indices"],
		"topk_probs": j["topk_probs"],
		"rest_mass": j["rest_mass"],
		"method": method,
	}
	r = requests.post(f"{server_base}/analyze_topk_compact", json=body, timeout=30)
	r.raise_for_status()
	return r.json()


def _draw_progress(step: int, total: int, prefix: str = "") -> None:
	if total <= 0:
		return
	width = 40
	pct = min(max(step / total, 0.0), 1.0)
	filled = int(width * pct)
	bar = "#" * filled + "-" * (width - filled)
	sys.stdout.write(f"\r{prefix} [{bar}] {int(pct*100)}% ({step}/{total})")
	sys.stdout.flush()


def grid_calibrate_lambda_tau(hbars: List[Tuple[float,int]]) -> Tuple[float,float,float]:
	# hbars: list of (hbar, label) label=1 for failing, 0 for passing
	best = (5.0, 1.0, 1e9)
	for lam in [x/10 for x in range(5, 101, 5)]:
		for tau in [x/50 for x in range(0, 151, 5)]:
			loss = 0.0
			for h,y in hbars:
				p = 1.0/(1.0+math.exp(-lam*(h - tau)))
				# logistic log-loss
				p = max(1e-6, min(1-1e-6, p))
				loss += -(y*math.log(p) + (1-y)*math.log(1-p))
			if loss < best[2]:
				best = (lam, tau, loss)
	return best[0], best[1], best[2]


def main() -> int:
	args = parse_args()
	methods = [m.strip() for m in args.methods.split(',') if m.strip()]
	pairs = default_pairs()
	models = load_models(args.models_config)
	cfg = load_models_config(args.models_config)
	if not models:
		models = [os.environ.get('UNDENIABLE_MODEL','sshleifer/tiny-gpt2')]

	report = {"server_base": args.server_base, "hf_base": args.hf_base, "methods": methods, "results": []}

	total_models = len(models)
	model_index = 0
	for model_id in models:
		model_index += 1
		_draw_progress(model_index-1, total_models, prefix="Models")
		if args.only_model and model_id != args.only_model:
			continue
		model_entry = {"model": model_id, "methods": {}}
		m_cfg = get_model_entry(cfg, model_id)
		ollama_model_name = derive_ollama_model_name(m_cfg, model_id)
		hf_model_name = derive_hf_model_name(m_cfg, model_id)
		for method in methods:
			rows = []
			cal_data: List[Tuple[float,int]] = []
			total_pairs = len(pairs)
			pair_index = 0
			for pr in pairs:
				pair_index += 1
				_draw_progress(pair_index, total_pairs, prefix=f"{model_id} {method}")
				if args.provider == "hf":
					jf = hf_topk(args.hf_base, hf_model_name, pr.prompt, pr.failing, args.top_k)
					jp = hf_topk(args.hf_base, hf_model_name, pr.prompt, pr.passing, args.top_k)
					df = analyze_topk(args.server_base, model_id, jf, method)
					dp = analyze_topk(args.server_base, model_id, jp, method)
				else:
					jf = ollama_topk(args.ollama_base, ollama_model_name, pr.prompt, pr.failing, args.top_k, args.ollama_num_samples)
					jp = ollama_topk(args.ollama_base, ollama_model_name, pr.prompt, pr.passing, args.top_k, args.ollama_num_samples)
					df = analyze_topk_compact(args.server_base, model_id, jf, method)
					dp = analyze_topk_compact(args.server_base, model_id, jp, method)
			rows.append({
				"prompt": pr.prompt,
				"failing": {"hbar_s": df["hbar_s"], "p_fail": df["p_fail"], "free_energy": df.get("free_energy")},
				"passing": {"hbar_s": dp["hbar_s"], "p_fail": dp["p_fail"], "free_energy": dp.get("free_energy")},
			})
			cal_data.append((float(df["hbar_s"]), 1))
			cal_data.append((float(dp["hbar_s"]), 0))
		lam, tau, loss = grid_calibrate_lambda_tau(cal_data)
		# compute pass rate with calibrated pfail
		passes = 0
		for r in rows:
			pf_f = 1.0/(1.0+math.exp(-lam*(float(r["failing"]["hbar_s"]) - tau)))
			pf_p = 1.0/(1.0+math.exp(-lam*(float(r["passing"]["hbar_s"]) - tau)))
			r["failing"]["p_fail_cal"] = pf_f
			r["passing"]["p_fail_cal"] = pf_p
			if float(r["failing"]["hbar_s"])>float(r["passing"]["hbar_s"]) and pf_f>pf_p:
				passes += 1
		model_entry["methods"][method] = {
			"lambda": lam, "tau": tau, "calibration_loss": loss,
			"pass_rate": passes/len(rows) if rows else 0.0,
			"rows": rows,
		}
		report["results"].append(model_entry)
		_draw_progress(model_index, total_models, prefix="Models")

	with open(args.report, "w") as f:
		json.dump(report, f, indent=2)
	print(f"\nWrote report to {args.report}")

	if args.write_models:
		updated = False
		for result in report.get("results", []):
			mid = result.get("model")
			entry = get_model_entry(cfg, mid)
			if not entry:
				continue
			# prefer full_fim_dir if present, else any
			meth = None
			if "full_fim_dir" in result.get("methods", {}):
				meth = result["methods"]["full_fim_dir"]
			elif result.get("methods"):
				# take the first method
				mk = sorted(result["methods"].keys())[0]
				meth = result["methods"][mk]
			if not meth:
				continue
			entry.setdefault("failure_law", {})
			entry["failure_law"]["lambda"] = float(meth["lambda"])  # type: ignore
			entry["failure_law"]["tau"] = float(meth["tau"])        # type: ignore
			updated = True
		if updated:
			with open(args.models_config, "w") as f:
				json.dump(cfg, f, indent=2)
			print(f"Updated {args.models_config} with calibrated failure_law values")
	return 0

if __name__ == "__main__":
	sys.exit(main()) 