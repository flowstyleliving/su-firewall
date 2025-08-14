#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

try:
	from datasets import load_dataset
except ImportError:
	print("Warning: 'datasets' library not found. Hugging Face dataset loading disabled.", file=sys.stderr)


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
		Pair("What is the chemical symbol for sodium?", "The symbol for sodium is So.", "The symbol for sodium is Na."),
		Pair("Is the Sun a planet or a star?", "The Sun is a planet.", "The Sun is a star."),
		Pair("Where is the Great Barrier Reef located?", "The Great Barrier Reef is in the Mediterranean Sea.", "The Great Barrier Reef is off the coast of Australia."),
		Pair("What is the chemical formula for water?", "Water's formula is H3O.", "Water's formula is H2O."),
		Pair("Which country uses the Yen?", "South Korea uses the Yen.", "Japan uses the Yen."),
		Pair("Is Mount Everest the tallest mountain above sea level?", "No, Mount Everest is not the tallest.", "Yes, Mount Everest is the tallest above sea level."),
	]


def load_pairs_from_file(path: str) -> List[Pair]:
	with open(path, "r") as f:
		items = json.load(f)
	pairs: List[Pair] = []
	for it in items:
		pairs.append(Pair(prompt=it["prompt"], failing=it["failing"], passing=it["passing"]))
	return pairs


def load_truthfulqa(split: str = "validation", max_samples: Optional[int] = None) -> List[Pair]:
	try:
		dataset = load_dataset("truthful_qa", "generation", split=split)
	except ImportError:
		raise ValueError("Hugging Face 'datasets' library required for TruthfulQA loading.")
	pairs = []
	for row in dataset:
		if row['correct_answers'] and row['incorrect_answers']:
			passing = random.choice(row['correct_answers'])
			failing = random.choice(row['incorrect_answers'])
			pairs.append(Pair(prompt=row['question'], failing=failing, passing=passing))
	if max_samples:
		pairs = random.sample(pairs, min(max_samples, len(pairs)))
	return pairs


def load_halucval(task: str = "qa", max_samples: Optional[int] = None) -> List[Pair]:
	urls = {
		"qa": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
		"dialogue": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json",
		"summarization": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json",
		"general": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json",
	}
	if task not in urls:
		raise ValueError(f"Invalid HaluEval task: {task}. Choices: qa, dialogue, summarization, general")
	r = requests.get(urls[task])
	r.raise_for_status()
	items = json.loads(r.text)
	pairs = []
	for it in items:
		prompt_key = "question" if task == "qa" else "dialogue_history" if task == "dialogue" else "document" if task == "summarization" else "user_query"
		failing_key = "hallucinated_answer" if task == "qa" else "hallucinated_response" if task == "dialogue" else "hallucinated_summary" if task == "summarization" else "chatgpt_response"
		passing_key = "right_answer" if task == "qa" else "right_response" if task == "dialogue" else "right_summary" if task == "summarization" else None
		if task == "general" and it.get("hallucination_label") == "Yes":
			pairs.append(Pair(prompt=it[prompt_key], failing=it[failing_key], passing=it.get("chatgpt_response") if it.get("hallucination_label") == "No" else ""))
		else:
			pairs.append(Pair(prompt=it[prompt_key], failing=it[failing_key], passing=it[passing_key]))
	if max_samples:
		pairs = random.sample(pairs, min(max_samples, len(pairs)))
	return pairs


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run Comprehensive Hallucination Detection Test against analyze API")
	p.add_argument("--base", default="http://127.0.0.1:3000/api/v1", help="API base (default: %(default)s)")
	p.add_argument("--method", default="diag_fim_dir", help="Analysis method to request")
	p.add_argument("--repeats", type=int, default=1, help="Repeat evaluations N times to assess stability")
	p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
	p.add_argument("--threshold", type=float, default=0.8, help="Required pass rate to return 0 exit code")
	p.add_argument("--pairs", type=str, default=None, help="Optional JSON file with list of {prompt,failing,passing}")
	p.add_argument("--dataset", type=str, default="default", choices=["default", "truthfulqa", "halu_qa", "halu_dialogue", "halu_summarization", "halu_general"], help="Dataset source")
	p.add_argument("--max_samples", type=int, default=None, help="Max samples from dataset")
	p.add_argument("--lam", type=float, default=None, help="Override lambda for pfail mapping")
	p.add_argument("--tau", type=float, default=None, help="Override tau for pfail mapping")
	p.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots (e.g., roc.png, calibration.png)")
	return p.parse_args()


def analyze(base: str, method: str, prompt: str, output: str, timeout: float) -> Dict:
	url = f"{base}/analyze"
	r = requests.post(url, json={"prompt": prompt, "output": output, "method": method}, timeout=timeout)
	r.raise_for_status()
	resp = r.json()
	return resp if isinstance(resp, dict) else {}


def safe_float(v: Optional[float], default: float = 0.0) -> float:
	try:
		return float(v)
	except Exception:
		return default


def compute_ece(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
	prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy='uniform')
	ece = np.sum(np.abs(prob_true - prob_pred) * (np.histogram(probs, bins=n_bins, range=(0,1))[0] / len(probs)))
	return ece


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, dir: Optional[str]):
	plt.figure()
	plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc='lower right')
	if dir:
		plt.savefig(os.path.join(dir, 'roc.png'))
	else:
		plt.show()
	plt.close()


def plot_calibration(probs: List[float], labels: List[int], dir: Optional[str], n_bins: int = 10):
	prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy='uniform')
	plt.figure()
	plt.plot(prob_pred, prob_true, marker='o')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('Predicted Probability')
	plt.ylabel('Observed Probability')
	plt.title('Calibration Curve')
	if dir:
		plt.savefig(os.path.join(dir, 'calibration.png'))
	else:
		plt.show()
	plt.close()


def main() -> int:
	args = parse_args()

	if args.dataset == "default":
		pairs = load_pairs_from_file(args.pairs) if args.pairs else default_pairs()
	elif args.dataset == "truthfulqa":
		pairs = load_truthfulqa(max_samples=args.max_samples)
	elif "halu_" in args.dataset:
		task = args.dataset.replace("halu_", "")
		pairs = load_halucval(task, max_samples=args.max_samples)
	else:
		raise ValueError(f"Unknown dataset: {args.dataset}")

	# Health sanity
	try:
		hr = requests.get(f"{args.base}/health", timeout=args.timeout)
		hr.raise_for_status()
	except Exception as e:
		print(f"ERROR: Health check failed for {args.base}/health: {e}", file=sys.stderr)
		return 2

	print(f"Running Comprehensive Test: {len(pairs)} pairs x {args.repeats} repeats against {args.base} method={args.method} dataset={args.dataset}")

	passes = 0
	total = 0
	f_hbars: List[float] = []
	p_hbars: List[float] = []
	f_pfails: List[float] = []
	p_pfails: List[float] = []
	f_fep: List[float] = []
	p_fep: List[float] = []
	all_pfails: List[float] = []  # For ROC: all scores
	all_labels: List[int] = []   # 1 for failing, 0 for passing

	for r in range(args.repeats):
		for i, pr in enumerate(pairs):
			f_res = analyze(args.base, args.method, pr.prompt, pr.failing, args.timeout)
			p_res = analyze(args.base, args.method, pr.prompt, pr.passing, args.timeout)
			f_h = safe_float(f_res.get("hbar_s"))
			p_h = safe_float(p_res.get("hbar_s"))
			f_pf = safe_float(f_res.get("p_fail"))
			p_pf = safe_float(p_res.get("p_fail"))
			f_fe = safe_float(f_res.get("free_energy"))
			p_fe = safe_float(p_res.get("free_energy"))
			# Override pfail if params provided
			if args.lam is not None and args.tau is not None:
				lam = args.lam
				tau = args.tau
				f_pf = 1.0 / (1.0 + math.exp(-lam * (f_h - tau)))
				p_pf = 1.0 / (1.0 + math.exp(-lam * (p_h - tau)))

			ok = (f_h > p_h) and (f_pf > p_pf)
			passes += 1 if ok else 0
			total += 1
			f_hbars.append(f_h)
			p_hbars.append(p_h)
			f_pfails.append(f_pf)
			p_pfails.append(p_pf)
			if f_fe is not None: f_fep.append(f_fe)
			if p_fe is not None: p_fep.append(p_fe)

			# For ROC/Calibration
			all_pfails.append(f_pf)
			all_labels.append(1)  # failing
			all_pfails.append(p_pf)
			all_labels.append(0)  # passing

			print(json.dumps({
				"pair_index": i,
				"repeat": r,
				"prompt": pr.prompt,
				"failing_hbar_s": round(f_h, 6),
				"passing_hbar_s": round(p_h, 6),
				"failing_p_fail": round(f_pf, 6),
				"passing_p_fail": round(p_pf, 6),
				"failing_free_energy": round(f_fe, 6) if f_fe is not None else None,
				"passing_free_energy": round(p_fe, 6) if p_fe is not None else None,
				"pass": ok,
			}))

	pass_rate = passes / total if total else 0.0
	avg_f_h = np.mean(f_hbars) if f_hbars else 0.0
	avg_p_h = np.mean(p_hbars) if p_hbars else 0.0
	avg_f_pf = np.mean(f_pfails) if f_pfails else 0.0
	avg_p_pf = np.mean(p_pfails) if p_pfails else 0.0
	avg_f_fe = np.mean(f_fep) if f_fep else 0.0
	avg_p_fe = np.mean(p_fep) if p_fep else 0.0

	# Advanced Metrics
	if all_pfails and all_labels:
		roc_auc = roc_auc_score(all_labels, all_pfails)
		brier = brier_score_loss(all_labels, all_pfails)
		ece = compute_ece(all_pfails, all_labels)
		t_stat, p_val = stats.ttest_ind(f_pfails, p_pfails, equal_var=False)  # Welch's t-test on pfail deltas
	else:
		roc_auc = brier = ece = t_stat = p_val = None

	# Plots
	if args.plot_dir:
		os.makedirs(args.plot_dir, exist_ok=True)
		from sklearn.metrics import roc_curve
		fpr, tpr, _ = roc_curve(all_labels, all_pfails)
		plot_roc(fpr, tpr, roc_auc, args.plot_dir)
		plot_calibration(all_pfails, all_labels, args.plot_dir)

	summary = {
		"dataset": args.dataset,
		"pairs": len(pairs),
		"repeats": args.repeats,
		"total_cases": total,
		"passes": passes,
		"pass_rate": round(pass_rate, 4),
		"avg_hbar_failing": round(avg_f_h, 6),
		"avg_hbar_passing": round(avg_p_h, 6),
		"avg_pfail_failing": round(avg_f_pf, 6),
		"avg_pfail_passing": round(avg_p_pf, 6),
		"avg_free_energy_failing": round(avg_f_fe, 6),
		"avg_free_energy_passing": round(avg_p_fe, 6),
		"delta_hbar": round(avg_f_h - avg_p_h, 6),
		"delta_pfail": round(avg_f_pf - avg_p_pf, 6),
		"delta_free_energy": round(avg_f_fe - avg_p_fe, 6),
		"roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
		"brier_score": round(brier, 4) if brier is not None else None,
		"ece": round(ece, 4) if ece is not None else None,
		"t_test_p_value": round(p_val, 6) if p_val is not None else None,
	}
	print("=== Comprehensive Summary ===")
	print(json.dumps(summary, indent=2))

	return 0 if pass_rate >= args.threshold else 1


if __name__ == "__main__":
	sys.exit(main()) 