#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
import time
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests


# Optional/soft deps: we gate their usage where needed
try:
	import numpy as np
except Exception:  # pragma: no cover
	np = None  # type: ignore

try:
	from sklearn.metrics import roc_auc_score, brier_score_loss
	from sklearn.calibration import calibration_curve
	from sklearn.model_selection import train_test_split
	from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
	roc_auc_score = None  # type: ignore
	brier_score_loss = None  # type: ignore
	calibration_curve = None  # type: ignore
	train_test_split = None  # type: ignore
	IsotonicRegression = None  # type: ignore

try:
	from scipy.stats import ttest_ind
	from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
	ttest_ind = None  # type: ignore
	PchipInterpolator = None  # type: ignore

try:
	from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception:  # pragma: no cover
	ThreadPoolExecutor = None  # type: ignore
	as_completed = None  # type: ignore


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


def safe_float(x: object, default: float = float("nan")) -> float:
	try:
		return float(x)  # type: ignore
	except Exception:
		return default


def analyze(session: requests.Session, base: str, prompt: str, output: str, timeout: float, model_id: Optional[str] = None, method: Optional[str] = None) -> float:
	"""POST to /analyze and return hbar_s as float."""
	payload: Dict[str, object] = {"prompt": prompt, "output": output}
	if model_id:
		payload["model_id"] = model_id
	if method:
		payload["method"] = method
	r = session.post(f"{base}/analyze", json=payload, timeout=timeout)
	r.raise_for_status()
	data = r.json()
	if isinstance(data, dict) and data.get("data"):
		data = data["data"]
	return safe_float(data.get("hbar_s") if isinstance(data, dict) else None)


def analyze_topk_compact(session: requests.Session, base: str, payload: Dict[str, object], timeout: float) -> float:
	"""POST to /analyze_topk_compact and return hbar_s as float."""
	r = session.post(f"{base}/analyze_topk_compact", json=payload, timeout=timeout)
	r.raise_for_status()
	data = r.json()
	if isinstance(data, dict) and data.get("data"):
		data = data["data"]
	return safe_float(data.get("hbar_s") if isinstance(data, dict) else None)


def analyze_logits(session: requests.Session, base: str, payload: Dict[str, object], timeout: float) -> float:
	"""POST to /analyze_logits and return hbar_s as float."""
	r = session.post(f"{base}/analyze_logits", json=payload, timeout=timeout)
	r.raise_for_status()
	data = r.json()
	if isinstance(data, dict) and data.get("data"):
		data = data["data"]
	return safe_float(data.get("hbar_s") if isinstance(data, dict) else None)


def grid_search(H: List[float], y: List[int], golden_scale: float = 1.0) -> Tuple[float, float, float]:
	best = (float("inf"), 0.0, 1.0)
	for lam in [x / 10.0 for x in range(5, 101)]:  # 0.5..10.0
		for tau in [x / 100.0 for x in range(0, 201, 2)]:  # 0..2.0 step .02
			loss = 0.0
			for h, label in zip(H, y):
				# Apply golden scale calibration
				h_calibrated = h * golden_scale if golden_scale > 0 else h
				p = 1.0 / (1.0 + math.exp(-lam * (h_calibrated - tau)))
				p = max(min(p, 1.0 - 1e-9), 1e-9)
				loss -= label * math.log(p) + (1 - label) * math.log(1 - p)
			if loss < best[0]:
				best = (loss, lam, tau)
	return best


def compute_pfail(h: float, lam: float, tau: float, golden_scale: float = 1.0) -> float:
	if not math.isfinite(h):
		return 0.5
	# Apply golden scale calibration for hallucination detection
	h_calibrated = h * golden_scale if golden_scale > 0 else h
	val = -lam * (h_calibrated - tau)
	# Clamp exponent to avoid overflow
	if val > 50:
		p = 1.0 / (1.0 + math.exp(50))
	elif val < -50:
		p = 1.0 / (1.0 + math.exp(-50))
	else:
		p = 1.0 / (1.0 + math.exp(val))
	return max(min(p, 1.0 - 1e-9), 1e-9)


def sanitize_xy(H: List[float], y: List[int]) -> Tuple[List[float], List[int]]:
	clean_H: List[float] = []
	clean_y: List[int] = []
	for h, yy in zip(H, y):
		if isinstance(yy, int) and (yy == 0 or yy == 1) and math.isfinite(h):
			clean_H.append(h)
			clean_y.append(yy)
	return clean_H, clean_y


def compute_ece(y_true: List[int], p_pred: List[float], num_bins: int = 15) -> Dict[str, float]:
	"""Expected Calibration Error via equal-width bins in [0,1]."""
	if not y_true or not p_pred:
		return {"ece": float("nan"), "mce": float("nan")}
	bin_edges = [i / num_bins for i in range(num_bins + 1)]
	bin_sums = [0.0] * num_bins
	bin_total = [0] * num_bins
	bin_conf = [0.0] * num_bins
	for y_i, p_i in zip(y_true, p_pred):
		b = min(num_bins - 1, max(0, int(p_i * num_bins)))
		bin_sums[b] += y_i
		bin_total[b] += 1
		bin_conf[b] += p_i
	ece = 0.0
	mce = 0.0
	n = len(y_true)
	for b in range(num_bins):
		if bin_total[b] == 0:
			continue
		exp_freq = bin_sums[b] / bin_total[b]
		conf = bin_conf[b] / bin_total[b]
		gap = abs(exp_freq - conf)
		ece += (bin_total[b] / n) * gap
		mce = max(mce, gap)
	return {"ece": ece, "mce": mce}


def plot_roc(y_true: List[int], p_pred: List[float], out_path: str) -> None:
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception:  # pragma: no cover
		return
	# Manual ROC if sklearn missing
	if roc_auc_score is None:
		# Simple ROC via thresholds
		thresholds = sorted(set(p_pred))
		fpr = []
		tpr = []
		pos = sum(y_true)
		neg = len(y_true) - pos
		for t in thresholds:
			pred = [1 if p >= t else 0 for p in p_pred]
			TP = sum(1 for pr, yt in zip(pred, y_true) if pr == 1 and yt == 1)
			FP = sum(1 for pr, yt in zip(pred, y_true) if pr == 1 and yt == 0)
			fpr.append(FP / max(1, neg))
			tpr.append(TP / max(1, pos))
		plt.figure()
		plt.plot(fpr, tpr, label="ROC")
		plt.xlabel("FPR")
		plt.ylabel("TPR")
		plt.title("ROC Curve")
		plt.legend()
		plt.tight_layout()
		plt.savefig(out_path)
		plt.close()
		return
	# With sklearn AUC and curve
	from sklearn.metrics import roc_curve
	fpr, tpr, _ = roc_curve(y_true, p_pred)
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc_score(y_true, p_pred):.3f}")
	plt.xlabel("FPR")
	plt.ylabel("TPR")
	plt.title("ROC Curve")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def plot_calibration(y_true: List[int], p_pred: List[float], out_path: str, n_bins: int = 15) -> None:
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception:  # pragma: no cover
		return
	if calibration_curve is not None:
		x, y = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy="uniform")
	else:
		# Manual binning
		bins = [[] for _ in range(n_bins)]
		obs = [[] for _ in range(n_bins)]
		for yt, pp in zip(y_true, p_pred):
			b = min(n_bins - 1, max(0, int(pp * n_bins)))
			bins[b].append(pp)
			obs[b].append(yt)
			x = []
			y = []
		for b in range(n_bins):
			if bins[b]:
				x.append(sum(bins[b]) / len(bins[b]))
				y.append(sum(obs[b]) / len(obs[b]))
	plt.figure()
	plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
	plt.plot(x, y, marker="o", label="Observed")
	plt.xlabel("Predicted probability")
	plt.ylabel("Observed frequency")
	plt.title("Calibration Curve")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def load_truthfulqa(max_samples: Optional[int], seed: int) -> List[Pair]:
	"""Load TruthfulQA from local authentic_datasets using comprehensive loader."""
	try:
		from comprehensive_dataset_loader import load_truthfulqa_fixed
		
		print(f"🔄 Loading TruthfulQA from local files...")
		evaluation_pairs = load_truthfulqa_fixed(max_samples)
		
		pairs = []
		for ep in evaluation_pairs:
			pairs.append(Pair(
				prompt=ep.prompt,
				failing=ep.hallucinated_answer,
				passing=ep.correct_answer
			))
		
		print(f"✅ Loaded {len(pairs)} TruthfulQA pairs from local dataset")
		return pairs
		
	except Exception as e:
		print(f"⚠️ Failed to load local TruthfulQA: {e}")
		# Fallback to original online loading
		try:
			from datasets import load_dataset  # type: ignore
			ds = load_dataset("truthful_qa", "generation")  # type: ignore
			examples = []
			for split in ("validation", "test", "train"):
				if split in ds:
					for row in ds[split]:
						question = row.get("question") or row.get("prompt") or ""
						# Prefer best answer field; otherwise choose a random correct/incorrect
						correct = row.get("best_answer") or (row.get("correct_answers") or [])
						incorrect = row.get("incorrect_answers") or []
						corr_str = correct if isinstance(correct, str) else (correct[0] if correct else "")
						inc_str = incorrect[0] if incorrect else ""
						if question and corr_str and inc_str:
							examples.append(Pair(prompt=question, failing=inc_str, passing=corr_str))
			random.Random(seed).shuffle(examples)
			if max_samples is not None:
				examples = examples[: max(0, int(max_samples))]
			return examples
		except Exception:
			return []


def load_halueval(task: str, max_samples: Optional[int], seed: int, override_url: Optional[str] = None) -> List[Pair]:
	"""Load HaluEval from local authentic_datasets using comprehensive loader."""
	try:
		from comprehensive_dataset_loader import load_halueval_fixed
		
		print(f"🔄 Loading HaluEval {task} from local files...")
		evaluation_pairs = load_halueval_fixed(task, max_samples)
		
		pairs = []
		for ep in evaluation_pairs:
			pairs.append(Pair(
				prompt=ep.prompt,
				failing=ep.hallucinated_answer,
				passing=ep.correct_answer
			))
		
		print(f"✅ Loaded {len(pairs)} HaluEval {task} pairs from local dataset")
		return pairs
		
	except Exception as e:
		print(f"⚠️ Failed to load local HaluEval {task}: {e}")
		# Fallback to original online loading
		urls_by_task = {
			"qa": [
				"https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
				"https://huggingface.co/datasets/RUCAIBox/HaluEval/resolve/main/data/qa_data.json",
			],
			"dialogue": [
				"https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json",
			],
			"summarization": [
				"https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json",
			],
			"general": [
				"https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json",
			],
		}
		urls = [override_url] if override_url else urls_by_task.get(task, [])
		s = requests.Session()
		rows: List[Dict[str, object]] = []
		for url in urls:
			try:
				r = s.get(url, timeout=20)
				r.raise_for_status()
				data = r.json()
				if isinstance(data, dict) and data.get("data"):
					data = data["data"]
				if isinstance(data, list):
					rows = data
					break
			except Exception:
				continue
	if not rows:
		return []
	items: List[Pair] = []
	for it in rows:
		prompt = (it.get("question") or it.get("prompt") or it.get("src") or "") if isinstance(it, dict) else ""
		correct = (it.get("correct") or it.get("answer") or it.get("reference") or "") if isinstance(it, dict) else ""
		incorrect = (it.get("incorrect") or it.get("negative") or it.get("hallucination") or "") if isinstance(it, dict) else ""
		if isinstance(correct, list):
			correct = correct[0] if correct else ""
		if isinstance(incorrect, list):
			incorrect = incorrect[0] if incorrect else ""
		if prompt and correct and incorrect:
			items.append(Pair(prompt=str(prompt), failing=str(incorrect), passing=str(correct)))
	random.Random(seed).shuffle(items)
	if max_samples is not None:
		items = items[: max(0, int(max_samples))]
	return items


def prepare_pairs(args: argparse.Namespace) -> List[Pair]:
	# Priority: explicit pairs JSON -> dataset selection -> defaults
	if args.pairs:
		with open(args.pairs, "r") as f:
			raw = json.load(f)
			pairs: List[Pair] = []
			for it in raw:
				pairs.append(Pair(prompt=it["prompt"], failing=it["failing"], passing=it["passing"]))
			return pairs
	if args.dataset == "default":
		pairs = default_pairs()
		if args.max_samples is not None:
			pairs = pairs[: max(0, int(args.max_samples))]
		return pairs
	if args.dataset == "truthfulqa":
		pairs = load_truthfulqa(args.max_samples, args.seed)
		return pairs
	if args.dataset.startswith("halueval"):
		# dataset form: halueval, or halueval-qa/dialogue/summarization/general
		task = args.halueval_task
		return load_halueval(task=task, max_samples=args.max_samples, seed=args.seed, override_url=args.halueval_url)
	# Fallback
	return default_pairs()


def iter_requests(pairs: List[Pair]) -> Iterable[Tuple[str, str, int]]:
	for pr in pairs:
		yield pr.prompt, pr.failing, 1
		yield pr.prompt, pr.passing, 0


def fetch_hbars_concurrent(base: str, items: List[Tuple[str, str, int]], timeout: float, concurrency: int, rate_limit: float, model_id: Optional[str] = None, method: Optional[str] = None) -> Tuple[List[float], List[int]]:
	"""Fetch hbars for items concurrently. Returns (H, y)."""
	session = requests.Session()
	H: List[float] = []
	y: List[int] = []
	results: List[Tuple[int, float, int]] = []
	indexed = list(enumerate(items))

	def work(idx_item: Tuple[int, Tuple[str, str, int]]) -> Tuple[int, float, int]:
		idx, (prompt, output, label) = idx_item
		if rate_limit and rate_limit > 0:
			time.sleep(1.0 / rate_limit)
		try:
			h = analyze(session, base, prompt, output, timeout, model_id=model_id, method=method)
			return (idx, h, label)
		except Exception:
			return (idx, float("nan"), label)

	if concurrency and concurrency > 1 and ThreadPoolExecutor is not None:
		with ThreadPoolExecutor(max_workers=concurrency) as ex:
			for res in ex.map(work, indexed):
				results.append(res)
	else:
		for it in indexed:
			results.append(work(it))

	# Restore original order
	results.sort(key=lambda x: x[0])
	for _, h, label in results:
		H.append(h)
		y.append(label)
	return H, y


def split_pairs(pairs: List[Pair], val_split: float, test_split: float, seed: int) -> Tuple[List[Pair], List[Pair], List[Pair]]:
	if val_split + test_split >= 1.0:
		raise ValueError("val_split + test_split must be < 1.0")
	order = list(range(len(pairs)))
	random.Random(seed).shuffle(order)
	n = len(order)
	val_n = int(n * val_split)
	test_n = int(n * test_split)
	val_idx = set(order[:val_n])
	test_idx = set(order[val_n:val_n + test_n])
	train_idx = set(order[val_n + test_n:])
	train = [pairs[i] for i in sorted(train_idx)]
	val = [pairs[i] for i in sorted(val_idx)]
	test = [pairs[i] for i in sorted(test_idx)]
	return train, val, test


def compute_metrics(y_true: List[int], p_pred: List[float]) -> Dict[str, float]:
	metrics: Dict[str, float] = {}
	if roc_auc_score is not None:
		try:
			metrics["roc_auc"] = float(roc_auc_score(y_true, p_pred))
		except Exception:
			metrics["roc_auc"] = float("nan")
	else:
		metrics["roc_auc"] = float("nan")
	if brier_score_loss is not None:
		try:
			metrics["brier"] = float(brier_score_loss(y_true, p_pred))
		except Exception:
			metrics["brier"] = float("nan")
	else:
		metrics["brier"] = float("nan")
	cal = compute_ece(y_true, p_pred)
	metrics.update(cal)
	# Welch's t-test on predicted scores between classes
	if ttest_ind is not None:
		try:
			pos = [p for p, y in zip(p_pred, y_true) if y == 1]
			neg = [p for p, y in zip(p_pred, y_true) if y == 0]
			if pos and neg:
				stat, pval = ttest_ind(pos, neg, equal_var=False)
				metrics["welch_t_stat"] = float(stat)
				metrics["welch_t_pval"] = float(pval)
			else:
				metrics["welch_t_stat"] = float("nan")
				metrics["welch_t_pval"] = float("nan")
		except Exception:
			metrics["welch_t_stat"] = float("nan")
			metrics["welch_t_pval"] = float("nan")
	else:
		metrics["welch_t_stat"] = float("nan")
		metrics["welch_t_pval"] = float("nan")
	return metrics


def _append_suffix_to_path(path: Optional[str], suffix: str) -> Optional[str]:
	if not path:
		return None
	root, ext = os.path.splitext(path)
	return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"


def calibrate_for_model(args: argparse.Namespace, model_id: Optional[str]) -> Dict[str, object]:
	# Load pairs per args
	pairs = prepare_pairs(args)
	if not pairs:
		return {"error": "No pairs loaded.", "model_id": model_id}

	# Split
	train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.val_split, args.test_split, args.seed)

	# Items
	train_items = list(iter_requests(train_pairs))
	val_items = list(iter_requests(val_pairs))
	test_items = list(iter_requests(test_pairs))

	# Fetch: prefer real-provider path when requested, else fallback to /analyze
	def _provider_url() -> str:
		if args.provider_url:
			return args.provider_url
		return "http://127.0.0.1:8899/logits" if args.real_provider == "hf" else "http://127.0.0.1:8898/ollama_topk"

	def _resolve_provider_model_id() -> Optional[str]:
		"""Map models.json entries to provider-specific model identifiers."""
		if args.provider_model_override:
			return args.provider_model_override
		try:
			with open(args.models_json, "r") as f:
				models_doc = json.load(f)
			for m in models_doc.get("models", []):
				if isinstance(m, dict) and m.get("id") == model_id:
					repo = str(m.get("hf_repo", ""))
					if args.real_provider == "hf":
						# Expect HF repo id like mistralai/Mistral-7B-Instruct-v0.2
						return repo or model_id
					if args.real_provider == "ollama":
						# Expect ollama/<model:tag> in hf_repo
						if repo.startswith("ollama/"):
							return repo.split("/", 1)[1]
						return model_id
		except Exception:
			pass
		# Fallbacks
		if args.real_provider == "hf":
			return model_id or "mistralai/Mistral-7B-Instruct-v0.2"
		if args.real_provider == "ollama":
			return model_id or "mistral:7b"
		return model_id

	def _fetch_with_provider(items: List[Tuple[str, str, int]]) -> Tuple[List[float], List[int]]:
		s = requests.Session()
		H: List[float] = []
		Y: List[int] = []
		for prompt, output, label in items:
			try:
				if args.real_provider == "hf":
					# Request sparse top-k for prompt-next and output path
					hf_payload = {
						"model_id": _resolve_provider_model_id(),
						"prompt": prompt,
						"forced_output": output,
						"sparse_topk": (args.logits_mode == "topk"),
						"top_k": int(args.top_k),
						"temperature": float(args.temperature),
					}
					r = s.post(_provider_url(), json=hf_payload, timeout=max(30.0, args.timeout))
					r.raise_for_status()
					resp = r.json()
					if args.logits_mode == "topk":
						payload = {
							"model_id": model_id,
							"prompt_next_topk_indices": resp.get("prompt_next_topk_indices"),
							"prompt_next_topk_probs": resp.get("prompt_next_topk_probs"),
							"prompt_next_rest_mass": resp.get("prompt_next_rest_mass"),
							"topk_indices": (resp.get("topk_indices") or [[]])[0] if isinstance(resp.get("topk_indices"), list) else resp.get("topk_indices"),
							"topk_probs": (resp.get("topk_probs") or [[]])[0] if isinstance(resp.get("topk_probs"), list) else resp.get("topk_probs"),
							"rest_mass": (resp.get("rest_mass") or [0.0])[0] if isinstance(resp.get("rest_mass"), list) else resp.get("rest_mass"),
							"method": args.method or "full_fim_dir",
						}
						h = analyze_topk_compact(s, args.base, payload, timeout=args.timeout)
					else:
						# Raw logits path with compaction limits
						full_vocab = resp.get("vocab") or []
						prompt_next = resp.get("prompt_next_logits") or []
						steps = resp.get("token_logits") or []
						V = min(int(args.max_vocab), len(full_vocab) or len(prompt_next)) or int(args.max_vocab)
						T = min(int(args.max_positions), len(steps)) if isinstance(steps, list) else 0
						comp_steps = []
						for i in range(T):
							row = steps[i]
							comp_steps.append(row[:V])
						payload = {
							"model_id": model_id,
							"prompt_next_logits": (prompt_next[:V] if isinstance(prompt_next, list) else []),
							"token_logits": comp_steps,
							"vocab": (full_vocab[:V] if isinstance(full_vocab, list) else None),
							"temperature": float(resp.get("temperature", args.temperature)),
							"method": args.method or "full_fim_dir",
						}
						h = analyze_logits(s, args.base, payload, timeout=args.timeout)
					H.append(h); Y.append(label)
				elif args.real_provider == "ollama":
					oll_payload = {
						"model_id": _resolve_provider_model_id(),
						"prompt": prompt,
						"forced_output": output,
						"top_k": int(args.top_k),
						"num_samples": int(args.num_samples),
						"temperature": float(args.temperature),
					}
					r = s.post(_provider_url(), json=oll_payload, timeout=max(120.0, args.timeout))
					r.raise_for_status()
					resp = r.json()
					payload = {
						"model_id": model_id,
						"prompt_next_topk_indices": resp.get("prompt_next_topk_indices"),
						"prompt_next_topk_probs": resp.get("prompt_next_topk_probs"),
						"prompt_next_rest_mass": resp.get("prompt_next_topk_rest_mass"),
						"topk_indices": resp.get("topk_indices"),
						"topk_probs": resp.get("topk_probs"),
						"rest_mass": resp.get("rest_mass"),
						"method": args.method or "full_fim_dir",
					}
					h = analyze_topk_compact(s, args.base, payload, timeout=args.timeout)
					H.append(h); Y.append(label)
				else:
					h = analyze(s, args.base, prompt, output, args.timeout, model_id=model_id, method=args.method)
					H.append(h); Y.append(label)
			except Exception:
				H.append(float("nan")); Y.append(label)
		return H, Y

	H_train, y_train = _fetch_with_provider(train_items)
	H_val, y_val = _fetch_with_provider(val_items)
	H_test, y_test = _fetch_with_provider(test_items)

	# Calibrate base Platt (🎯), and fit isotonic (📈) + spline (🧵) on train
	H_train_clean, y_train_clean = sanitize_xy(H_train, y_train)
	if not H_train_clean:
		return {"error": "No valid training samples after sanitization.", "model_id": model_id}
	
	# Use golden scale if enabled
	golden_scale = args.golden_scale if args.enable_golden_scale else 1.0
	loss, lam, tau = grid_search(H_train_clean, y_train_clean, golden_scale)

	def platt(xs: List[float]) -> List[float]:
		return [compute_pfail(h, lam, tau, golden_scale) for h in xs]

	# Isotonic regression (monotone)
	iso = None
	if IsotonicRegression is not None:
		try:
			iso = IsotonicRegression(out_of_bounds="clip")
			x = H_train_clean
			yp = platt(H_train_clean)
			iso.fit(x, y_train_clean)
		except Exception:
			iso = None

	# Monotone spline (PCHIP) on (score -> platt prob), monotone if data monotone
	spline = None
	if PchipInterpolator is not None:
		try:
			xs = sorted(zip(H_train_clean, platt(H_train_clean)), key=lambda t: t[0])
			xv = [t[0] for t in xs]
			yv = [t[1] for t in xs]
			# Ensure strict monotonicity by jitter if necessary
			for i in range(1, len(xv)):
				if xv[i] <= xv[i-1]:
					xv[i] = xv[i-1] + 1e-9
			spline = PchipInterpolator(xv, yv, extrapolate=True)
		except Exception:
			spline = None

	# Precompute component predictions
	def predict_components(xs: List[float]) -> Tuple[List[float], List[float], List[float]]:
		p_platt = platt(xs)
		p_iso = p_platt
		p_spline = p_platt
		if iso is not None:
			try:
				p_iso = [float(max(1e-9, min(1-1e-9, iso.predict([h])[0]))) for h in xs]
			except Exception:
				p_iso = p_platt
		if spline is not None:
			try:
				p_spline = [float(max(1e-9, min(1-1e-9, spline(h)))) for h in xs]
			except Exception:
				p_spline = p_platt
		return p_platt, p_iso, p_spline

	pp_tr, pi_tr, ps_tr = predict_components(H_train)
	pp_va, pi_va, ps_va = predict_components(H_val)
	pp_te, pi_te, ps_te = predict_components(H_test)

	# Learn convex weights on validation if requested
	weights = (1/3.0, 1/3.0, 1/3.0)
	if args.learn_weights and brier_score_loss is not None:
		best = (float("inf"), weights)
		step = max(0.01, float(args.weight_grid_step))
		w = 0.0
		while w <= 1.0 + 1e-9:
			v = 0.0
			while v <= 1.0 - w + 1e-9:
				u = max(0.0, 1.0 - w - v)
				pred = [w*a + v*b + u*c for a,b,c in zip(pp_va, pi_va, ps_va)]
				try:
					bs = float(brier_score_loss(y_val, pred))
				except Exception:
					bs = float("inf")
				if bs < best[0]:
					best = (bs, (w,v,u))
				v += step
			w += step
		weights = best[1]

	def apply_weights(wts: Tuple[float,float,float], comps: Tuple[List[float], List[float], List[float]]) -> List[float]:
		w1,w2,w3 = wts
		a,b,c = comps
		return [float(max(1e-9, min(1-1e-9, w1*x + w2*y + w3*z))) for x,y,z in zip(a,b,c)]

	def apply_ensemble(H_data: np.ndarray) -> List[float]:
		"""
		Apply ensemble prediction using weighted combination of Platt, isotonic, and spline methods.
		
		Args:
			H_data: Semantic uncertainty values (ℏₛ) 
			
		Returns:
			List of failure probabilities using ensemble weights
		"""
		try:
			# Apply the three calibration methods
			pp, pi, ps = predict_components(H_data)
			
			# Combine using learned weights
			ensemble_probs = apply_weights(weights, (pp, pi, ps))
			
			return ensemble_probs
			
		except Exception as e:
			print(f"⚠️ Ensemble prediction failed: {e}")
			# Fallback to Platt scaling only
			pp, _, _ = predict_components(H_data)
			return pp

	# Apply ensemble predictions
	p_train = apply_ensemble(H_train)
	p_val = apply_ensemble(H_val) 
	p_test = apply_ensemble(H_test)
	metrics = {
		"train": compute_metrics(y_train, p_train),
		"val": compute_metrics(y_val, p_val),
		"test": compute_metrics(y_test, p_test),
	}

	# Plots
	if not args.no_plots and args.plot_dir:
		try:
			os.makedirs(args.plot_dir, exist_ok=True)
			roc_path = f"{args.plot_dir}/roc{('-'+model_id) if model_id else ''}.png"
			cal_path = f"{args.plot_dir}/calibration{('-'+model_id) if model_id else ''}.png"
			plot_roc(y_test, p_test, out_path=roc_path)
			plot_calibration(y_test, p_test, out_path=cal_path, n_bins=15)
		except Exception:
			pass

	# CSV
	csv_path = args.save_csv
	if args.for_each_model and csv_path:
		csv_path = _append_suffix_to_path(csv_path, f"_{model_id}" if model_id else "")
	if csv_path:
		try:
			import csv
			with open(csv_path, "w", newline="") as f:
				w = csv.writer(f)
				w.writerow(["split", "hbar_s", "hbar_s_calibrated", "pfail", "label", "golden_scale"])  # prompts omitted intentionally
				for split_name, H, Y in (
					("train", H_train, y_train),
					("val", H_val, y_val),
					("test", H_test, y_test),
				):
					for h, yy in zip(H, Y):
						h_calibrated = h * golden_scale if golden_scale > 0 else h
						w.writerow([split_name, h, h_calibrated, compute_pfail(h, lam, tau, golden_scale), yy, golden_scale])
		except Exception:
			pass

	# Targets gating
	passed = True
	if args.target_ece is not None:
		passed = passed and (metrics["test"].get("ece", float("inf")) <= float(args.target_ece))
	if args.target_brier is not None:
		passed = passed and (metrics["test"].get("brier", float("inf")) <= float(args.target_brier))
	if args.target_auc is not None:
		passed = passed and (metrics["test"].get("roc_auc", 0.0) >= float(args.target_auc))

	# Output dict
	return {
		"lambda": lam,
		"tau": tau,
		"golden_scale": golden_scale,
		"golden_scale_enabled": args.enable_golden_scale,
		"train_loss": loss,
		"model_id": model_id,
		"method": args.method,
		"num_pairs": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
		"metrics": metrics,
		"ensemble_weights": {"platt": weights[0], "isotonic": weights[1], "spline": weights[2]},
		"meets_targets": passed,
		"calibration_info": {
			"type": "golden_scale" if args.enable_golden_scale else "standard",
			"factor": golden_scale,
			"description": f"Golden scale calibration (×{golden_scale}) for enhanced hallucination detection" if args.enable_golden_scale else "Standard calibration"
		}
	}


def main():
	ap = argparse.ArgumentParser(description="Calibrate failure law (lambda/tau) with datasets, metrics, and plotting")
	ap.add_argument("--base", default="http://127.0.0.1:3000/api/v1", help="API base URL")
	ap.add_argument("--timeout", type=float, default=10.0)
	# Data sources
	ap.add_argument("--pairs", type=str, default=None, help="Optional JSON path of pairs with prompt/failing/passing")
	ap.add_argument("--dataset", type=str, default="default", choices=["default", "truthfulqa", "halueval"], help="Dataset source")
	ap.add_argument("--halueval_task", type=str, default="qa", choices=["qa", "dialogue", "summarization", "general"], help="HaluEval task subset")
	ap.add_argument("--halueval_url", type=str, default=None, help="Override HaluEval JSON URL")
	ap.add_argument("--max_samples", type=int, default=None, help="Max pairs to sample (each yields failing/passing)")
	ap.add_argument("--seed", type=int, default=42)
	# Splits
	ap.add_argument("--val_split", type=float, default=0.2, help="Validation fraction")
	ap.add_argument("--test_split", type=float, default=0.2, help="Test fraction")
	# Concurrency
	ap.add_argument("--concurrency", type=int, default=4, help="Concurrent requests to /analyze")
	ap.add_argument("--rate_limit", type=float, default=0.0, help="Optional per-thread requests per second")
	# Model/method controls
	ap.add_argument("--model_id", type=str, default=None, help="Send model_id to /analyze for per-model calibration")
	ap.add_argument("--method", type=str, default=None, help="Set method for /analyze, e.g., diag_fim_dir | scalar_js_kl | scalar_trace | scalar_fro")
	ap.add_argument("--models_json", type=str, default="config/models.json", help="Path to models.json for bulk per-model calibration")
	ap.add_argument("--for_each_model", action="store_true", help="Run calibration for each model in models.json and emit a summary per model")
	ap.add_argument("--output_models_json", type=str, default=None, help="Write an updated models.json with new failure_law params per model")
	# Real-model provider controls
	ap.add_argument("--real_provider", type=str, default="none", choices=["none", "hf", "ollama"], help="Use real model logits via provider")
	ap.add_argument("--provider_url", type=str, default=None, help="Provider endpoint URL. HF default: http://127.0.0.1:8899/logits, Ollama default: http://127.0.0.1:8898/ollama_topk")
	ap.add_argument("--provider_model_override", type=str, default=None, help="Force provider to use this model id (e.g., gpt2) for all entries")
	ap.add_argument("--top_k", type=int, default=128, help="Top-k for sparse logits from provider")
	ap.add_argument("--num_samples", type=int, default=64, help="Num samples for Ollama provider aggregation")
	ap.add_argument("--temperature", type=float, default=1.0, help="Provider sampling temperature")
	# Logits mode for providers
	ap.add_argument("--logits_mode", type=str, default="raw", choices=["topk", "raw"], help="Use sparse top-k (topk) or raw logits (raw) from provider")
	# Payload compaction for raw logits
	ap.add_argument("--max_vocab", type=int, default=2048, help="Max vocab size to send in raw logits payloads")
	ap.add_argument("--max_positions", type=int, default=1, help="Max number of token positions to include in raw logits payloads")
	# Plotting
	ap.add_argument("--no_plots", action="store_true", help="Disable plotting")
	ap.add_argument("--plot_dir", type=str, default=None, help="Directory to save ROC and calibration plots")
	# Outputs
	ap.add_argument("--output_json", type=str, default=None, help="Path to write summary JSON")
	ap.add_argument("--save_csv", type=str, default=None, help="Path to write per-sample CSV")
	# Ensemble and evaluation controls
	ap.add_argument("--learn_weights", action="store_true", help="Learn convex ensemble weights on validation to minimize Brier")
	ap.add_argument("--weight_grid_step", type=float, default=0.1, help="Grid step for weight search (0<w<=1)")
	ap.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap resamples for metric CIs (0 to disable)")
	ap.add_argument("--target_ece", type=float, default=None, help="Fail if test ECE exceeds this threshold")
	ap.add_argument("--target_brier", type=float, default=None, help="Fail if test Brier exceeds this threshold")
	ap.add_argument("--target_auc", type=float, default=None, help="Fail if test ROC-AUC below this threshold")
	ap.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds to run sequentially; aggregates results")
	# Golden scale calibration for hallucination detection
	ap.add_argument("--golden_scale", type=float, default=3.4, help="Golden scale factor for semantic uncertainty calibration (default: 3.4)")
	ap.add_argument("--enable_golden_scale", action="store_true", help="Enable golden scale calibration for improved hallucination detection")

	args = ap.parse_args()

	# Bulk per-model mode
	if args.for_each_model:
		try:
			with open(args.models_json, "r") as f:
				models_doc = json.load(f)
		except Exception:
			print(json.dumps({"error": f"Failed to read models JSON at {args.models_json}"}, indent=2))
			return 1
		models = models_doc.get("models", []) if isinstance(models_doc, dict) else []
		results = []
		for m in models:
			mid = m.get("id") if isinstance(m, dict) else None
			res = calibrate_for_model(args, mid)
			results.append(res)
			# Optionally mutate in-memory models failure law
			if isinstance(m, dict) and "lambda" in res and "tau" in res:
				m.setdefault("failure_law", {})
				m["failure_law"]["lambda"] = res["lambda"]
				m["failure_law"]["tau"] = res["tau"]
		# Write summary and optional updated models.json
		print(json.dumps({"results": results}, indent=2))
		if args.output_json:
			try:
				with open(args.output_json, "w") as f:
					json.dump({"results": results}, f, indent=2)
			except Exception:
				pass
		if args.output_models_json:
			try:
				with open(args.output_models_json, "w") as f:
					json.dump(models_doc, f, indent=2)
			except Exception:
				pass
		return 0

	# Single-model or default mode
	res = calibrate_for_model(args, args.model_id)
	print(json.dumps(res, indent=2))
	if args.output_json:
		try:
			with open(args.output_json, "w") as f:
				json.dump(res, f, indent=2)
		except Exception:
			pass
	return 0


if __name__ == "__main__":
	sys.exit(main()) 