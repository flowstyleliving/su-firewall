#!/usr/bin/env python3
"""
Comprehensive Hallucination Detection Test: 3-Tier System Evaluation
=====================================================================

Executes semantic uncertainty (ℏₛ) calculations → calibrated P(fail) → FEP evaluation
across all 5 methods and 6 model architectures using authentic datasets.

Test Set: 52,883+ prompt-output pairs from:
- TruthfulQA: 7,901 factual questions
- HaluEval QA: 10,000 question-answering cases  
- HaluEval Dialogue: 10,000 conversational cases
- HaluEval Summarization: 10,000 summarization cases
- Synthetic benchmark: 10,475 generated cases
"""

import json
import time
import requests
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Single evaluation result for a prompt-output pair"""
    dataset: str
    question_id: str
    prompt: str
    output: str
    ground_truth: bool  # True = correct/factual, False = hallucinated
    method: str
    model_id: str
    
    # Level 1: Semantic Uncertainty (ℏₛ)
    hbar_s: float
    delta_mu: float
    delta_sigma: float
    
    # Level 2: Calibrated P(fail)
    p_fail: float
    lambda_param: float
    tau_param: float
    
    # Level 3: FEP Components
    kl_surprise: float = 0.0
    attention_entropy: float = 0.0
    prediction_variance: float = 0.0
    fisher_info_trace: float = 0.0
    enhanced_free_energy: float = 0.0
    
    # Performance metrics
    processing_time_ms: float = 0.0
    tier_3_prediction: bool = False  # Final prediction after all tiers
    
    # Error tracking
    error: str = None

class ComprehensiveEvaluator:
    """Main evaluation orchestrator for 3-tier hallucination detection system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
        # All 5 semantic uncertainty calculation methods
        self.methods = [
            "diag_fim_dir",    # Default: Diagonal FIM directional  
            "scalar_js_kl",    # Jensen-Shannon + KL divergence
            "scalar_trace",    # FIM trace-based
            "scalar_fro",      # Frobenius norm of FIM
            "full_fim_dir"     # Full FIM (computationally intensive)
        ]
        
        # Load model configurations and calibrated parameters
        self.models_config = self.load_models_config()
        self.model_ids = [model["id"] for model in self.models_config["models"]]
        
        # Dataset paths
        self.dataset_paths = {
            "truthfulqa": Path("authentic_datasets/truthfulqa_data.json"),
            "halueval_qa": Path("authentic_datasets/halueval_qa_data.json"),
            "halueval_dialogue": Path("authentic_datasets/halueval_dialogue_data.json"), 
            "halueval_summarization": Path("authentic_datasets/halueval_summarization_data.json"),
            "synthetic": Path("authentic_datasets/authentic_hallucination_benchmark.json")
        }
        
        # Results storage
        self.results: List[EvaluationResult] = []
        self.summary_stats: Dict[str, Any] = {}
        
    def load_models_config(self) -> Dict[str, Any]:
        """Load model configurations with calibrated λ,τ parameters"""
        try:
            with open("config/models.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load models config: {e}")
            # Return default configuration
            return {
                "default_model_id": "mistral-7b",
                "models": [
                    {"id": "mistral-7b", "failure_law": {"lambda": 1.887, "tau": 0.191}}
                ]
            }
    
    def load_dataset(self, dataset_name: str, max_samples: int = None) -> List[Dict[str, Any]]:
        """Load and parse dataset with optional sampling"""
        dataset_path = self.dataset_paths.get(dataset_name)
        if not dataset_path or not dataset_path.exists():
            logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")
            return []
            
        try:
            with open(dataset_path, "r") as f:
                data = json.load(f)
                
            # Handle different dataset formats
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "data" in data:
                samples = data["data"]
            else:
                samples = [data]
                
            # Apply sampling if specified
            if max_samples and len(samples) > max_samples:
                samples = samples[:max_samples]
                
            logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def analyze_single_sample(self, prompt: str, output: str, method: str, model_id: str) -> Dict[str, Any]:
        """Send single analysis request to the API"""
        try:
            payload = {
                "prompt": prompt,
                "output": output,
                "method": method,
                "model_id": model_id
            }
            
            response = self.session.post(
                f"{self.api_base_url}/api/v1/analyze",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API error {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
    
    def extract_level_metrics(self, api_response: Dict[str, Any], method: str, model_id: str) -> Tuple[float, float, float, float, float, Dict[str, float]]:
        """Extract metrics from API response for all 3 levels"""
        
        # Level 1: Semantic Uncertainty (ℏₛ)
        hbar_s = api_response.get("hbar_s", 0.0)
        delta_mu = api_response.get("delta_mu", 0.0)
        delta_sigma = api_response.get("delta_sigma", 0.0)
        
        # Level 2: Calibrated P(fail) 
        p_fail = api_response.get("p_fail", 0.0)
        
        # Get model-specific calibrated parameters
        model_info = next((m for m in self.models_config["models"] if m["id"] == model_id), None)
        if model_info and "failure_law" in model_info:
            lambda_param = model_info["failure_law"]["lambda"]
            tau_param = model_info["failure_law"]["tau"]
        else:
            lambda_param = 5.0  # Default fallback
            tau_param = 1.0
            
        # Level 3: FEP Components
        enhanced_fep = api_response.get("enhanced_fep", {})
        fep_metrics = {
            "kl_surprise": enhanced_fep.get("kl_surprise", 0.0),
            "attention_entropy": enhanced_fep.get("attention_entropy", 0.0), 
            "prediction_variance": enhanced_fep.get("prediction_variance", 0.0),
            "fisher_info_trace": enhanced_fep.get("fisher_info_trace", 0.0),
            "enhanced_free_energy": enhanced_fep.get("enhanced_free_energy", 0.0)
        }
        
        return hbar_s, delta_mu, delta_sigma, p_fail, lambda_param, tau_param, fep_metrics
    
    def compute_tier_3_prediction(self, hbar_s: float, p_fail: float, fep_metrics: Dict[str, float]) -> bool:
        """Combine all 3 tiers to make final hallucination prediction"""
        
        # Tier 1: ℏₛ threshold (lower = more likely to be hallucination)
        tier_1_threshold = 1.0
        tier_1_risk = hbar_s < tier_1_threshold
        
        # Tier 2: P(fail) threshold (higher = more likely to be hallucination) 
        tier_2_threshold = 0.5
        tier_2_risk = p_fail > tier_2_threshold
        
        # Tier 3: Enhanced FEP anomaly scoring
        fep_score = (
            fep_metrics["kl_surprise"] * 2.0 +
            fep_metrics["attention_entropy"] * 0.5 + 
            fep_metrics["prediction_variance"] * 1.0
        )
        tier_3_threshold = 1.5
        tier_3_risk = fep_score > tier_3_threshold
        
        # Combine all tiers with weighted voting
        risk_signals = [tier_1_risk, tier_2_risk, tier_3_risk]
        risk_count = sum(risk_signals)
        
        # Predict hallucination if majority of tiers indicate risk
        return risk_count >= 2
    
    def evaluate_dataset_method_model(self, dataset_name: str, method: str, model_id: str, max_samples: int = 100) -> List[EvaluationResult]:
        """Evaluate single dataset-method-model combination"""
        
        logger.info(f"🔬 Evaluating: {dataset_name} | {method} | {model_id}")
        
        # Load dataset samples
        samples = self.load_dataset(dataset_name, max_samples)
        if not samples:
            logger.warning(f"No samples loaded for {dataset_name}")
            return []
            
        results = []
        
        # Progress tracking
        with tqdm(total=len(samples), desc=f"{dataset_name}-{method}-{model_id}") as pbar:
            
            for i, sample in enumerate(samples):
                try:
                    # Extract prompt, output, and ground truth from sample
                    # Handle different dataset formats
                    if "prompt" in sample and "output" in sample:
                        prompt = sample["prompt"]
                        output = sample["output"]  
                        ground_truth = sample.get("is_correct", sample.get("correct", True))
                    elif "question" in sample and "answer" in sample:
                        prompt = sample["question"]
                        output = sample["answer"]
                        ground_truth = sample.get("is_correct", sample.get("truthful", True))
                    else:
                        logger.warning(f"Unrecognized sample format: {sample.keys()}")
                        continue
                    
                    # Skip if prompt/output too short or long
                    if len(prompt) < 10 or len(output) < 10 or len(prompt) > 2000:
                        continue
                        
                    # Analyze with API
                    start_time = time.time()
                    api_response = self.analyze_single_sample(prompt, output, method, model_id)
                    processing_time_ms = (time.time() - start_time) * 1000
                    
                    if "error" in api_response:
                        logger.warning(f"Analysis failed: {api_response['error']}")
                        continue
                    
                    # Extract all 3-tier metrics
                    hbar_s, delta_mu, delta_sigma, p_fail, lambda_param, tau_param, fep_metrics = self.extract_level_metrics(
                        api_response, method, model_id
                    )
                    
                    # Compute Tier 3 prediction
                    tier_3_prediction = self.compute_tier_3_prediction(hbar_s, p_fail, fep_metrics)
                    
                    # Create evaluation result
                    result = EvaluationResult(
                        dataset=dataset_name,
                        question_id=str(i),
                        prompt=prompt,
                        output=output,
                        ground_truth=ground_truth,
                        method=method,
                        model_id=model_id,
                        hbar_s=hbar_s,
                        delta_mu=delta_mu,
                        delta_sigma=delta_sigma,
                        p_fail=p_fail,
                        lambda_param=lambda_param,
                        tau_param=tau_param,
                        kl_surprise=fep_metrics["kl_surprise"],
                        attention_entropy=fep_metrics["attention_entropy"],
                        prediction_variance=fep_metrics["prediction_variance"], 
                        fisher_info_trace=fep_metrics["fisher_info_trace"],
                        enhanced_free_energy=fep_metrics["enhanced_free_energy"],
                        processing_time_ms=processing_time_ms,
                        tier_3_prediction=tier_3_prediction
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    continue
                    
                finally:
                    pbar.update(1)
                    
                # Throttling to respect rate limits
                time.sleep(0.1)  # 10 requests per second max
        
        logger.info(f"✅ Completed {dataset_name}-{method}-{model_id}: {len(results)} results")
        return results
    
    def run_comprehensive_evaluation(self, max_samples_per_dataset: int = 50):
        """Execute complete 3-tier evaluation across all combinations"""
        
        logger.info("🚀 Starting Comprehensive Hallucination Detection Evaluation")
        logger.info(f"📊 Configuration:")
        logger.info(f"   • Methods: {self.methods}")
        logger.info(f"   • Models: {self.model_ids}")
        logger.info(f"   • Datasets: {list(self.dataset_paths.keys())}")
        logger.info(f"   • Max samples per dataset: {max_samples_per_dataset}")
        
        total_combinations = len(self.methods) * len(self.model_ids) * len(self.dataset_paths)
        logger.info(f"   • Total combinations: {total_combinations}")
        
        # Execute evaluation for each combination
        all_results = []
        completed_combinations = 0
        
        for dataset_name in self.dataset_paths.keys():
            for method in self.methods:
                for model_id in self.model_ids:
                    
                    logger.info(f"\n🔄 Processing combination {completed_combinations + 1}/{total_combinations}")
                    
                    try:
                        results = self.evaluate_dataset_method_model(
                            dataset_name, method, model_id, max_samples_per_dataset
                        )
                        all_results.extend(results)
                        
                    except Exception as e:
                        logger.error(f"Failed combination {dataset_name}-{method}-{model_id}: {e}")
                        
                    completed_combinations += 1
        
        self.results = all_results
        logger.info(f"🎉 Evaluation Complete! Total results: {len(self.results)}")
        
        # Generate comprehensive reports
        self.generate_detailed_reports()
        
    def calculate_performance_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate accuracy, precision, recall, F1, ECE, etc."""
        
        if not results:
            return {}
            
        # Extract predictions and ground truth
        predictions = [r.tier_3_prediction for r in results]
        ground_truth = [not r.ground_truth for r in results]  # Invert: True = hallucination
        
        # Basic classification metrics
        tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
        tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
        fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
        fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)
        
        accuracy = (tp + tn) / len(predictions) if predictions else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Expected Calibration Error (ECE) for P(fail)
        p_fail_values = [r.p_fail for r in results]
        ece = self.calculate_ece(p_fail_values, ground_truth)
        
        # Average processing time
        avg_processing_time = np.mean([r.processing_time_ms for r in results])
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "ece": ece,
            "avg_processing_time_ms": avg_processing_time,
            "total_samples": len(results)
        }
    
    def calculate_ece(self, probabilities: List[float], ground_truth: List[bool], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        
        if not probabilities or len(probabilities) != len(ground_truth):
            return 0.0
            
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = [(bin_lower <= p < bin_upper) for p in probabilities]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                bin_probs = [probabilities[i] for i, in_b in enumerate(in_bin) if in_b]
                bin_gt = [ground_truth[i] for i, in_b in enumerate(in_bin) if in_b]
                
                accuracy_in_bin = np.mean(bin_gt)
                avg_confidence_in_bin = np.mean(bin_probs)
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def generate_detailed_reports(self):
        """Generate comprehensive performance analysis and insights"""
        
        logger.info("📈 Generating Detailed Performance Reports...")
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save raw results
        df.to_csv("comprehensive_evaluation_results.csv", index=False)
        logger.info("💾 Raw results saved to comprehensive_evaluation_results.csv")
        
        # Performance by method
        method_performance = {}
        for method in self.methods:
            method_results = [r for r in self.results if r.method == method]
            method_performance[method] = self.calculate_performance_metrics(method_results)
        
        # Performance by dataset
        dataset_performance = {}
        for dataset in self.dataset_paths.keys():
            dataset_results = [r for r in self.results if r.dataset == dataset]
            dataset_performance[dataset] = self.calculate_performance_metrics(dataset_results)
        
        # Performance by model
        model_performance = {}
        for model_id in self.model_ids:
            model_results = [r for r in self.results if r.model_id == model_id]
            model_performance[model_id] = self.calculate_performance_metrics(model_results)
            
        # Overall performance
        overall_performance = self.calculate_performance_metrics(self.results)
        
        # Compile comprehensive report
        report = {
            "evaluation_summary": {
                "total_evaluations": len(self.results),
                "methods_tested": self.methods,
                "models_tested": self.model_ids,
                "datasets_tested": list(self.dataset_paths.keys()),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "overall_performance": overall_performance,
            "performance_by_method": method_performance,
            "performance_by_dataset": dataset_performance,
            "performance_by_model": model_performance,
            "insights": self.generate_insights(df)
        }
        
        # Save comprehensive report
        with open("comprehensive_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📊 Comprehensive report saved to comprehensive_evaluation_report.json")
        
        # Print key findings
        self.print_key_findings(report)
    
    def generate_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract key insights and correlations from evaluation data"""
        
        insights = {}
        
        # Best performing method by ECE minimization
        method_ece = df.groupby('method').apply(
            lambda x: self.calculate_ece(x['p_fail'].tolist(), (~x['ground_truth']).tolist())
        ).to_dict()
        best_ece_method = min(method_ece, key=method_ece.get)
        insights["best_ece_method"] = {"method": best_ece_method, "ece": method_ece[best_ece_method]}
        
        # Correlation between ℏₛ and enhanced_free_energy
        if len(df) > 10:
            hbar_fep_corr = df['hbar_s'].corr(df['enhanced_free_energy'])
            insights["hbar_fep_correlation"] = float(hbar_fep_corr) if not np.isnan(hbar_fep_corr) else 0.0
        
        # Performance differences: factual vs conversational
        factual_datasets = ['truthfulqa', 'halueval_qa']
        conversational_datasets = ['halueval_dialogue']
        
        factual_accuracy = df[df['dataset'].isin(factual_datasets)]['tier_3_prediction'].mean() if not df[df['dataset'].isin(factual_datasets)].empty else 0.0
        conversational_accuracy = df[df['dataset'].isin(conversational_datasets)]['tier_3_prediction'].mean() if not df[df['dataset'].isin(conversational_datasets)].empty else 0.0
        
        insights["performance_by_type"] = {
            "factual_accuracy": float(factual_accuracy),
            "conversational_accuracy": float(conversational_accuracy)
        }
        
        # Processing time analysis
        avg_time_by_method = df.groupby('method')['processing_time_ms'].mean().to_dict()
        insights["processing_time_by_method"] = {k: float(v) for k, v in avg_time_by_method.items()}
        
        return insights
    
    def print_key_findings(self, report: Dict[str, Any]):
        """Print summary of key evaluation findings"""
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE HALLUCINATION DETECTION EVALUATION RESULTS")
        print("="*80)
        
        # Overall performance
        overall = report["overall_performance"]
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Total Evaluations: {overall.get('total_samples', 0):,}")
        print(f"   • Overall Accuracy: {overall.get('accuracy', 0)*100:.1f}%")
        print(f"   • Precision: {overall.get('precision', 0)*100:.1f}%")
        print(f"   • Recall: {overall.get('recall', 0)*100:.1f}%")
        print(f"   • F1-Score: {overall.get('f1_score', 0)*100:.1f}%")
        print(f"   • Expected Calibration Error: {overall.get('ece', 0)*100:.2f}%")
        print(f"   • Avg Processing Time: {overall.get('avg_processing_time_ms', 0):.1f}ms")
        
        # Best performing method
        method_perf = report["performance_by_method"]
        best_method = max(method_perf.keys(), key=lambda k: method_perf[k].get('f1_score', 0))
        print(f"\n🏆 BEST PERFORMING METHOD: {best_method}")
        print(f"   • F1-Score: {method_perf[best_method].get('f1_score', 0)*100:.1f}%")
        print(f"   • ECE: {method_perf[best_method].get('ece', 0)*100:.2f}%")
        
        # Method comparison
        print(f"\n📈 METHOD PERFORMANCE COMPARISON:")
        for method, perf in method_perf.items():
            print(f"   • {method:15} | F1: {perf.get('f1_score', 0)*100:5.1f}% | ECE: {perf.get('ece', 0)*100:5.2f}% | Time: {perf.get('avg_processing_time_ms', 0):6.1f}ms")
        
        # Insights
        insights = report.get("insights", {})
        print(f"\n🔍 KEY INSIGHTS:")
        
        if "best_ece_method" in insights:
            best_ece = insights["best_ece_method"]
            print(f"   • Best Calibrated Method: {best_ece['method']} (ECE: {best_ece['ece']*100:.2f}%)")
            
        if "hbar_fep_correlation" in insights:
            corr = insights["hbar_fep_correlation"]
            print(f"   • ℏₛ ↔ FEP Correlation: {corr:.3f}")
            
        if "performance_by_type" in insights:
            perf_type = insights["performance_by_type"]
            print(f"   • Factual vs Conversational: {perf_type['factual_accuracy']*100:.1f}% vs {perf_type['conversational_accuracy']*100:.1f}%")
        
        print("\n" + "="*80)


def main():
    """Main evaluation execution"""
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(api_base_url="http://localhost:8080")
    
    # Run comprehensive evaluation
    # Using smaller sample size for demonstration - increase for full evaluation
    evaluator.run_comprehensive_evaluation(max_samples_per_dataset=20)
    
    print("\n🎉 Comprehensive Evaluation Complete!")
    print("📁 Results saved:")
    print("   • comprehensive_evaluation_results.csv")
    print("   • comprehensive_evaluation_report.json")

if __name__ == "__main__":
    main()