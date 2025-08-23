#!/usr/bin/env python3
"""
Focused Extended Evaluation: Complete 6-Model ℏₛ Tier Analysis
=============================================================

Completes evaluation of all 6 models with tier-specific ℏₛ analysis
using optimized approach for faster completion.
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FocusedEvaluationResult:
    """Focused evaluation result for tier analysis"""
    dataset: str
    question_id: str
    prompt: str
    output: str
    ground_truth: bool
    method: str
    model_id: str
    
    # Core metrics for tier analysis
    hbar_s: float
    delta_mu: float
    delta_sigma: float
    p_fail: float
    lambda_param: float
    tau_param: float
    
    # Tier predictions
    tier_1_prediction: bool = False
    tier_2_prediction: bool = False
    tier_3_prediction: bool = False
    
    processing_time_ms: float = 0.0
    error: str = None

class FocusedExtendedEvaluator:
    """Focused evaluator for completing all 6 models efficiently"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
        # Focus on best performing methods to save time
        self.methods = [
            "scalar_js_kl",    # Best performer from previous evaluation
            "diag_fim_dir"     # Standard method for comparison
        ]
        
        # All 6 models as requested
        self.model_ids = [
            "mixtral-8x7b",
            "mistral-7b", 
            "qwen2.5-7b",
            "pythia-6.9b",
            "dialogpt-medium",
            "ollama-mistral-7b"
        ]
        
        # Load model configurations
        self.models_config = self.load_models_config()
        
        # Results storage
        self.results: List[FocusedEvaluationResult] = []
        
    def load_models_config(self) -> Dict[str, Any]:
        """Load model configurations"""
        try:
            with open("config/models.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load models config: {e}")
            return {
                "default_model_id": "mistral-7b",
                "models": [
                    {"id": "mistral-7b", "failure_law": {"lambda": 1.887, "tau": 0.191}}
                ]
            }
    
    def load_truthfulqa_dataset(self, max_samples: int = 12) -> List[Dict[str, Any]]:
        """Load TruthfulQA dataset efficiently"""
        try:
            with open("authentic_datasets/truthfulqa_data.json", "r") as f:
                data = json.load(f)
                
            samples = []
            raw_samples = data.get("validation", [])
            
            for sample in raw_samples[:max_samples//2]:
                # Correct version
                correct_sample = {
                    "prompt": sample["Question"],
                    "output": sample["Best Answer"],
                    "is_correct": True,
                    "category": sample.get("Category", "unknown")
                }
                # Incorrect version
                incorrect_sample = {
                    "prompt": sample["Question"],
                    "output": sample["Best Incorrect Answer"],
                    "is_correct": False,
                    "category": sample.get("Category", "unknown")
                }
                samples.extend([correct_sample, incorrect_sample])
                
            logger.info(f"Loaded {len(samples)} TruthfulQA samples")
            return samples
            
        except Exception as e:
            logger.error(f"Error loading TruthfulQA: {e}")
            return []
    
    def analyze_single_sample(self, prompt: str, output: str, method: str, model_id: str) -> Dict[str, Any]:
        """Send analysis request to API"""
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
                timeout=12
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def compute_tier_predictions(self, hbar_s: float, p_fail: float) -> Tuple[bool, bool, bool]:
        """Compute predictions for each tier"""
        
        # Tier 1: ℏₛ only (lower = more likely hallucination)
        tier_1_threshold = 1.0
        tier_1_prediction = hbar_s < tier_1_threshold
        
        # Tier 2: ℏₛ + P(fail) (higher P(fail) = more likely hallucination)
        tier_2_threshold = 0.5
        tier_2_prediction = (hbar_s < tier_1_threshold) or (p_fail > tier_2_threshold)
        
        # Tier 3: Combined heuristic
        tier_3_prediction = tier_2_prediction or (hbar_s < 0.8 and p_fail > 0.7)
        
        return tier_1_prediction, tier_2_prediction, tier_3_prediction
    
    def extract_metrics(self, api_response: Dict[str, Any], method: str, model_id: str):
        """Extract all metrics from API response"""
        
        # Basic metrics
        hbar_s = api_response.get("hbar_s", 0.0)
        delta_mu = api_response.get("delta_mu", 0.0)
        delta_sigma = api_response.get("delta_sigma", 0.0)
        p_fail = api_response.get("p_fail", 0.0)
        
        # Model-specific parameters
        model_info = next((m for m in self.models_config["models"] if m["id"] == model_id), None)
        if model_info and "failure_law" in model_info:
            lambda_param = model_info["failure_law"]["lambda"]
            tau_param = model_info["failure_law"]["tau"]
        else:
            lambda_param = 5.0
            tau_param = 1.0
            
        return hbar_s, delta_mu, delta_sigma, p_fail, lambda_param, tau_param
    
    def run_focused_extended_evaluation(self, max_samples: int = 12):
        """Run focused evaluation on all 6 models"""
        
        logger.info("🚀 Starting Focused Extended Model Evaluation")
        logger.info(f"📊 Configuration:")
        logger.info(f"   • Methods: {len(self.methods)} methods ({', '.join(self.methods)})")
        logger.info(f"   • Models: {len(self.model_ids)} models")
        logger.info(f"   • Max samples: {max_samples}")
        
        # Load samples
        samples = self.load_truthfulqa_dataset(max_samples)
        if not samples:
            logger.error("No samples loaded!")
            return
            
        total_combinations = len(self.methods) * len(self.model_ids)
        logger.info(f"   • Total combinations: {total_combinations}")
        
        all_results = []
        completed = 0
        
        # Test each method-model combination
        for method in self.methods:
            for model_id in self.model_ids:
                
                completed += 1
                logger.info(f"\n🔄 Processing {completed}/{total_combinations}: {method} | {model_id}")
                
                method_results = []
                
                # Progress bar for samples
                with tqdm(total=len(samples), desc=f"{method}-{model_id}") as pbar:
                    
                    for i, sample in enumerate(samples):
                        try:
                            prompt = sample["prompt"]
                            output = sample["output"] 
                            ground_truth = sample["is_correct"]
                            
                            # Skip invalid samples
                            if len(prompt) < 10 or len(output) < 10:
                                continue
                                
                            # Analyze sample
                            start_time = time.time()
                            api_response = self.analyze_single_sample(prompt, output, method, model_id)
                            processing_time = (time.time() - start_time) * 1000
                            
                            if "error" in api_response:
                                logger.warning(f"API error: {api_response['error']}")
                                continue
                            
                            # Extract metrics
                            hbar_s, delta_mu, delta_sigma, p_fail, lambda_param, tau_param = self.extract_metrics(
                                api_response, method, model_id
                            )
                            
                            # Compute tier predictions
                            tier_1_pred, tier_2_pred, tier_3_pred = self.compute_tier_predictions(hbar_s, p_fail)
                            
                            # Create result
                            result = FocusedEvaluationResult(
                                dataset="truthfulqa",
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
                                tier_1_prediction=tier_1_pred,
                                tier_2_prediction=tier_2_pred,
                                tier_3_prediction=tier_3_pred,
                                processing_time_ms=processing_time
                            )
                            
                            method_results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Error processing sample {i}: {e}")
                            continue
                            
                        finally:
                            pbar.update(1)
                            
                        # Rate limiting
                        time.sleep(0.25)  # Conservative rate limiting
                
                all_results.extend(method_results)
                logger.info(f"✅ {method}-{model_id}: {len(method_results)} results")
        
        self.results = all_results
        logger.info(f"🎉 Focused extended evaluation complete! Total results: {len(self.results)}")
        
        # Generate reports
        self.generate_focused_reports()
    
    def generate_focused_reports(self):
        """Generate focused reports with tier-specific ℏₛ analysis"""
        
        logger.info("📈 Generating Focused Analysis Reports...")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save raw results
        df.to_csv("focused_extended_evaluation_results.csv", index=False)
        
        # Generate tier-specific ℏₛ analysis by model
        model_tier_analysis = self.analyze_hbar_by_model_and_tier(df)
        
        # Generate method comparison
        method_analysis = self.analyze_by_method(df)
        
        # Generate overall statistics
        overall_stats = self.calculate_overall_statistics(df)
        
        # Compile comprehensive report
        report = {
            "evaluation_summary": {
                "total_evaluations": len(self.results),
                "methods_tested": list(set(r.method for r in self.results)),
                "models_tested": list(set(r.model_id for r in self.results)),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "model_tier_hbar_analysis": model_tier_analysis,
            "method_performance": method_analysis,
            "overall_statistics": overall_stats,
            "key_insights": self.generate_key_insights(df)
        }
        
        # Save report
        with open("focused_extended_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📊 Focused report saved to focused_extended_evaluation_report.json")
        
        # Print results
        self.print_focused_results(report)
    
    def analyze_hbar_by_model_and_tier(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ℏₛ values by model with tier-specific breakdown"""
        
        model_analysis = {}
        
        for model_id in df['model_id'].unique():
            model_data = df[df['model_id'] == model_id]
            
            # Overall ℏₛ statistics for this model
            hbar_stats = {
                "mean": float(model_data['hbar_s'].mean()),
                "std": float(model_data['hbar_s'].std()),
                "min": float(model_data['hbar_s'].min()),
                "max": float(model_data['hbar_s'].max()),
                "median": float(model_data['hbar_s'].median()),
                "samples": len(model_data)
            }
            
            # Tier-specific analysis
            tier_breakdown = {}
            for tier in [1, 2, 3]:
                pred_col = f'tier_{tier}_prediction'
                if pred_col in model_data.columns:
                    tier_predictions = model_data[pred_col]
                    ground_truth = ~model_data['ground_truth']  # Invert for hallucination detection
                    
                    # ℏₛ stats for samples predicted as hallucinations in this tier
                    hallucination_samples = model_data[tier_predictions == True]
                    if len(hallucination_samples) > 0:
                        hbar_hallucination = {
                            "mean": float(hallucination_samples['hbar_s'].mean()),
                            "count": len(hallucination_samples)
                        }
                    else:
                        hbar_hallucination = {"mean": 0.0, "count": 0}
                    
                    # ℏₛ stats for samples predicted as correct in this tier
                    correct_samples = model_data[tier_predictions == False]
                    if len(correct_samples) > 0:
                        hbar_correct = {
                            "mean": float(correct_samples['hbar_s'].mean()),
                            "count": len(correct_samples)
                        }
                    else:
                        hbar_correct = {"mean": 0.0, "count": 0}
                    
                    # Performance metrics for this tier
                    if len(tier_predictions) > 0:
                        accuracy = (tier_predictions == ground_truth).mean()
                        precision = ((tier_predictions) & (ground_truth)).sum() / tier_predictions.sum() if tier_predictions.sum() > 0 else 0.0
                    else:
                        accuracy = 0.0
                        precision = 0.0
                        
                    tier_breakdown[f"tier_{tier}"] = {
                        "hbar_hallucination_predicted": hbar_hallucination,
                        "hbar_correct_predicted": hbar_correct,
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "predictions_made": int(tier_predictions.sum())
                    }
            
            # P(fail) statistics
            pfail_stats = {
                "mean": float(model_data['p_fail'].mean()),
                "std": float(model_data['p_fail'].std()),
                "min": float(model_data['p_fail'].min()),
                "max": float(model_data['p_fail'].max())
            }
            
            # Processing time
            time_stats = {
                "mean_ms": float(model_data['processing_time_ms'].mean()),
                "std_ms": float(model_data['processing_time_ms'].std())
            }
            
            model_analysis[model_id] = {
                "hbar_statistics": hbar_stats,
                "pfail_statistics": pfail_stats,
                "processing_time": time_stats,
                "tier_breakdown": tier_breakdown
            }
            
        return model_analysis
    
    def analyze_by_method(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by method"""
        
        method_analysis = {}
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            # Basic stats
            stats = {
                "samples": len(method_data),
                "avg_hbar": float(method_data['hbar_s'].mean()),
                "avg_pfail": float(method_data['p_fail'].mean()),
                "avg_processing_time_ms": float(method_data['processing_time_ms'].mean())
            }
            
            method_analysis[method] = stats
            
        return method_analysis
    
    def calculate_overall_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall statistics across all evaluations"""
        
        if len(df) == 0:
            return {}
        
        return {
            "total_samples": len(df),
            "overall_hbar_mean": float(df['hbar_s'].mean()),
            "overall_hbar_std": float(df['hbar_s'].std()),
            "overall_pfail_mean": float(df['p_fail'].mean()),
            "overall_processing_time_ms": float(df['processing_time_ms'].mean()),
            "models_evaluated": len(df['model_id'].unique()),
            "methods_evaluated": len(df['method'].unique())
        }
    
    def generate_key_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate key insights from the data"""
        
        insights = {}
        
        if len(df) > 10:
            # Model rankings by ℏₛ
            model_hbar_means = df.groupby('model_id')['hbar_s'].mean().sort_values(ascending=False)
            insights["model_hbar_ranking"] = model_hbar_means.to_dict()
            
            # Model rankings by P(fail)
            model_pfail_means = df.groupby('model_id')['p_fail'].mean().sort_values(ascending=False)
            insights["model_pfail_ranking"] = model_pfail_means.to_dict()
            
            # Processing time rankings
            model_time_means = df.groupby('model_id')['processing_time_ms'].mean().sort_values()
            insights["model_speed_ranking"] = model_time_means.to_dict()
        
        return insights
    
    def print_focused_results(self, report: Dict[str, Any]):
        """Print focused results with emphasis on tier-specific ℏₛ analysis"""
        
        print("\n" + "="*100)
        print("🎯 FOCUSED EXTENDED EVALUATION - ALL 6 MODELS + TIER-SPECIFIC ℏₛ ANALYSIS")
        print("="*100)
        
        # Summary
        summary = report["evaluation_summary"]
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   • Total Evaluations: {summary['total_evaluations']:,}")
        print(f"   • Methods Tested: {len(summary['methods_tested'])} ({', '.join(summary['methods_tested'])})")
        print(f"   • Models Tested: {len(summary['models_tested'])} models")
        
        # Model-specific ℏₛ analysis by tier (main requested output)
        print(f"\n🧮 TIER-SPECIFIC ℏₛ ANALYSIS BY MODEL:")
        model_analysis = report["model_tier_hbar_analysis"]
        
        for model_id, analysis in model_analysis.items():
            hbar_stats = analysis["hbar_statistics"]
            pfail_stats = analysis["pfail_statistics"]
            
            print(f"\n   🤖 {model_id.upper()}:")
            print(f"      • Overall ℏₛ Mean: {hbar_stats['mean']:.3f}")
            print(f"      • Overall ℏₛ Range: {hbar_stats['min']:.3f} - {hbar_stats['max']:.3f}")
            print(f"      • P(fail) Mean: {pfail_stats['mean']:.3f}")
            print(f"      • Samples: {hbar_stats['samples']}")
            
            # Tier breakdown
            tier_breakdown = analysis.get("tier_breakdown", {})
            for tier_name, tier_data in tier_breakdown.items():
                hbar_halluc = tier_data["hbar_hallucination_predicted"]
                hbar_correct = tier_data["hbar_correct_predicted"]
                print(f"      • {tier_name.upper()}: ℏₛ_halluc={hbar_halluc['mean']:.3f} ({hbar_halluc['count']} samples), ℏₛ_correct={hbar_correct['mean']:.3f} ({hbar_correct['count']} samples)")
        
        # Key insights
        insights = report["key_insights"]
        if "model_hbar_ranking" in insights:
            print(f"\n🏆 MODEL RANKING BY ℏₛ (Higher = More Certain):")
            for model_id, hbar_mean in insights["model_hbar_ranking"].items():
                print(f"   • {model_id:20} | ℏₛ: {hbar_mean:.3f}")
        
        if "model_pfail_ranking" in insights:
            print(f"\n⚠️ MODEL RANKING BY P(fail) (Higher = More Failures Expected):")
            for model_id, pfail_mean in insights["model_pfail_ranking"].items():
                print(f"   • {model_id:20} | P(fail): {pfail_mean:.3f}")
        
        # Overall statistics
        overall = report["overall_statistics"]
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   • Total Evaluations: {overall['total_samples']:,}")
        print(f"   • Mean ℏₛ Across All Models: {overall['overall_hbar_mean']:.3f}")
        print(f"   • Mean P(fail) Across All Models: {overall['overall_pfail_mean']:.3f}")
        print(f"   • Average Processing Time: {overall['overall_processing_time_ms']:.1f}ms")
        
        print("\n" + "="*100)

def main():
    """Main execution"""
    
    evaluator = FocusedExtendedEvaluator(api_base_url="http://localhost:8080")
    
    # Run focused extended evaluation with all models
    evaluator.run_focused_extended_evaluation(max_samples=12)  # 12 samples = 6 correct + 6 incorrect
    
    print("\n🎉 Focused Extended Evaluation Complete!")
    print("📁 Files generated:")
    print("   • focused_extended_evaluation_results.csv")
    print("   • focused_extended_evaluation_report.json")

if __name__ == "__main__":
    main()