#!/usr/bin/env python3
"""
Final Comprehensive Evaluation: All 6 Models with Improved P(fail) Calibration
=============================================================================

Tests the improved calibration parameters across all models and methods.
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
class FinalEvaluationResult:
    """Final evaluation result with improved calibration"""
    dataset: str
    question_id: str
    prompt: str
    output: str
    ground_truth: bool
    method: str
    model_id: str
    
    # Core metrics
    hbar_s: float
    delta_mu: float
    delta_sigma: float
    p_fail: float
    lambda_param: float
    tau_param: float
    
    # FEP metrics
    kl_surprise: float = 0.0
    attention_entropy: float = 0.0
    prediction_variance: float = 0.0
    enhanced_free_energy: float = 0.0
    
    # Tier predictions
    tier_1_prediction: bool = False
    tier_2_prediction: bool = False
    tier_3_prediction: bool = False
    
    processing_time_ms: float = 0.0
    error: str = None

class FinalComprehensiveEvaluator:
    """Final evaluator with improved calibration"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
        # All 5 methods
        self.methods = [
            "scalar_js_kl",
            "diag_fim_dir", 
            "scalar_trace",
            "scalar_fro",
            "full_fim_dir"
        ]
        
        # All 6 models
        self.model_ids = [
            "mixtral-8x7b",
            "mistral-7b",
            "qwen2.5-7b",
            "pythia-6.9b", 
            "dialogpt-medium",
            "ollama-mistral-7b"
        ]
        
        # Load updated model configurations
        self.models_config = self.load_models_config()
        
        # Results storage
        self.results: List[FinalEvaluationResult] = []
        
    def load_models_config(self) -> Dict[str, Any]:
        """Load updated model configurations with improved calibration"""
        try:
            with open("config/models.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load models config: {e}")
            return {"default_model_id": "mistral-7b", "models": []}
    
    def load_truthfulqa_dataset(self, max_samples: int = 15) -> List[Dict[str, Any]]:
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
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def compute_tier_predictions(self, hbar_s: float, p_fail: float, fep_metrics: Dict[str, float]) -> Tuple[bool, bool, bool]:
        """Compute predictions for each tier with improved thresholds"""
        
        # Tier 1: ℏₛ with method-specific thresholds
        tier_1_threshold = 1.0  # Conservative threshold
        tier_1_prediction = hbar_s < tier_1_threshold
        
        # Tier 2: Improved P(fail) thresholds (more balanced)
        tier_2_threshold = 0.5  # 50% threshold
        tier_2_prediction = (hbar_s < tier_1_threshold) or (p_fail > tier_2_threshold)
        
        # Tier 3: Enhanced FEP integration
        fep_score = (
            fep_metrics.get("kl_surprise", 0) * 2.0 +
            fep_metrics.get("attention_entropy", 0) * 0.5 + 
            fep_metrics.get("prediction_variance", 0) * 1.0
        )
        fep_threshold = 1.5
        tier_3_prediction = tier_2_prediction or (fep_score > fep_threshold)
        
        return tier_1_prediction, tier_2_prediction, tier_3_prediction
    
    def extract_metrics(self, api_response: Dict[str, Any], method: str, model_id: str):
        """Extract all metrics from API response"""
        
        # Basic metrics
        hbar_s = api_response.get("hbar_s", 0.0)
        delta_mu = api_response.get("delta_mu", 0.0)
        delta_sigma = api_response.get("delta_sigma", 0.0)
        p_fail = api_response.get("p_fail", 0.0)
        
        # Model-specific parameters (now improved)
        model_info = next((m for m in self.models_config["models"] if m["id"] == model_id), None)
        if model_info and "failure_law" in model_info:
            lambda_param = model_info["failure_law"]["lambda"]
            tau_param = model_info["failure_law"]["tau"]
        else:
            lambda_param = 1.0
            tau_param = 1.0
            
        # FEP components
        enhanced_fep = api_response.get("enhanced_fep", {})
        fep_metrics = {
            "kl_surprise": enhanced_fep.get("kl_surprise", 0.0),
            "attention_entropy": enhanced_fep.get("attention_entropy", 0.0),
            "prediction_variance": enhanced_fep.get("prediction_variance", 0.0),
            "enhanced_free_energy": enhanced_fep.get("enhanced_free_energy", 0.0)
        }
        
        return hbar_s, delta_mu, delta_sigma, p_fail, lambda_param, tau_param, fep_metrics
    
    def run_final_comprehensive_evaluation(self, max_samples: int = 15):
        """Run final evaluation with all models and improved calibration"""
        
        logger.info("🚀 Starting Final Comprehensive Evaluation with Improved Calibration")
        logger.info(f"📊 Configuration:")
        logger.info(f"   • Methods: {len(self.methods)} methods")
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
                            hbar_s, delta_mu, delta_sigma, p_fail, lambda_param, tau_param, fep_metrics = self.extract_metrics(
                                api_response, method, model_id
                            )
                            
                            # Compute tier predictions
                            tier_1_pred, tier_2_pred, tier_3_pred = self.compute_tier_predictions(
                                hbar_s, p_fail, fep_metrics
                            )
                            
                            # Create result
                            result = FinalEvaluationResult(
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
                                kl_surprise=fep_metrics["kl_surprise"],
                                attention_entropy=fep_metrics["attention_entropy"],
                                prediction_variance=fep_metrics["prediction_variance"],
                                enhanced_free_energy=fep_metrics["enhanced_free_energy"],
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
                        time.sleep(0.2)
                
                all_results.extend(method_results)
                logger.info(f"✅ {method}-{model_id}: {len(method_results)} results")
                
                # Brief pause between combinations
                time.sleep(1.0)
        
        self.results = all_results
        logger.info(f"🎉 Final comprehensive evaluation complete! Total results: {len(self.results)}")
        
        # Generate reports
        self.generate_final_reports()
    
    def calculate_performance_metrics(self, results: List[FinalEvaluationResult]) -> Dict[str, float]:
        """Calculate classification metrics"""
        
        if not results:
            return {}
            
        # Extract predictions and ground truth
        predictions = [r.tier_3_prediction for r in results]  # Final tier prediction
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
        
        # Average processing time
        avg_processing_time = np.mean([r.processing_time_ms for r in results])
        
        # P(fail) distribution
        pfail_values = [r.p_fail for r in results]
        pfail_stats = {
            "mean": float(np.mean(pfail_values)),
            "std": float(np.std(pfail_values)),
            "min": float(min(pfail_values)),
            "max": float(max(pfail_values))
        }
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_processing_time_ms": avg_processing_time,
            "total_samples": len(results),
            "pfail_distribution": pfail_stats
        }
    
    def generate_final_reports(self):
        """Generate comprehensive final reports"""
        
        logger.info("📈 Generating Final Analysis Reports...")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save raw results
        df.to_csv("final_comprehensive_evaluation_results.csv", index=False)
        
        # Performance by method
        method_performance = {}
        for method in set(r.method for r in self.results):
            method_results = [r for r in self.results if r.method == method]
            method_performance[method] = self.calculate_performance_metrics(method_results)
        
        # Performance by model
        model_performance = {}
        for model_id in set(r.model_id for r in self.results):
            model_results = [r for r in self.results if r.model_id == model_id]
            model_performance[model_id] = self.calculate_performance_metrics(model_results)
        
        # Overall performance
        overall_performance = self.calculate_performance_metrics(self.results)
        
        # Model-tier analysis
        model_tier_analysis = self.analyze_model_tier_performance(df)
        
        # Compile comprehensive report
        report = {
            "evaluation_summary": {
                "total_evaluations": len(self.results),
                "methods_tested": list(set(r.method for r in self.results)),
                "models_tested": list(set(r.model_id for r in self.results)),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "calibration_status": "improved_parameters"
            },
            "overall_performance": overall_performance,
            "performance_by_method": method_performance,
            "performance_by_model": model_performance,
            "model_tier_analysis": model_tier_analysis,
            "calibration_improvements": self.analyze_calibration_improvements()
        }
        
        # Save report
        with open("final_comprehensive_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📊 Final report saved to final_comprehensive_evaluation_report.json")
        
        # Print results
        self.print_final_results(report)
    
    def analyze_model_tier_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by model and tier"""
        
        model_analysis = {}
        
        for model_id in df['model_id'].unique():
            model_data = df[df['model_id'] == model_id]
            
            # Tier-specific analysis
            tier_performance = {}
            for tier in [1, 2, 3]:
                pred_col = f'tier_{tier}_prediction'
                if pred_col in model_data.columns:
                    predictions = model_data[pred_col]
                    ground_truth = ~model_data['ground_truth']  # Invert for hallucination detection
                    
                    # Calculate metrics
                    if len(predictions) > 0:
                        accuracy = (predictions == ground_truth).mean()
                        precision = ((predictions) & (ground_truth)).sum() / predictions.sum() if predictions.sum() > 0 else 0.0
                        recall = ((predictions) & (ground_truth)).sum() / ground_truth.sum() if ground_truth.sum() > 0 else 0.0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        tier_performance[f"tier_{tier}"] = {
                            "accuracy": float(accuracy),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1_score": float(f1),
                            "predictions_made": int(predictions.sum())
                        }
            
            # Overall model stats
            model_stats = {
                "samples": len(model_data),
                "avg_hbar": float(model_data['hbar_s'].mean()),
                "avg_pfail": float(model_data['p_fail'].mean()),
                "avg_processing_time_ms": float(model_data['processing_time_ms'].mean()),
                "lambda_param": float(model_data['lambda_param'].iloc[0]) if len(model_data) > 0 else 0.0,
                "tau_param": float(model_data['tau_param'].iloc[0]) if len(model_data) > 0 else 0.0
            }
            
            model_analysis[model_id] = {
                "model_statistics": model_stats,
                "tier_performance": tier_performance
            }
            
        return model_analysis
    
    def analyze_calibration_improvements(self) -> Dict[str, Any]:
        """Analyze improvements from new calibration"""
        
        if not self.results:
            return {}
        
        # P(fail) distribution analysis
        pfail_values = [r.p_fail for r in self.results]
        
        # Count samples in different P(fail) ranges
        low_risk = sum(1 for p in pfail_values if p < 0.2)      # < 20%
        medium_risk = sum(1 for p in pfail_values if 0.2 <= p < 0.8)  # 20-80%
        high_risk = sum(1 for p in pfail_values if p >= 0.8)    # > 80%
        
        distribution_analysis = {
            "low_risk_count": low_risk,
            "medium_risk_count": medium_risk, 
            "high_risk_count": high_risk,
            "total_samples": len(pfail_values),
            "low_risk_percent": (low_risk / len(pfail_values)) * 100,
            "medium_risk_percent": (medium_risk / len(pfail_values)) * 100,
            "high_risk_percent": (high_risk / len(pfail_values)) * 100
        }
        
        return {
            "pfail_distribution_analysis": distribution_analysis,
            "overall_pfail_stats": {
                "mean": float(np.mean(pfail_values)),
                "std": float(np.std(pfail_values)),
                "min": float(min(pfail_values)),
                "max": float(max(pfail_values))
            }
        }
    
    def print_final_results(self, report: Dict[str, Any]):
        """Print final comprehensive results"""
        
        print("\n" + "="*100)
        print("🎯 FINAL COMPREHENSIVE EVALUATION - ALL 6 MODELS + IMPROVED CALIBRATION")
        print("="*100)
        
        # Summary
        summary = report["evaluation_summary"]
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   • Total Evaluations: {summary['total_evaluations']:,}")
        print(f"   • Methods Tested: {len(summary['methods_tested'])} methods")
        print(f"   • Models Tested: {len(summary['models_tested'])} models")
        print(f"   • Calibration Status: {summary['calibration_status']}")
        
        # Overall performance
        overall = report["overall_performance"]
        print(f"\n🏆 OVERALL PERFORMANCE (IMPROVED CALIBRATION):")
        print(f"   • Accuracy: {overall['accuracy']*100:.1f}%")
        print(f"   • Precision: {overall['precision']*100:.1f}%")
        print(f"   • Recall: {overall['recall']*100:.1f}%")
        print(f"   • F1-Score: {overall['f1_score']*100:.1f}%")
        print(f"   • Avg Processing Time: {overall['avg_processing_time_ms']:.1f}ms")
        
        # P(fail) distribution improvements
        if "calibration_improvements" in report:
            calib = report["calibration_improvements"]
            pfail_dist = calib["pfail_distribution_analysis"]
            pfail_stats = calib["overall_pfail_stats"]
            
            print(f"\n📈 P(FAIL) CALIBRATION IMPROVEMENTS:")
            print(f"   • P(fail) Range: {pfail_stats['min']:.3f} - {pfail_stats['max']:.3f}")
            print(f"   • P(fail) Mean: {pfail_stats['mean']:.3f} ± {pfail_stats['std']:.3f}")
            print(f"   • Low Risk (<20%): {pfail_dist['low_risk_percent']:.1f}% ({pfail_dist['low_risk_count']} samples)")
            print(f"   • Medium Risk (20-80%): {pfail_dist['medium_risk_percent']:.1f}% ({pfail_dist['medium_risk_count']} samples)")
            print(f"   • High Risk (>80%): {pfail_dist['high_risk_percent']:.1f}% ({pfail_dist['high_risk_count']} samples)")
        
        # Method comparison
        print(f"\n📊 METHOD PERFORMANCE RANKING (by F1-Score):")
        method_perf = report["performance_by_method"]
        sorted_methods = sorted(method_perf.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        for method, perf in sorted_methods:
            print(f"   • {method:15} | F1: {perf['f1_score']*100:5.1f}% | Acc: {perf['accuracy']*100:5.1f}% | P(fail): {perf['pfail_distribution']['mean']:.3f}")
        
        # Model comparison
        print(f"\n🤖 MODEL PERFORMANCE RANKING (by F1-Score):")
        model_perf = report["performance_by_model"]
        sorted_models = sorted(model_perf.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        for model_id, perf in sorted_models:
            print(f"   • {model_id:20} | F1: {perf['f1_score']*100:5.1f}% | P(fail): {perf['pfail_distribution']['mean']:.3f}")
        
        print("\n" + "="*100)

def main():
    """Main execution"""
    
    evaluator = FinalComprehensiveEvaluator(api_base_url="http://localhost:8080")
    
    # Run final comprehensive evaluation with improved calibration
    evaluator.run_final_comprehensive_evaluation(max_samples=15)
    
    print("\n🎉 Final Comprehensive Evaluation Complete!")
    print("📁 Files generated:")
    print("   • final_comprehensive_evaluation_results.csv")
    print("   • final_comprehensive_evaluation_report.json")

if __name__ == "__main__":
    main()