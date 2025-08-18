#!/usr/bin/env python3
"""
Extended Evaluation: All Models + Tier-Specific ℏₛ Analysis
==========================================================

Runs evaluation on all 6 models and provides detailed ℏₛ calculations by tier.
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
class ExtendedEvaluationResult:
    """Extended evaluation result with tier-specific analysis"""
    dataset: str
    question_id: str
    prompt: str
    output: str
    ground_truth: bool
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
    
    # Tier predictions
    tier_1_prediction: bool = False  # Based on ℏₛ only
    tier_2_prediction: bool = False  # Based on ℏₛ + P(fail)
    tier_3_prediction: bool = False  # Based on ℏₛ + P(fail) + FEP
    
    processing_time_ms: float = 0.0
    error: str = None

class ExtendedEvaluator:
    """Extended evaluator for all models with tier-specific ℏₛ analysis"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
        # All 5 methods
        self.methods = [
            "diag_fim_dir",
            "scalar_js_kl", 
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
        
        # Load model configurations
        self.models_config = self.load_models_config()
        
        # Results storage
        self.results: List[ExtendedEvaluationResult] = []
        
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
    
    def load_truthfulqa_dataset(self, max_samples: int = 20) -> List[Dict[str, Any]]:
        """Load TruthfulQA dataset with proper handling"""
        try:
            with open("authentic_datasets/truthfulqa_data.json", "r") as f:
                data = json.load(f)
                
            samples = []
            raw_samples = data.get("validation", [])
            
            for sample in raw_samples[:max_samples//2]:  # Half for correct, half for incorrect
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
        """Compute predictions for each tier"""
        
        # Tier 1: ℏₛ only (lower = more likely hallucination)
        tier_1_threshold = 1.0
        tier_1_prediction = hbar_s < tier_1_threshold
        
        # Tier 2: ℏₛ + P(fail) (higher P(fail) = more likely hallucination)
        tier_2_threshold = 0.5
        tier_2_prediction = (hbar_s < tier_1_threshold) or (p_fail > tier_2_threshold)
        
        # Tier 3: ℏₛ + P(fail) + FEP
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
        
        # Model-specific parameters
        model_info = next((m for m in self.models_config["models"] if m["id"] == model_id), None)
        if model_info and "failure_law" in model_info:
            lambda_param = model_info["failure_law"]["lambda"]
            tau_param = model_info["failure_law"]["tau"]
        else:
            lambda_param = 5.0
            tau_param = 1.0
            
        # FEP components
        enhanced_fep = api_response.get("enhanced_fep", {})
        fep_metrics = {
            "kl_surprise": enhanced_fep.get("kl_surprise", 0.0),
            "attention_entropy": enhanced_fep.get("attention_entropy", 0.0),
            "prediction_variance": enhanced_fep.get("prediction_variance", 0.0),
            "fisher_info_trace": enhanced_fep.get("fisher_info_trace", 0.0),
            "enhanced_free_energy": enhanced_fep.get("enhanced_free_energy", 0.0)
        }
        
        return hbar_s, delta_mu, delta_sigma, p_fail, lambda_param, tau_param, fep_metrics
    
    def run_extended_evaluation(self, max_samples: int = 20):
        """Run evaluation on all models with tier analysis"""
        
        logger.info("🚀 Starting Extended Model Evaluation")
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
                logger.info(f"\\n🔄 Processing {completed}/{total_combinations}: {method} | {model_id}")
                
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
                            result = ExtendedEvaluationResult(
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
                                fisher_info_trace=fep_metrics["fisher_info_trace"],
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
                        time.sleep(0.3)  # Conservative rate limiting
                
                all_results.extend(method_results)
                logger.info(f"✅ {method}-{model_id}: {len(method_results)} results")
        
        self.results = all_results
        logger.info(f"🎉 Extended evaluation complete! Total results: {len(self.results)}")
        
        # Generate reports
        self.generate_extended_reports()
    
    def generate_extended_reports(self):
        """Generate extended reports with tier-specific ℏₛ analysis"""
        
        logger.info("📈 Generating Extended Analysis Reports...")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save raw results
        df.to_csv("extended_evaluation_results.csv", index=False)
        
        # Generate tier-specific ℏₛ analysis
        tier_analysis = self.analyze_hbar_by_tiers(df)
        
        # Generate model comparison
        model_analysis = self.analyze_by_model(df)
        
        # Generate method analysis
        method_analysis = self.analyze_by_method(df)
        
        # Compile comprehensive report
        report = {
            "evaluation_summary": {
                "total_evaluations": len(self.results),
                "methods_tested": list(set(r.method for r in self.results)),
                "models_tested": list(set(r.model_id for r in self.results)),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "tier_hbar_analysis": tier_analysis,
            "model_performance": model_analysis,
            "method_performance": method_analysis,
            "detailed_insights": self.generate_detailed_insights(df)
        }
        
        # Save report
        with open("extended_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📊 Extended report saved to extended_evaluation_report.json")
        
        # Print results
        self.print_extended_results(report)
    
    def analyze_hbar_by_tiers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ℏₛ values by tier and method"""
        
        tier_analysis = {}
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            # Calculate ℏₛ stats for this method
            hbar_stats = {
                "mean": float(method_data['hbar_s'].mean()),
                "std": float(method_data['hbar_s'].std()),
                "min": float(method_data['hbar_s'].min()),
                "max": float(method_data['hbar_s'].max()),
                "median": float(method_data['hbar_s'].median())
            }
            
            # Tier performance analysis
            tier_performance = {}
            for tier in [1, 2, 3]:
                pred_col = f'tier_{tier}_prediction'
                if pred_col in method_data.columns:
                    predictions = method_data[pred_col]
                    ground_truth = ~method_data['ground_truth']  # Invert for hallucination detection
                    
                    # Calculate metrics
                    accuracy = (predictions == ground_truth).mean()
                    if predictions.sum() > 0:
                        precision = ((predictions) & (ground_truth)).sum() / predictions.sum()
                    else:
                        precision = 0.0
                        
                    tier_performance[f"tier_{tier}"] = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "predictions_made": int(predictions.sum())
                    }
            
            tier_analysis[method] = {
                "hbar_statistics": hbar_stats,
                "tier_performance": tier_performance
            }
        
        return tier_analysis
    
    def analyze_by_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by model"""
        
        model_analysis = {}
        
        for model_id in df['model_id'].unique():
            model_data = df[df['model_id'] == model_id]
            
            # ℏₛ statistics for this model
            hbar_stats = {
                "mean": float(model_data['hbar_s'].mean()),
                "std": float(model_data['hbar_s'].std()),
                "samples": len(model_data)
            }
            
            # P(fail) statistics
            pfail_stats = {
                "mean": float(model_data['p_fail'].mean()),
                "std": float(model_data['p_fail'].std())
            }
            
            # Processing time
            time_stats = {
                "mean_ms": float(model_data['processing_time_ms'].mean()),
                "std_ms": float(model_data['processing_time_ms'].std())
            }
            
            model_analysis[model_id] = {
                "hbar_statistics": hbar_stats,
                "pfail_statistics": pfail_stats,
                "processing_time": time_stats
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
                "avg_processing_time_ms": float(method_data['processing_time_ms'].mean())
            }
            
            method_analysis[method] = stats
            
        return method_analysis
    
    def generate_detailed_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed insights"""
        
        insights = {}
        
        if len(df) > 10:
            # Model rankings by ℏₛ
            model_hbar_means = df.groupby('model_id')['hbar_s'].mean().sort_values(ascending=False)
            insights["model_hbar_ranking"] = model_hbar_means.to_dict()
            
            # Method rankings by ℏₛ
            method_hbar_means = df.groupby('method')['hbar_s'].mean().sort_values(ascending=False)
            insights["method_hbar_ranking"] = method_hbar_means.to_dict()
            
            # Processing time rankings
            method_time_means = df.groupby('method')['processing_time_ms'].mean().sort_values()
            insights["method_speed_ranking"] = method_time_means.to_dict()
        
        return insights
    
    def print_extended_results(self, report: Dict[str, Any]):
        """Print extended results"""
        
        print("\\n" + "="*100)
        print("🎯 EXTENDED MODEL EVALUATION - ALL 6 MODELS + TIER ANALYSIS")
        print("="*100)
        
        # Summary
        summary = report["evaluation_summary"]
        print(f"\\n📊 EVALUATION SUMMARY:")
        print(f"   • Total Evaluations: {summary['total_evaluations']:,}")
        print(f"   • Methods Tested: {len(summary['methods_tested'])}")
        print(f"   • Models Tested: {len(summary['models_tested'])}")
        
        # Tier ℏₛ Analysis
        print(f"\\n🧮 TIER-SPECIFIC ℏₛ ANALYSIS BY METHOD:")
        tier_analysis = report["tier_hbar_analysis"]
        
        for method, analysis in tier_analysis.items():
            hbar_stats = analysis["hbar_statistics"]
            print(f"\\n   📈 {method.upper()}:")
            print(f"      • Average ℏₛ: {hbar_stats['mean']:.3f}")
            print(f"      • ℏₛ Range: {hbar_stats['min']:.3f} - {hbar_stats['max']:.3f}")
            print(f"      • ℏₛ Std Dev: {hbar_stats['std']:.3f}")
            
            # Tier performance
            tier_perf = analysis.get("tier_performance", {})
            for tier_name, perf in tier_perf.items():
                print(f"      • {tier_name.upper()}: Acc {perf['accuracy']*100:.1f}%, Prec {perf['precision']*100:.1f}%")
        
        # Model Analysis
        print(f"\\n🤖 MODEL COMPARISON (ℏₛ Analysis):")
        model_analysis = report["model_performance"]
        
        for model_id, analysis in model_analysis.items():
            hbar_stats = analysis["hbar_statistics"]
            time_stats = analysis["processing_time"]
            print(f"   • {model_id:20} | Avg ℏₛ: {hbar_stats['mean']:.3f} | Samples: {hbar_stats['samples']:3d} | Time: {time_stats['mean_ms']:.1f}ms")
        
        # Method Speed Ranking
        insights = report["detailed_insights"]
        if "method_speed_ranking" in insights:
            print(f"\\n⚡ METHOD SPEED RANKING:")
            for method, time_ms in insights["method_speed_ranking"].items():
                print(f"   • {method:15} | {time_ms:.1f}ms")
        
        print("\\n" + "="*100)

def main():
    """Main execution"""
    
    evaluator = ExtendedEvaluator(api_base_url="http://localhost:8080")
    
    # Run extended evaluation with all models
    evaluator.run_extended_evaluation(max_samples=24)  # 24 samples = 12 correct + 12 incorrect
    
    print("\\n🎉 Extended Evaluation Complete!")
    print("📁 Files generated:")
    print("   • extended_evaluation_results.csv")
    print("   • extended_evaluation_report.json")

if __name__ == "__main__":
    main()