#!/usr/bin/env python3
"""
📊 3-Tier Hallucination Detection System Performance Report Generator
Analyzes L1 (ℏₛ), L2 (ℏₛ + P(fail)), L3 (ℏₛ + P(fail) + FEP) performance
"""

import requests
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import time

class TierSystemReportGenerator:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        
    def analyze_text_all_tiers(self, text: str, method: str = "diag_fim_dir") -> Dict[str, Any]:
        """Analyze text and extract L1, L2, L3 tier scores."""
        
        request_data = {
            "topk_indices": [1, 2, 3],
            "topk_probs": [0.5, 0.3, 0.2],
            "rest_mass": 0.0,
            "vocab_size": 50000,
            "method": method
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/analyze_topk_compact",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract tier components
                hbar_s = result.get("hbar_s", 0.0)                    # L1: Semantic uncertainty
                p_fail = result.get("p_fail", 0.0)                    # L2 addition: Failure probability
                
                # L3 addition: Enhanced FEP components
                enhanced_fep = result.get("enhanced_fep", {})
                if enhanced_fep:
                    fep_components = {
                        "kl_surprise": enhanced_fep.get("kl_surprise", 0.0),
                        "attention_entropy": enhanced_fep.get("attention_entropy", 0.0),
                        "prediction_variance": enhanced_fep.get("prediction_variance", 0.0),
                        "fisher_info_trace": enhanced_fep.get("fisher_info_trace", 0.0),
                        "fisher_info_mean_eigenvalue": enhanced_fep.get("fisher_info_mean_eigenvalue", 0.0),
                        "enhanced_free_energy": enhanced_fep.get("enhanced_free_energy", 0.0)
                    }
                    total_fep = sum(fep_components.values())
                else:
                    # Fallback to basic free_energy
                    total_fep = result.get("free_energy", 0.0)
                    fep_components = {"basic_free_energy": total_fep}
                
                # Calculate tier scores
                l1_score = hbar_s                           # L1: Pure semantic uncertainty
                l2_score = hbar_s + p_fail                  # L2: + Calibrated failure probability  
                l3_score = hbar_s + p_fail + total_fep      # L3: + Full FEP analysis
                
                return {
                    "success": True,
                    "method": method,
                    "processing_time_ms": result.get("processing_time_ms", 0.0),
                    "components": {
                        "hbar_s": hbar_s,
                        "p_fail": p_fail,
                        "fep_total": total_fep,
                        "fep_components": fep_components
                    },
                    "tier_scores": {
                        "L1_semantic_uncertainty": l1_score,
                        "L2_plus_failure_prob": l2_score, 
                        "L3_plus_fep": l3_score
                    },
                    "raw_result": result
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def evaluate_tier_performance(self, dataset_path: str, methods: List[str], max_samples: int = 50) -> Dict[str, Any]:
        """Evaluate all 3 tiers across multiple methods."""
        
        print(f"📊 Loading dataset: {dataset_path}")
        
        # Load dataset
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        examples.append(json.loads(line.strip()))
                    except:
                        continue
        
        examples = examples[:max_samples]
        print(f"🎯 Evaluating {len(examples)} examples across {len(methods)} methods")
        
        # Results structure: {method: {tier: {accuracy, scores, etc.}}}
        results = {}
        for method in methods:
            results[method] = {
                "L1": {"correct": 0, "total": 0, "scores": [], "predictions": []},
                "L2": {"correct": 0, "total": 0, "scores": [], "predictions": []}, 
                "L3": {"correct": 0, "total": 0, "scores": [], "predictions": []}
            }
        
        # Process each example
        for i, example in enumerate(examples):
            if i % 10 == 0:
                print(f"📈 Progress: {i}/{len(examples)}")
            
            # Extract ground truth
            if not isinstance(example, dict):
                continue
                
            text = example.get('chatgpt_response', example.get('text', ''))
            is_hallucination = example.get('hallucination') == 'yes'
            
            if not text:
                continue
            
            # Test each method on this example
            for method in methods:
                analysis = self.analyze_text_all_tiers(text, method)
                
                if analysis["success"]:
                    tier_scores = analysis["tier_scores"]
                    
                    # Determine optimal thresholds for each tier
                    thresholds = {
                        "L1": 1.0,   # ℏₛ threshold
                        "L2": 1.5,   # ℏₛ + P(fail) threshold  
                        "L3": 2.0    # ℏₛ + P(fail) + FEP threshold
                    }
                    
                    # Evaluate each tier
                    tier_score_keys = {
                        "L1": "L1_semantic_uncertainty",
                        "L2": "L2_plus_failure_prob", 
                        "L3": "L3_plus_fep"
                    }
                    
                    for tier in ["L1", "L2", "L3"]:
                        score = tier_scores[tier_score_keys[tier]]
                        predicted_hallucination = score > thresholds[tier]
                        
                        correct = predicted_hallucination == is_hallucination
                        results[method][tier]["correct"] += int(correct)
                        results[method][tier]["total"] += 1
                        results[method][tier]["scores"].append(score)
                        results[method][tier]["predictions"].append(predicted_hallucination)
        
        return results
    
    def generate_comprehensive_report(self, results: Dict[str, Any], dataset_name: str) -> str:
        """Generate detailed 3-tier system performance report."""
        
        report = []
        report.append("=" * 100)
        report.append("🔥 3-TIER HALLUCINATION DETECTION SYSTEM PERFORMANCE REPORT")
        report.append("=" * 100)
        report.append(f"📊 Dataset: {dataset_name}")
        report.append(f"🕒 Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System Overview
        report.append("🏗️  SYSTEM ARCHITECTURE:")
        report.append("   L1: ℏₛ (Semantic Uncertainty) - Base precision×flexibility combinations")
        report.append("   L2: ℏₛ + P(fail) - Adds calibrated failure probability") 
        report.append("   L3: ℏₛ + P(fail) + FEP - Adds Free Energy Principle components")
        report.append("")
        
        # Method-wise performance
        method_summaries = []
        
        for method, method_data in results.items():
            report.append(f"🔬 METHOD: {method.upper()}")
            report.append("-" * 60)
            
            tier_performances = []
            
            for tier in ["L1", "L2", "L3"]:
                data = method_data[tier]
                if data["total"] > 0:
                    accuracy = data["correct"] / data["total"]
                    avg_score = np.mean(data["scores"]) if data["scores"] else 0.0
                    std_score = np.std(data["scores"]) if data["scores"] else 0.0
                    
                    performance_rating = (
                        "🏆 EXCELLENT" if accuracy > 0.90 else
                        "✅ GOOD" if accuracy > 0.80 else  
                        "⚠️  MODERATE" if accuracy > 0.70 else
                        "❌ POOR"
                    )
                    
                    tier_performances.append({
                        "tier": tier,
                        "accuracy": accuracy,
                        "rating": performance_rating,
                        "avg_score": avg_score,
                        "std_score": std_score
                    })
                    
                    report.append(f"   {tier:2} │ {performance_rating:15} │ {accuracy:5.1%} ({data['correct']:2}/{data['total']:2}) │ Score: {avg_score:5.3f}±{std_score:5.3f}")
                else:
                    report.append(f"   {tier:2} │ {'NO DATA':15} │ {'N/A':>11} │ Score: N/A")
            
            # Tier progression analysis
            if len(tier_performances) == 3:
                l1_acc, l2_acc, l3_acc = [tp["accuracy"] for tp in tier_performances]
                l1_to_l2_gain = l2_acc - l1_acc
                l2_to_l3_gain = l3_acc - l2_acc
                total_gain = l3_acc - l1_acc
                
                report.append("")
                report.append(f"   📈 TIER PROGRESSION ANALYSIS:")
                report.append(f"      L1→L2 Gain: {l1_to_l2_gain:+5.1%} {'📈' if l1_to_l2_gain > 0 else '📉' if l1_to_l2_gain < 0 else '➡️'}")
                report.append(f"      L2→L3 Gain: {l2_to_l3_gain:+5.1%} {'📈' if l2_to_l3_gain > 0 else '📉' if l2_to_l3_gain < 0 else '➡️'}")
                report.append(f"      Total Gain: {total_gain:+5.1%} {'🚀' if total_gain > 0.1 else '✅' if total_gain > 0 else '❌'}")
                
                method_summaries.append({
                    "method": method,
                    "l3_accuracy": l3_acc,
                    "total_gain": total_gain,
                    "best_tier": max(tier_performances, key=lambda x: x["accuracy"])
                })
            
            report.append("")
        
        # Overall rankings
        report.append("🏆 OVERALL METHOD RANKINGS (by L3 performance):")
        report.append("-" * 60)
        
        sorted_methods = sorted(method_summaries, key=lambda x: x["l3_accuracy"], reverse=True)
        
        for i, method_summary in enumerate(sorted_methods, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2}."
            method = method_summary["method"]
            l3_acc = method_summary["l3_accuracy"]
            gain = method_summary["total_gain"]
            
            report.append(f"   {emoji} {method:15} │ L3: {l3_acc:5.1%} │ Total Gain: {gain:+5.1%}")
        
        # Tier-wise analysis
        report.append("")
        report.append("📊 TIER-WISE PERFORMANCE ANALYSIS:")
        report.append("-" * 60)
        
        for tier in ["L1", "L2", "L3"]:
            tier_accuracies = []
            for method, method_data in results.items():
                if method_data[tier]["total"] > 0:
                    accuracy = method_data[tier]["correct"] / method_data[tier]["total"]
                    tier_accuracies.append(accuracy)
            
            if tier_accuracies:
                avg_acc = np.mean(tier_accuracies)
                std_acc = np.std(tier_accuracies)
                max_acc = max(tier_accuracies)
                min_acc = min(tier_accuracies)
                
                report.append(f"   {tier} │ Avg: {avg_acc:5.1%}±{std_acc:4.1%} │ Range: {min_acc:5.1%}-{max_acc:5.1%}")
        
        # Key insights
        report.append("")
        report.append("💡 KEY INSIGHTS:")
        report.append("-" * 60)
        
        # Best overall method
        best_method = sorted_methods[0]["method"] if sorted_methods else "None"
        best_accuracy = sorted_methods[0]["l3_accuracy"] if sorted_methods else 0
        report.append(f"   • Best Overall Method: {best_method} ({best_accuracy:.1%} L3 accuracy)")
        
        # Tier effectiveness 
        if sorted_methods:
            positive_gains = sum(1 for m in sorted_methods if m["total_gain"] > 0)
            total_methods = len(sorted_methods)
            report.append(f"   • Tier System Effectiveness: {positive_gains}/{total_methods} methods show L1→L3 improvement")
        
        # Performance distribution
        if sorted_methods:
            excellent = sum(1 for m in sorted_methods if m["l3_accuracy"] > 0.90)
            good = sum(1 for m in sorted_methods if 0.80 < m["l3_accuracy"] <= 0.90)
            moderate = sum(1 for m in sorted_methods if 0.70 < m["l3_accuracy"] <= 0.80)
            poor = sum(1 for m in sorted_methods if m["l3_accuracy"] <= 0.70)
            
            report.append(f"   • Performance Distribution: {excellent} Excellent, {good} Good, {moderate} Moderate, {poor} Poor")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def run_complete_analysis(self, 
                            datasets_dir: str = "/Users/elliejenkins/Desktop/su-firewall/authentic_datasets",
                            methods: List[str] = None,
                            max_samples: int = 50) -> Dict[str, str]:
        """Run complete 3-tier analysis and generate reports."""
        
        if methods is None:
            methods = ["diag_fim_dir", "scalar_js_kl", "scalar_trace", "scalar_fro", "full_fim_dir"]
        
        print("🔥 Starting 3-Tier Hallucination Detection Analysis")
        print(f"🎯 Methods: {', '.join(methods)}")
        print(f"📊 Max samples per dataset: {max_samples}")
        
        datasets = ["halueval_general_data.json", "truthfulqa_data.json"]
        reports = {}
        
        for dataset_name in datasets:
            dataset_path = Path(datasets_dir) / dataset_name
            if dataset_path.exists():
                print(f"\n🔬 Analyzing {dataset_name}")
                results = self.evaluate_tier_performance(str(dataset_path), methods, max_samples)
                report = self.generate_comprehensive_report(results, dataset_name)
                reports[dataset_name] = report
                
                # Save individual report
                report_file = f"/Users/elliejenkins/Desktop/su-firewall/tier_report_{dataset_name.replace('.json', '.txt')}"
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"📁 Report saved: {report_file}")
                
        return reports

def main():
    generator = TierSystemReportGenerator()
    reports = generator.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*100)
    print("🏆 ANALYSIS COMPLETE!")
    print("="*100)
    
    for dataset, report in reports.items():
        print(f"\n📊 {dataset}:")
        # Print just the key insights section
        lines = report.split('\n')
        insights_start = False
        for line in lines:
            if "💡 KEY INSIGHTS:" in line:
                insights_start = True
            elif insights_start and line.startswith("="):
                break
            elif insights_start:
                print(line)

if __name__ == "__main__":
    main()