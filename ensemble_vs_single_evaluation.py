#!/usr/bin/env python3
"""
Ensemble vs Single ℏₛ Performance Evaluation

Compares the performance, reliability, and accuracy of:
1. Single ℏₛ calculation (current approach)
2. Ensemble ℏₛ with multiple calculation methods
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from ensemble_uncertainty_system import (
    EnsembleUncertaintySystem, 
    EnsembleMethod, 
    UncertaintyCalculationMethod
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'scripts')
from calibrate_failure_law import compute_pfail

def evaluate_ensemble_vs_single():
    """Comprehensive evaluation of ensemble vs single ℏₛ approaches."""
    print("🔄 ENSEMBLE vs SINGLE ℏₛ EVALUATION")
    print("=" * 80)
    
    ensemble_system = EnsembleUncertaintySystem(golden_scale=3.4)
    lambda_val, tau_val = 5.0, 2.0
    
    # Comprehensive test scenarios
    test_scenarios = [
        # High confidence scenarios (should have low uncertainty)
        {"name": "Very High Confidence", "p": [0.95, 0.03, 0.01, 0.01], "q": [0.90, 0.05, 0.03, 0.02], "label": "truthful"},
        {"name": "High Confidence", "p": [0.85, 0.10, 0.03, 0.02], "q": [0.80, 0.12, 0.05, 0.03], "label": "truthful"},
        {"name": "Moderate Confidence", "p": [0.70, 0.20, 0.07, 0.03], "q": [0.65, 0.25, 0.07, 0.03], "label": "truthful"},
        
        # Low confidence scenarios (should have high uncertainty/hallucination risk)
        {"name": "Low Confidence", "p": [0.45, 0.30, 0.15, 0.10], "q": [0.40, 0.35, 0.15, 0.10], "label": "uncertain"},
        {"name": "Very Low Confidence", "p": [0.30, 0.30, 0.25, 0.15], "q": [0.25, 0.35, 0.25, 0.15], "label": "hallucination"},
        {"name": "Conflicting Info", "p": [0.60, 0.25, 0.10, 0.05], "q": [0.20, 0.30, 0.30, 0.20], "label": "hallucination"},
        
        # Edge cases
        {"name": "Near Uniform", "p": [0.26, 0.25, 0.25, 0.24], "q": [0.24, 0.26, 0.25, 0.25], "label": "uncertain"},
        {"name": "Sharp Disagreement", "p": [0.80, 0.15, 0.03, 0.02], "q": [0.10, 0.20, 0.30, 0.40], "label": "hallucination"},
        {"name": "Slight Mismatch", "p": [0.75, 0.15, 0.07, 0.03], "q": [0.70, 0.18, 0.08, 0.04], "label": "truthful"},
    ]
    
    # Evaluation metrics
    results = []
    single_method_times = []
    ensemble_method_times = []
    
    print("\n📊 DETAILED COMPARISON")
    print("-" * 120)
    print(f"{'Scenario':<20} | {'Single ℏₛ':<12} | {'Ensemble ℏₛ':<15} | {'Single P(fail)':<13} | {'Ensemble P(fail)':<15} | {'Reliability':<12} | {'Ground Truth':<12}")
    print("-" * 120)
    
    for scenario in test_scenarios:
        name = scenario["name"]
        p_dist = scenario["p"]
        q_dist = scenario["q"]
        ground_truth = scenario["label"]
        
        # Single ℏₛ calculation (current method)
        start_time = datetime.now()
        p = np.array(p_dist)
        q = np.array(q_dist)
        
        # Jensen-Shannon + KL divergence
        m = 0.5 * (p + q)
        js_div = 0.5 * (np.sum(p * np.log((p + 1e-12) / (m + 1e-12))) + 
                        np.sum(q * np.log((q + 1e-12) / (m + 1e-12))))
        kl_div = np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))
        
        single_hbar_s = np.sqrt(js_div * kl_div) * 3.4  # Apply golden scale
        single_p_fail = compute_pfail(single_hbar_s / 3.4, lambda_val, tau_val, 3.4)
        
        single_time = (datetime.now() - start_time).total_seconds() * 1000
        single_method_times.append(single_time)
        
        # Ensemble ℏₛ calculation
        try:
            ensemble_result = ensemble_system.calculate_ensemble_uncertainty(
                p_dist, q_dist,
                ensemble_method=EnsembleMethod.CONFIDENCE_WEIGHTED
            )
            
            ensemble_hbar_s = ensemble_result.ensemble_hbar_s
            ensemble_p_fail = ensemble_result.final_p_fail
            reliability = ensemble_result.reliability_score
            ensemble_method_times.append(ensemble_result.computation_time_ms)
            
        except Exception as e:
            print(f"❌ Ensemble failed for {name}: {e}")
            ensemble_hbar_s = single_hbar_s
            ensemble_p_fail = single_p_fail
            reliability = 0.0
            ensemble_method_times.append(0.0)
        
        results.append({
            "scenario": name,
            "ground_truth": ground_truth,
            "single_hbar_s": single_hbar_s,
            "ensemble_hbar_s": ensemble_hbar_s,
            "single_p_fail": single_p_fail,
            "ensemble_p_fail": ensemble_p_fail,
            "reliability": reliability,
            "single_time_ms": single_time,
            "ensemble_time_ms": ensemble_method_times[-1]
        })
        
        print(f"{name:<20} | {single_hbar_s:<12.6f} | {ensemble_hbar_s:<15.6f} | {single_p_fail:<13.6f} | {ensemble_p_fail:<15.6f} | {reliability:<12.3f} | {ground_truth:<12}")
    
    # Performance analysis
    print(f"\n📈 PERFORMANCE ANALYSIS")
    print("-" * 80)
    
    # Timing analysis
    avg_single_time = np.mean(single_method_times)
    avg_ensemble_time = np.mean(ensemble_method_times)
    time_overhead = (avg_ensemble_time - avg_single_time) / avg_single_time * 100
    
    print(f"⏱️ Timing Comparison:")
    print(f"   Single Method:  {avg_single_time:.2f}ms (avg)")
    print(f"   Ensemble:       {avg_ensemble_time:.2f}ms (avg)")
    print(f"   Overhead:       {time_overhead:.1f}%")
    
    # Classification performance analysis
    def classify_prediction(p_fail, threshold=0.001):
        """Classify based on failure probability."""
        return "hallucination" if p_fail > threshold else "truthful"
    
    # Test different thresholds
    thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    
    print(f"\n🎯 CLASSIFICATION ACCURACY ANALYSIS")
    print(f"{'Threshold':<10} | {'Single Acc':<12} | {'Ensemble Acc':<15} | {'Single F1':<12} | {'Ensemble F1':<15}")
    print("-" * 80)
    
    best_ensemble_threshold = 0.001
    best_ensemble_f1 = 0.0
    
    for threshold in thresholds:
        # Calculate accuracy and F1 for both approaches
        single_correct = 0
        ensemble_correct = 0
        single_tp = single_fp = single_tn = single_fn = 0
        ensemble_tp = ensemble_fp = ensemble_tn = ensemble_fn = 0
        
        for result in results:
            true_label = result["ground_truth"]
            single_pred = classify_prediction(result["single_p_fail"], threshold)
            ensemble_pred = classify_prediction(result["ensemble_p_fail"], threshold)
            
            # Accuracy
            if (true_label == "hallucination" and single_pred == "hallucination") or \
               (true_label != "hallucination" and single_pred == "truthful"):
                single_correct += 1
            
            if (true_label == "hallucination" and ensemble_pred == "hallucination") or \
               (true_label != "hallucination" and ensemble_pred == "truthful"):
                ensemble_correct += 1
            
            # F1 metrics (treating hallucination as positive class)
            actual_positive = true_label == "hallucination"
            
            # Single method
            if single_pred == "hallucination" and actual_positive:
                single_tp += 1
            elif single_pred == "hallucination" and not actual_positive:
                single_fp += 1
            elif single_pred == "truthful" and not actual_positive:
                single_tn += 1
            else:
                single_fn += 1
            
            # Ensemble method
            if ensemble_pred == "hallucination" and actual_positive:
                ensemble_tp += 1
            elif ensemble_pred == "hallucination" and not actual_positive:
                ensemble_fp += 1
            elif ensemble_pred == "truthful" and not actual_positive:
                ensemble_tn += 1
            else:
                ensemble_fn += 1
        
        # Calculate metrics
        single_accuracy = single_correct / len(results)
        ensemble_accuracy = ensemble_correct / len(results)
        
        # F1 scores
        single_precision = single_tp / (single_tp + single_fp) if (single_tp + single_fp) > 0 else 0
        single_recall = single_tp / (single_tp + single_fn) if (single_tp + single_fn) > 0 else 0
        single_f1 = 2 * (single_precision * single_recall) / (single_precision + single_recall) if (single_precision + single_recall) > 0 else 0
        
        ensemble_precision = ensemble_tp / (ensemble_tp + ensemble_fp) if (ensemble_tp + ensemble_fp) > 0 else 0
        ensemble_recall = ensemble_tp / (ensemble_tp + ensemble_fn) if (ensemble_tp + ensemble_fn) > 0 else 0
        ensemble_f1 = 2 * (ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall) if (ensemble_precision + ensemble_recall) > 0 else 0
        
        if ensemble_f1 > best_ensemble_f1:
            best_ensemble_f1 = ensemble_f1
            best_ensemble_threshold = threshold
        
        print(f"{threshold:<10.4f} | {single_accuracy:<12.3f} | {ensemble_accuracy:<15.3f} | {single_f1:<12.3f} | {ensemble_f1:<15.3f}")
    
    # Reliability analysis
    print(f"\n🔒 RELIABILITY ANALYSIS")
    print("-" * 80)
    
    avg_reliability = np.mean([r["reliability"] for r in results])
    reliability_std = np.std([r["reliability"] for r in results])
    
    print(f"📊 Average Reliability Score: {avg_reliability:.3f} ± {reliability_std:.3f}")
    print(f"🎯 Best Ensemble F1 Score: {best_ensemble_f1:.3f} (threshold: {best_ensemble_threshold})")
    
    # Consensus analysis
    uncertainty_differences = [abs(r["ensemble_hbar_s"] - r["single_hbar_s"]) for r in results]
    avg_difference = np.mean(uncertainty_differences)
    max_difference = max(uncertainty_differences)
    
    print(f"📈 Uncertainty Difference Analysis:")
    print(f"   Average |Ensemble - Single|: {avg_difference:.6f}")
    print(f"   Maximum difference: {max_difference:.6f}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 80)
    
    if best_ensemble_f1 > 0.1 and avg_reliability > 0.4:
        recommendation = "✅ RECOMMEND ENSEMBLE: Significant improvement in accuracy and reliability"
        deploy_ensemble = True
    elif time_overhead < 500:  # Less than 5x overhead
        recommendation = "⚖️ CONSIDER ENSEMBLE: Modest improvement with acceptable overhead"
        deploy_ensemble = True
    else:
        recommendation = "❌ STICK WITH SINGLE: Limited benefit for the computational cost"
        deploy_ensemble = False
    
    print(f"🏆 {recommendation}")
    print(f"📊 Performance Summary:")
    print(f"   • Best F1 improvement: {best_ensemble_f1:.3f}")
    print(f"   • Average reliability: {avg_reliability:.3f}")
    print(f"   • Time overhead: {time_overhead:.1f}%")
    print(f"   • Optimal threshold: {best_ensemble_threshold}")
    
    return {
        "evaluation_type": "ensemble_vs_single_hbar_s",
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "performance_summary": {
            "avg_single_time_ms": avg_single_time,
            "avg_ensemble_time_ms": avg_ensemble_time,
            "time_overhead_percent": time_overhead,
            "best_ensemble_f1": best_ensemble_f1,
            "best_threshold": best_ensemble_threshold,
            "avg_reliability": avg_reliability,
            "recommendation": recommendation,
            "deploy_ensemble": deploy_ensemble
        }
    }

if __name__ == "__main__":
    results = evaluate_ensemble_vs_single()
    
    # Save results
    with open("test_results/ensemble_vs_single_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to test_results/ensemble_vs_single_evaluation.json")