#!/usr/bin/env python3
"""
Comprehensive Golden Scale Calibration Evaluation Comparison
Compares performance with and without golden scale calibration
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'scripts')
from calibrate_failure_law import compute_pfail

def evaluate_golden_scale_performance():
    """Evaluate and compare golden scale performance."""
    print("🔬 GOLDEN SCALE CALIBRATION PERFORMANCE EVALUATION")
    print("=" * 80)
    
    # Configuration parameters
    lambda_val = 5.0
    tau_val = 2.0
    golden_scale_factor = 3.4
    
    # Test scenarios with realistic ℏₛ values and ground truth
    test_scenarios = [
        # Hallucinations (should be detected - high P(fail))
        {"h_s": 0.1, "label": "hallucination", "description": "Complete fabrication"},
        {"h_s": 0.15, "label": "hallucination", "description": "False medical claim"},
        {"h_s": 0.2, "label": "hallucination", "description": "Made-up historical fact"},
        {"h_s": 0.25, "label": "hallucination", "description": "Incorrect scientific claim"},
        {"h_s": 0.3, "label": "hallucination", "description": "Fabricated statistics"},
        {"h_s": 0.35, "label": "hallucination", "description": "Wrong geographical info"},
        {"h_s": 0.4, "label": "hallucination", "description": "False biographical data"},
        {"h_s": 0.45, "label": "hallucination", "description": "Incorrect technical info"},
        
        # Truthful content (should pass through - low P(fail))
        {"h_s": 0.8, "label": "truthful", "description": "Common knowledge"},
        {"h_s": 0.9, "label": "truthful", "description": "Well-known fact"},
        {"h_s": 1.0, "label": "truthful", "description": "Verified information"},
        {"h_s": 1.1, "label": "truthful", "description": "Obvious truth"},
        {"h_s": 1.2, "label": "truthful", "description": "Scientific fact"},
        {"h_s": 1.3, "label": "truthful", "description": "Historical accuracy"},
        {"h_s": 1.4, "label": "truthful", "description": "Mathematical truth"},
        {"h_s": 1.5, "label": "truthful", "description": "Established principle"},
        
        # Borderline/uncertain cases
        {"h_s": 0.5, "label": "uncertain", "description": "Questionable claim"},
        {"h_s": 0.6, "label": "uncertain", "description": "Ambiguous statement"},
        {"h_s": 0.7, "label": "uncertain", "description": "Context-dependent info"},
    ]
    
    # Evaluate both configurations
    baseline_results = []
    golden_scale_results = []
    
    print("\n📊 DETAILED PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"{'Description':<25} | {'ℏₛ':<6} | {'Baseline P(fail)':<15} | {'Golden P(fail)':<15} | {'Golden Effect':<12} | {'Ground Truth':<12}")
    print("-" * 80)
    
    for scenario in test_scenarios:
        h_s = scenario["h_s"]
        label = scenario["label"]
        desc = scenario["description"]
        
        # Baseline (no golden scale)
        p_baseline = compute_pfail(h_s, lambda_val, tau_val, 1.0)
        
        # Golden scale enabled
        p_golden = compute_pfail(h_s, lambda_val, tau_val, golden_scale_factor)
        
        # Calculate improvement factor
        if p_baseline > 0:
            improvement = p_golden / p_baseline
        else:
            improvement = float('inf') if p_golden > 0 else 1.0
            
        effect = "High" if improvement > 10 else "Medium" if improvement > 2 else "Low"
        
        baseline_results.append({
            "h_s": h_s,
            "p_fail": p_baseline,
            "label": label,
            "description": desc
        })
        
        golden_scale_results.append({
            "h_s": h_s,
            "p_fail": p_golden,
            "label": label,
            "description": desc
        })
        
        print(f"{desc:<25} | {h_s:<6.2f} | {p_baseline:<15.6f} | {p_golden:<15.6f} | {effect:<12} | {label:<12}")
    
    # Calculate performance metrics
    def calculate_metrics(results, threshold=0.5):
        """Calculate binary classification metrics."""
        tp = fp = tn = fn = 0
        
        for result in results:
            predicted_hallucination = result["p_fail"] > threshold
            actual_hallucination = result["label"] == "hallucination"
            
            if predicted_hallucination and actual_hallucination:
                tp += 1
            elif predicted_hallucination and not actual_hallucination:
                fp += 1
            elif not predicted_hallucination and not actual_hallucination:
                tn += 1
            else:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy
        }
    
    print("\n📈 PERFORMANCE METRICS COMPARISON")
    print("=" * 80)
    
    baseline_metrics = calculate_metrics(baseline_results)
    golden_metrics = calculate_metrics(golden_scale_results)
    
    print("Baseline Configuration (No Golden Scale):")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.3f}")
    print(f"  Precision: {baseline_metrics['precision']:.3f}")
    print(f"  Recall:    {baseline_metrics['recall']:.3f}")
    print(f"  F1-Score:  {baseline_metrics['f1']:.3f}")
    print(f"  TP: {baseline_metrics['tp']}, FP: {baseline_metrics['fp']}, TN: {baseline_metrics['tn']}, FN: {baseline_metrics['fn']}")
    
    print("\nGolden Scale Configuration (3.4x scaling):")
    print(f"  Accuracy:  {golden_metrics['accuracy']:.3f}")
    print(f"  Precision: {golden_metrics['precision']:.3f}")
    print(f"  Recall:    {golden_metrics['recall']:.3f}")
    print(f"  F1-Score:  {golden_metrics['f1']:.3f}")
    print(f"  TP: {golden_metrics['tp']}, FP: {golden_metrics['fp']}, TN: {golden_metrics['tn']}, FN: {golden_metrics['fn']}")
    
    # Calculate improvements
    print(f"\n🚀 GOLDEN SCALE IMPROVEMENTS:")
    print(f"  Accuracy Δ:  {golden_metrics['accuracy'] - baseline_metrics['accuracy']:+.3f}")
    print(f"  Precision Δ: {golden_metrics['precision'] - baseline_metrics['precision']:+.3f}")
    print(f"  Recall Δ:    {golden_metrics['recall'] - baseline_metrics['recall']:+.3f}")
    print(f"  F1-Score Δ:  {golden_metrics['f1'] - baseline_metrics['f1']:+.3f}")
    
    # Threshold sensitivity analysis
    print(f"\n🎯 THRESHOLD SENSITIVITY ANALYSIS")
    print("-" * 80)
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    
    print(f"{'Threshold':<10} | {'Baseline F1':<12} | {'Golden F1':<10} | {'Improvement':<12}")
    print("-" * 50)
    
    best_golden_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        baseline_f1 = calculate_metrics(baseline_results, threshold)['f1']
        golden_f1 = calculate_metrics(golden_scale_results, threshold)['f1']
        improvement = golden_f1 - baseline_f1
        
        if golden_f1 > best_golden_f1:
            best_golden_f1 = golden_f1
            best_threshold = threshold
            
        print(f"{threshold:<10.1f} | {baseline_f1:<12.3f} | {golden_f1:<10.3f} | {improvement:+<12.3f}")
    
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"  Best Threshold: {best_threshold}")
    print(f"  Best F1-Score:  {best_golden_f1:.3f}")
    
    # Save detailed results
    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "golden_scale_comparison",
        "configuration": {
            "lambda": lambda_val,
            "tau": tau_val,
            "golden_scale_factor": golden_scale_factor
        },
        "baseline_results": {
            "configuration": "no_golden_scale",
            "metrics": baseline_metrics,
            "detailed_results": baseline_results
        },
        "golden_scale_results": {
            "configuration": f"golden_scale_{golden_scale_factor}",
            "metrics": golden_metrics,
            "detailed_results": golden_scale_results
        },
        "improvements": {
            "accuracy_delta": golden_metrics['accuracy'] - baseline_metrics['accuracy'],
            "precision_delta": golden_metrics['precision'] - baseline_metrics['precision'],
            "recall_delta": golden_metrics['recall'] - baseline_metrics['recall'],
            "f1_delta": golden_metrics['f1'] - baseline_metrics['f1']
        },
        "optimal_threshold": best_threshold,
        "best_f1_score": best_golden_f1,
        "summary": {
            "golden_scale_effective": golden_metrics['f1'] > baseline_metrics['f1'],
            "improvement_magnitude": "significant" if golden_metrics['f1'] - baseline_metrics['f1'] > 0.1 else "moderate" if golden_metrics['f1'] - baseline_metrics['f1'] > 0.05 else "minimal",
            "recommended": golden_metrics['f1'] > baseline_metrics['f1']
        }
    }
    
    return comparison_results

if __name__ == "__main__":
    results = evaluate_golden_scale_performance()
    
    # Save results
    with open("test_results/golden_scale_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to test_results/golden_scale_comparison_results.json")
    print(f"🎯 Summary: Golden Scale calibration {'RECOMMENDED' if results['summary']['recommended'] else 'NOT RECOMMENDED'}")
    print(f"📊 Performance improvement: {results['summary']['improvement_magnitude'].upper()}")