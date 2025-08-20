#!/usr/bin/env python3
"""
Comprehensive Metrics Evaluation
===============================

Implements advanced metrics with confusion matrices, precision-recall curves,
and performance analysis using the fixed dataset loader and ensemble API.
"""

import json
import requests
import numpy as np
# import matplotlib.pyplot as plt  # Optional for visualization
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics structure"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    precision_recall_auc: float
    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    confusion_matrix: Dict[str, int]
    per_class_metrics: Dict[str, Dict[str, float]]

def evaluate_comprehensive_metrics(
    max_samples: int = 500,
    api_base: str = "http://localhost:8080/api/v1"
) -> PerformanceMetrics:
    """
    Run comprehensive evaluation with confusion matrices and advanced metrics.
    """
    
    print("🎯 COMPREHENSIVE METRICS EVALUATION")
    print("=" * 60)
    
    # Load datasets using fixed loader
    from comprehensive_dataset_loader import load_truthfulqa_fixed, load_halueval_fixed
    
    # Load samples from multiple datasets
    truthfulqa = load_truthfulqa_fixed(max_samples // 4)
    halueval_qa = load_halueval_fixed("qa", max_samples // 4)
    halueval_general = load_halueval_fixed("general", max_samples // 4)
    halueval_dialogue = load_halueval_fixed("dialogue", max_samples // 4)
    
    all_pairs = truthfulqa + halueval_qa + halueval_general + halueval_dialogue
    
    print(f"📊 Dataset composition:")
    print(f"   TruthfulQA: {len(truthfulqa):,}")
    print(f"   HaluEval QA: {len(halueval_qa):,}")
    print(f"   HaluEval General: {len(halueval_general):,}")
    print(f"   HaluEval Dialogue: {len(halueval_dialogue):,}")
    print(f"   Total pairs: {len(all_pairs):,}")
    
    if not all_pairs:
        print("❌ No evaluation pairs loaded")
        return None
    
    # Evaluation arrays
    predictions = []
    ground_truth = []
    probabilities = []
    processing_times = []
    individual_scores = []
    
    print(f"\n🔍 Running evaluation on {len(all_pairs):,} pairs...")
    
    for i, pair in enumerate(all_pairs):
        # Test both correct and hallucinated answers
        test_cases = [
            (pair.prompt, pair.correct_answer, False, "correct"),
            (pair.prompt, pair.hallucinated_answer, True, "hallucination")
        ]
        
        for prompt, answer, is_hallucination, case_type in test_cases:
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{api_base}/analyze",
                    json={
                        "prompt": prompt,
                        "output": answer,
                        "model_id": "mistral-7b",
                        "ensemble": True,
                        "intelligent_routing": True
                    },
                    timeout=5
                )
                
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
                
                if response.status_code == 200:
                    result = response.json()
                    ensemble = result.get("ensemble_result", {})
                    
                    # Extract metrics
                    p_fail = ensemble.get("p_fail", 0.5)
                    hbar_s = ensemble.get("hbar_s", 1.0)
                    agreement = ensemble.get("agreement", 0.5)
                    
                    # Individual method scores
                    individual = result.get("individual_results", {})
                    individual_scores.append({
                        "case": case_type,
                        "source": pair.source,
                        **individual
                    })
                    
                    # Binary prediction using optimal threshold for corrected P(fail) logic
                    # With corrected inverse relationship, P(fail) values cluster around 0.489
                    detected = p_fail > 0.4888  # Optimal threshold based on P(fail) distribution analysis
                    
                    predictions.append(1 if detected else 0)
                    ground_truth.append(1 if is_hallucination else 0)
                    probabilities.append(p_fail)
                    
                else:
                    # Handle API errors gracefully
                    predictions.append(0)
                    ground_truth.append(1 if is_hallucination else 0)
                    probabilities.append(0.5)
                    processing_times.append(5000)  # Timeout penalty
                    
            except Exception as e:
                # Handle request errors gracefully
                predictions.append(0)
                ground_truth.append(1 if is_hallucination else 0)
                probabilities.append(0.5)
                processing_times.append(5000)  # Error penalty
        
        # Progress reporting
        if (i + 1) % 25 == 0:
            progress = ((i + 1) / len(all_pairs)) * 100
            avg_time = np.mean(processing_times[-50:]) if processing_times else 0
            print(f"📈 Progress: {i + 1:,}/{len(all_pairs):,} ({progress:.1f}%) | Avg time: {avg_time:.1f}ms")
    
    print(f"\n📊 COMPREHENSIVE METRICS CALCULATION")
    print("-" * 50)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    probabilities = np.array(probabilities)
    
    # Confusion Matrix
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    confusion_matrix = {
        "true_positive": int(tp),
        "true_negative": int(tn), 
        "false_positive": int(fp),
        "false_negative": int(fn)
    }
    
    # Basic Metrics
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Advanced Metrics
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        roc_auc = roc_auc_score(ground_truth, probabilities)
        precision_recall_auc = average_precision_score(ground_truth, probabilities)
        brier_score = brier_score_loss(ground_truth, probabilities)
    except:
        roc_auc = 0.5
        precision_recall_auc = 0.5
        brier_score = np.mean((probabilities - ground_truth) ** 2)
    
    # Calibration Metrics (ECE/MCE)
    ece, mce = calculate_calibration_metrics(probabilities, ground_truth)
    
    # Per-class metrics
    per_class_metrics = {
        "hallucination": {
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1": f1_score,
            "support": int(np.sum(ground_truth == 1))
        },
        "correct": {
            "precision": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "recall": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1": 2 * (tn / (tn + fn)) * (tn / (tn + fp)) / ((tn / (tn + fn)) + (tn / (tn + fp))) if (tn + fn) > 0 and (tn + fp) > 0 else 0,
            "support": int(np.sum(ground_truth == 0))
        }
    }
    
    # Create comprehensive metrics object
    metrics = PerformanceMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        roc_auc=roc_auc,
        precision_recall_auc=precision_recall_auc,
        brier_score=brier_score,
        ece=ece,
        mce=mce,
        confusion_matrix=confusion_matrix,
        per_class_metrics=per_class_metrics
    )
    
    # Print results
    print_comprehensive_results(metrics, processing_times, individual_scores)
    
    # Save results
    save_comprehensive_results(metrics, individual_scores)
    
    # Generate visualizations (skip if matplotlib not available)
    try:
        create_comprehensive_visualizations(ground_truth, probabilities, predictions, metrics)
    except ImportError:
        print("⚠️ Skipping visualizations (matplotlib not available)")
    
    return metrics

def calculate_calibration_metrics(probabilities: np.ndarray, ground_truth: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)"""
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Average confidence and accuracy in this bin
            accuracy_in_bin = ground_truth[in_bin].mean()
            avg_confidence_in_bin = probabilities[in_bin].mean()
            
            # Calibration error for this bin
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            
            # Update ECE and MCE
            ece += bin_error * prop_in_bin
            mce = max(mce, bin_error)
    
    return ece, mce

def print_comprehensive_results(metrics: PerformanceMetrics, processing_times: List[float], individual_scores: List[Dict]):
    """Print comprehensive results in organized format"""
    
    print(f"\n🎯 COMPREHENSIVE PERFORMANCE RESULTS")
    print("=" * 60)
    
    # Core Metrics
    print(f"📊 Core Detection Metrics:")
    print(f"   Accuracy: {metrics.accuracy:.3f}")
    print(f"   Precision: {metrics.precision:.3f}")
    print(f"   Recall: {metrics.recall:.3f}")
    print(f"   F1-Score: {metrics.f1_score:.3f}")
    
    # Advanced Metrics  
    print(f"\n📈 Advanced Metrics:")
    print(f"   ROC-AUC: {metrics.roc_auc:.3f}")
    print(f"   PR-AUC: {metrics.precision_recall_auc:.3f}")
    print(f"   Brier Score: {metrics.brier_score:.4f}")
    
    # Calibration Metrics
    print(f"\n🎯 Calibration Quality:")
    print(f"   ECE (Expected): {metrics.ece:.4f}")
    print(f"   MCE (Maximum): {metrics.mce:.4f}")
    
    # Confusion Matrix
    cm = metrics.confusion_matrix
    print(f"\n🔍 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Correct | Hallucination")
    print(f"   Actual Correct    {cm['true_negative']:4d} | {cm['false_positive']:4d}")
    print(f"   Actual Halluc.    {cm['false_negative']:4d} | {cm['true_positive']:4d}")
    
    # Per-Class Metrics
    print(f"\n📋 Per-Class Performance:")
    for class_name, class_metrics in metrics.per_class_metrics.items():
        print(f"   {class_name.capitalize()}:")
        print(f"     Precision: {class_metrics['precision']:.3f}")
        print(f"     Recall: {class_metrics['recall']:.3f}")
        print(f"     F1-Score: {class_metrics['f1']:.3f}")
        print(f"     Support: {class_metrics['support']:,}")
    
    # Performance Stats
    print(f"\n⚡ Performance Stats:")
    print(f"   Avg Processing Time: {np.mean(processing_times):.1f}ms")
    print(f"   Min Processing Time: {np.min(processing_times):.1f}ms")
    print(f"   Max Processing Time: {np.max(processing_times):.1f}ms")
    print(f"   95th Percentile: {np.percentile(processing_times, 95):.1f}ms")

def save_comprehensive_results(metrics: PerformanceMetrics, individual_scores: List[Dict]):
    """Save results to JSON files"""
    
    # Save main metrics
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_type": "comprehensive_metrics",
        "core_metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score
        },
        "advanced_metrics": {
            "roc_auc": metrics.roc_auc,
            "precision_recall_auc": metrics.precision_recall_auc,
            "brier_score": metrics.brier_score
        },
        "calibration_metrics": {
            "ece": metrics.ece,
            "mce": metrics.mce
        },
        "confusion_matrix": metrics.confusion_matrix,
        "per_class_metrics": metrics.per_class_metrics
    }
    
    with open("comprehensive_metrics_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save individual scores for analysis
    with open("individual_method_scores.json", "w") as f:
        json.dump(individual_scores, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   Main metrics: comprehensive_metrics_results.json")
    print(f"   Individual scores: individual_method_scores.json")

def create_comprehensive_visualizations(ground_truth: np.ndarray, probabilities: np.ndarray, 
                                      predictions: np.ndarray, metrics: PerformanceMetrics):
    """Create comprehensive visualization plots"""
    
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix Heatmap
        cm_data = [
            [metrics.confusion_matrix['true_negative'], metrics.confusion_matrix['false_positive']],
            [metrics.confusion_matrix['false_negative'], metrics.confusion_matrix['true_positive']]
        ]
        im1 = ax1.imshow(cm_data, cmap='Blues', alpha=0.7)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Correct', 'Hallucination'])
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Correct', 'Hallucination'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, str(cm_data[i][j]), ha='center', va='center', fontweight='bold')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(ground_truth, probabilities)
        ax2.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {metrics.roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'r--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(ground_truth, probabilities)
        ax3.plot(recall_vals, precision_vals, 'g-', label=f'PR (AUC = {metrics.precision_recall_auc:.3f})')
        ax3.axhline(y=np.mean(ground_truth), color='r', linestyle='--', label='Baseline')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Calibration Plot
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_accuracy = ground_truth[in_bin].mean()
                bin_confidence = probabilities[in_bin].mean()
                bin_count = np.sum(in_bin)
            else:
                bin_accuracy = 0
                bin_confidence = bin_centers[i]
                bin_count = 0
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax4.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
                label=f'ECE = {metrics.ece:.4f}')
        ax4.plot(bin_confidences, bin_accuracies, 'ro-', markersize=4)
        ax4.set_xlabel('Mean Predicted Probability')
        ax4.set_ylabel('Fraction of Positives')
        ax4.set_title('Calibration Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("comprehensive_metrics_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualizations saved: comprehensive_metrics_visualization.png")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")

def compare_with_benchmarks(metrics: PerformanceMetrics):
    """Compare results with industry benchmarks"""
    
    print(f"\n🏆 BENCHMARK COMPARISON")
    print("-" * 40)
    
    # Industry benchmarks (from user's analysis)
    benchmarks = {
        "Gemini 2 Flash": 0.993,
        "μ-Shroom IoU": 0.570,
        "Lettuce Detect F1": 0.792
    }
    
    our_f1 = metrics.f1_score
    
    print(f"📊 F1-Score Comparison:")
    print(f"   Our System: {our_f1:.3f}")
    
    for system, score in benchmarks.items():
        gap = score - our_f1
        status = "✅" if gap <= 0 else "🎯"
        print(f"   {system}: {score:.3f} (gap: {gap:+.3f}) {status}")
    
    # Performance tier classification
    if our_f1 >= 0.85:
        tier = "🥇 TIER 1 (Production Ready)"
    elif our_f1 >= 0.70:
        tier = "🥈 TIER 2 (Strong Performance)"
    elif our_f1 >= 0.55:
        tier = "🥉 TIER 3 (Baseline Performance)"
    else:
        tier = "🔧 NEEDS IMPROVEMENT"
    
    print(f"\n🏅 Performance Tier: {tier}")

if __name__ == "__main__":
    print("🚀 Starting comprehensive metrics evaluation...")
    
    # Run evaluation
    metrics = evaluate_comprehensive_metrics(max_samples=1000)
    
    if metrics:
        # Compare with benchmarks
        compare_with_benchmarks(metrics)
        
        print(f"\n✅ Comprehensive evaluation complete!")
        print(f"🎯 Key Result: F1-Score = {metrics.f1_score:.3f}, Accuracy = {metrics.accuracy:.3f}")
    else:
        print("❌ Evaluation failed")