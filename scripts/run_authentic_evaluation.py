#!/usr/bin/env python3
"""
🎯 DIRECT AUTHENTIC EVALUATION
Run L1→L2→L3 on authentic dataset without complex inheritance
"""

import sys
sys.path.append('.')
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from scripts.world_class_benchmark_runner import WorldClassBenchmarkRunner

def run_direct_authentic_evaluation():
    """Run direct evaluation on authentic dataset."""
    
    print("🎯 DIRECT AUTHENTIC EVALUATION")
    print("Using real HaluEval + TruthfulQA datasets")
    print("=" * 60)
    
    # Load authentic dataset
    authentic_file = Path("authentic_datasets/authentic_hallucination_benchmark.json")
    if not authentic_file.exists():
        print("❌ Authentic dataset not found. Run download_authentic_datasets_fixed.py first")
        return None
        
    with open(authentic_file, 'r') as f:
        authentic_data = json.load(f)
    
    print(f"📊 Loaded {authentic_data['metadata']['total_cases']} authentic cases")
    print(f"   HaluEval: {authentic_data['metadata']['data_sources']['halueval']}")
    print(f"   TruthfulQA: {authentic_data['metadata']['data_sources']['truthfulqa']}")
    
    # Initialize runner with original synthetic data, then override
    runner = WorldClassBenchmarkRunner()
    runner.benchmark_data = authentic_data  # Override with authentic data
    
    print(f"\n🔄 Switched to authentic dataset for evaluation")
    
    # Test subset for performance
    test_cases = authentic_data["test_cases"][:100]  # Test first 100 cases
    model = runner.models_config["models"][0]  # Use Mixtral
    
    print(f"\n🚀 Evaluating {len(test_cases)} cases with {model['display_name']}")
    print(f"⚖️  λ={model['failure_law']['lambda']:.3f}, τ={model['failure_law']['tau']:.3f}")
    
    # Track results
    level_results = {
        "L1_best_combo": {"correct": 0, "total": 0},
        "L2_hbar_plus_pfail": {"correct": 0, "total": 0},
        "L3_hbar_pfail_fep": {"correct": 0, "total": 0}
    }
    
    detailed_results = []
    
    for i, test_case in enumerate(test_cases):
        if i % 25 == 0:
            print(f"Processing case {i+1}/{len(test_cases)}...")
            
        result = runner.evaluate_test_case(test_case, model)
        if not result:
            continue
            
        level_results["L1_best_combo"]["total"] += 1
        level_results["L2_hbar_plus_pfail"]["total"] += 1
        level_results["L3_hbar_pfail_fep"]["total"] += 1
        
        # Extract accuracy results
        if result.get("level_accuracies", {}).get("L1_best_combination", 0) > 0:
            level_results["L1_best_combo"]["correct"] += 1
            
        if result.get("level_accuracies", {}).get("L2_best_hbar_pfail", 0) > 0:
            level_results["L2_hbar_plus_pfail"]["correct"] += 1
            
        if result.get("level_accuracies", {}).get("L3_best_hbar_pfail_fep", 0) > 0:
            level_results["L3_hbar_pfail_fep"]["correct"] += 1
        
        detailed_results.append({
            "case_id": i,
            "source": test_case.get("source", "unknown"),
            "domain": test_case.get("domain", "unknown"),
            "L1_accuracy": result.get("level_accuracies", {}).get("L1_best_combination", 0),
            "L2_accuracy": result.get("level_accuracies", {}).get("L2_best_hbar_pfail", 0),
            "L3_accuracy": result.get("level_accuracies", {}).get("L3_best_hbar_pfail_fep", 0),
            "best_combo": result.get("best_combination", "unknown")
        })
    
    # Calculate final accuracies
    results_summary = {}
    for level, data in level_results.items():
        if data["total"] > 0:
            accuracy = data["correct"] / data["total"]
            results_summary[level] = accuracy
        else:
            results_summary[level] = 0.0
    
    # Display results
    print(f"\n🏆 AUTHENTIC DATASET EVALUATION RESULTS")
    print("=" * 60)
    
    for level, accuracy in results_summary.items():
        emoji = "🎯" if accuracy >= 0.99 else "📈" if accuracy >= 0.85 else "📉"
        level_name = level.replace("_", " ").title()
        print(f"{level_name}: {accuracy:.1%} {emoji}")
    
    # Analysis by source
    print(f"\n📊 PERFORMANCE BY DATA SOURCE:")
    halueval_results = [r for r in detailed_results if "halueval" in r["source"]]
    truthfulqa_results = [r for r in detailed_results if "truthfulqa" in r["source"]]
    
    if halueval_results:
        halueval_l3 = np.mean([r["L3_accuracy"] for r in halueval_results])
        print(f"HaluEval L3 Accuracy:   {halueval_l3:.1%} ({len(halueval_results)} cases)")
        
    if truthfulqa_results:
        truthfulqa_l3 = np.mean([r["L3_accuracy"] for r in truthfulqa_results])
        print(f"TruthfulQA L3 Accuracy: {truthfulqa_l3:.1%} ({len(truthfulqa_results)} cases)")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    if results_summary["L3_hbar_pfail_fep"] >= 0.99:
        print("🎯 SUCCESS: Achieved 99%+ accuracy target with authentic data!")
    elif results_summary["L3_hbar_pfail_fep"] >= 0.90:
        print("📈 EXCELLENT: >90% accuracy - authentic data shows strong discrimination")
    elif results_summary["L3_hbar_pfail_fep"] >= 0.80:
        print("📊 GOOD: >80% accuracy - significant improvement over synthetic plateau")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Still below 80% - may need FEP enhancements")
        
    # Compare to synthetic baseline
    if results_summary["L3_hbar_pfail_fep"] > 0.80:
        improvement = (results_summary["L3_hbar_pfail_fep"] - 0.80) / 0.80 * 100
        print(f"🚀 Improvement over synthetic data plateau: +{improvement:.1f}%")
    
    # Save results
    final_results = {
        "evaluation_type": "authentic_dataset",
        "dataset_info": {
            "total_cases_evaluated": len(detailed_results),
            "halueval_cases": len(halueval_results),
            "truthfulqa_cases": len(truthfulqa_results)
        },
        "level_accuracies": results_summary,
        "detailed_results": detailed_results,
        "target_achieved": results_summary["L3_hbar_pfail_fep"] >= 0.99
    }
    
    output_file = "authentic_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return final_results

def main():
    """Main execution."""
    
    try:
        results = run_direct_authentic_evaluation()
        
        if results and results["target_achieved"]:
            print("\n✅ 99% ACCURACY TARGET ACHIEVED WITH AUTHENTIC DATA!")
        elif results:
            print(f"\n📈 Evaluation completed. L3 accuracy: {results['level_accuracies']['L3_hbar_pfail_fep']:.1%}")
        else:
            print("\n❌ Evaluation failed")
            
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()