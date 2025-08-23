#!/usr/bin/env python3
"""
Analyze and compare large-scale evaluation results from HaluEval and TruthfulQA.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def analyze_evaluation_results(halueval_file: str, truthfulqa_file: str):
    """Comprehensive analysis of both evaluation datasets"""
    
    print("📊 LARGE-SCALE EVALUATION ANALYSIS")
    print("=" * 80)
    
    results = {}
    
    # Load results
    for name, filepath in [("HaluEval", halueval_file), ("TruthfulQA", truthfulqa_file)]:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            results[name] = data
            print(f"✅ Loaded {name}: {filepath}")
        else:
            print(f"⏳ {name}: File not found - {filepath}")
            results[name] = None
    
    print("\n" + "=" * 80)
    
    # Summary comparison
    if all(results.values()):
        print("🎯 COMPARATIVE SUMMARY")
        print("-" * 40)
        
        for dataset_name, data in results.items():
            if 'summary' in data:
                summary = data['summary']
                print(f"\n📋 {dataset_name} Results:")
                print(f"   🔢 Total Samples: {summary.get('total_samples', 'N/A')}")
                print(f"   📈 ROC-AUC: {summary.get('roc_auc', 'N/A'):.4f}")
                print(f"   📉 Brier Score: {summary.get('brier_score', 'N/A'):.4f}")
                print(f"   🎯 ECE: {summary.get('ece_uniform_10', 'N/A'):.4f}")
                print(f"   📊 MCE: {summary.get('mce_uniform_10', 'N/A'):.4f}")
                if 'processing_time_seconds' in summary:
                    print(f"   ⏱️  Processing Time: {summary['processing_time_seconds']:.1f}s")
                if 'statistical_significance' in summary:
                    stats = summary['statistical_significance']
                    print(f"   📈 Statistical Significance (p-value): {stats.get('welch_t_test_p_value', 'N/A'):.6f}")
        
        # Cross-dataset comparison
        print(f"\n🔬 CROSS-DATASET INSIGHTS")
        print("-" * 40)
        
        halueval_auc = results["HaluEval"]['summary'].get('roc_auc', 0)
        truthfulqa_auc = results["TruthfulQA"]['summary'].get('roc_auc', 0)
        
        if halueval_auc and truthfulqa_auc:
            auc_diff = abs(halueval_auc - truthfulqa_auc)
            print(f"   📊 ROC-AUC Difference: {auc_diff:.4f}")
            
            if auc_diff < 0.05:
                print("   ✅ Consistent performance across datasets")
            elif halueval_auc > truthfulqa_auc:
                print("   🔍 Better detection on HaluEval (synthetic hallucinations)")
            else:
                print("   🔍 Better detection on TruthfulQA (factual misconceptions)")
        
        # Method performance analysis
        print(f"\n🔧 METHOD PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        for dataset_name, data in results.items():
            if 'method_analysis' in data:
                method_analysis = data['method_analysis']
                print(f"\n   {dataset_name} Top Methods:")
                
                # Sort methods by performance
                method_scores = []
                for method, scores in method_analysis.items():
                    if isinstance(scores, dict) and 'roc_auc' in scores:
                        method_scores.append((method, scores['roc_auc']))
                
                method_scores.sort(key=lambda x: x[1], reverse=True)
                
                for i, (method, score) in enumerate(method_scores[:3], 1):
                    print(f"   {i}. {method}: {score:.4f} ROC-AUC")
    
    # Individual detailed analysis
    for dataset_name, data in results.items():
        if data and 'detailed_results' in data:
            print(f"\n📋 {dataset_name.upper()} DETAILED ANALYSIS")
            print("-" * 60)
            
            detailed = data['detailed_results']
            
            # Calculate distribution statistics
            hbar_values = [item['hbar_s'] for item in detailed if 'hbar_s' in item]
            p_fail_values = [item['p_fail'] for item in detailed if 'p_fail' in item]
            
            if hbar_values:
                print(f"   📊 Semantic Uncertainty (ℏₛ) Distribution:")
                print(f"      Mean: {np.mean(hbar_values):.4f}")
                print(f"      Std:  {np.std(hbar_values):.4f}")
                print(f"      Min:  {np.min(hbar_values):.4f}")
                print(f"      Max:  {np.max(hbar_values):.4f}")
                
                # Risk categorization
                critical = sum(1 for h in hbar_values if h < 0.8)
                warning = sum(1 for h in hbar_values if 0.8 <= h < 1.2)
                safe = sum(1 for h in hbar_values if h >= 1.2)
                
                print(f"   🚨 Risk Distribution:")
                print(f"      Critical (ℏₛ < 0.8): {critical} ({critical/len(hbar_values)*100:.1f}%)")
                print(f"      Warning (0.8 ≤ ℏₛ < 1.2): {warning} ({warning/len(hbar_values)*100:.1f}%)")
                print(f"      Safe (ℏₛ ≥ 1.2): {safe} ({safe/len(hbar_values)*100:.1f}%)")
    
    # Generate final recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("=" * 80)
    
    if all(results.values()):
        overall_samples = sum(r['summary'].get('total_samples', 0) for r in results.values() if 'summary' in r)
        avg_auc = np.mean([r['summary'].get('roc_auc', 0) for r in results.values() if 'summary' in r and 'roc_auc' in r['summary']])
        
        print(f"✅ Successfully evaluated {overall_samples} samples with real model inference")
        print(f"📈 Average ROC-AUC across datasets: {avg_auc:.4f}")
        
        if avg_auc > 0.75:
            print("🎉 EXCELLENT: System demonstrates strong hallucination detection capability")
        elif avg_auc > 0.65:
            print("✅ GOOD: System shows reliable detection with room for improvement")
        else:
            print("⚠️  NEEDS WORK: Consider ensemble rebalancing or threshold adjustment")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Deploy to production with current calibration")
        print(f"   2. Monitor real-world performance metrics")
        print(f"   3. Consider multi-model ensemble for enhanced accuracy")
        
    else:
        print("⏳ Waiting for evaluation completion...")
    
    print("\n" + "=" * 80)

def main():
    """Main execution"""
    halueval_file = "halueval_large_scale_real_logits.json"
    truthfulqa_file = "truthfulqa_large_scale_real_logits.json" 
    
    analyze_evaluation_results(halueval_file, truthfulqa_file)

if __name__ == "__main__":
    main()