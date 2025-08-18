#!/usr/bin/env python3
"""
Golden Scale Universality Analysis
Tests whether the 3.4 golden scale factor normalizes across different model behaviors
or if per-model λ/τ tuning is still necessary.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'scripts')
from calibrate_failure_law import compute_pfail

def analyze_golden_scale_universality():
    """Analyze whether golden scale normalizes model behavior differences."""
    print("🎭 GOLDEN SCALE UNIVERSALITY ANALYSIS")
    print("=" * 80)
    
    # Extract model configurations from models.json
    model_configs = {
        "mixtral-8x7b": {"lambda": 0.1, "tau": 0.3, "type": "MoE SOTA"},
        "mistral-7b": {"lambda": 0.1, "tau": 1.115, "type": "7B baseline", "golden_enabled": True},
        "qwen2.5-7b": {"lambda": 0.1, "tau": 0.3, "type": "Multi-lingual"},
        "pythia-6.9b": {"lambda": 0.1, "tau": 0.3, "type": "Research baseline"},
        "dialogpt-medium": {"lambda": 1.0, "tau": 2.083, "type": "Conversational"},
        "ollama-mistral-7b": {"lambda": 0.1, "tau": 0.3, "type": "Edge deployment"}
    }
    
    # Test scenarios with varying semantic uncertainty levels
    test_scenarios = [
        {"h_s": 0.1, "description": "High confidence hallucination"},
        {"h_s": 0.3, "description": "Medium confidence hallucination"},
        {"h_s": 0.5, "description": "Low confidence statement"},
        {"h_s": 0.8, "description": "Moderate truth"},
        {"h_s": 1.0, "description": "High confidence truth"},
        {"h_s": 1.5, "description": "Very high confidence truth"},
    ]
    
    golden_scale = 3.4
    universal_params = {"lambda": 5.0, "tau": 2.0}  # Our current universal config
    
    print("\n🔬 PARAMETER VARIATION ANALYSIS")
    print("-" * 80)
    print("Current per-model λ/τ variations:")
    
    lambda_values = [config["lambda"] for config in model_configs.values()]
    tau_values = [config["tau"] for config in model_configs.values()]
    
    print(f"λ range: {min(lambda_values):.1f} - {max(lambda_values):.1f} (variation: {max(lambda_values)/min(lambda_values):.1f}×)")
    print(f"τ range: {min(tau_values):.3f} - {max(tau_values):.3f} (variation: {max(tau_values)/min(tau_values):.1f}×)")
    
    # Test 1: Per-model parameters WITHOUT golden scale
    print(f"\n📊 TEST 1: PER-MODEL PARAMETERS (No Golden Scale)")
    print("-" * 80)
    print(f"{'Scenario':<30} | {'ℏₛ':<6} | ", end="")
    for model in model_configs.keys():
        print(f"{model:<15}", end=" | ")
    print()
    print("-" * 150)
    
    per_model_without_golden = {}
    for scenario in test_scenarios:
        h_s = scenario["h_s"]
        desc = scenario["description"]
        print(f"{desc:<30} | {h_s:<6.1f} | ", end="")
        
        per_model_without_golden[desc] = {}
        for model, config in model_configs.items():
            p_fail = compute_pfail(h_s, config["lambda"], config["tau"], 1.0)
            per_model_without_golden[desc][model] = p_fail
            print(f"{p_fail:<15.6f}", end=" | ")
        print()
    
    # Test 2: Per-model parameters WITH golden scale
    print(f"\n📊 TEST 2: PER-MODEL PARAMETERS + GOLDEN SCALE (3.4×)")
    print("-" * 80)
    print(f"{'Scenario':<30} | {'ℏₛ':<6} | ", end="")
    for model in model_configs.keys():
        print(f"{model:<15}", end=" | ")
    print()
    print("-" * 150)
    
    per_model_with_golden = {}
    for scenario in test_scenarios:
        h_s = scenario["h_s"]
        desc = scenario["description"]
        print(f"{desc:<30} | {h_s:<6.1f} | ", end="")
        
        per_model_with_golden[desc] = {}
        for model, config in model_configs.items():
            p_fail = compute_pfail(h_s, config["lambda"], config["tau"], golden_scale)
            per_model_with_golden[desc][model] = p_fail
            print(f"{p_fail:<15.6f}", end=" | ")
        print()
    
    # Test 3: Universal parameters WITH golden scale
    print(f"\n📊 TEST 3: UNIVERSAL PARAMETERS + GOLDEN SCALE (λ={universal_params['lambda']}, τ={universal_params['tau']})")
    print("-" * 80)
    print(f"{'Scenario':<30} | {'ℏₛ':<6} | {'P(fail)':<15} | {'Golden Effect':<15}")
    print("-" * 80)
    
    universal_with_golden = {}
    for scenario in test_scenarios:
        h_s = scenario["h_s"]
        desc = scenario["description"]
        
        p_fail_baseline = compute_pfail(h_s, universal_params["lambda"], universal_params["tau"], 1.0)
        p_fail_golden = compute_pfail(h_s, universal_params["lambda"], universal_params["tau"], golden_scale)
        
        effect = f"{p_fail_golden/p_fail_baseline:.1f}×" if p_fail_baseline > 0 else "∞"
        
        universal_with_golden[desc] = p_fail_golden
        print(f"{desc:<30} | {h_s:<6.1f} | {p_fail_golden:<15.6f} | {effect:<15}")
    
    # Analysis: Calculate variance across models
    print(f"\n🎯 VARIANCE ANALYSIS")
    print("-" * 80)
    
    def calculate_coefficient_of_variation(values):
        """Calculate coefficient of variation (std/mean)."""
        if len(values) == 0 or np.mean(values) == 0:
            return float('inf')
        return np.std(values) / np.mean(values)
    
    print("Coefficient of Variation (lower = more normalized):")
    print(f"{'Scenario':<30} | {'Without Golden':<15} | {'With Golden':<15} | {'Universal Golden':<15} | {'Normalization':<15}")
    print("-" * 100)
    
    normalization_scores = {}
    for scenario in test_scenarios:
        desc = scenario["description"]
        
        # Values without golden scale (per-model params)
        values_without = list(per_model_without_golden[desc].values())
        cv_without = calculate_coefficient_of_variation(values_without)
        
        # Values with golden scale (per-model params) 
        values_with = list(per_model_with_golden[desc].values())
        cv_with = calculate_coefficient_of_variation(values_with)
        
        # Universal golden scale value
        universal_value = universal_with_golden[desc]
        
        # Normalization effect (lower CV = better normalization)
        normalization_effect = "BETTER" if cv_with < cv_without else "WORSE"
        normalization_factor = cv_without / cv_with if cv_with > 0 else float('inf')
        
        normalization_scores[desc] = {
            "cv_without": cv_without,
            "cv_with": cv_with, 
            "universal_value": universal_value,
            "normalization_factor": normalization_factor,
            "effect": normalization_effect
        }
        
        print(f"{desc:<30} | {cv_without:<15.3f} | {cv_with:<15.3f} | {universal_value:<15.6f} | {normalization_effect:<15}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS")
    print("-" * 80)
    
    # Calculate average normalization
    valid_normalizations = [s["normalization_factor"] for s in normalization_scores.values() 
                           if s["normalization_factor"] != float('inf')]
    avg_normalization = np.mean(valid_normalizations) if valid_normalizations else 0
    
    better_count = sum(1 for s in normalization_scores.values() if s["effect"] == "BETTER")
    total_count = len(normalization_scores)
    
    print(f"1. Golden Scale Normalization Effect:")
    print(f"   - Scenarios improved: {better_count}/{total_count} ({better_count/total_count*100:.1f}%)")
    print(f"   - Average normalization factor: {avg_normalization:.2f}×")
    
    # Parameter range analysis
    param_sensitivity = {}
    for desc, scores in normalization_scores.items():
        # Compare universal vs best/worst per-model performance
        per_model_values = list(per_model_with_golden[desc].values())
        universal_value = scores["universal_value"]
        
        best_per_model = max(per_model_values) if per_model_values else 0
        worst_per_model = min(per_model_values) if per_model_values else 0
        
        param_sensitivity[desc] = {
            "universal": universal_value,
            "best_per_model": best_per_model,
            "worst_per_model": worst_per_model,
            "universal_vs_best": abs(universal_value - best_per_model) / best_per_model if best_per_model > 0 else float('inf'),
            "universal_vs_worst": abs(universal_value - worst_per_model) / worst_per_model if worst_per_model > 0 else float('inf')
        }
    
    print(f"\n2. Universal Parameters vs Per-Model Tuning:")
    avg_deviation_best = np.mean([s["universal_vs_best"] for s in param_sensitivity.values() 
                                 if s["universal_vs_best"] != float('inf')])
    avg_deviation_worst = np.mean([s["universal_vs_worst"] for s in param_sensitivity.values() 
                                  if s["universal_vs_worst"] != float('inf')])
    
    print(f"   - Average deviation from best per-model: {avg_deviation_best*100:.1f}%")
    print(f"   - Average deviation from worst per-model: {avg_deviation_worst*100:.1f}%")
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATION")
    print("-" * 80)
    
    if avg_normalization > 1.5 and better_count >= total_count * 0.7:
        recommendation = "Golden Scale provides SIGNIFICANT normalization - Universal parameters viable"
        per_model_needed = False
    elif avg_normalization > 1.2 and better_count >= total_count * 0.5:
        recommendation = "Golden Scale provides MODERATE normalization - Universal parameters mostly viable"  
        per_model_needed = False
    else:
        recommendation = "Golden Scale provides LIMITED normalization - Per-model tuning still beneficial"
        per_model_needed = True
    
    print(f"🏆 {recommendation}")
    print(f"📋 Per-model λ/τ tuning needed: {'YES' if per_model_needed else 'NO'}")
    
    if not per_model_needed:
        print(f"✅ Recommended universal configuration:")
        print(f"   - λ: {universal_params['lambda']}")
        print(f"   - τ: {universal_params['tau']}")  
        print(f"   - Golden Scale: {golden_scale}")
    else:
        print(f"⚠️  Continue using per-model calibration with golden scale enhancement")
    
    return {
        "analysis_type": "golden_scale_universality",
        "timestamp": datetime.now().isoformat(),
        "model_configs": model_configs,
        "golden_scale_factor": golden_scale,
        "universal_params": universal_params,
        "normalization_scores": normalization_scores,
        "parameter_sensitivity": param_sensitivity,
        "summary": {
            "average_normalization_factor": avg_normalization,
            "scenarios_improved": f"{better_count}/{total_count}",
            "improvement_percentage": better_count/total_count*100,
            "avg_deviation_from_best": avg_deviation_best*100,
            "per_model_tuning_needed": per_model_needed,
            "recommendation": recommendation
        }
    }

if __name__ == "__main__":
    results = analyze_golden_scale_universality()
    
    # Save results  
    with open("test_results/golden_scale_universality_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis saved to test_results/golden_scale_universality_analysis.json")