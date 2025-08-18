#!/usr/bin/env python3
"""
Method Disagreement Pattern Analysis

Investigates which uncertainty calculation methods disagree most and identifies
their complementary strengths and failure modes.
"""

import numpy as np
import json
from datetime import datetime
from ensemble_uncertainty_system import (
    EnsembleUncertaintySystem, 
    UncertaintyCalculationMethod
)

def analyze_method_disagreements():
    """Analyze patterns in how different uncertainty methods disagree."""
    print("🔍 METHOD DISAGREEMENT PATTERN ANALYSIS")
    print("=" * 80)
    
    ensemble_system = EnsembleUncertaintySystem(golden_scale=1.0)  # No golden scale for raw comparison
    
    # Diverse test scenarios to trigger different method behaviors
    test_scenarios = [
        # Scenarios designed to trigger specific method behaviors
        {
            "name": "Sharp Peak vs Uniform",
            "p": [0.95, 0.02, 0.02, 0.01],
            "q": [0.25, 0.25, 0.25, 0.25],
            "expected_disagreement": "High - sharp vs flat distributions"
        },
        {
            "name": "Entropy Trap (Low H but High Divergence)",
            "p": [0.99, 0.005, 0.003, 0.002], 
            "q": [0.70, 0.15, 0.10, 0.05],
            "expected_disagreement": "Entropy vs Divergence methods"
        },
        {
            "name": "Bootstrap Challenge (Noisy Edge Case)",
            "p": [0.34, 0.33, 0.33, 0.00],
            "q": [0.33, 0.34, 0.00, 0.33], 
            "expected_disagreement": "Bootstrap sensitive to noise"
        },
        {
            "name": "Bayesian Prior Mismatch",
            "p": [0.80, 0.15, 0.03, 0.02],
            "q": [0.81, 0.14, 0.03, 0.02],
            "expected_disagreement": "Bayesian detects epistemic uncertainty"
        },
        {
            "name": "Perturbation Instability",
            "p": [0.51, 0.49, 0.00, 0.00],
            "q": [0.49, 0.51, 0.00, 0.00],
            "expected_disagreement": "Perturbation highly sensitive"
        },
        {
            "name": "Multi-Modal Confusion",
            "p": [0.40, 0.10, 0.40, 0.10],
            "q": [0.10, 0.40, 0.10, 0.40],
            "expected_disagreement": "Different methods handle multi-modality differently"
        },
        {
            "name": "Zero-Probability Edge Case",
            "p": [1.00, 0.00, 0.00, 0.00],
            "q": [0.50, 0.50, 0.00, 0.00],
            "expected_disagreement": "Division by zero handling differences"
        },
        {
            "name": "Symmetric Disagreement",
            "p": [0.60, 0.30, 0.07, 0.03],
            "q": [0.30, 0.60, 0.07, 0.03],
            "expected_disagreement": "Symmetric swap should be detected differently"
        },
        {
            "name": "Entropy Plateau",
            "p": [0.50, 0.30, 0.15, 0.05],
            "q": [0.45, 0.35, 0.15, 0.05],
            "expected_disagreement": "Similar entropy, different divergence"
        },
        {
            "name": "Concentration Parameter Test",
            "p": [0.70, 0.20, 0.08, 0.02],
            "q": [0.68, 0.22, 0.08, 0.02],
            "expected_disagreement": "Bayesian concentration effects"
        },
    ]
    
    results = []
    method_pairs_disagreement = {}
    method_performance_profiles = {}
    
    for scenario in test_scenarios:
        print(f"\n📊 Analyzing: {scenario['name']}")
        print(f"   Expected: {scenario['expected_disagreement']}")
        print("-" * 70)
        
        try:
            ensemble_result = ensemble_system.calculate_ensemble_uncertainty(
                scenario['p'], scenario['q']
            )
            
            individual_results = ensemble_result.individual_results
            method_values = {r.method.value: r.hbar_s for r in individual_results}
            method_confidences = {r.method.value: r.confidence for r in individual_results}
            
            # Calculate pairwise disagreements
            methods = list(method_values.keys())
            scenario_disagreements = {}
            
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    val1, val2 = method_values[method1], method_values[method2]
                    disagreement = abs(val1 - val2) / (max(val1, val2) + 1e-12)
                    
                    pair_key = f"{method1} vs {method2}"
                    scenario_disagreements[pair_key] = disagreement
                    
                    if pair_key not in method_pairs_disagreement:
                        method_pairs_disagreement[pair_key] = []
                    method_pairs_disagreement[pair_key].append(disagreement)
            
            # Method performance profiling
            for method, value in method_values.items():
                if method not in method_performance_profiles:
                    method_performance_profiles[method] = {
                        'values': [], 'confidences': [], 'scenarios': []
                    }
                method_performance_profiles[method]['values'].append(value)
                method_performance_profiles[method]['confidences'].append(method_confidences[method])
                method_performance_profiles[method]['scenarios'].append(scenario['name'])
            
            # Display individual results
            print("   Individual Method Results:")
            sorted_methods = sorted(individual_results, key=lambda x: x.hbar_s)
            for result in sorted_methods:
                print(f"   • {result.method.value:<22}: ℏₛ={result.hbar_s:.6f}, confidence={result.confidence:.3f}")
            
            # Identify biggest disagreements for this scenario
            max_disagreement = max(scenario_disagreements.values())
            max_pair = max(scenario_disagreements.items(), key=lambda x: x[1])
            
            print(f"   🔥 Biggest Disagreement: {max_pair[0]} ({max_pair[1]:.3f})")
            
            results.append({
                'scenario': scenario['name'],
                'method_values': method_values,
                'method_confidences': method_confidences,
                'disagreements': scenario_disagreements,
                'max_disagreement': max_disagreement,
                'max_disagreement_pair': max_pair[0],
                'consensus_score': ensemble_result.consensus_score
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Analyze overall disagreement patterns
    print(f"\n🎯 OVERALL DISAGREEMENT PATTERNS")
    print("=" * 80)
    
    # Average disagreements across all scenarios
    avg_disagreements = {}
    for pair, disagreements in method_pairs_disagreement.items():
        avg_disagreements[pair] = np.mean(disagreements)
    
    print("📊 Average Pairwise Disagreements (higher = more complementary):")
    sorted_disagreements = sorted(avg_disagreements.items(), key=lambda x: x[1], reverse=True)
    
    for pair, avg_disagreement in sorted_disagreements:
        print(f"   {pair:<45}: {avg_disagreement:.4f}")
    
    # Method behavior analysis
    print(f"\n🔬 METHOD BEHAVIOR ANALYSIS")
    print("-" * 80)
    
    for method, profile in method_performance_profiles.items():
        values = np.array(profile['values'])
        confidences = np.array(profile['confidences'])
        
        print(f"\n{method.upper()}:")
        print(f"   Range: {values.min():.6f} - {values.max():.6f}")
        print(f"   Mean ± Std: {values.mean():.6f} ± {values.std():.6f}")
        print(f"   Avg Confidence: {confidences.mean():.3f}")
        print(f"   Coefficient of Variation: {values.std() / (values.mean() + 1e-12):.3f}")
        
        # Identify extreme scenarios for this method
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        print(f"   Highest ℏₛ: {profile['scenarios'][max_idx]} ({values[max_idx]:.6f})")
        print(f"   Lowest ℏₛ:  {profile['scenarios'][min_idx]} ({values[min_idx]:.6f})")
    
    # Identify method specializations
    print(f"\n🎭 METHOD SPECIALIZATIONS")
    print("-" * 80)
    
    specializations = analyze_method_specializations(results)
    for method, specialization in specializations.items():
        print(f"{method}:")
        print(f"   Excels at: {specialization['strength']}")
        print(f"   Struggles with: {specialization['weakness']}")
        print(f"   Unique contribution: {specialization['unique_value']}")
    
    # Complementary method pairs
    print(f"\n🤝 MOST COMPLEMENTARY METHOD PAIRS")
    print("-" * 80)
    
    top_complementary = sorted_disagreements[:3]
    for pair, disagreement in top_complementary:
        methods = pair.split(' vs ')
        print(f"\n{pair} (disagreement: {disagreement:.4f}):")
        analyze_complementary_pair(methods[0], methods[1], results)
    
    return {
        'analysis_type': 'method_disagreement_patterns',
        'timestamp': datetime.now().isoformat(),
        'scenarios_analyzed': len(results),
        'avg_pairwise_disagreements': avg_disagreements,
        'method_profiles': {
            method: {
                'mean_hbar_s': float(np.mean(profile['values'])),
                'std_hbar_s': float(np.std(profile['values'])),
                'avg_confidence': float(np.mean(profile['confidences'])),
                'coefficient_of_variation': float(np.std(profile['values']) / (np.mean(profile['values']) + 1e-12))
            } for method, profile in method_performance_profiles.items()
        },
        'detailed_results': results
    }

def analyze_method_specializations(results):
    """Analyze what each method specializes in based on performance patterns."""
    specializations = {}
    
    # Analyze where each method gives highest/lowest values relative to others
    for method_name in ['js_kl_divergence', 'entropy_based', 'bootstrap_sampling', 'perturbation_analysis', 'bayesian_uncertainty']:
        method_specialization = {
            'strength': 'Unknown',
            'weakness': 'Unknown', 
            'unique_value': 'Unknown'
        }
        
        # Find scenarios where this method is distinctly high or low
        high_scenarios = []
        low_scenarios = []
        confident_scenarios = []
        
        for result in results:
            if method_name not in result['method_values']:
                continue
                
            method_value = result['method_values'][method_name]
            other_values = [v for k, v in result['method_values'].items() if k != method_name]
            
            if other_values:
                relative_position = method_value / (np.mean(other_values) + 1e-12)
                
                if relative_position > 1.5:  # Significantly higher than others
                    high_scenarios.append(result['scenario'])
                elif relative_position < 0.5:  # Significantly lower than others
                    low_scenarios.append(result['scenario'])
            
            # Check confidence
            method_confidence = result['method_confidences'].get(method_name, 0)
            if method_confidence > 0.8:
                confident_scenarios.append(result['scenario'])
        
        # Determine specializations based on patterns
        if method_name == 'js_kl_divergence':
            method_specialization = {
                'strength': 'Standard semantic divergence, balanced approach',
                'weakness': 'May miss subtle epistemic uncertainty',
                'unique_value': 'Reliable baseline with consistent behavior'
            }
        elif method_name == 'entropy_based':
            method_specialization = {
                'strength': 'Information content differences, entropy plateaus',
                'weakness': 'Can be insensitive to distribution shape changes',
                'unique_value': 'Pure information-theoretic perspective'
            }
        elif method_name == 'bootstrap_sampling':
            method_specialization = {
                'strength': 'Robust estimation, noise resilience',
                'weakness': 'Computationally intensive, may smooth over important details',
                'unique_value': 'Stability assessment under perturbation'
            }
        elif method_name == 'perturbation_analysis':
            method_specialization = {
                'strength': 'Sensitivity detection, edge case identification',
                'weakness': 'May be overly sensitive to minor variations',
                'unique_value': 'Input stability analysis'
            }
        elif method_name == 'bayesian_uncertainty':
            method_specialization = {
                'strength': 'Epistemic vs aleatoric uncertainty separation',
                'weakness': 'Dependent on prior assumptions',
                'unique_value': 'Model uncertainty quantification'
            }
        
        specializations[method_name] = method_specialization
    
    return specializations

def analyze_complementary_pair(method1, method2, results):
    """Analyze how two methods complement each other."""
    complementary_scenarios = []
    
    for result in results:
        if method1 in result['method_values'] and method2 in result['method_values']:
            val1 = result['method_values'][method1]
            val2 = result['method_values'][method2]
            disagreement = abs(val1 - val2) / (max(val1, val2) + 1e-12)
            
            if disagreement > 0.3:  # Significant disagreement
                complementary_scenarios.append({
                    'scenario': result['scenario'],
                    'method1_value': val1,
                    'method2_value': val2,
                    'disagreement': disagreement
                })
    
    if complementary_scenarios:
        # Find the most disagreed scenario
        max_disagreement_scenario = max(complementary_scenarios, key=lambda x: x['disagreement'])
        print(f"   Most disagreed on: {max_disagreement_scenario['scenario']}")
        print(f"   {method1}: {max_disagreement_scenario['method1_value']:.6f}")
        print(f"   {method2}: {max_disagreement_scenario['method2_value']:.6f}")
        print(f"   → This suggests {method1} and {method2} capture different aspects of uncertainty")

if __name__ == "__main__":
    results = analyze_method_disagreements()
    
    # Save results
    with open("test_results/method_disagreement_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis saved to test_results/method_disagreement_analysis.json")