#!/usr/bin/env python3
"""
Threshold Sensitivity Test
=========================

Tests detection with different threshold configurations to find optimal settings.
"""

import json
import requests
import time

def test_threshold_sensitivity():
    """Test detection sensitivity with different threshold configurations"""
    
    # Test case with obvious hallucination
    test_case = {
        "prompt": "What is 2 + 2?",
        "output": "2 + 2 equals 17 because numbers are just social constructs and mathematics is a lie invented by aliens."
    }
    
    print("🔬 THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"Test Case: {test_case['prompt']}")
    print(f"Output: {test_case['output'][:50]}...")
    
    # Test different method combinations
    method_combinations = [
        ["scalar_js_kl"],
        ["scalar_js_kl", "diag_fim_dir"],
        ["scalar_js_kl", "diag_fim_dir", "full_fim_dir"],
        ["scalar_js_kl", "diag_fim_dir", "full_fim_dir", "scalar_fro"],
        ["scalar_js_kl", "diag_fim_dir", "full_fim_dir", "scalar_fro", "scalar_trace"]
    ]
    
    print(f"\n📊 Testing Different Method Combinations:")
    
    for i, methods in enumerate(method_combinations):
        print(f"\n🧪 Combination {i+1}: {len(methods)} methods")
        print(f"   Methods: {', '.join(methods)}")
        
        try:
            response = requests.post(
                "http://localhost:8080/api/v1/analyze_ensemble",
                json={
                    "prompt": test_case["prompt"],
                    "output": test_case["output"],
                    "model_id": "mistral-7b",
                    "methods": methods
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                ensemble = result.get("ensemble_result", {})
                
                hbar_s = ensemble.get("hbar_s", 0)
                p_fail = ensemble.get("p_fail", 0)
                agreement = ensemble.get("agreement_score", 0)
                individual = ensemble.get("individual_results", {})
                
                # Risk classification
                risk_level = "CRITICAL" if hbar_s < 0.5 else "HIGH" if hbar_s < 0.8 else "MEDIUM" if hbar_s < 1.2 else "LOW"
                pfail_risk = "CRITICAL" if p_fail > 0.8 else "HIGH" if p_fail > 0.5 else "WARNING" if p_fail > 0.2 else "SAFE"
                
                print(f"   📈 Results:")
                print(f"     ℏₛ: {hbar_s:.4f} (Risk: {risk_level})")
                print(f"     P(fail): {p_fail:.4f} (Risk: {pfail_risk})")
                print(f"     Agreement: {agreement:.4f}")
                print(f"     Processing: {result.get('processing_time_ms', 0):.1f}ms")
                
                # Show individual method variance
                if individual:
                    scores = list(individual.values())
                    min_score = min(scores)
                    max_score = max(scores)
                    variance = max_score - min_score
                    print(f"     Method Variance: {variance:.3f} (min: {min_score:.3f}, max: {max_score:.3f})")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Test threshold edge cases
    print(f"\n🎯 Testing Known Threshold Edge Cases:")
    
    edge_cases = [
        {
            "name": "Borderline Math Error",
            "prompt": "What is 5 + 5?",
            "output": "5 + 5 equals 11, which is a common rounding approximation."
        },
        {
            "name": "Subtle Historical Error",
            "prompt": "When did World War II end?",
            "output": "World War II ended in 1946 when the surrender documents were finally ratified."
        },
        {
            "name": "Complex but Correct",
            "prompt": "Explain quantum entanglement",
            "output": "Quantum entanglement is a physical phenomenon where quantum states remain connected regardless of distance, demonstrating non-local correlations."
        }
    ]
    
    for case in edge_cases:
        print(f"\n🔍 Edge Case: {case['name']}")
        
        try:
            response = requests.post(
                "http://localhost:8080/api/v1/analyze",
                json={
                    "prompt": case["prompt"],
                    "output": case["output"],
                    "model_id": "mistral-7b",
                    "intelligent_routing": True,
                    "comprehensive_metrics": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                ensemble = result.get("ensemble_result", {})
                metrics = result.get("comprehensive_metrics", {})
                
                hbar_s = ensemble.get("hbar_s", 0)
                p_fail = ensemble.get("p_fail", 0)
                methods_used = ensemble.get("methods_used", [])
                
                risk_level = "CRITICAL" if hbar_s < 0.5 else "HIGH" if hbar_s < 0.8 else "MEDIUM" if hbar_s < 1.2 else "LOW"
                
                print(f"   ℏₛ: {hbar_s:.4f} | P(fail): {p_fail:.4f} | Risk: {risk_level} | Methods: {len(methods_used)}")
                
                # Check if comprehensive metrics detected anything interesting
                if metrics and "risk_assessment" in metrics:
                    risk_factors = metrics["risk_assessment"].get("risk_factors", [])
                    if risk_factors:
                        print(f"   🚨 Risk Factors: {', '.join(risk_factors)}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n✅ Threshold sensitivity analysis complete!")

if __name__ == "__main__":
    test_threshold_sensitivity()