#!/usr/bin/env python3
"""
Test if Golden Scale calibration is actually being applied
"""
import requests
import json

def test_golden_scale():
    """Test raw vs calibrated uncertainty values"""
    
    print("🔬 TESTING GOLDEN SCALE CALIBRATION")
    print("=" * 60)
    
    test_case = {
        "prompt": "What is the capital of France?",
        "output": "The capital of France is Paris.",
        "model_id": "mistral-7b"
    }
    
    # Test single method to see raw calculation
    response = requests.post(
        'http://localhost:8080/api/v1/analyze',
        json={
            **test_case,
            "method": "standard_js_kl"
        },
        timeout=10
    )
    
    if response.status_code == 200:
        single_result = response.json()
        print(f"📊 Single Method (standard_js_kl):")
        print(f"   ℏₛ: {single_result['hbar_s']:.6f}")
        print(f"   δμ: {single_result['delta_mu']:.6f}")
        print(f"   δσ: {single_result['delta_sigma']:.6f}")
        print(f"   Expected with Golden Scale: {(3.4 * single_result['delta_mu'] * single_result['delta_sigma'])**0.5:.6f}")
        print()
    
    # Test ensemble method
    response = requests.post(
        'http://localhost:8080/api/v1/analyze_ensemble',
        json=test_case,
        timeout=10
    )
    
    if response.status_code == 200:
        ensemble_result = response.json()['ensemble_result']
        print(f"📊 Ensemble Method:")
        print(f"   ℏₛ: {ensemble_result['hbar_s']:.6f}")
        print(f"   Individual Results:")
        for method, score in ensemble_result['individual_results'].items():
            print(f"     {method}: {score:.6f}")
        print()
        
        # Check if standard_js_kl has Golden Scale applied
        standard_score = ensemble_result['individual_results'].get('standard_js_kl', 0)
        print(f"🔬 Analysis:")
        print(f"   standard_js_kl score: {standard_score:.6f}")
        print(f"   Without Golden Scale would be: {(standard_score / (3.4**0.5)):.6f}")
        
        # Calculate expected ℏₛ with 3.4x factor
        js_div = single_result['delta_mu']  # This should be JS divergence
        kl_div = single_result['delta_sigma']  # This should be KL divergence  
        expected_hbar = (3.4 * js_div * kl_div)**0.5
        
        print(f"   Expected ℏₛ with Golden Scale: {expected_hbar:.6f}")
        print(f"   Actual ℏₛ: {ensemble_result['hbar_s']:.6f}")
        
        if abs(standard_score - expected_hbar) < 0.01:
            print("✅ Golden Scale appears to be applied correctly!")
        else:
            print("⚠️  Golden Scale may not be applied or is being overridden")

if __name__ == "__main__":
    test_golden_scale()