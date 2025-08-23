#!/usr/bin/env python3
"""
Live demonstration of Golden Scale Calibration via API
"""

import requests
import json

def test_golden_scale_api():
    """Test the golden scale calibration through the live API."""
    print("🌟 GOLDEN SCALE CALIBRATION - LIVE API DEMONSTRATION")
    print("🎯 Testing Enhanced Hallucination Detection via HTTP API")
    print("=" * 80)
    
    base_url = "http://localhost:8080/api/v1"
    
    # Test different uncertainty scenarios
    test_cases = [
        {
            "name": "High Hallucination Risk (Low Uncertainty)",
            "description": "Fabricated content with low semantic uncertainty",
            "topk_probs": [0.9, 0.05, 0.03, 0.02],
            "expected_risk": "HIGH"
        },
        {
            "name": "Moderate Hallucination Risk",
            "description": "Questionable content with moderate uncertainty", 
            "topk_probs": [0.6, 0.2, 0.1, 0.1],
            "expected_risk": "MEDIUM"
        },
        {
            "name": "Low Hallucination Risk (High Uncertainty)",
            "description": "Truthful content with higher uncertainty",
            "topk_probs": [0.4, 0.3, 0.2, 0.1], 
            "expected_risk": "LOW"
        },
        {
            "name": "Very Low Risk (Balanced Distribution)",
            "description": "Well-calibrated truthful response",
            "topk_probs": [0.35, 0.25, 0.25, 0.15],
            "expected_risk": "VERY_LOW"
        }
    ]
    
    print(f"{'Test Case':>35} | {'Raw ℏₛ':>10} | {'Golden ℏₛ':>12} | {'P(fail)':>10} | {'Risk Assessment':>15}")
    print("-" * 95)
    
    for i, test_case in enumerate(test_cases):
        # Prepare API request
        payload = {
            "model_id": "mistral-7b",  # This model has golden scale enabled
            "method": "scalar_js_kl",
            "topk_indices": list(range(len(test_case["topk_probs"]))),
            "topk_probs": test_case["topk_probs"],
            "rest_mass": 1.0 - sum(test_case["topk_probs"]),
            "prompt_next_topk_indices": [10, 11, 12, 13],
            "prompt_next_topk_probs": [0.5, 0.3, 0.15, 0.05], 
            "prompt_next_rest_mass": 0.0
        }
        
        try:
            # Make API call
            response = requests.post(f"{base_url}/analyze_topk_compact", 
                                   json=payload, 
                                   timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract results
            raw_hbar = (result["delta_mu"] * result["delta_sigma"]) ** 0.5
            calibrated_hbar = result["hbar_s"]  # This includes golden scale calibration
            p_fail = result["p_fail"]
            
            # Determine risk level
            if p_fail > 0.8:
                risk_level = "CRITICAL 🚨"
            elif p_fail > 0.6:
                risk_level = "HIGH ⚠️"
            elif p_fail > 0.4:
                risk_level = "MEDIUM 👀"
            else:
                risk_level = "LOW ✅"
            
            print(f"{test_case['name']:>35} | {raw_hbar:>10.4f} | {calibrated_hbar:>12.4f} | {p_fail:>10.4f} | {risk_level:>15}")
            
        except requests.exceptions.RequestException as e:
            print(f"{test_case['name']:>35} | {'ERROR':>10} | {'ERROR':>12} | {'ERROR':>10} | {'API ERROR':>15}")
            print(f"  Error: {e}")
    
    print("\n" + "=" * 80)
    print("GOLDEN SCALE EFFECT ANALYSIS")
    print("=" * 80)
    
    # Test with different models to show the calibration difference
    models_to_test = [
        ("mistral-7b", "Golden Scale Enabled"),
        ("mixtral-8x7b", "Default Calibration")
    ]
    
    print(f"{'Model':>15} | {'Calibration':>20} | {'Raw ℏₛ':>10} | {'Calibrated ℏₛ':>15} | {'P(fail)':>10}")
    print("-" * 85)
    
    # Use the same test case for comparison
    test_payload = {
        "method": "scalar_js_kl",
        "topk_indices": [0, 1, 2, 3],
        "topk_probs": [0.7, 0.2, 0.08, 0.02],
        "rest_mass": 0.0,
        "prompt_next_topk_indices": [4, 5, 6, 7],
        "prompt_next_topk_probs": [0.6, 0.25, 0.1, 0.05],
        "prompt_next_rest_mass": 0.0
    }
    
    for model_id, calibration_type in models_to_test:
        test_payload["model_id"] = model_id
        
        try:
            response = requests.post(f"{base_url}/analyze_topk_compact", 
                                   json=test_payload, 
                                   timeout=10)
            response.raise_for_status()
            result = response.json()
            
            raw_hbar = (result["delta_mu"] * result["delta_sigma"]) ** 0.5
            calibrated_hbar = result["hbar_s"]
            p_fail = result["p_fail"]
            
            print(f"{model_id:>15} | {calibration_type:>20} | {raw_hbar:>10.4f} | {calibrated_hbar:>15.4f} | {p_fail:>10.4f}")
            
        except Exception as e:
            print(f"{model_id:>15} | {calibration_type:>20} | {'ERROR':>10} | {'ERROR':>15} | {'ERROR':>10}")
    
    print("\n" + "=" * 80)
    print("CONFIGURATION DETAILS")
    print("=" * 80)
    
    try:
        # Check models configuration
        with open("config/models.json", "r") as f:
            models_config = json.load(f)
        
        for model in models_config.get("models", []):
            if model.get("id") == "mistral-7b":
                print(f"✅ Model: {model['id']}")
                print(f"   Display Name: {model['display_name']}")
                print(f"   Calibration Mode: {model.get('calibration_mode', 'DEFAULT')}")
                if 'calibration_mode' in model:
                    cal_mode = model['calibration_mode']
                    print(f"   Golden Scale: {cal_mode.get('scaling', 'N/A')}")
                    print(f"   Thresholds: abort={cal_mode.get('abort_threshold')}, warn={cal_mode.get('warn_threshold')}")
                
                break
        
        # Check failure law configuration
        with open("config/failure_law.json", "r") as f:
            failure_law = json.load(f)
            
        print(f"\n📊 Default Failure Law Configuration:")
        print(f"   Lambda: {failure_law.get('lambda')}")
        print(f"   Tau: {failure_law.get('tau')}")
        print(f"   Golden Scale: {failure_law.get('golden_scale')}")
        print(f"   Golden Scale Enabled: {failure_law.get('golden_scale_enabled')}")
        
    except Exception as e:
        print(f"⚠️ Could not read configuration files: {e}")
    
    print(f"\n🎉 Golden Scale Calibration Demonstration Complete!")
    print(f"✨ Enhanced semantic uncertainty analysis for hallucination detection")
    print(f"🔬 Raw ℏₛ values multiplied by 3.4 for improved discrimination")
    print(f"🛡️ Production-ready semantic uncertainty firewall")

if __name__ == "__main__":
    test_golden_scale_api()