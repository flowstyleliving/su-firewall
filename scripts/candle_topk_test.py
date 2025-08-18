#!/usr/bin/env python3
"""
🔥 Candle ML TopK Integration Test
Test Candle with realistic probability distributions
"""

import requests
import json

def test_candle_topk():
    """Test Candle with topk compact endpoint"""
    print("🔥 Testing Candle ML TopK Integration")
    print("=" * 50)
    
    # Simulate realistic probability distribution from a language model
    test_data = {
        "topk_indices": [262, 783, 257, 318, 465],  # Token IDs
        "topk_probs": [0.35, 0.25, 0.15, 0.10, 0.05],  # Probabilities
        "rest_mass": 0.10,  # Remaining probability mass
        "vocab_size": 50257,
        "method": "diag_fim_dir"
    }
    
    print("📊 Test data:")
    print(f"   Top-K tokens: {test_data['topk_indices']}")
    print(f"   Probabilities: {test_data['topk_probs']}")
    print(f"   Rest mass: {test_data['rest_mass']}")
    print()
    
    methods = ["diag_fim_dir", "scalar_js_kl", "full_fim_dir", "scalar_trace", "scalar_fro"]
    
    print("🧪 Testing all methods with TopK endpoint...")
    print("-" * 60)
    
    results = []
    
    for method in methods:
        test_data_method = test_data.copy()
        test_data_method["method"] = method
        
        try:
            response = requests.post(
                "http://localhost:8080/api/v1/analyze_topk_compact",
                json=test_data_method,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                hbar_s = result.get('hbar_s', 0.0)
                delta_mu = result.get('delta_mu', 0.0)
                delta_sigma = result.get('delta_sigma', 0.0)
                p_fail = result.get('p_fail', 0.0)
                free_energy = result.get('free_energy', 0.0)
                processing_time = result.get('processing_time_ms', 0.0)
                
                results.append({
                    'method': method,
                    'hbar_s': hbar_s,
                    'delta_mu': delta_mu,
                    'delta_sigma': delta_sigma,
                    'p_fail': p_fail,
                    'free_energy': free_energy,
                    'processing_time': processing_time
                })
                
                print(f"   {method:15} | ℏₛ: {hbar_s:6.3f} | δμ: {delta_mu:6.3f} | δσ: {delta_sigma:6.3f} | P(fail): {p_fail:6.3f} | FE: {free_energy:6.3f}")
            else:
                print(f"   {method:15} | ❌ Failed: {response.status_code}")
                print(f"      Response: {response.text[:100]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"   {method:15} | ❌ Error: {e}")
    
    print()
    
    # Detailed analysis
    if len(results) >= 3:
        print("📈 Detailed Analysis:")
        print("-" * 40)
        
        hbar_values = [r['hbar_s'] for r in results]
        delta_mu_values = [r['delta_mu'] for r in results]
        delta_sigma_values = [r['delta_sigma'] for r in results]
        pfail_values = [r['p_fail'] for r in results]
        
        print(f"   ℏₛ range: {min(hbar_values):.3f} - {max(hbar_values):.3f}")
        print(f"   δμ range: {min(delta_mu_values):.3f} - {max(delta_mu_values):.3f}")
        print(f"   δσ range: {min(delta_sigma_values):.3f} - {max(delta_sigma_values):.3f}")
        print(f"   P(fail) range: {min(pfail_values):.3f} - {max(pfail_values):.3f}")
        
        # Check for method differentiation
        hbar_range = max(hbar_values) - min(hbar_values)
        if hbar_range > 0.5:
            print("   ✅ Excellent method differentiation")
        elif hbar_range > 0.1:
            print("   ✅ Good method differentiation")
        else:
            print("   ⚠️  Limited method differentiation")
        
        # Find best performing methods
        sorted_by_hbar = sorted(results, key=lambda x: x['hbar_s'])
        print(f"   🏆 Lowest uncertainty: {sorted_by_hbar[0]['method']} (ℏₛ={sorted_by_hbar[0]['hbar_s']:.3f})")
        print(f"   📈 Highest uncertainty: {sorted_by_hbar[-1]['method']} (ℏₛ={sorted_by_hbar[-1]['hbar_s']:.3f})")
        
        # Performance analysis
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        print(f"   ⚡ Average processing time: {avg_time:.2f}ms")
        
        fastest = min(results, key=lambda x: x['processing_time'])
        print(f"   🚀 Fastest method: {fastest['method']} ({fastest['processing_time']:.2f}ms)")
    
    print()
    print("🔥 Candle ML TopK Integration Test Complete!")

if __name__ == "__main__":
    test_candle_topk()