#!/usr/bin/env python3
"""
🔥 Candle ML Integration Test
Test the Candle ML integration with semantic uncertainty analysis
"""

import requests
import json
import time

def test_candle_integration():
    """Test Candle integration via API"""
    print("🔥 Testing Candle ML Integration")
    print("=" * 50)
    
    # Test data - using the same data as our successful tests
    test_data = {
        "prompt": "The capital of France is",
        "output": "Paris",  # Expected completion
        "method": "diag_fim_dir",
        "model_id": "candle-gpt2"
    }
    
    # Test the standard analyze endpoint
    try:
        print("📡 Testing /api/v1/analyze endpoint...")
        response = requests.post(
            "http://localhost:8080/api/v1/analyze",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analyze endpoint successful!")
            print(f"   Request ID: {result.get('request_id', 'N/A')}")
            print(f"   ℏₛ (hbar_s): {result.get('hbar_s', 'N/A'):.4f}")
            print(f"   δμ (delta_mu): {result.get('delta_mu', 'N/A'):.4f}")
            print(f"   δσ (delta_sigma): {result.get('delta_sigma', 'N/A'):.4f}")
            print(f"   P(fail): {result.get('p_fail', 'N/A'):.4f}")
            print(f"   Method: {result.get('method', 'N/A')}")
            print(f"   Processing time: {result.get('processing_time_ms', 'N/A'):.2f}ms")
        else:
            print(f"❌ Analyze endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        
    print()
    
    # Test multiple methods to verify differentiation works
    methods = ["diag_fim_dir", "scalar_js_kl", "full_fim_dir", "scalar_trace", "scalar_fro"]
    
    print("🧪 Testing method differentiation...")
    print("-" * 40)
    
    results = []
    
    for method in methods:
        test_data_method = test_data.copy()
        test_data_method["method"] = method
        
        try:
            response = requests.post(
                "http://localhost:8080/api/v1/analyze",
                json=test_data_method,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                hbar_s = result.get('hbar_s', 0.0)
                p_fail = result.get('p_fail', 0.0)
                processing_time = result.get('processing_time_ms', 0.0)
                
                results.append({
                    'method': method,
                    'hbar_s': hbar_s,
                    'p_fail': p_fail,
                    'processing_time': processing_time
                })
                
                print(f"   {method:15} | ℏₛ: {hbar_s:6.3f} | P(fail): {p_fail:6.3f} | {processing_time:5.1f}ms")
            else:
                print(f"   {method:15} | ❌ Failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   {method:15} | ❌ Error: {e}")
    
    print()
    
    # Analyze differentiation
    if len(results) >= 2:
        hbar_values = [r['hbar_s'] for r in results]
        pfail_values = [r['p_fail'] for r in results]
        
        hbar_range = max(hbar_values) - min(hbar_values)
        pfail_range = max(pfail_values) - min(pfail_values)
        
        print("📊 Method Differentiation Analysis:")
        print(f"   ℏₛ value range: {hbar_range:.4f}")
        print(f"   P(fail) range: {pfail_range:.4f}")
        
        if hbar_range > 0.1:
            print("   ✅ Methods show good differentiation in ℏₛ values")
        else:
            print("   ⚠️  Methods show limited differentiation in ℏₛ values")
            
        if pfail_range > 0.1:
            print("   ✅ Methods show good differentiation in P(fail) values")
        else:
            print("   ⚠️  Methods show limited differentiation in P(fail) values")
    else:
        print("❌ Not enough successful results for differentiation analysis")
    
    print()
    print("🔥 Candle ML Integration Test Complete!")


if __name__ == "__main__":
    test_candle_integration()