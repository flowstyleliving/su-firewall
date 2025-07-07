#!/usr/bin/env python3
"""
🚀 Semantic Uncertainty API Test Script for John Yue
Quick test to verify the API is working correctly.
"""

import requests
import json
import time

# API Configuration
API_ENDPOINT = "https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev"
API_KEY = "your-production-api-key"

def test_health_check():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_ENDPOINT}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Engine: {data.get('engine')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_analysis():
    """Test single prompt analysis"""
    print("\n🔍 Testing single prompt analysis...")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    data = {
        "prompt": "Explain artificial intelligence in simple terms",
        "model": "gpt4"
    }
    
    try:
        response = requests.post(f"{API_ENDPOINT}/api/v1/analyze", 
                               headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single analysis passed!")
            
            analysis = result['data']
            print(f"   Prompt: {analysis['prompt'][:50]}...")
            print(f"   Model: {analysis['model']}")
            print(f"   Semantic Uncertainty (ℏₛ): {analysis['semantic_uncertainty']}")
            print(f"   Precision (Δμ): {analysis['precision']}")
            print(f"   Flexibility (Δσ): {analysis['flexibility']}")
            print(f"   Risk Level: {analysis['risk_level']}")
            print(f"   Processing Time: {analysis['processing_time']}ms")
            return True
        else:
            print(f"❌ Single analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Single analysis error: {e}")
        return False

def test_batch_analysis():
    """Test batch prompt analysis"""
    print("\n📦 Testing batch analysis...")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    data = {
        "prompts": [
            "Write a poem about technology",
            "Explain quantum computing",
            "Create a business plan outline"
        ],
        "model": "claude3"
    }
    
    try:
        response = requests.post(f"{API_ENDPOINT}/api/v1/batch", 
                               headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch analysis passed!")
            
            batch_data = result['data']
            print(f"   Total Prompts: {batch_data['total_prompts']}")
            print(f"   Total Time: {batch_data['total_time']}ms")
            print(f"   Average ℏₛ: {batch_data['average_h_bar']:.4f}")
            
            print("\n   📊 Individual Results:")
            for i, analysis in enumerate(batch_data['results'], 1):
                print(f"     {i}. {analysis['prompt'][:30]}...")
                print(f"        ℏₛ: {analysis['h_bar']:.4f} | Risk: {analysis['risk_level']}")
            
            return True
        else:
            print(f"❌ Batch analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch analysis error: {e}")
        return False

def test_error_handling():
    """Test API error handling"""
    print("\n🚨 Testing error handling...")
    
    # Test without API key
    try:
        response = requests.post(f"{API_ENDPOINT}/api/v1/analyze",
                               headers={"Content-Type": "application/json"},
                               json={"prompt": "Test", "model": "gpt4"})
        
        if response.status_code == 401:
            print("✅ Unauthorized error handling works!")
            return True
        else:
            print(f"❌ Expected 401, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def analyze_risk_levels():
    """Demonstrate different risk levels"""
    print("\n🎯 Demonstrating risk level analysis...")
    
    test_prompts = [
        "Hello world",  # Simple, stable
        "Write a comprehensive analysis of quantum mechanics with mathematical proofs",  # Complex
        "Generate infinite creative writing ideas",  # High flexibility
    ]
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    for prompt in test_prompts:
        try:
            response = requests.post(f"{API_ENDPOINT}/api/v1/analyze", 
                                   headers=headers, 
                                   json={"prompt": prompt, "model": "gpt4"})
            
            if response.status_code == 200:
                result = response.json()['data']
                h_bar = result['semantic_uncertainty']
                risk = result['risk_level']
                
                # Risk level emojis
                risk_emoji = {
                    'stable': '✅',
                    'moderate_instability': '⚠️',
                    'high_collapse_risk': '🔥'
                }.get(risk, '❓')
                
                print(f"   {risk_emoji} \"{prompt[:40]}...\"")
                print(f"      ℏₛ: {h_bar:.4f} | Risk: {risk}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing: {prompt[:30]}... - {e}")

def main():
    """Run all tests"""
    print("🚀 SEMANTIC UNCERTAINTY API TEST SUITE")
    print("=" * 50)
    print(f"Testing endpoint: {API_ENDPOINT}")
    print(f"Using API key: {API_KEY[:20]}...")
    print()
    
    tests = [
        test_health_check,
        test_single_analysis,
        test_batch_analysis,
        test_error_handling,
        analyze_risk_levels
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(1)  # Brief pause between tests
        except KeyboardInterrupt:
            print("\n❌ Tests interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {test.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working perfectly.")
        print("\n💡 Next steps:")
        print("   1. Try your own prompts with the API")
        print("   2. Integrate into your applications")
        print("   3. Monitor semantic uncertainty values")
        print("   4. Use risk levels to filter prompts")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("   - Verify your internet connection")
        print("   - Check if the API key is correct")
        print("   - Wait a few minutes for DNS propagation")

if __name__ == "__main__":
    main() 