#!/usr/bin/env python3
"""
Test direct Candle model loading vs Ollama performance
"""
import json
import requests
import time

def test_direct_candle():
    """Test direct Candle model loading"""
    
    print("🔥 TESTING DIRECT CANDLE MODEL LOADING")
    print("=" * 60)
    
    # Test data
    test_requests = [
        {
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "model_id": "mistral-7b"  # Should use Candle if safetensors exists
        },
        {
            "prompt": "What is the capital of France?", 
            "output": "The capital of France is Paris.",
            "model_id": "ollama-mistral-7b"  # Should use Ollama
        }
    ]
    
    results = {}
    
    for test_req in test_requests:
        model_id = test_req["model_id"]
        print(f"\n🧮 Testing {model_id}...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                'http://localhost:8080/api/v1/analyze_ensemble',
                json=test_req,
                timeout=10
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                ensemble = data['ensemble_result']
                
                results[model_id] = {
                    'status': 'success',
                    'hbar_s': ensemble['hbar_s'],
                    'p_fail': ensemble['p_fail'],
                    'processing_time_s': processing_time,
                    'server_processing_ms': data.get('processing_time_ms', 0),
                    'methods_used': ensemble['methods_used']
                }
                
                print(f"  ✅ Success: ℏₛ={ensemble['hbar_s']:.4f}, Time={processing_time:.2f}s")
                
            else:
                results[model_id] = {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text[:100]}",
                    'processing_time_s': processing_time
                }
                print(f"  ❌ Error: {response.status_code} - {response.text[:50]}")
                
        except Exception as e:
            results[model_id] = {
                'status': 'exception',
                'error': str(e),
                'processing_time_s': time.time() - start_time
            }
            print(f"  💥 Exception: {e}")
    
    # Performance comparison
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    for model_id, result in results.items():
        if result['status'] == 'success':
            deployment_type = "🔥 Candle" if model_id == "mistral-7b" else "🦙 Ollama"
            print(f"{deployment_type} ({model_id}): {result['processing_time_s']:.2f}s")
            print(f"   ℏₛ: {result['hbar_s']:.4f}")
            print(f"   Methods: {len(result['methods_used'])}")
    
    # Export results
    with open('candle_vs_ollama_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: candle_vs_ollama_test.json")
    
    return results

if __name__ == "__main__":
    # Make sure server is running
    try:
        health_check = requests.get('http://localhost:8080/health', timeout=5)
        if health_check.status_code == 200:
            print("✅ Server is running")
            test_direct_candle()
        else:
            print("❌ Server health check failed")
    except:
        print("❌ Server not running. Start with: cargo run -p server")