#!/usr/bin/env python3
"""
Test Ensemble API Implementation
=================================

Tests the new weighted ensemble method endpoint.
"""

import json
import requests
import time

def test_ensemble_endpoint():
    """Test the ensemble endpoint"""
    
    # Test data
    test_samples = [
        {
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "expected": "correct"
        },
        {
            "prompt": "What is the capital of France?", 
            "output": "The capital of France is London.",
            "expected": "hallucination"
        },
        {
            "prompt": "Explain quantum physics",
            "output": "Quantum physics involves particles that can exist in multiple states simultaneously through superposition.",
            "expected": "correct"
        }
    ]
    
    print("🚀 Testing Ensemble API Implementation")
    print("=" * 60)
    
    # Test ensemble endpoint
    for i, sample in enumerate(test_samples):
        print(f"\n🔍 Test {i+1}: {sample['expected']}")
        print(f"Prompt: {sample['prompt'][:50]}...")
        
        try:
            # Test ensemble endpoint
            response = requests.post(
                "http://localhost:8080/api/v1/analyze_ensemble",
                json={
                    "prompt": sample["prompt"],
                    "output": sample["output"],
                    "model_id": "mistral-7b"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                ensemble = result.get("ensemble_result", {})
                
                print(f"✅ Ensemble ℏₛ: {ensemble.get('hbar_s', 0):.4f}")
                print(f"   P(fail): {ensemble.get('p_fail', 0):.4f}")
                print(f"   Agreement: {ensemble.get('agreement_score', 0):.4f}")
                print(f"   Methods: {', '.join(ensemble.get('methods_used', []))}")
                print(f"   Processing: {result.get('processing_time_ms', 0):.1f}ms")
                
                # Show individual method results
                individual = ensemble.get("individual_results", {})
                print(f"   Individual: {', '.join([f'{k}:{v:.3f}' for k, v in individual.items()])}")
                
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Test regular endpoint with ensemble flag
    print(f"\n🔍 Testing regular endpoint with ensemble=true")
    try:
        response = requests.post(
            "http://localhost:8080/api/v1/analyze",
            json={
                "prompt": "What is 2+2?",
                "output": "2+2 equals 5",
                "model_id": "mistral-7b",
                "ensemble": True
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Regular endpoint with ensemble=true works")
            print(f"   Response type: {type(result)}")
            if "ensemble_result" in result:
                ensemble = result["ensemble_result"]
                print(f"   Ensemble ℏₛ: {ensemble.get('hbar_s', 0):.4f}")
                print(f"   Agreement: {ensemble.get('agreement_score', 0):.4f}")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print(f"\n✅ Ensemble API test complete!")

if __name__ == "__main__":
    test_ensemble_endpoint()