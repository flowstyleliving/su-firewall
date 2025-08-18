#!/usr/bin/env python3
"""
Test Dynamic Threshold Adjustment
=================================

Tests the dynamic threshold adjustment based on method agreement.
"""

import json
import requests
import time

def test_dynamic_thresholds():
    """Test dynamic threshold adjustment"""
    
    # Test samples with varying complexity to test agreement-based threshold adjustment
    test_samples = [
        {
            "name": "High Agreement Expected",
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "description": "Simple fact - methods should agree"
        },
        {
            "name": "Medium Agreement Expected",
            "prompt": "Explain machine learning",
            "output": "Machine learning is a subset of AI that uses algorithms to learn from data.",
            "description": "Complex topic - moderate agreement"
        },
        {
            "name": "Low Agreement Expected", 
            "prompt": "What is consciousness?",
            "output": "Consciousness is the quantum field interaction between neural microtubules and cosmic background radiation.",
            "description": "Philosophical/complex - methods may disagree"
        },
        {
            "name": "Clear Hallucination",
            "prompt": "When was the internet invented?",
            "output": "The internet was invented by Napoleon Bonaparte in 1815.",
            "description": "Obvious error - should trigger high-risk routing"
        }
    ]
    
    print("🎯 Testing Dynamic Threshold Adjustment")
    print("=" * 60)
    
    for i, sample in enumerate(test_samples):
        print(f"\n🔍 Test {i+1}: {sample['name']}")
        print(f"Description: {sample['description']}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Output: {sample['output'][:70]}...")
        
        try:
            # Test with static thresholds
            static_response = requests.post(
                "http://localhost:8080/api/v1/analyze",
                json={
                    "prompt": sample["prompt"],
                    "output": sample["output"],
                    "model_id": "mistral-7b",
                    "intelligent_routing": True,
                    "dynamic_thresholds": False
                },
                timeout=10
            )
            
            # Test with dynamic thresholds
            dynamic_response = requests.post(
                "http://localhost:8080/api/v1/analyze",
                json={
                    "prompt": sample["prompt"],
                    "output": sample["output"],
                    "model_id": "mistral-7b",
                    "intelligent_routing": True,
                    "dynamic_thresholds": True
                },
                timeout=10
            )
            
            if static_response.status_code == 200 and dynamic_response.status_code == 200:
                static_result = static_response.json()
                dynamic_result = dynamic_response.json()
                
                static_ensemble = static_result.get("ensemble_result", {})
                dynamic_ensemble = dynamic_result.get("ensemble_result", {})
                
                print(f"📊 Static Thresholds:")
                print(f"   Methods: {', '.join(static_ensemble.get('methods_used', []))} ({len(static_ensemble.get('methods_used', []))} methods)")
                print(f"   Agreement: {static_ensemble.get('agreement_score', 0):.4f}")
                print(f"   P(fail): {static_ensemble.get('p_fail', 0):.4f}")
                print(f"   Processing: {static_result.get('processing_time_ms', 0):.1f}ms")
                
                print(f"🎯 Dynamic Thresholds:")
                print(f"   Methods: {', '.join(dynamic_ensemble.get('methods_used', []))} ({len(dynamic_ensemble.get('methods_used', []))} methods)")
                print(f"   Agreement: {dynamic_ensemble.get('agreement_score', 0):.4f}")
                print(f"   P(fail): {dynamic_ensemble.get('p_fail', 0):.4f}")
                print(f"   Processing: {dynamic_result.get('processing_time_ms', 0):.1f}ms")
                
                # Compare routing decisions
                static_methods = len(static_ensemble.get('methods_used', []))
                dynamic_methods = len(dynamic_ensemble.get('methods_used', []))
                
                if static_methods != dynamic_methods:
                    print(f"✨ Dynamic adjustment changed routing: {static_methods} → {dynamic_methods} methods")
                else:
                    print(f"➡️ Same routing decision with both approaches")
                
            else:
                print(f"❌ Error: Static={static_response.status_code}, Dynamic={dynamic_response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Brief pause between tests
        time.sleep(0.5)
    
    print(f"\n✅ Dynamic threshold adjustment test complete!")

if __name__ == "__main__":
    test_dynamic_thresholds()