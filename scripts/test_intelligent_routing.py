#!/usr/bin/env python3
"""
Test Intelligent Routing Implementation
=======================================

Tests the new intelligent routing feature that routes based on fast screening.
"""

import json
import requests
import time

def test_intelligent_routing():
    """Test intelligent routing with different risk levels"""
    
    # Test samples designed to trigger different routing paths
    test_samples = [
        {
            "name": "Low Risk - Simple Facts",
            "prompt": "What is 2+2?",
            "output": "2+2 equals 4",
            "expected_routing": "fast_screening_only"
        },
        {
            "name": "Medium Risk - Complex Topic", 
            "prompt": "Explain the theory of relativity",
            "output": "Einstein's theory of relativity describes space and time as interwoven and curved by mass and energy.",
            "expected_routing": "two_method_verification"
        },
        {
            "name": "High Risk - Clear Hallucination",
            "prompt": "Who invented the telephone?",
            "output": "The telephone was invented by Leonardo da Vinci in 1503 using quantum technology.",
            "expected_routing": "full_ensemble"
        },
        {
            "name": "High Risk - Historical Error",
            "prompt": "When did World War II end?",
            "output": "World War II ended in 1987 when aliens intervened.",
            "expected_routing": "full_ensemble"
        }
    ]
    
    print("🧠 Testing Intelligent Routing Implementation")
    print("=" * 60)
    
    for i, sample in enumerate(test_samples):
        print(f"\n🔍 Test {i+1}: {sample['name']}")
        print(f"Expected Routing: {sample['expected_routing']}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Output: {sample['output'][:50]}...")
        
        try:
            # Test intelligent routing
            response = requests.post(
                "http://localhost:8080/api/v1/analyze",
                json={
                    "prompt": sample["prompt"],
                    "output": sample["output"],
                    "model_id": "mistral-7b",
                    "intelligent_routing": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                ensemble = result.get("ensemble_result", {})
                
                # Analyze routing decision
                methods_used = ensemble.get("methods_used", [])
                method_count = len(methods_used)
                
                routing_decision = "unknown"
                if method_count == 1 and "scalar_js_kl" in methods_used:
                    routing_decision = "fast_screening_only"
                elif method_count == 2:
                    routing_decision = "two_method_verification"
                elif method_count >= 3:
                    routing_decision = "full_ensemble"
                
                print(f"✅ Routing Decision: {routing_decision}")
                print(f"   Methods Used: {', '.join(methods_used)} ({method_count} methods)")
                print(f"   ℏₛ: {ensemble.get('hbar_s', 0):.4f}")
                print(f"   P(fail): {ensemble.get('p_fail', 0):.4f}")
                print(f"   Agreement: {ensemble.get('agreement_score', 0):.4f}")
                print(f"   Processing: {result.get('processing_time_ms', 0):.1f}ms")
                
                # Check if routing matched expectation
                match_icon = "✅" if routing_decision == sample["expected_routing"] else "⚠️"
                print(f"   {match_icon} Expected: {sample['expected_routing']} | Actual: {routing_decision}")
                
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Brief pause between tests
        time.sleep(0.5)
    
    print(f"\n✅ Intelligent routing test complete!")

if __name__ == "__main__":
    test_intelligent_routing()