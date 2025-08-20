#!/usr/bin/env python3
"""
Quick test of the full ensemble system with real model configurations.
Tests all 6 configured models from config/models.json
"""

import json
import requests
import time
from typing import Dict, Any

# Test cases - mix of correct and potentially hallucinated content
test_cases = [
    {
        "name": "Correct_Geography",
        "prompt": "What is the capital of France?",
        "output": "The capital of France is Paris, known as the city of lights.",
        "expected_risk": "low"
    },
    {
        "name": "Hallucinated_Geography", 
        "prompt": "What is the capital of France?",
        "output": "The capital of France is Lyon, a beautiful city with ancient Roman ruins.",
        "expected_risk": "high"
    },
    {
        "name": "Correct_Science",
        "prompt": "What is the speed of light in vacuum?",
        "output": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "expected_risk": "low"
    },
    {
        "name": "Hallucinated_Science",
        "prompt": "What is the speed of light in vacuum?", 
        "output": "The speed of light in vacuum is approximately 186,000 miles per hour, which varies with atmospheric pressure.",
        "expected_risk": "high"
    }
]

# Models from config/models.json
models = [
    {"id": "mistral-7b", "name": "Mistral-7B (Apache-2.0)"},
    {"id": "mixtral-8x7b", "name": "Mixtral-8x7B (Apache-2.0)"},
    {"id": "qwen2.5-7b", "name": "Qwen2.5-7B (Apache-2.0)"},
    {"id": "pythia-6.9b", "name": "Pythia-6.9B (Apache-2.0)"},
    {"id": "dialogpt-medium", "name": "DialoGPT-medium (HF)"},
    {"id": "ollama-mistral-7b", "name": "Ollama Mistral:7B (logits-only)"}
]

def test_ensemble_system():
    """Test the 5-method ensemble system with all configured models"""
    
    results = []
    
    print("🔥 Testing Ensemble Uncertainty System with Real Model Configurations")
    print("=" * 80)
    
    for model in models:
        print(f"\n📊 Testing Model: {model['name']} (ID: {model['id']})")
        print("-" * 60)
        
        model_results = []
        
        for test_case in test_cases:
            print(f"  Testing: {test_case['name']}")
            
            try:
                # Test ensemble endpoint
                start_time = time.time()
                response = requests.post(
                    'http://localhost:8080/api/v1/analyze_ensemble',
                    json={
                        'prompt': test_case['prompt'],
                        'output': test_case['output'], 
                        'model_id': model['id']
                    },
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    ensemble = data['ensemble_result']
                    
                    # Extract key metrics
                    hbar_s = ensemble['hbar_s']
                    p_fail = ensemble['p_fail']
                    agreement = ensemble['agreement_score']
                    processing_time = end_time - start_time
                    
                    # Assess risk based on uncertainty
                    risk_assessment = "low" if hbar_s > 1.2 else ("medium" if hbar_s > 0.8 else "high")
                    
                    result = {
                        'model_id': model['id'],
                        'test_case': test_case['name'],
                        'expected_risk': test_case['expected_risk'],
                        'hbar_s': round(hbar_s, 4),
                        'p_fail': round(p_fail, 4),
                        'agreement_score': round(agreement, 4),
                        'risk_assessment': risk_assessment,
                        'processing_time_ms': round(processing_time * 1000, 1),
                        'methods_used': ensemble['methods_used'],
                        'individual_results': {k: round(v, 4) for k, v in ensemble['individual_results'].items()}
                    }
                    
                    model_results.append(result)
                    results.append(result)
                    
                    # Print results
                    status_emoji = "✅" if risk_assessment == test_case['expected_risk'] else "⚠️"
                    print(f"    {status_emoji} ℏₛ={hbar_s:.3f}, P(fail)={p_fail:.3f}, Agreement={agreement:.3f} ({processing_time*1000:.1f}ms)")
                    
                else:
                    print(f"    ❌ HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # Model summary
        if model_results:
            avg_hbar = sum(r['hbar_s'] for r in model_results) / len(model_results)
            avg_pfail = sum(r['p_fail'] for r in model_results) / len(model_results)
            avg_agreement = sum(r['agreement_score'] for r in model_results) / len(model_results)
            avg_time = sum(r['processing_time_ms'] for r in model_results) / len(model_results)
            
            print(f"  📈 Model Summary: ℏₛ_avg={avg_hbar:.3f}, P(fail)_avg={avg_pfail:.3f}, Agreement={avg_agreement:.3f} ({avg_time:.1f}ms avg)")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("🎯 ENSEMBLE SYSTEM EVALUATION RESULTS")
    print("=" * 80)
    
    if results:
        # Calculate overall metrics
        correct_predictions = sum(1 for r in results if r['risk_assessment'] == r['expected_risk'])
        accuracy = correct_predictions / len(results)
        
        avg_hbar_all = sum(r['hbar_s'] for r in results) / len(results)
        avg_pfail_all = sum(r['p_fail'] for r in results) / len(results)
        avg_agreement_all = sum(r['agreement_score'] for r in results) / len(results)
        avg_processing_time = sum(r['processing_time_ms'] for r in results) / len(results)
        
        print(f"✅ Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{len(results)} correct)")
        print(f"📊 Average ℏₛ: {avg_hbar_all:.4f}")
        print(f"📊 Average P(fail): {avg_pfail_all:.4f}")
        print(f"📊 Average Agreement: {avg_agreement_all:.4f}")
        print(f"⏱️  Average Processing Time: {avg_processing_time:.1f}ms")
        
        # Show method usage
        all_methods = set()
        for r in results:
            all_methods.update(r['methods_used'])
        print(f"🔧 Methods Used: {sorted(all_methods)}")
        
        # Export detailed results
        with open('ensemble_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'accuracy': accuracy,
                    'total_tests': len(results),
                    'correct_predictions': correct_predictions,
                    'avg_hbar_s': avg_hbar_all,
                    'avg_p_fail': avg_pfail_all,
                    'avg_agreement_score': avg_agreement_all,
                    'avg_processing_time_ms': avg_processing_time
                },
                'detailed_results': results
            }, f, indent=2)
        
        print(f"📁 Detailed results saved to: ensemble_test_results.json")
        
        # Success criteria check
        if accuracy >= 0.75 and avg_agreement_all >= 0.5:
            print("\n🎉 SUCCESS: Ensemble system performing within expected parameters!")
            return True
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: Accuracy={accuracy:.1%}, Agreement={avg_agreement_all:.3f}")
            return False
    else:
        print("❌ No results obtained")
        return False

if __name__ == "__main__":
    success = test_ensemble_system()
    exit(0 if success else 1)