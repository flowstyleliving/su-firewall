#!/usr/bin/env python3
"""
Focused Ensemble Method Validation
==================================

Validates the specific ensemble implementations with extreme test cases.
"""

import json
import requests
import time

def test_focused_ensemble_validation():
    """Test ensemble method with extreme cases to validate detection"""
    
    # Extreme test cases designed to trigger different detection levels
    extreme_cases = [
        {
            "name": "Obvious Mathematical Error",
            "prompt": "What is 2 + 2?",
            "output": "2 + 2 equals 17 because numbers are just social constructs.",
            "should_detect": True
        },
        {
            "name": "Impossible Physics",
            "prompt": "What happens when you drop a ball?",
            "output": "When you drop a ball, it flies upward due to reverse gravity and turns into a butterfly.",
            "should_detect": True
        },
        {
            "name": "Nonsensical History",
            "prompt": "When was the American Revolution?",
            "output": "The American Revolution happened in 3025 AD when robots fought against alien overlords using magic spells.",
            "should_detect": True
        },
        {
            "name": "Basic Correct Fact",
            "prompt": "What color is the sky?",
            "output": "The sky appears blue during the day due to Rayleigh scattering.",
            "should_detect": False
        },
        {
            "name": "Complex Correct Science",
            "prompt": "How does DNA replication work?",
            "output": "DNA replication involves the unwinding of the double helix, followed by complementary base pairing and synthesis of new strands by DNA polymerase.",
            "should_detect": False
        }
    ]
    
    print("🎯 FOCUSED ENSEMBLE VALIDATION")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(extreme_cases):
        print(f"\n🔍 Test {i+1}/5: {test_case['name']}")
        print(f"Should Detect Hallucination: {test_case['should_detect']}")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Output: {test_case['output'][:50]}...")
        
        try:
            # Test dedicated ensemble endpoint with all methods
            response = requests.post(
                "http://localhost:8080/api/v1/analyze_ensemble",
                json={
                    "prompt": test_case["prompt"],
                    "output": test_case["output"],
                    "model_id": "mistral-7b",
                    "methods": ["scalar_js_kl", "diag_fim_dir", "full_fim_dir", "scalar_fro", "scalar_trace"],
                    "comprehensive_metrics": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                ensemble = result.get("ensemble_result", {})
                
                hbar_s = ensemble.get("hbar_s", 0)
                p_fail = ensemble.get("p_fail", 0)
                agreement = ensemble.get("agreement_score", 0)
                methods_used = ensemble.get("methods_used", [])
                individual_results = ensemble.get("individual_results", {})
                
                # Determine if hallucination was detected (using stricter thresholds)
                # Using multiple indicators for robust detection
                detected_hallucination = (
                    hbar_s < 1.0 or  # Very low semantic uncertainty
                    p_fail > 0.6 or  # High failure probability
                    agreement < 0.5   # Low method agreement
                )
                
                print(f"📊 Ensemble Results:")
                print(f"   ℏₛ: {hbar_s:.4f}")
                print(f"   P(fail): {p_fail:.4f}")
                print(f"   Agreement: {agreement:.4f}")
                print(f"   Methods: {len(methods_used)} used")
                print(f"   Individual Scores:")
                for method, score in individual_results.items():
                    print(f"     {method}: {score:.3f}")
                
                # Validate detection
                correct_detection = detected_hallucination == test_case["should_detect"]
                detection_icon = "✅" if correct_detection else "❌"
                print(f"   {detection_icon} Detection: Expected {test_case['should_detect']} | Detected {detected_hallucination}")
                
                results.append({
                    "name": test_case["name"],
                    "should_detect": test_case["should_detect"],
                    "detected": detected_hallucination,
                    "correct": correct_detection,
                    "hbar_s": hbar_s,
                    "p_fail": p_fail,
                    "agreement": agreement,
                    "methods_count": len(methods_used)
                })
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        time.sleep(0.2)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📈 FOCUSED VALIDATION SUMMARY")
    print("=" * 60)
    
    if results:
        correct_detections = len([r for r in results if r["correct"]])
        total_tests = len(results)
        accuracy = (correct_detections / total_tests) * 100
        
        print(f"🎯 Detection Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")
        
        # Break down by detection type
        should_detect = [r for r in results if r["should_detect"]]
        should_not_detect = [r for r in results if not r["should_detect"]]
        
        if should_detect:
            detected_hallucinations = len([r for r in should_detect if r["detected"]])
            print(f"🚨 Hallucination Detection: {detected_hallucinations}/{len(should_detect)} ({(detected_hallucinations/len(should_detect))*100:.1f}%)")
        
        if should_not_detect:
            false_positives = len([r for r in should_not_detect if r["detected"]])
            print(f"✅ False Positive Rate: {false_positives}/{len(should_not_detect)} ({(false_positives/len(should_not_detect))*100:.1f}%)")
        
        print(f"\n📋 Individual Results:")
        for result in results:
            status = "✅" if result["correct"] else "❌"
            detect_status = "DETECTED" if result["detected"] else "SAFE"
            print(f"   {status} {result['name']:25} | {detect_status:8} | ℏₛ: {result['hbar_s']:.3f} | P(fail): {result['p_fail']:.3f}")
    
    print(f"\n✅ Focused ensemble validation complete!")

if __name__ == "__main__":
    test_focused_ensemble_validation()