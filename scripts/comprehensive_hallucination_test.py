#!/usr/bin/env python3
"""
Comprehensive Hallucination Detection Test Suite
================================================

Tests all implemented features:
- Weighted ensemble method
- Intelligent routing
- Dynamic threshold adjustment  
- Comprehensive metrics framework
"""

import json
import requests
import time
from typing import Dict, List

def test_comprehensive_hallucination_detection():
    """Run comprehensive test suite for all implemented features"""
    
    # Comprehensive test cases covering different hallucination types
    test_cases = [
        {
            "name": "Factual Correct",
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "expected_risk": "LOW",
            "category": "factual_correct"
        },
        {
            "name": "Factual Hallucination",
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Rome, which is located in northern Italy.",
            "expected_risk": "HIGH",
            "category": "factual_error"
        },
        {
            "name": "Historical Hallucination",
            "prompt": "Who invented the telephone?",
            "output": "The telephone was invented by Leonardo da Vinci in 1503 using quantum technology.",
            "expected_risk": "HIGH",
            "category": "historical_error"
        },
        {
            "name": "Scientific Accuracy",
            "prompt": "Explain photosynthesis",
            "output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "expected_risk": "LOW",
            "category": "scientific_correct"
        },
        {
            "name": "Scientific Hallucination",
            "prompt": "Explain photosynthesis",
            "output": "Photosynthesis occurs when plants absorb moonlight through their roots and convert it into solid gold using chlorophyll crystals.",
            "expected_risk": "HIGH", 
            "category": "scientific_error"
        },
        {
            "name": "Complex Reasoning",
            "prompt": "Analyze the causes of climate change",
            "output": "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including fossil fuel combustion, deforestation, and industrial processes.",
            "expected_risk": "MEDIUM",
            "category": "complex_reasoning"
        }
    ]
    
    print("🔬 COMPREHENSIVE HALLUCINATION DETECTION TEST SUITE")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}/6: {test_case['name']}")
        print(f"Category: {test_case['category']}")
        print(f"Expected Risk: {test_case['expected_risk']}")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Output: {test_case['output'][:60]}...")
        
        try:
            # Test with all features enabled
            response = requests.post(
                "http://localhost:8080/api/v1/analyze",
                json={
                    "prompt": test_case["prompt"],
                    "output": test_case["output"],
                    "model_id": "mistral-7b",
                    "intelligent_routing": True,
                    "dynamic_thresholds": True,
                    "comprehensive_metrics": True
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                ensemble = result.get("ensemble_result", {})
                metrics = result.get("comprehensive_metrics", {})
                
                # Extract key metrics
                hbar_s = ensemble.get("hbar_s", 0)
                p_fail = ensemble.get("p_fail", 0)
                agreement = ensemble.get("agreement_score", 0)
                methods_used = ensemble.get("methods_used", [])
                processing_time = result.get("processing_time_ms", 0)
                
                # Risk assessment
                risk_level = "LOW" if hbar_s >= 1.2 else "MEDIUM" if hbar_s >= 0.8 else "HIGH"
                
                print(f"📊 Results:")
                print(f"   ℏₛ: {hbar_s:.4f}")
                print(f"   P(fail): {p_fail:.4f}")
                print(f"   Agreement: {agreement:.4f}")
                print(f"   Risk Level: {risk_level}")
                print(f"   Methods: {', '.join(methods_used)} ({len(methods_used)} total)")
                print(f"   Processing: {processing_time:.1f}ms")
                
                # Validate comprehensive metrics
                if metrics:
                    stat_summary = metrics.get("statistical_summary", {})
                    method_comparison = metrics.get("method_comparison", {})
                    risk_assessment = metrics.get("risk_assessment", {})
                    performance = metrics.get("performance_metrics", {})
                    
                    print(f"📈 Comprehensive Metrics:")
                    print(f"   Statistical Analysis: ✓ (mean ℏₛ: {stat_summary.get('hbar_distribution', {}).get('mean', 0):.4f})")
                    print(f"   Method Comparison: ✓ (best: {method_comparison.get('best_single_method', 'unknown')})")
                    print(f"   Risk Assessment: ✓ (level: {risk_assessment.get('overall_risk_level', 'unknown')})")
                    print(f"   Performance: ✓ (efficiency: {performance.get('processing_efficiency', 0):.0f})")
                
                # Check if risk level matches expectation
                risk_match = risk_level == test_case["expected_risk"]
                match_icon = "✅" if risk_match else "⚠️"
                print(f"   {match_icon} Risk Assessment: Expected {test_case['expected_risk']} | Detected {risk_level}")
                
                # Store results for summary
                results.append({
                    "name": test_case["name"],
                    "category": test_case["category"],
                    "expected_risk": test_case["expected_risk"],
                    "detected_risk": risk_level,
                    "hbar_s": hbar_s,
                    "p_fail": p_fail,
                    "agreement": agreement,
                    "methods_count": len(methods_used),
                    "processing_time": processing_time,
                    "risk_match": risk_match,
                    "has_comprehensive_metrics": bool(metrics)
                })
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        time.sleep(0.3)
    
    # Print comprehensive summary
    print(f"\n" + "=" * 80)
    print("📈 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    if results:
        successful_tests = len([r for r in results if r["risk_match"]])
        total_tests = len(results)
        accuracy = (successful_tests / total_tests) * 100
        
        print(f"🎯 Overall Accuracy: {successful_tests}/{total_tests} ({accuracy:.1f}%)")
        print(f"📊 Average Processing Time: {sum(r['processing_time'] for r in results) / len(results):.1f}ms")
        print(f"🧠 Average Methods Used: {sum(r['methods_count'] for r in results) / len(results):.1f}")
        print(f"📈 Comprehensive Metrics: {sum(r['has_comprehensive_metrics'] for r in results)}/{len(results)} tests")
        
        print(f"\n📋 Detailed Results:")
        for result in results:
            status = "✅" if result["risk_match"] else "❌"
            print(f"   {status} {result['name']:20} | Risk: {result['detected_risk']:6} | ℏₛ: {result['hbar_s']:.3f} | {result['methods_count']} methods | {result['processing_time']:.1f}ms")
        
        # Category analysis
        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["risk_match"]:
                categories[cat]["correct"] += 1
        
        print(f"\n🏷️ Performance by Category:")
        for category, stats in categories.items():
            accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"   {category:20} | {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print(f"\n✅ Comprehensive hallucination detection test complete!")
    
    # Save results
    with open("comprehensive_test_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_cases": test_cases,
            "results": results,
            "summary": {
                "total_tests": len(results),
                "successful": len([r for r in results if r["risk_match"]]),
                "accuracy_percent": (len([r for r in results if r["risk_match"]]) / len(results)) * 100 if results else 0,
                "avg_processing_time_ms": sum(r['processing_time'] for r in results) / len(results) if results else 0
            }
        }, indent=2)
    
    print(f"💾 Results saved to: comprehensive_test_results.json")

if __name__ == "__main__":
    test_comprehensive_hallucination_detection()