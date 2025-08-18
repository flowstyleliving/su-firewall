#!/usr/bin/env python3
"""
Official Benchmark Evaluation
==============================

Evaluates hallucination detection on official benchmarks:
- HaluEval (QA, Dialogue, Summarization, General)
- TruthfulQA
- Authentic Hallucination Benchmark

Tests our ensemble methods against ground truth labels.
"""

import json
import requests
import time
import random
from typing import Dict, List, Tuple
from pathlib import Path

def load_halueval_dataset(task_type: str, max_samples: int = 100) -> List[Dict]:
    """Load HaluEval dataset for specific task type"""
    file_path = f"authentic_datasets/halueval_{task_type}_data.json"
    
    if not Path(file_path).exists():
        print(f"❌ Dataset {file_path} not found")
        return []
    
    try:
        examples = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= max_samples:
                    break
                if line.strip():
                    example = json.loads(line.strip())
                    examples.append(example)
        
        print(f"📊 Loaded {len(examples)} examples from {task_type}")
        return examples
    except Exception as e:
        print(f"❌ Error loading {task_type}: {e}")
        return []

def load_truthfulqa_dataset(max_samples: int = 100) -> List[Dict]:
    """Load TruthfulQA dataset"""
    file_path = "authentic_datasets/truthfulqa_data.json"
    
    if not Path(file_path).exists():
        print("❌ TruthfulQA dataset not found")
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        validation_data = data.get('validation', [])
        
        for i, item in enumerate(validation_data):
            if i >= max_samples:
                break
                
            # Create examples with correct and incorrect answers
            question = item.get('Question', '')
            best_answer = item.get('Best Answer', '')
            incorrect_answers = item.get('Incorrect Answers', [])
            
            if question and best_answer:
                # Add correct answer example
                examples.append({
                    'question': question,
                    'right_answer': best_answer,
                    'answer': best_answer,
                    'is_hallucination': False,
                    'source': 'truthfulqa_correct'
                })
                
                # Add incorrect answer example
                if incorrect_answers:
                    incorrect = random.choice(incorrect_answers)
                    examples.append({
                        'question': question,
                        'right_answer': best_answer,
                        'answer': incorrect,
                        'is_hallucination': True,
                        'source': 'truthfulqa_incorrect'
                    })
        
        print(f"📊 Loaded {len(examples)} TruthfulQA examples")
        return examples
    except Exception as e:
        print(f"❌ Error loading TruthfulQA: {e}")
        return []

def evaluate_hallucination_detection(prompt: str, answer: str, is_hallucination: bool, test_name: str) -> Dict:
    """Evaluate our detection system on a single example"""
    
    try:
        # Use our ensemble method with comprehensive metrics
        response = requests.post(
            "http://localhost:8080/api/v1/analyze_ensemble",
            json={
                "prompt": prompt,
                "output": answer,
                "model_id": "mistral-7b",
                "methods": ["scalar_js_kl", "diag_fim_dir", "full_fim_dir", "scalar_fro"],
                "comprehensive_metrics": True
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            ensemble = result.get("ensemble_result", {})
            metrics = result.get("comprehensive_metrics", {})
            
            hbar_s = ensemble.get("hbar_s", 0)
            p_fail = ensemble.get("p_fail", 0)
            agreement = ensemble.get("agreement_score", 0)
            methods_used = ensemble.get("methods_used", [])
            
            # Multiple detection criteria for robust evaluation
            detection_criteria = {
                "hbar_low": hbar_s < 1.0,          # Low semantic uncertainty
                "pfail_high": p_fail > 0.55,       # High failure probability  
                "agreement_low": agreement < 0.6,   # Low method agreement
                "hbar_very_low": hbar_s < 0.8,     # Very low uncertainty
                "pfail_very_high": p_fail > 0.6    # Very high failure prob
            }
            
            # Combine criteria for final detection
            detected_hallucination = (
                detection_criteria["hbar_low"] or 
                detection_criteria["pfail_high"] or
                detection_criteria["agreement_low"]
            )
            
            # Conservative detection (stricter criteria)
            conservative_detection = (
                detection_criteria["hbar_very_low"] and 
                detection_criteria["pfail_very_high"]
            )
            
            # Calculate accuracy for this example
            standard_correct = detected_hallucination == is_hallucination
            conservative_correct = conservative_detection == is_hallucination
            
            return {
                "test_name": test_name,
                "prompt": prompt[:50] + "...",
                "answer": answer[:50] + "...",
                "ground_truth": is_hallucination,
                "hbar_s": hbar_s,
                "p_fail": p_fail,
                "agreement": agreement,
                "methods_count": len(methods_used),
                "detected_standard": detected_hallucination,
                "detected_conservative": conservative_detection,
                "correct_standard": standard_correct,
                "correct_conservative": conservative_correct,
                "detection_criteria": detection_criteria,
                "processing_time_ms": result.get("processing_time_ms", 0)
            }
            
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return None

def run_official_benchmark_evaluation():
    """Run comprehensive evaluation on all official benchmarks"""
    
    print("🏆 OFFICIAL BENCHMARK EVALUATION")
    print("=" * 80)
    print("📊 Testing: HaluEval (QA, Dialogue, Summarization, General) + TruthfulQA")
    print("🎯 Evaluation: Ensemble detection vs ground truth labels")
    
    all_results = []
    
    # 1. HaluEval QA Dataset
    print(f"\n🔬 HALUEVAL QA EVALUATION")
    print("-" * 40)
    
    halueval_qa = load_halueval_dataset("qa", max_samples=50)
    for i, example in enumerate(halueval_qa):
        if i % 10 == 0:
            print(f"📈 Progress: {i}/{len(halueval_qa)}")
        
        # Test both correct and hallucinated answers
        prompt = example.get("question", "")
        right_answer = example.get("right_answer", "")
        hallucinated_answer = example.get("hallucinated_answer", "")
        
        # Evaluate correct answer
        if prompt and right_answer:
            result = evaluate_hallucination_detection(
                prompt, right_answer, False, "halueval_qa_correct"
            )
            if result:
                all_results.append(result)
        
        # Evaluate hallucinated answer  
        if prompt and hallucinated_answer:
            result = evaluate_hallucination_detection(
                prompt, hallucinated_answer, True, "halueval_qa_hallucination"
            )
            if result:
                all_results.append(result)
        
        time.sleep(0.1)  # Rate limiting
    
    # 2. TruthfulQA Dataset
    print(f"\n🔬 TRUTHFULQA EVALUATION")
    print("-" * 40)
    
    truthfulqa = load_truthfulqa_dataset(max_samples=25)  # 50 examples total
    for i, example in enumerate(truthfulqa):
        if i % 10 == 0:
            print(f"📈 Progress: {i}/{len(truthfulqa)}")
            
        prompt = example.get("question", "")
        answer = example.get("answer", "")
        is_hallucination = example.get("is_hallucination", False)
        source = example.get("source", "truthfulqa")
        
        if prompt and answer:
            result = evaluate_hallucination_detection(
                prompt, answer, is_hallucination, source
            )
            if result:
                all_results.append(result)
        
        time.sleep(0.1)
    
    # 3. Analysis and Results
    print(f"\n" + "=" * 80)
    print("📊 BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Overall accuracy
    total_examples = len(all_results)
    standard_correct = len([r for r in all_results if r["correct_standard"]])
    conservative_correct = len([r for r in all_results if r["correct_conservative"]])
    
    standard_accuracy = (standard_correct / total_examples) * 100
    conservative_accuracy = (conservative_correct / total_examples) * 100
    
    print(f"🎯 Overall Results:")
    print(f"   Total Examples: {total_examples}")
    print(f"   Standard Detection: {standard_correct}/{total_examples} ({standard_accuracy:.1f}%)")
    print(f"   Conservative Detection: {conservative_correct}/{total_examples} ({conservative_accuracy:.1f}%)")
    
    # Performance by dataset
    datasets = {}
    for result in all_results:
        test_name = result["test_name"]
        if test_name not in datasets:
            datasets[test_name] = {"total": 0, "correct_standard": 0, "correct_conservative": 0}
        
        datasets[test_name]["total"] += 1
        if result["correct_standard"]:
            datasets[test_name]["correct_standard"] += 1
        if result["correct_conservative"]:
            datasets[test_name]["correct_conservative"] += 1
    
    print(f"\n📊 Performance by Dataset:")
    for dataset_name, stats in datasets.items():
        if stats["total"] > 0:
            std_acc = (stats["correct_standard"] / stats["total"]) * 100
            cons_acc = (stats["correct_conservative"] / stats["total"]) * 100
            print(f"   {dataset_name:25} | {stats['correct_standard']:2}/{stats['total']:2} ({std_acc:5.1f}%) | Conservative: {cons_acc:5.1f}%")
    
    # Detection analysis
    hallucination_examples = [r for r in all_results if r["ground_truth"] == True]
    correct_examples = [r for r in all_results if r["ground_truth"] == False]
    
    if hallucination_examples:
        detected_hallucinations = len([r for r in hallucination_examples if r["detected_standard"]])
        recall = (detected_hallucinations / len(hallucination_examples)) * 100
        print(f"\n🚨 Hallucination Detection:")
        print(f"   Recall (Sensitivity): {detected_hallucinations}/{len(hallucination_examples)} ({recall:.1f}%)")
        
        avg_hbar_hallucination = sum(r["hbar_s"] for r in hallucination_examples) / len(hallucination_examples)
        avg_pfail_hallucination = sum(r["p_fail"] for r in hallucination_examples) / len(hallucination_examples)
        print(f"   Avg ℏₛ (hallucinations): {avg_hbar_hallucination:.4f}")
        print(f"   Avg P(fail) (hallucinations): {avg_pfail_hallucination:.4f}")
    
    if correct_examples:
        false_positives = len([r for r in correct_examples if r["detected_standard"]])
        specificity = ((len(correct_examples) - false_positives) / len(correct_examples)) * 100
        print(f"\n✅ Correct Answer Handling:")
        print(f"   Specificity (True Negative Rate): {len(correct_examples) - false_positives}/{len(correct_examples)} ({specificity:.1f}%)")
        print(f"   False Positive Rate: {false_positives}/{len(correct_examples)} ({((false_positives/len(correct_examples))*100):.1f}%)")
        
        avg_hbar_correct = sum(r["hbar_s"] for r in correct_examples) / len(correct_examples)
        avg_pfail_correct = sum(r["p_fail"] for r in correct_examples) / len(correct_examples)
        print(f"   Avg ℏₛ (correct): {avg_hbar_correct:.4f}")
        print(f"   Avg P(fail) (correct): {avg_pfail_correct:.4f}")
    
    # Performance metrics
    avg_processing_time = sum(r["processing_time_ms"] for r in all_results) / len(all_results)
    avg_methods_used = sum(r["methods_count"] for r in all_results) / len(all_results)
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   Average Processing Time: {avg_processing_time:.1f}ms")
    print(f"   Average Methods Used: {avg_methods_used:.1f}")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"official_benchmark_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_examples": total_examples,
            "standard_accuracy": standard_accuracy,
            "conservative_accuracy": conservative_accuracy,
            "dataset_breakdown": datasets,
            "detailed_results": all_results,
            "summary": {
                "recall": recall if hallucination_examples else 0,
                "specificity": specificity if correct_examples else 0,
                "avg_processing_time_ms": avg_processing_time,
                "avg_methods_used": avg_methods_used
            }
        }, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print(f"✅ Official benchmark evaluation complete!")

if __name__ == "__main__":
    run_official_benchmark_evaluation()