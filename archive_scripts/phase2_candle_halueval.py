#!/usr/bin/env python3
"""
PHASE 2: Production-Grade Hallucination Detection Evaluation
Using Candle ML + Rust tiered system + HaluEval dataset + config/models.json
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
from datasets import load_dataset
import statistics

@dataclass
class EvaluationSample:
    prompt: str
    output: str
    is_hallucination: bool
    domain: str
    difficulty: str
    sample_id: str

@dataclass
class ModelResult:
    model_id: str
    hbar_s: float
    p_fail: float
    agreement_score: float
    processing_time_ms: float
    methods_used: List[str]
    individual_results: Dict[str, float]

class HaluEvalDatasetLoader:
    """Load and prepare HaluEval dataset for evaluation"""
    
    def load_halueval_dataset(self, subset: str = "qa", max_samples: int = 2000) -> List[EvaluationSample]:
        """Load HaluEval dataset"""
        
        print(f"📊 Loading HaluEval {subset} dataset (max {max_samples} samples)...")
        
        try:
            # Load HaluEval dataset
            dataset = load_dataset("pminervini/HaluEval", subset, split="data")
            
            samples = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                # Extract fields (HaluEval structure)
                if subset == "qa":
                    prompt = item.get("question", "")
                    output = item.get("answer", "")
                    is_hallucination = item.get("hallucination", "yes") == "yes"
                elif subset == "dialogue":
                    prompt = item.get("dialogue_history", "")
                    output = item.get("response", "")
                    is_hallucination = item.get("hallucination", "yes") == "yes"
                else:  # summarization
                    prompt = item.get("document", "")
                    output = item.get("summary", "")
                    is_hallucination = item.get("hallucination", "yes") == "yes"
                
                if prompt and output:
                    sample = EvaluationSample(
                        prompt=prompt[:1000],  # Truncate long prompts
                        output=output[:1000],   # Truncate long outputs
                        is_hallucination=is_hallucination,
                        domain=subset,
                        difficulty="medium",
                        sample_id=f"{subset}_{i}"
                    )
                    samples.append(sample)
            
            print(f"✅ Loaded {len(samples)} {subset} samples")
            print(f"📊 Hallucination rate: {sum(s.is_hallucination for s in samples) / len(samples):.1%}")
            
            return samples
            
        except Exception as e:
            print(f"❌ Error loading HaluEval {subset}: {e}")
            print("🔄 Falling back to synthetic data...")
            return self._generate_synthetic_samples(subset, max_samples // 2)
    
    def _generate_synthetic_samples(self, domain: str, count: int) -> List[EvaluationSample]:
        """Generate synthetic evaluation samples as fallback"""
        
        synthetic_cases = [
            # Geography - Truth
            ("What is the capital of France?", "The capital of France is Paris.", False),
            ("What is the largest city in Japan?", "The largest city in Japan is Tokyo.", False),
            ("Which country is Rome located in?", "Rome is located in Italy.", False),
            
            # Geography - Hallucination  
            ("What is the capital of France?", "The capital of France is Lyon.", True),
            ("What is the largest city in Japan?", "The largest city in Japan is Osaka.", True),
            ("Which country is Rome located in?", "Rome is located in Greece.", True),
            
            # Science - Truth
            ("What is the speed of light?", "The speed of light is approximately 299,792,458 meters per second.", False),
            ("How many planets are in our solar system?", "There are 8 planets in our solar system.", False),
            ("What is the chemical symbol for water?", "The chemical symbol for water is H2O.", False),
            
            # Science - Hallucination
            ("What is the speed of light?", "The speed of light is approximately 186,000 miles per hour.", True),
            ("How many planets are in our solar system?", "There are 12 planets in our solar system.", True),
            ("What is the chemical symbol for water?", "The chemical symbol for water is H3O.", True),
            
            # Math - Truth
            ("What is 15 + 27?", "15 + 27 equals 42.", False),
            ("What is the square root of 64?", "The square root of 64 is 8.", False),
            ("What is 12 × 8?", "12 × 8 equals 96.", False),
            
            # Math - Hallucination
            ("What is 15 + 27?", "15 + 27 equals 43.", True),
            ("What is the square root of 64?", "The square root of 64 is 6.", True),
            ("What is 12 × 8?", "12 × 8 equals 84.", True),
        ]
        
        samples = []
        target_count = min(count, len(synthetic_cases))
        
        for i in range(target_count):
            prompt, output, is_halluc = synthetic_cases[i % len(synthetic_cases)]
            sample = EvaluationSample(
                prompt=prompt,
                output=output,
                is_hallucination=is_halluc,
                domain="synthetic",
                difficulty="easy" if i < target_count // 2 else "medium",
                sample_id=f"synthetic_{i}"
            )
            samples.append(sample)
        
        print(f"✅ Generated {len(samples)} synthetic samples for {domain}")
        return samples

class CandleRustEvaluator:
    """Evaluator using Candle ML + Rust tiered system"""
    
    def __init__(self, api_endpoint: str = "http://localhost:8080"):
        self.api_endpoint = api_endpoint
        self.models = self._load_model_configs()
    
    def _load_model_configs(self) -> List[Dict]:
        """Load models from config/models.json"""
        try:
            with open('config/models.json', 'r') as f:
                config = json.load(f)
            return config['models']
        except Exception as e:
            print(f"⚠️  Could not load config/models.json: {e}")
            return [{"id": "mistral-7b", "display_name": "Mistral-7B (fallback)"}]
    
    async def evaluate_sample_with_model(self, session: aiohttp.ClientSession, 
                                       sample: EvaluationSample, model_id: str) -> Optional[ModelResult]:
        """Evaluate single sample with specific model using Candle ML + Rust ensemble"""
        
        try:
            async with session.post(
                f'{self.api_endpoint}/api/v1/analyze_ensemble',
                json={
                    'prompt': sample.prompt,
                    'output': sample.output,
                    'model_id': model_id
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    ensemble_result = data['ensemble_result']
                    
                    return ModelResult(
                        model_id=model_id,
                        hbar_s=ensemble_result['hbar_s'],
                        p_fail=ensemble_result['p_fail'],
                        agreement_score=ensemble_result['agreement_score'],
                        processing_time_ms=data.get('processing_time_ms', 0),
                        methods_used=ensemble_result['methods_used'],
                        individual_results=ensemble_result['individual_results']
                    )
                else:
                    print(f"⚠️  HTTP {response.status} for {model_id}: {await response.text()}")
                    return None
                    
        except Exception as e:
            print(f"⚠️  Error evaluating {sample.sample_id} with {model_id}: {e}")
            return None
    
    async def batch_evaluate(self, samples: List[EvaluationSample], 
                           model_ids: List[str], batch_size: int = 10) -> Dict[str, List[ModelResult]]:
        """Evaluate samples in batches across multiple models"""
        
        print(f"🚀 Starting batch evaluation:")
        print(f"   📊 Samples: {len(samples)}")
        print(f"   🤖 Models: {model_ids}")
        print(f"   📦 Batch size: {batch_size}")
        
        results = {model_id: [] for model_id in model_ids}
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                batch_start = time.time()
                
                print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")
                
                # Process each model for this batch
                for model_id in model_ids:
                    print(f"   🤖 {model_id}: ", end="")
                    
                    tasks = [self.evaluate_sample_with_model(session, sample, model_id) 
                            for sample in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    successful = 0
                    for result in batch_results:
                        if isinstance(result, ModelResult):
                            results[model_id].append(result)
                            successful += 1
                    
                    print(f"{successful}/{len(batch)} successful")
                
                batch_time = time.time() - batch_start
                rate = len(batch) * len(model_ids) / batch_time
                print(f"   ⚡ Rate: {rate:.1f} evaluations/sec")
        
        return results

def calculate_metrics(results: List[ModelResult], ground_truth: List[bool]) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    
    if len(results) != len(ground_truth):
        print(f"⚠️  Mismatch: {len(results)} results vs {len(ground_truth)} ground truth")
        min_len = min(len(results), len(ground_truth))
        results = results[:min_len]
        ground_truth = ground_truth[:min_len]
    
    # Convert hbar_s to binary predictions (lower hbar_s = higher hallucination probability)
    # Using adaptive threshold
    hbar_values = [r.hbar_s for r in results]
    threshold = np.median(hbar_values)  # Adaptive threshold
    
    predictions = [r.hbar_s < threshold for r in results]  # Lower hbar_s = hallucination
    
    # Calculate confusion matrix
    tp = sum(1 for pred, true in zip(predictions, ground_truth) if pred and true)
    tn = sum(1 for pred, true in zip(predictions, ground_truth) if not pred and not true) 
    fp = sum(1 for pred, true in zip(predictions, ground_truth) if pred and not true)
    fn = sum(1 for pred, true in zip(predictions, ground_truth) if not pred and true)
    
    # Calculate metrics
    accuracy = (tp + tn) / len(ground_truth) if len(ground_truth) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1_score': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'threshold': threshold,
        'avg_hbar_s': statistics.mean(hbar_values),
        'avg_agreement': statistics.mean([r.agreement_score for r in results]),
        'avg_processing_time': statistics.mean([r.processing_time_ms for r in results])
    }

async def run_phase2_evaluation():
    """Run Phase 2: Comprehensive evaluation with Candle ML + Rust"""
    
    print("🔥 PHASE 2: CANDLE ML + RUST TIERED SYSTEM + HALUEVAL")
    print("=" * 80)
    
    # Load datasets
    loader = HaluEvalDatasetLoader()
    
    # Load multiple HaluEval subsets
    qa_samples = loader.load_halueval_dataset("qa", max_samples=500)
    
    # For now, start with QA subset - can expand later
    all_samples = qa_samples
    
    if not all_samples:
        print("❌ No samples loaded - cannot proceed")
        return
    
    print(f"📊 Total evaluation samples: {len(all_samples)}")
    
    # Load model configurations
    evaluator = CandleRustEvaluator()
    
    # Test with key models from config/models.json
    priority_models = ["mistral-7b", "qwen2.5-7b", "dialogpt-medium"]  # Start with fast models
    
    print(f"🤖 Priority models: {priority_models}")
    
    # Check server availability
    try:
        response = requests.get(f"{evaluator.api_endpoint}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Rust server is running and healthy")
        else:
            print(f"⚠️  Server health check returned {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to Rust server: {e}")
        print("💡 Please ensure: cargo run -p server --release")
        return
    
    # Run evaluation
    start_time = time.time()
    results = await evaluator.batch_evaluate(all_samples, priority_models, batch_size=5)
    end_time = time.time()
    
    # Analyze results
    print(f"\n🎯 PHASE 2 EVALUATION RESULTS")
    print("=" * 80)
    
    total_evaluations = sum(len(model_results) for model_results in results.values())
    evaluation_rate = total_evaluations / (end_time - start_time)
    
    print(f"⚡ Performance: {total_evaluations} evaluations in {end_time - start_time:.1f}s ({evaluation_rate:.1f}/sec)")
    
    # Calculate metrics per model
    ground_truth = [s.is_hallucination for s in all_samples]
    
    model_metrics = {}
    for model_id, model_results in results.items():
        if model_results:
            metrics = calculate_metrics(model_results, ground_truth[:len(model_results)])
            model_metrics[model_id] = metrics
            
            print(f"\n🤖 {model_id.upper()}:")
            print(f"   📊 Samples evaluated: {len(model_results)}")
            print(f"   🎯 F1-Score: {metrics['f1_score']:.3f}")
            print(f"   📈 Precision: {metrics['precision']:.3f}")  
            print(f"   📈 Recall: {metrics['recall']:.3f}")
            print(f"   🎯 Accuracy: {metrics['accuracy']:.3f}")
            print(f"   ⚡ Avg processing time: {metrics['avg_processing_time']:.1f}ms")
            print(f"   🧮 Avg ℏₛ: {metrics['avg_hbar_s']:.3f}")
            print(f"   🤝 Avg agreement: {metrics['avg_agreement']:.3f}")
    
    # Overall analysis
    if model_metrics:
        best_f1_model = max(model_metrics.items(), key=lambda x: x[1]['f1_score'])
        best_f1_score = best_f1_model[1]['f1_score']
        
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   🤖 Model: {best_f1_model[0]}")
        print(f"   🎯 F1-Score: {best_f1_score:.3f}")
        
        if best_f1_score > 0.7:
            print("🎉 EXCELLENT: F1-Score > 0.7 achieved!")
            success_status = "EXCELLENT"
        elif best_f1_score > 0.5:
            print("✅ GOOD: F1-Score > 0.5 achieved")
            success_status = "GOOD"
        else:
            print("⚠️  NEEDS IMPROVEMENT: F1-Score < 0.5")
            success_status = "NEEDS_IMPROVEMENT"
    else:
        success_status = "FAILED"
    
    # Save comprehensive results
    comprehensive_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "PHASE_2_CANDLE_RUST_HALUEVAL",
        "success_status": success_status,
        "total_samples": len(all_samples),
        "total_evaluations": total_evaluations,
        "evaluation_time_seconds": end_time - start_time,
        "evaluation_rate_per_second": evaluation_rate,
        "models_tested": priority_models,
        "model_metrics": model_metrics,
        "best_model": best_f1_model[0] if model_metrics else None,
        "best_f1_score": best_f1_score if model_metrics else 0,
        "dataset_info": {
            "sources": ["HaluEval_QA", "synthetic_fallback"],
            "hallucination_rate": sum(s.is_hallucination for s in all_samples) / len(all_samples)
        }
    }
    
    with open('phase2_candle_rust_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n📁 Comprehensive results saved: phase2_candle_rust_results.json")
    
    return success_status == "EXCELLENT" or success_status == "GOOD"

if __name__ == "__main__":
    print("🚀 STARTING PHASE 2: PRODUCTION-GRADE EVALUATION")
    print("🔥 Candle ML + Rust Tiered System + HaluEval Dataset")
    
    success = asyncio.run(run_phase2_evaluation())
    exit(0 if success else 1)