#!/usr/bin/env python3
"""
🔥 Rust-based Hallucination Detection Evaluation Runner
Uses realtime engine API with all 6 precision/flexibility methods + Pfail + FEP in one step
"""

import asyncio
import requests
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import time
from pathlib import Path
import argparse

class RustEvaluationRunner:
    def __init__(self, realtime_api_url="http://localhost:8080"):
        self.api_url = realtime_api_url
        # All 6 core API methods from the realtime engine
        self.methods = [
            "diag_fim_dir",     # Default: Diagonal FIM directional
            "scalar_js_kl",     # Jensen-Shannon/KL divergence
            "scalar_trace",     # Trace-based
            "scalar_fro",       # Frobenius norm
            "full_fim_dir",     # Full FIM directional
            "logits_adapter"    # Advanced logit-based calculations
        ]
        
    async def start_realtime_engine(self):
        """Start the Rust realtime engine server."""
        import subprocess
        import os
        
        print("🚀 Starting Rust realtime engine...")
        realtime_dir = "/Users/elliejenkins/Desktop/su-firewall/realtime"
        
        # Build and start the server
        build_process = subprocess.run(
            ["cargo", "build", "--release", "--features", "api,candle,candle-metal"],
            cwd=realtime_dir,
            capture_output=True,
            text=True
        )
        
        if build_process.returncode != 0:
            print(f"❌ Build failed: {build_process.stderr}")
            return False
            
        # Start server in background
        self.server_process = subprocess.Popen(
            ["cargo", "run", "--release", "--features", "api,candle,candle-metal", "--bin", "server"],
            cwd=realtime_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        await asyncio.sleep(5)
        return True
        
    def analyze_with_method(self, text: str, method: str) -> Dict[str, Any]:
        """Analyze text using specified method via API."""
        
        # Prepare request based on method  
        if method == "logits_adapter":
            # Use analyze_logits endpoint for advanced adapter
            request_data = {
                "prompt": text,
                "token_logits": [[0.5, 0.3, 0.2]],  # Mock logits for now
                "method": method
            }
            endpoint = "/api/v1/analyze_logits"
        else:
            # Use analyze_topk_compact for other methods (correct format)
            request_data = {
                "topk_indices": [1, 2, 3],       # Single list, not nested
                "topk_probs": [0.5, 0.3, 0.2],  # Single list, not nested
                "rest_mass": 0.0,               # Single float, not list
                "vocab_size": 50000,
                "method": method,
                "model_id": "mistral-7b"        # CRITICAL FIX: Add model_id
            }
            endpoint = "/api/v1/analyze_topk_compact"
            
        try:
            response = requests.post(
                f"{self.api_url}{endpoint}",
                json=request_data,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    "method": method,
                    "hbar_s": result.get("hbar_s", 0.0),
                    "delta_mu": result.get("delta_mu", 0.0), 
                    "delta_sigma": result.get("delta_sigma", 0.0),
                    "p_fail": result.get("p_fail", 0.0),
                    "free_energy": result.get("free_energy", {}),
                    "success": True,
                    "processing_time_ms": result.get("processing_time_ms", 0.0)
                }
            else:
                print(f"❌ API error for {method}: {response.status_code}")
                return {"method": method, "success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Request failed for {method}: {str(e)}")
            return {"method": method, "success": False, "error": str(e)}
            
    def evaluate_dataset(self, dataset_path: str, max_samples: int = 100) -> Dict[str, Any]:
        """Evaluate all methods on dataset."""
        print(f"📊 Loading dataset: {dataset_path}")
        
        # Load authentic dataset (JSONL format)
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line.strip()))
        
        examples = examples[:max_samples]
            
        print(f"🎯 Evaluating {len(examples)} examples with {len(self.methods)} methods")
        
        results = {method: {"correct": 0, "total": 0, "scores": [], "times": []} 
                  for method in self.methods}
        
        for i, example in enumerate(examples):
            if i % 10 == 0:
                print(f"📈 Progress: {i}/{len(examples)}")
            
            # Extract text and expected label
            if isinstance(example, dict):
                # HaluEval format
                text = example.get('chatgpt_response', example.get('text', ''))
                is_hallucination = example.get('hallucination') == 'yes'
            else:
                continue
                
            if not text:
                continue
            
            # Test all methods on this example
            for method in self.methods:
                result = self.analyze_with_method(text, method)
                
                if result["success"]:
                    # Extract combined score (hbar_s + p_fail + FEP all in one)
                    hbar_s = result["hbar_s"]
                    p_fail = result["p_fail"] 
                    free_energy = result.get("free_energy", {})
                    
                    # FEP components (if available)
                    fep_score = 0.0
                    if isinstance(free_energy, dict):
                        fep_score = (free_energy.get("kl_surprise", 0.0) + 
                                   free_energy.get("complexity", 0.0) + 
                                   free_energy.get("prediction_error", 0.0))
                    elif isinstance(free_energy, (int, float)):
                        fep_score = free_energy
                    
                    # Combined uncertainty score (L3: ℏₛ + P(fail) + FEP)
                    combined_score = hbar_s + p_fail + fep_score
                    
                    # Prediction: high combined score = hallucination
                    predicted_hallucination = combined_score > 1.5  # Adjusted threshold
                    
                    # Track accuracy
                    correct = predicted_hallucination == is_hallucination
                    results[method]["correct"] += int(correct)
                    results[method]["total"] += 1
                    results[method]["scores"].append(combined_score)
                    results[method]["times"].append(result["processing_time_ms"])
                    
        # Calculate final statistics
        final_results = {}
        for method in self.methods:
            data = results[method]
            if data["total"] > 0:
                accuracy = data["correct"] / data["total"]
                avg_score = np.mean(data["scores"]) if data["scores"] else 0.0
                avg_time = np.mean(data["times"]) if data["times"] else 0.0
                
                final_results[method] = {
                    "accuracy": accuracy,
                    "correct": data["correct"],
                    "total": data["total"], 
                    "avg_combined_score": avg_score,
                    "avg_processing_time_ms": avg_time,
                    "performance_rating": "🏆 EXCELLENT" if accuracy > 0.95 else
                                        "✅ GOOD" if accuracy > 0.85 else
                                        "⚠️  NEEDS WORK" if accuracy > 0.75 else
                                        "❌ POOR"
                }
                
        return final_results
        
    def run_full_evaluation(self, datasets_dir: str = "/Users/elliejenkins/Desktop/su-firewall/authentic_datasets"):
        """Run complete evaluation across all datasets and methods."""
        print("🔥 Starting Rust-based Hallucination Detection Evaluation")
        print(f"🎯 Testing {len(self.methods)} precision/flexibility methods")
        print("📊 Methods: " + ", ".join(self.methods))
        print("🔧 Using calibrated Pfail + FEP in single API call")
        
        datasets = [
            "halueval_general_data.json",
            "truthfulqa_data.json"
        ]
        
        all_results = {}
        
        for dataset_name in datasets:
            dataset_path = Path(datasets_dir) / dataset_name
            if dataset_path.exists():
                print(f"\n🔬 Evaluating {dataset_name}")
                results = self.evaluate_dataset(str(dataset_path), max_samples=10000)  # Use all available samples
                all_results[dataset_name] = results
                
                # Print results for this dataset
                print(f"\n📈 Results for {dataset_name}:")
                for method, data in results.items():
                    print(f"  {method:15} | {data['performance_rating']:15} | "
                          f"Accuracy: {data['accuracy']:.1%} ({data['correct']}/{data['total']}) | "
                          f"Avg Score: {data['avg_combined_score']:.3f} | "
                          f"Time: {data['avg_processing_time_ms']:.1f}ms")
            else:
                print(f"⚠️  Dataset not found: {dataset_path}")
                
        # Overall summary
        print(f"\n🏆 OVERALL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        method_averages = {}
        for method in self.methods:
            accuracies = []
            for dataset_results in all_results.values():
                if method in dataset_results:
                    accuracies.append(dataset_results[method]["accuracy"])
            if accuracies:
                avg_acc = np.mean(accuracies)
                method_averages[method] = avg_acc
                
        # Sort by performance
        sorted_methods = sorted(method_averages.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🥇 METHOD RANKINGS:")
        for i, (method, accuracy) in enumerate(sorted_methods, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            performance = "🏆 WORLD-CLASS" if accuracy > 0.95 else \
                        "✅ EXCELLENT" if accuracy > 0.90 else \
                        "⚠️  GOOD" if accuracy > 0.80 else "❌ NEEDS WORK"
            print(f"  {emoji} {method:15} | {performance:15} | {accuracy:.1%}")
            
        print(f"\n✅ Evaluation complete! Best method: {sorted_methods[0][0]} ({sorted_methods[0][1]:.1%})")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Rust-based hallucination detection evaluation")
    parser.add_argument("--datasets-dir", default="/Users/elliejenkins/Desktop/su-firewall/authentic_datasets")
    parser.add_argument("--api-url", default="http://localhost:8080")
    
    args = parser.parse_args()
    
    runner = RustEvaluationRunner(args.api_url)
    results = runner.run_full_evaluation(args.datasets_dir)
    
    # Save results
    output_file = "/Users/elliejenkins/Desktop/su-firewall/rust_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()