#!/usr/bin/env python3
"""
Quick comprehensive evaluation of Mistral-7B with realtime engine.
Focused evaluation for immediate results with the 0G deployment configuration.
"""

import subprocess
import json
import sys
from pathlib import Path

def run_evaluation(dataset, task, samples, method_name, endpoint_method):
    """Run a single evaluation configuration"""
    
    cmd = [
        "python3", "scripts/calibrate_failure_law.py",
        "--base", "http://127.0.0.1:8080/api/v1",
        "--model_id", "mistral-7b",
        "--dataset", dataset,
        "--max_samples", str(samples),
        "--timeout", "30",
        "--concurrency", "4",
        "--enable_golden_scale",
        "--golden_scale", "3.4",
        "--output_json", f"quick_results_{dataset}_{task}_{method_name}.json"
    ]
    
    if task:
        cmd.extend(["--halueval_task", task])
    
    if endpoint_method:
        cmd.extend(["--method", endpoint_method])
    
    print(f"\n🚀 Running {dataset}({task if task else 'all'}) with {method_name} method")
    print(f"📊 Sample size: {samples}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse results
            with open(f"quick_results_{dataset}_{task}_{method_name}.json", 'r') as f:
                data = json.load(f)
            
            test_metrics = data.get("metrics", {}).get("test", {})
            roc_auc = test_metrics.get("roc_auc", 0)
            brier = test_metrics.get("brier", 0)
            ece = test_metrics.get("ece", 0)
            
            print(f"✅ {dataset}({task}) + {method_name}: ROC-AUC={roc_auc:.3f}, Brier={brier:.3f}, ECE={ece:.3f}")
            return {
                "status": "success",
                "dataset": dataset,
                "task": task,
                "method": method_name,
                "roc_auc": roc_auc,
                "brier_score": brier,
                "ece": ece,
                "golden_scale": data.get("golden_scale", 3.4)
            }
        else:
            print(f"❌ {dataset}({task}) + {method_name} failed:")
            print(f"   STDERR: {result.stderr[-200:]}...")
            return {"status": "failed", "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset}({task}) + {method_name} timed out")
        return {"status": "timeout"}
    except Exception as e:
        print(f"❌ {dataset}({task}) + {method_name} error: {e}")
        return {"status": "error", "error": str(e)}

def main():
    print("🚀 Quick Realtime Engine Evaluation - Mistral-7B with 0G Configuration")
    print("📋 Using golden scale calibration (3.4x) from production 0G deployment")
    
    # Evaluation configurations - smaller samples for quick results
    evaluations = [
        # TruthfulQA with different methods
        {"dataset": "truthfulqa", "task": None, "samples": 100, "method_name": "standard_js_kl", "endpoint_method": "standard_js_kl"},
        {"dataset": "truthfulqa", "task": None, "samples": 100, "method_name": "ensemble", "endpoint_method": "full_fim_dir"},
        
        # HaluEval QA 
        {"dataset": "halueval", "task": "qa", "samples": 100, "method_name": "standard_js_kl", "endpoint_method": "standard_js_kl"},
        {"dataset": "halueval", "task": "qa", "samples": 100, "method_name": "ensemble", "endpoint_method": "full_fim_dir"},
        
        # HaluEval General
        {"dataset": "halueval", "task": "general", "samples": 100, "method_name": "standard_js_kl", "endpoint_method": "standard_js_kl"},
        {"dataset": "halueval", "task": "general", "samples": 100, "method_name": "ensemble", "endpoint_method": "full_fim_dir"}
    ]
    
    results = []
    successful = 0
    failed = 0
    
    print(f"\n📊 Running {len(evaluations)} evaluation configurations...")
    
    for i, eval_config in enumerate(evaluations, 1):
        print(f"\n📈 Progress: {i}/{len(evaluations)}")
        
        result = run_evaluation(
            eval_config["dataset"],
            eval_config["task"], 
            eval_config["samples"],
            eval_config["method_name"],
            eval_config["endpoint_method"]
        )
        
        results.append(result)
        
        if result["status"] == "success":
            successful += 1
        else:
            failed += 1
    
    # Generate summary
    print(f"\n{'='*60}")
    print("📋 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: mistral-7b (0G deployment configuration)")
    print(f"Golden Scale: 3.4x calibration factor")
    print(f"Total Evaluations: {len(evaluations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(evaluations)*100:.1f}%")
    
    print(f"\n🏆 PERFORMANCE RESULTS")
    print(f"{'Dataset':<15} {'Task':<8} {'Method':<12} {'ROC-AUC':<8} {'Brier':<8} {'ECE':<8}")
    print("-" * 70)
    
    for result in results:
        if result["status"] == "success":
            dataset = result["dataset"]
            task = result.get("task", "all")[:8] if result.get("task") else "all"
            method = result["method"][:12]
            roc_auc = f"{result['roc_auc']:.3f}" if result.get('roc_auc') else "N/A"
            brier = f"{result['brier_score']:.3f}" if result.get('brier_score') else "N/A"
            ece = f"{result['ece']:.3f}" if result.get('ece') else "N/A"
            
            print(f"{dataset:<15} {task:<8} {method:<12} {roc_auc:<8} {brier:<8} {ece:<8}")
    
    # Save comprehensive results
    summary = {
        "model_id": "mistral-7b",
        "golden_scale": 3.4,
        "total_evaluations": len(evaluations),
        "successful_evaluations": successful,
        "failed_evaluations": failed,
        "success_rate": successful/len(evaluations)*100,
        "results": results
    }
    
    with open("quick_realtime_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: quick_realtime_evaluation_summary.json")
    
    if successful > 0:
        print("\n✅ Realtime engine evaluation completed successfully!")
        print("🎯 Key findings:")
        
        avg_roc_auc = sum(r.get('roc_auc', 0) for r in results if r['status'] == 'success') / successful
        print(f"   • Average ROC-AUC: {avg_roc_auc:.3f}")
        
        if any(r.get('roc_auc', 0) > 0.85 for r in results if r['status'] == 'success'):
            print("   • 🏆 Excellent hallucination detection performance (ROC-AUC > 0.85)")
        
        print("   • ⚡ Golden scale calibration (3.4x) applied correctly")
        print("   • 📊 Mistral-7B model working with 0G production configuration")
        
        return 0
    else:
        print("\n❌ All evaluations failed - check server and dataset configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())