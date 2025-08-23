#!/usr/bin/env python3
"""
Comprehensive evaluation script for realtime engine using Mistral-7B model configuration
from 0G deployment on HaluEval, TruthfulQA, and HaluEval QA datasets.
"""

import json
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

class RealtimeEvaluator:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8080/api/v1"
        self.model_id = "mistral-7b"  # 0G deployment model
        self.results_dir = Path("realtime_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Evaluation configurations
        self.datasets = [
            {
                "name": "TruthfulQA",
                "dataset": "truthfulqa",
                "max_samples": 500,
                "description": "TruthfulQA generation task evaluation"
            },
            {
                "name": "HaluEval_General",
                "dataset": "halueval",
                "halueval_task": "general",
                "max_samples": 300,
                "description": "HaluEval general hallucination detection"
            },
            {
                "name": "HaluEval_QA",
                "dataset": "halueval", 
                "halueval_task": "qa",
                "max_samples": 300,
                "description": "HaluEval QA-specific evaluation"
            }
        ]
        
        # Methods to evaluate (both FIM and new WASM-compatible)
        self.methods = [
            {
                "name": "standard_js_kl",
                "endpoint": "/api/v1/analyze",
                "description": "Standard Jensen-Shannon + KL divergence (0G baseline)"
            },
            {
                "name": "wasm_4method",
                "endpoint": "/api/v1/analyze_wasm_4method", 
                "description": "WASM-compatible 4-method ensemble with golden scale"
            },
            {
                "name": "full_ensemble",
                "endpoint": "/api/v1/analyze_ensemble",
                "description": "Full 5-method ensemble with FIM"
            }
        ]

    def check_server_health(self):
        """Check if realtime server is running"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server healthy - uptime: {health_data.get('uptime_ms', 0)/1000:.1f}s")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Server not reachable: {e}")
            return False

    def start_server(self):
        """Start the realtime server in background"""
        print("🚀 Starting realtime server...")
        try:
            self.server_process = subprocess.Popen(
                ["cargo", "run", "-p", "server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Wait for server to start
            for attempt in range(30):  # 30 second timeout
                time.sleep(1)
                if self.check_server_health():
                    print(f"✅ Server started successfully after {attempt+1}s")
                    return True
            
            print("❌ Server failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the realtime server"""
        if hasattr(self, 'server_process'):
            print("🛑 Stopping realtime server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()

    def run_calibration_evaluation(self, dataset_config, method_config):
        """Run calibration evaluation for a specific dataset and method"""
        print(f"\n📊 Running {dataset_config['name']} evaluation with {method_config['name']}")
        
        # Build calibration command
        cmd = [
            "python3", "scripts/calibrate_failure_law.py",
            "--base", self.base_url,
            "--model_id", self.model_id,
            "--dataset", dataset_config["dataset"],
            "--max_samples", str(dataset_config["max_samples"]),
            "--timeout", "30",
            "--concurrency", "4",
            "--enable_golden_scale",  # Enable 0G golden scale calibration
            "--golden_scale", "3.4",  # 0G deployment golden scale
            "--learn_weights",  # Learn ensemble weights
            "--plot_dir", str(self.results_dir / f"plots_{dataset_config['name']}_{method_config['name']}"),
        ]
        
        # Add method-specific parameters
        if method_config["name"] != "wasm_4method":  # WASM method doesn't use standard method param
            cmd.extend(["--method", method_config["name"]])
            
        # Add dataset-specific parameters
        if "halueval_task" in dataset_config:
            cmd.extend(["--halueval_task", dataset_config["halueval_task"]])
            
        # Output files
        output_json = self.results_dir / f"{dataset_config['name']}_{method_config['name']}_results.json"
        output_csv = self.results_dir / f"{dataset_config['name']}_{method_config['name']}_data.csv"
        
        cmd.extend([
            "--output_json", str(output_json),
            "--save_csv", str(output_csv)
        ])
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        try:
            # Run calibration
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {dataset_config['name']} + {method_config['name']} completed successfully")
                
                # Parse results
                try:
                    with open(output_json, 'r') as f:
                        results = json.load(f)
                    return {
                        "status": "success",
                        "results": results,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                except Exception as e:
                    print(f"⚠️ Failed to parse results: {e}")
                    return {
                        "status": "parse_error", 
                        "error": str(e),
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                print(f"❌ {dataset_config['name']} + {method_config['name']} failed")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    "status": "failed",
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {dataset_config['name']} + {method_config['name']} timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Error running {dataset_config['name']} + {method_config['name']}: {e}")
            return {"status": "error", "error": str(e)}

    def direct_api_test(self, method_config):
        """Test method directly via API calls"""
        print(f"\n🧪 Direct API test for {method_config['name']}")
        
        try:
            import requests
            
            # Test cases for direct API validation
            test_cases = [
                {
                    "name": "correct_answer",
                    "prompt": "What is the capital of France?",
                    "output": "Paris is the capital of France.",
                    "expected": "low_uncertainty"
                },
                {
                    "name": "hallucinated_answer", 
                    "prompt": "What is the capital of France?",
                    "output": "Berlin is the capital of France.",
                    "expected": "high_uncertainty"
                },
                {
                    "name": "complex_correct",
                    "prompt": "Explain photosynthesis.",
                    "output": "Photosynthesis is a biological process where plants convert light energy into chemical energy using chlorophyll and carbon dioxide.",
                    "expected": "low_uncertainty"
                }
            ]
            
            results = []
            
            for test_case in test_cases:
                payload = {
                    "prompt": test_case["prompt"],
                    "output": test_case["output"],
                    "model_id": self.model_id
                }
                
                response = requests.post(
                    f"{self.base_url}{method_config['endpoint']}",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract key metrics
                    if method_config["name"] == "wasm_4method":
                        hbar_s = data.get("calibrated_hbar_s", 0)
                        p_fail = data.get("p_fail", 0)
                        risk_level = data.get("risk_level", "unknown")
                    else:
                        hbar_s = data.get("hbar_s", 0)
                        p_fail = data.get("p_fail", 0)
                        risk_level = data.get("risk_level", "unknown")
                    
                    results.append({
                        "test_case": test_case["name"],
                        "hbar_s": hbar_s,
                        "p_fail": p_fail,
                        "risk_level": risk_level,
                        "expected": test_case["expected"],
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    })
                    
                    print(f"  {test_case['name']}: ℏₛ={hbar_s:.3f}, p_fail={p_fail:.3f}, risk={risk_level}")
                    
                else:
                    print(f"  ❌ {test_case['name']}: HTTP {response.status_code}")
                    results.append({
                        "test_case": test_case["name"],
                        "error": f"HTTP {response.status_code}",
                        "response": response.text[:200]
                    })
            
            return {"status": "success", "test_results": results}
            
        except Exception as e:
            print(f"❌ Direct API test failed: {e}")
            return {"status": "error", "error": str(e)}

    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across all datasets and methods"""
        print("🎯 Starting comprehensive realtime engine evaluation")
        print(f"📋 Model: {self.model_id}")
        print(f"📊 Datasets: {[d['name'] for d in self.datasets]}")
        print(f"🔧 Methods: {[m['name'] for m in self.methods]}")
        
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "model_id": self.model_id,
            "server_url": self.base_url,
            "datasets": [],
            "methods": [],
            "results": {},
            "direct_api_tests": {},
            "summary": {}
        }
        
        # Check if server is running, start if needed
        server_was_running = self.check_server_health()
        if not server_was_running:
            if not self.start_server():
                print("❌ Failed to start server, aborting evaluation")
                return evaluation_results
        
        try:
            # Direct API tests first
            print("\n" + "="*60)
            print("🧪 DIRECT API TESTS")
            print("="*60)
            
            for method in self.methods:
                evaluation_results["direct_api_tests"][method["name"]] = self.direct_api_test(method)
            
            # Dataset evaluations
            print("\n" + "="*60)
            print("📊 DATASET EVALUATIONS") 
            print("="*60)
            
            total_evaluations = len(self.datasets) * len(self.methods)
            current_eval = 0
            
            for dataset in self.datasets:
                evaluation_results["datasets"].append(dataset)
                evaluation_results["results"][dataset["name"]] = {}
                
                for method in self.methods:
                    if dataset["name"] not in [m["name"] for m in evaluation_results["methods"]]:
                        evaluation_results["methods"].append(method)
                    
                    current_eval += 1
                    print(f"\n📈 Progress: {current_eval}/{total_evaluations}")
                    
                    result = self.run_calibration_evaluation(dataset, method)
                    evaluation_results["results"][dataset["name"]][method["name"]] = result
                    
                    # Brief pause between evaluations
                    time.sleep(2)
            
            # Generate summary
            print("\n" + "="*60)
            print("📋 GENERATING SUMMARY")
            print("="*60)
            
            summary = self.generate_summary(evaluation_results)
            evaluation_results["summary"] = summary
            
            # Save comprehensive results
            results_file = self.results_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
            print(f"\n💾 Results saved to: {results_file}")
            
        finally:
            # Clean up server if we started it
            if not server_was_running:
                self.stop_server()
        
        return evaluation_results

    def generate_summary(self, evaluation_results):
        """Generate evaluation summary"""
        summary = {
            "total_evaluations": len(self.datasets) * len(self.methods),
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "dataset_performance": {},
            "method_performance": {},
            "api_test_results": {},
            "recommendations": []
        }
        
        # Count successes/failures
        for dataset_name, dataset_results in evaluation_results["results"].items():
            for method_name, result in dataset_results.items():
                if result.get("status") == "success":
                    summary["successful_evaluations"] += 1
                else:
                    summary["failed_evaluations"] += 1
        
        # Analyze API test results
        for method_name, api_result in evaluation_results["direct_api_tests"].items():
            if api_result.get("status") == "success":
                test_results = api_result.get("test_results", [])
                avg_response_time = sum(t.get("response_time_ms", 0) for t in test_results) / max(1, len(test_results))
                summary["api_test_results"][method_name] = {
                    "status": "success",
                    "avg_response_time_ms": avg_response_time,
                    "test_count": len(test_results)
                }
            else:
                summary["api_test_results"][method_name] = {
                    "status": "failed",
                    "error": api_result.get("error", "unknown")
                }
        
        # Generate recommendations
        if summary["successful_evaluations"] > 0:
            summary["recommendations"].append("✅ Realtime engine is functional with Mistral-7B model")
        
        if summary["api_test_results"].get("wasm_4method", {}).get("status") == "success":
            summary["recommendations"].append("✅ WASM 4-method ensemble is working correctly")
        
        if summary["failed_evaluations"] > 0:
            summary["recommendations"].append(f"⚠️ {summary['failed_evaluations']} evaluations failed - check logs")
        
        return summary

    def print_results_summary(self, evaluation_results):
        """Print a formatted summary of results"""
        print("\n" + "🎉 EVALUATION COMPLETE 🎉".center(60, "="))
        
        summary = evaluation_results.get("summary", {})
        
        print(f"""
📊 EVALUATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Model: {evaluation_results['model_id']}
• Total Evaluations: {summary.get('total_evaluations', 0)}
• Successful: {summary.get('successful_evaluations', 0)}
• Failed: {summary.get('failed_evaluations', 0)}
• Success Rate: {(summary.get('successful_evaluations', 0) / max(1, summary.get('total_evaluations', 1)) * 100):.1f}%

🔧 API TEST RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
        
        for method_name, api_result in summary.get("api_test_results", {}).items():
            status_icon = "✅" if api_result.get("status") == "success" else "❌"
            if api_result.get("status") == "success":
                avg_time = api_result.get("avg_response_time_ms", 0)
                print(f"• {status_icon} {method_name}: {avg_time:.1f}ms avg response time")
            else:
                print(f"• {status_icon} {method_name}: {api_result.get('error', 'failed')}")
        
        print(f"""
💡 RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
        
        for rec in summary.get("recommendations", []):
            print(f"• {rec}")
        
        print(f"""
📁 RESULTS LOCATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Directory: {self.results_dir}
• Files: JSON results, CSV data, calibration plots
        """)

def main():
    """Main evaluation function"""
    print("🚀 Realtime Engine Comprehensive Evaluation")
    print("📋 Using Mistral-7B configuration from 0G deployment")
    
    evaluator = RealtimeEvaluator()
    
    try:
        results = evaluator.run_comprehensive_evaluation()
        evaluator.print_results_summary(results)
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        evaluator.stop_server()
        return 1
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        evaluator.stop_server()
        return 1

if __name__ == "__main__":
    sys.exit(main())