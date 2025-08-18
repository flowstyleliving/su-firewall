#!/usr/bin/env python3
"""
Run calibrations for all models using Candle ML with Silicon chip optimization
"""

import json
import os
import sys
import subprocess
import time
import asyncio
from pathlib import Path
from typing import List, Dict

def load_models_config(config_path: str = "config/models.json") -> Dict:
    """Load models configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def start_candle_server() -> subprocess.Popen:
    """Start the server with Candle Metal acceleration."""
    print("🔥 Starting Candle server with Metal acceleration...")
    
    # Build with Candle Metal features first
    print("   📦 Building with Candle Metal support...")
    build_cmd = ["cargo", "build", "--release", "--features", "candle-metal", "-p", "server"]
    build_result = subprocess.run(build_cmd, capture_output=True, text=True)
    
    if build_result.returncode != 0:
        print(f"❌ Build failed: {build_result.stderr}")
        print("   🔄 Falling back to standard build...")
        build_cmd = ["cargo", "build", "--release", "-p", "server"]
        build_result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if build_result.returncode != 0:
            print(f"❌ Standard build also failed: {build_result.stderr}")
            sys.exit(1)
    
    # Start the server
    server_cmd = ["cargo", "run", "--release", "--features", "candle-metal", "-p", "server"]
    try:
        process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        print("   ⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Test health endpoint
        import requests
        try:
            response = requests.get("http://127.0.0.1:8080/health", timeout=10)
            if response.status_code == 200:
                print("   ✅ Candle server started successfully!")
                return process
        except requests.RequestException:
            pass
        
        # Fallback to standard server
        print("   🔄 Starting standard server...")
        process.kill()
        process = subprocess.Popen(
            ["cargo", "run", "--release", "-p", "server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(3)
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

def run_calibration(model_id: str, model_info: Dict, server_port: int = 8080) -> bool:
    """Run calibration for a single model."""
    print(f"\n🧮 Calibrating {model_info.get('display_name', model_id)}...")
    
    # Determine appropriate dataset
    dataset = "default"
    extra_args = []
    
    if 'mistral' in model_id.lower():
        dataset = "truthfulqa"
        extra_args = ["--max_samples", "2000"]
    elif 'qwen' in model_id.lower():
        dataset = "halueval"
        extra_args = ["--halueval_task", "qa", "--max_samples", "1500"]
    elif 'pythia' in model_id.lower():
        dataset = "default"
        extra_args = ["--max_samples", "1000"]
    elif 'dialogpt' in model_id.lower():
        dataset = "default"  
        extra_args = ["--max_samples", "800"]
    elif 'ollama' in model_id.lower():
        dataset = "truthfulqa"
        extra_args = ["--max_samples", "1000"]
    
    # Build calibration command
    cmd = [
        "python3", "scripts/calibrate_failure_law.py",
        "--base", f"http://127.0.0.1:{server_port}/api/v1",
        "--dataset", dataset,
        "--concurrency", "4",
        "--output_json", f"calibrations/{model_id}_calibration_summary.json",
        "--save_csv", f"calibrations/{model_id}_scores.csv",
        "--plot_dir", "calibrations/plots",
        "--seed", "42",
    ]
    cmd.extend(extra_args)
    
    print(f"   📋 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ Calibration completed for {model_id}")
            
            # Extract key results
            summary_file = f"calibrations/{model_id}_calibration_summary.json"
            if Path(summary_file).exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    lambda_val = summary.get('lambda', 'N/A')
                    tau_val = summary.get('tau', 'N/A')
                    train_loss = summary.get('train_loss', 'N/A')
                    print(f"      📊 Results: λ={lambda_val:.3f}, τ={tau_val:.3f}, loss={train_loss:.4f}")
            
            return True
        else:
            print(f"   ❌ Calibration failed for {model_id}")
            print(f"      Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Calibration timed out for {model_id}")
        return False
    except Exception as e:
        print(f"   💥 Exception during calibration for {model_id}: {e}")
        return False

def update_model_config(model_id: str) -> bool:
    """Update model config with calibrated parameters."""
    summary_file = f"calibrations/{model_id}_calibration_summary.json"
    if not Path(summary_file).exists():
        return False
    
    try:
        # Load calibration results
        with open(summary_file, 'r') as f:
            calibration_data = json.load(f)
        
        # Load current model config
        config = load_models_config()
        
        # Update the specific model
        for model in config['models']:
            if model['id'] == model_id:
                old_lambda = model['failure_law']['lambda']
                old_tau = model['failure_law']['tau']
                
                model['failure_law']['lambda'] = calibration_data['lambda']
                model['failure_law']['tau'] = calibration_data['tau']
                
                print(f"   🔄 Updated {model_id}: λ {old_lambda}→{calibration_data['lambda']:.3f}, τ {old_tau}→{calibration_data['tau']:.3f}")
                break
        
        # Save updated config
        with open("config/models.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to update config for {model_id}: {e}")
        return False

def main():
    print("🚀 Running calibrations for all models with Candle ML...")
    
    # Create output directories
    Path("calibrations/plots").mkdir(parents=True, exist_ok=True)
    
    # Load model configuration
    config = load_models_config()
    models = config.get('models', [])
    
    if not models:
        print("❌ No models found in configuration")
        sys.exit(1)
    
    # Start server with Candle acceleration
    server_process = start_candle_server()
    
    try:
        successful_calibrations = 0
        failed_calibrations = []
        
        # Run calibrations for each model
        for i, model in enumerate(models, 1):
            model_id = model['id']
            print(f"\n{'='*60}")
            print(f"📊 CALIBRATING MODEL {i}/{len(models)}: {model_id}")
            print(f"{'='*60}")
            
            success = run_calibration(model_id, model)
            
            if success:
                successful_calibrations += 1
                # Update model config with calibrated parameters
                if update_model_config(model_id):
                    print(f"   ✅ Config updated for {model_id}")
                else:
                    print(f"   ⚠️  Config update failed for {model_id}")
            else:
                failed_calibrations.append(model_id)
            
            # Brief pause between models
            if i < len(models):
                print("   ⏳ Pausing before next model...")
                time.sleep(2)
    
    finally:
        # Cleanup server
        print(f"\n🛑 Shutting down server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
        
    # Final summary
    print(f"\n{'='*60}")
    print(f"📈 CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"   Total Models: {len(models)}")
    print(f"   Successful: {successful_calibrations}")
    print(f"   Failed: {len(failed_calibrations)}")
    
    if failed_calibrations:
        print(f"\n❌ Failed Calibrations:")
        for model_id in failed_calibrations:
            print(f"   • {model_id}")
        print(f"\n   You can retry these manually using:")
        print(f"   python3 scripts/check_calibrations.py")
    
    if successful_calibrations == len(models):
        print(f"\n🎉 All models successfully calibrated!")
        print(f"   📁 Results saved in: calibrations/")
        print(f"   📊 Plots saved in: calibrations/plots/")
        print(f"   ⚙️  Model configs updated in: config/models.json")
    
    return len(failed_calibrations)

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n🛑 Calibration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)