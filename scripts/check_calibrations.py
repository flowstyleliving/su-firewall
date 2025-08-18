#!/usr/bin/env python3
"""
Check calibration status for all models in config/models.json
"""

import json
import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

def load_models_config(config_path: str = "config/models.json") -> Dict:
    """Load models configuration."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)

def check_calibration_files(model_id: str) -> Dict[str, bool]:
    """Check if calibration files exist for a model."""
    calibration_dir = Path("calibrations")
    calibration_dir.mkdir(exist_ok=True)
    
    files_to_check = {
        "failure_law": f"calibrations/{model_id}_failure_law.json",
        "calibration_summary": f"calibrations/{model_id}_calibration_summary.json", 
        "scores_csv": f"calibrations/{model_id}_scores.csv",
        "roc_plot": f"calibrations/plots/{model_id}_roc.png",
        "calibration_plot": f"calibrations/plots/{model_id}_calibration.png"
    }
    
    status = {}
    for file_type, file_path in files_to_check.items():
        status[file_type] = Path(file_path).exists()
    
    return status

def get_calibration_quality(model_id: str) -> Optional[Dict]:
    """Get calibration quality metrics if available."""
    summary_path = f"calibrations/{model_id}_calibration_summary.json"
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
            
            # Extract key quality metrics
            quality = {}
            if 'metrics' in data:
                metrics = data['metrics']
                for split in ['train', 'val', 'test']:
                    if split in metrics:
                        split_metrics = metrics[split]
                        quality[f"{split}_roc_auc"] = split_metrics.get('roc_auc')
                        quality[f"{split}_brier_score"] = split_metrics.get('brier_score')
                        quality[f"{split}_ece"] = split_metrics.get('ece')
                        
            quality['lambda'] = data.get('lambda')
            quality['tau'] = data.get('tau')
            quality['train_loss'] = data.get('train_loss')
            
            return quality
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def is_calibration_stale(model_id: str, max_age_days: int = 30) -> bool:
    """Check if calibration is older than max_age_days."""
    summary_path = Path(f"calibrations/{model_id}_calibration_summary.json")
    if not summary_path.exists():
        return True
        
    import time
    file_age_seconds = time.time() - summary_path.stat().st_mtime
    file_age_days = file_age_seconds / (24 * 3600)
    
    return file_age_days > max_age_days

def print_model_status(model_id: str, model_info: Dict, calibration_status: Dict, quality: Optional[Dict]):
    """Print detailed status for a model."""
    display_name = model_info.get('display_name', model_id)
    hf_repo = model_info.get('hf_repo', 'N/A')
    
    print(f"\n🤖 {display_name} ({model_id})")
    print(f"   📦 HF Repo: {hf_repo}")
    
    # Current failure law parameters
    failure_law = model_info.get('failure_law', {})
    current_lambda = failure_law.get('lambda', 'N/A')
    current_tau = failure_law.get('tau', 'N/A')
    print(f"   ⚙️  Current: λ={current_lambda}, τ={current_tau}")
    
    # Calibration file status
    files_exist = sum(calibration_status.values())
    total_files = len(calibration_status)
    completion = f"{files_exist}/{total_files}"
    
    if files_exist == total_files:
        status_emoji = "✅"
        status_text = "COMPLETE"
    elif files_exist > 0:
        status_emoji = "🔄" 
        status_text = "PARTIAL"
    else:
        status_emoji = "❌"
        status_text = "MISSING"
    
    print(f"   {status_emoji} Calibration: {status_text} ({completion})")
    
    # Detailed file status
    for file_type, exists in calibration_status.items():
        icon = "✓" if exists else "✗"
        print(f"      {icon} {file_type}")
    
    # Quality metrics if available
    if quality:
        print(f"   📊 Quality Metrics:")
        if quality.get('lambda') is not None:
            print(f"      Calibrated: λ={quality['lambda']:.3f}, τ={quality['tau']:.3f}")
        if quality.get('train_loss') is not None:
            print(f"      Train Loss: {quality['train_loss']:.4f}")
        if quality.get('val_roc_auc') is not None:
            print(f"      Val ROC-AUC: {quality['val_roc_auc']:.4f}")
        if quality.get('val_brier_score') is not None:
            print(f"      Val Brier: {quality['val_brier_score']:.4f}")
        if quality.get('val_ece') is not None:
            print(f"      Val ECE: {quality['val_ece']:.4f}")
    
    # Staleness check
    if is_calibration_stale(model_id):
        print(f"   ⚠️  Calibration may be stale (>30 days old)")

def generate_calibration_commands(models_to_calibrate: List[str]) -> List[str]:
    """Generate calibration commands for missing models."""
    commands = []
    
    for model_id in models_to_calibrate:
        # Base calibration command
        base_cmd = f"python scripts/calibrate_failure_law.py"
        
        # Add model-specific parameters
        cmd_parts = [
            base_cmd,
            f"--base http://127.0.0.1:3000/api/v1",
            f"--dataset default",  # Start with default, can be changed
            f"--max_samples 1000",
            f"--concurrency 4",
            f"--output_json calibrations/{model_id}_calibration_summary.json",
            f"--save_csv calibrations/{model_id}_scores.csv",
            f"--plot_dir calibrations/plots",
            f"--seed 42"
        ]
        
        # Add model-specific dataset if known
        if 'mistral' in model_id.lower():
            cmd_parts[3] = "--dataset truthfulqa"
        elif 'qwen' in model_id.lower():
            cmd_parts[3] = "--dataset halueval"
            cmd_parts.append("--halueval_task qa")
        
        commands.append(" ".join(cmd_parts))
    
    return commands

def main():
    print("🔍 Checking calibration status for all models...")
    
    # Load configuration
    config = load_models_config()
    models = config.get('models', [])
    
    if not models:
        print("❌ No models found in configuration")
        sys.exit(1)
    
    # Check each model
    models_needing_calibration = []
    models_needing_update = []
    
    for model in models:
        model_id = model['id']
        calibration_status = check_calibration_files(model_id)
        quality = get_calibration_quality(model_id)
        
        print_model_status(model_id, model, calibration_status, quality)
        
        # Determine if calibration is needed
        files_complete = all(calibration_status.values())
        is_stale = is_calibration_stale(model_id)
        
        if not files_complete:
            models_needing_calibration.append(model_id)
        elif is_stale:
            models_needing_update.append(model_id)
    
    # Summary
    total_models = len(models)
    calibrated_models = total_models - len(models_needing_calibration)
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total Models: {total_models}")
    print(f"   Fully Calibrated: {calibrated_models}")
    print(f"   Missing Calibration: {len(models_needing_calibration)}")
    print(f"   Stale Calibration: {len(models_needing_update)}")
    
    # Generate commands for missing calibrations
    if models_needing_calibration:
        print(f"\n🛠️  MODELS NEEDING CALIBRATION:")
        for model_id in models_needing_calibration:
            print(f"   • {model_id}")
        
        print(f"\n📋 SUGGESTED COMMANDS:")
        commands = generate_calibration_commands(models_needing_calibration)
        for i, cmd in enumerate(commands, 1):
            print(f"\n# {i}. {models_needing_calibration[i-1]}")
            print(cmd)
    
    if models_needing_update:
        print(f"\n⚠️  MODELS WITH STALE CALIBRATIONS:")
        for model_id in models_needing_update:
            print(f"   • {model_id}")
        print("   Consider re-running calibrations for these models.")
    
    if not models_needing_calibration and not models_needing_update:
        print(f"\n🎉 All models are fully calibrated and up-to-date!")
    
    # Create calibrations directory structure
    Path("calibrations/plots").mkdir(parents=True, exist_ok=True)
    
    return len(models_needing_calibration) + len(models_needing_update)

if __name__ == "__main__":
    sys.exit(main())