#!/usr/bin/env python3
"""
Monitor the progress of large-scale evaluations and provide real-time status.
"""
import time
import json
import os
from pathlib import Path

def monitor_evaluations():
    """Monitor both HaluEval and TruthfulQA evaluations"""
    
    files_to_monitor = [
        "halueval_large_scale_real_logits.json",
        "truthfulqa_large_scale_real_logits.json"
    ]
    
    print("🔍 Monitoring Large-Scale Evaluation Progress")
    print("=" * 60)
    
    while True:
        all_complete = True
        
        for filename in files_to_monitor:
            filepath = Path(filename)
            
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    if 'summary' in data:
                        # Complete evaluation
                        summary = data['summary']
                        print(f"✅ {filename}: COMPLETE")
                        print(f"   📊 Samples: {summary.get('total_samples', 'N/A')}")
                        print(f"   📈 ROC-AUC: {summary.get('roc_auc', 'N/A'):.4f}")
                        print(f"   📉 Brier Score: {summary.get('brier_score', 'N/A'):.4f}")
                        if 'processing_time_seconds' in summary:
                            print(f"   ⏱️  Time: {summary['processing_time_seconds']:.1f}s")
                        print()
                    elif 'error' in data:
                        print(f"❌ {filename}: ERROR - {data['error']}")
                        print()
                    else:
                        # Partial evaluation
                        print(f"🔄 {filename}: IN PROGRESS (partial data)")
                        print()
                        all_complete = False
                        
                except (json.JSONDecodeError, KeyError):
                    # File exists but incomplete/corrupted
                    file_size = filepath.stat().st_size
                    print(f"🔄 {filename}: WRITING ({file_size:,} bytes)")
                    print()
                    all_complete = False
                    
            else:
                print(f"⏳ {filename}: PENDING")
                print()
                all_complete = False
        
        if all_complete:
            print("🎉 All evaluations completed!")
            break
            
        print(f"📅 {time.strftime('%H:%M:%S')} - Checking again in 30 seconds...")
        print("-" * 60)
        time.sleep(30)

if __name__ == "__main__":
    monitor_evaluations()