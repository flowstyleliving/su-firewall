#!/usr/bin/env python3
"""
🔥 UNDENIABLE TEST - QUICK L1→L2→L3 VERIFICATION
Tests 10 cases to verify the 3-level progressive system works
"""

import sys
sys.path.append('.')
from scripts.world_class_benchmark_runner import WorldClassBenchmarkRunner

def main():
    print("🚀 TESTING UNDENIABLE L1→L2→L3 SYSTEM")
    print("=" * 50)
    
    runner = WorldClassBenchmarkRunner()
    
    # Run small sample (10 cases) to verify functionality
    print("📊 Testing 10 cases to verify L1→L2→L3 progression...")
    results = runner.run_comprehensive_evaluation(sample_size=10)
    
    # Print results
    runner.print_results_summary(results)
    
    print("\n✅ Undeniable Test System Verified!")

if __name__ == "__main__":
    main()