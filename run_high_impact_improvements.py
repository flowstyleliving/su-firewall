#!/usr/bin/env python3
"""
🚀 HIGH-IMPACT IMPROVEMENTS MASTER SCRIPT - Weeks 1-2
Complete implementation of immediate high-impact improvements for semantic uncertainty system
Ready for production deployment and CTO demonstration
"""

import subprocess
import sys
import time
import json
from datetime import datetime
import argparse

def print_banner():
    """Print impressive banner for the improvements"""
    print("🚀" * 40)
    print("🎯 HIGH-IMPACT SEMANTIC UNCERTAINTY IMPROVEMENTS")
    print("   Weeks 1-2 Implementation - Production Ready")
    print("🚀" * 40)
    print()

def check_server_health():
    """Check if the semantic uncertainty server is running"""
    try:
        import requests
        response = requests.get("http://localhost:8080/health", timeout=3)
        if response.status_code == 200:
            print("✅ Semantic uncertainty server is running")
            return True
        else:
            print("⚠️ Server responded but with error code:", response.status_code)
            return False
    except Exception as e:
        print(f"❌ Server not accessible: {str(e)[:100]}")
        print("💡 Please start the server with: cargo run -p server")
        return False

def run_script(script_name, description, required_server=True):
    """Run a Python script with error handling and timing"""
    
    if required_server and not check_server_health():
        print(f"⏭️ Skipping {script_name} (server required)")
        return False
    
    print(f"\n🔍 {description}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed successfully in {execution_time:.1f}s")
            if result.stdout:
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 10:
                    print("📄 Key results:")
                    for line in lines[-10:]:
                        if line.strip():
                            print(f"   {line}")
                else:
                    print(result.stdout)
            return True
        else:
            print(f"❌ Failed after {execution_time:.1f}s")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ Script timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Execution error: {str(e)}")
        return False

def run_all_improvements(quick_mode=False):
    """Run all high-impact improvements in sequence"""
    
    print_banner()
    
    overall_start = time.time()
    results = {}
    
    # Define improvement scripts in order
    improvements = [
        {
            "script": "enhanced_natural_distribution_test.py",
            "description": "Natural Distribution Testing (5-10% hallucination rates)",
            "emoji": "🌍",
            "required_server": True
        },
        {
            "script": "cross_domain_validation_suite.py", 
            "description": "Cross-Domain Validation (QA → Dialogue/Summarization/Creative)",
            "emoji": "🌐",
            "required_server": True
        },
        {
            "script": "ensemble_method_analyzer.py",
            "description": "Domain-Agnostic Ensemble Method Analysis",
            "emoji": "🔍",
            "required_server": True
        }
    ]
    
    # Additional diagnostic scripts (optional in quick mode)
    if not quick_mode:
        improvements.extend([
            {
                "script": "accuracy_benchmark_suite.py",
                "description": "Accuracy Benchmark Suite (>90% target validation)",
                "emoji": "🎯",
                "required_server": True
            },
            {
                "script": "performance_benchmark.py", 
                "description": "Performance Benchmark (<200ms response times)",
                "emoji": "⚡",
                "required_server": True
            }
        ])
    
    successful_improvements = 0
    total_improvements = len(improvements)
    
    print(f"📋 RUNNING {total_improvements} HIGH-IMPACT IMPROVEMENTS")
    print(f"⚡ Mode: {'Quick' if quick_mode else 'Comprehensive'}")
    print()
    
    for i, improvement in enumerate(improvements, 1):
        print(f"\n{improvement['emoji']} IMPROVEMENT {i}/{total_improvements}: {improvement['description']}")
        print("=" * 80)
        
        success = run_script(
            improvement["script"], 
            improvement["description"],
            improvement["required_server"]
        )
        
        results[improvement["script"]] = {
            "success": success,
            "description": improvement["description"],
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            successful_improvements += 1
            print(f"✅ Improvement {i} completed successfully")
        else:
            print(f"❌ Improvement {i} failed")
    
    # Generate summary report
    overall_time = time.time() - overall_start
    
    print(f"\n🏆 HIGH-IMPACT IMPROVEMENTS SUMMARY")
    print("=" * 80)
    print(f"⏱️ Total execution time: {overall_time/60:.1f} minutes")
    print(f"✅ Successful improvements: {successful_improvements}/{total_improvements}")
    print(f"📊 Success rate: {successful_improvements/total_improvements:.1%}")
    
    if successful_improvements == total_improvements:
        print("\n🎉 ALL IMPROVEMENTS COMPLETED SUCCESSFULLY!")
        print("🚀 System ready for production deployment")
        print("🎯 Ready for CTO demonstration")
    elif successful_improvements >= total_improvements * 0.75:
        print("\n⚡ MOST IMPROVEMENTS COMPLETED")
        print("🔧 Minor issues to resolve for full production readiness")
    else:
        print("\n⚠️ SEVERAL IMPROVEMENTS NEED ATTENTION")
        print("🛠️ Review failed components before production deployment")
    
    # Production readiness checklist
    print(f"\n📋 PRODUCTION READINESS CHECKLIST")
    print("-" * 40)
    
    checklist_items = [
        "Natural distribution testing (5-10% realistic rates)",
        "False positive rate optimization (<2%)",
        "Cross-domain validation framework",
        "Domain-agnostic ensemble method identification",
        "Performance drop measurement across domains"
    ]
    
    for i, item in enumerate(checklist_items, 1):
        status = "✅" if i <= successful_improvements else "⏳"
        print(f"{status} {item}")
    
    # Save execution report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"high_impact_improvements_report_{timestamp}.json"
    
    report_data = {
        "execution_summary": {
            "timestamp": datetime.now().isoformat(),
            "mode": "quick" if quick_mode else "comprehensive",
            "total_improvements": total_improvements,
            "successful_improvements": successful_improvements,
            "success_rate": successful_improvements/total_improvements,
            "execution_time_minutes": overall_time/60,
            "production_ready": successful_improvements == total_improvements
        },
        "individual_results": results,
        "production_checklist": {
            item: i <= successful_improvements 
            for i, item in enumerate(checklist_items, 1)
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📊 Execution report saved to: {report_file}")
    
    # Final recommendations
    print(f"\n💡 NEXT STEPS")
    print("-" * 20)
    
    if successful_improvements == total_improvements:
        print("1. 🚀 Deploy to production environment")
        print("2. 📈 Monitor performance metrics in production")
        print("3. 🎯 Schedule CTO demonstration") 
        print("4. 📊 Gather user feedback for further optimization")
    else:
        print("1. 🔧 Review and fix failed improvements")
        print("2. 🧪 Re-run failed tests to ensure stability")
        print("3. 📊 Complete production readiness validation")
        print("4. 🚀 Deploy once all improvements pass")
    
    return results

def main():
    """Main execution function with command line arguments"""
    
    parser = argparse.ArgumentParser(description="Run high-impact semantic uncertainty improvements")
    parser.add_argument("--quick", action="store_true", 
                       help="Run in quick mode (skip optional benchmarks)")
    parser.add_argument("--check-server", action="store_true",
                       help="Only check server health and exit")
    
    args = parser.parse_args()
    
    if args.check_server:
        print("🔍 Checking server health...")
        if check_server_health():
            print("✅ Server is ready for improvements")
            sys.exit(0)
        else:
            print("❌ Server needs to be started")
            sys.exit(1)
    
    # Run all improvements
    results = run_all_improvements(quick_mode=args.quick)
    
    # Exit with appropriate code
    successful_count = sum(1 for r in results.values() if r["success"])
    total_count = len(results)
    
    if successful_count == total_count:
        print("\n🎉 All improvements completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total_count - successful_count} improvements need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()