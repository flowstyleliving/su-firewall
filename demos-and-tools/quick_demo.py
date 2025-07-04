#!/usr/bin/env python3
"""
Quick Demo: Semantic Uncertainty Engine
======================================

Demonstrates the terminal-first approach using the semantic uncertainty equation:
ℏₛ(C) = √(Δμ × Δσ)

Usage:
    python demos-and-tools/quick_demo.py           # Terminal display only
    python demos-and-tools/quick_demo.py --save    # Also save to files
"""

import asyncio
import sys
import os
from pathlib import Path

# Add evaluation-frameworks to path
sys.path.append(str(Path(__file__).parent.parent / "evaluation-frameworks"))

from diagnostic_suite_simplified import SemanticDiagnosticSuite

async def main():
    """Quick demonstration of semantic uncertainty analysis"""
    print("🚀 SEMANTIC UNCERTAINTY QUICK DEMO")
    print("=" * 50)
    print("🧮 Equation: ℏₛ(C) = √(Δμ × Δσ)")
    print("📊 Δμ: Precision | 🎲 Δσ: Flexibility | ⚡ ℏₛ: Uncertainty")
    print("=" * 50)
    
    # Check for save flag
    save_results = "--save" in sys.argv or "-s" in sys.argv
    
    if save_results:
        print("💾 Demo results will be saved to data-and-results/diagnostic_outputs/")
    else:
        print("📺 Demo results will be displayed in terminal only")
        print("💡 Use --save flag to save results to files")
    
    print("\n🔬 Running diagnostic suite with reduced model set...")
    
    # Initialize with demo-focused settings
    suite = SemanticDiagnosticSuite(save_results=save_results)
    
    # Override models for quick demo
    suite.models = ['gpt4', 'claude3', 'gemini']  # Smaller set for demo
    
    # Run the diagnostic
    results = await suite.run_full_diagnostic()
    
    print("\n🎯 QUICK DEMO COMPLETE!")
    print("="*50)
    if save_results:
        print("📊 Launch dashboard: streamlit run demos-and-tools/dashboard.py")
        print("📁 View files in: data-and-results/diagnostic_outputs/")
    else:
        print("📊 Run with --save to persist results")
        print("🔄 Run again anytime for fresh analysis")
    
    print("\n💡 Next Steps:")
    print("   🧪 Full suite: python evaluation-frameworks/diagnostic_suite_simplified.py")
    print("   📈 LLM eval: python evaluation-frameworks/llm_evaluation.py")
    print("   📊 Dashboard: streamlit run demos-and-tools/dashboard.py")

if __name__ == "__main__":
    asyncio.run(main())
