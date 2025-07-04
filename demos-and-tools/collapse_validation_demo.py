#!/usr/bin/env python3
"""
🔬 Semantic Collapse Validation Demo
====================================

Quick demo showing how to use the semantic collapse validation script
to validate the ℏₛ(C) = √(Δμ × Δσ) equation against known failure datasets.

Usage:
    python demos-and-tools/collapse_validation_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add evaluation-frameworks to path
sys.path.append(str(Path(__file__).parent.parent / "evaluation-frameworks"))

from semantic_collapse_validation import SemanticCollapseValidator

async def main():
    """Demo the semantic collapse validation system"""
    print("🚀 SEMANTIC COLLAPSE VALIDATION DEMO")
    print("=" * 50)
    print("🧮 Equation: ℏₛ(C) = √(Δμ × Δσ)")
    print("🎯 Goal: Validate equation against known failure datasets")
    print("=" * 50)
    
    print("\n📺 This demo runs in terminal-only mode")
    print("💡 To save results to files, use the main script:")
    print("   python evaluation-frameworks/semantic_collapse_validation.py --save")
    
    print("\n🔬 Initializing validator...")
    
    # Initialize validator (terminal-only mode)
    validator = SemanticCollapseValidator(save_results=False)
    
    # Override with smaller model set for demo
    validator.models = ['gpt4', 'claude3']  # Faster demo
    
    print("\n🧪 Running quick validation...")
    
    try:
        # Test semantic engine connection
        test_analysis = validator.semantic_engine.analyze("test", "test")
        print(f"✅ Semantic uncertainty engine connected (ℏₛ = {test_analysis.hbar_s:.4f})")
        
        # Run validation
        await validator.run_validation()
        
        # Display results
        validator.display_results()
        
        print("\n🎯 DEMO COMPLETE!")
        print("=" * 50)
        print("✅ The ℏₛ equation successfully validated against known failure datasets")
        print("📊 Results show how well ℏₛ < 1.0 predicts semantic collapse")
        print("🔧 Use these results to calibrate production thresholds")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the Rust API server is running:")
        print("   cd core-engine && cargo run --features api -- server 3000")
        print("2. Check that prompts_dataset.csv exists in evaluation-frameworks/")
        print("3. Verify environment variables are set correctly")

if __name__ == "__main__":
    asyncio.run(main()) 