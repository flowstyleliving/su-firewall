#!/usr/bin/env python3
"""
Quick Phase 1: Validate Real Model Logit Extraction
Fast validation using lightweight model
"""

import sys
import os
sys.path.append('venv_eval/lib/python3.13/site-packages')

import torch
import numpy as np
import json
import time

def quick_validation():
    """Quick test without downloading large models"""
    
    print("🔥 PHASE 1: QUICK LOGIT VALIDATION")
    print("=" * 50)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print("📊 Loading GPT-2 (lightweight model)...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Vocab size: {model.config.vocab_size}")
        
        # Test cases
        test_cases = [
            ("Paris is the capital of", "France", "high_confidence"),
            ("Lyon is the capital of", "France", "low_confidence"),
            ("2 + 2 equals", "4", "high_confidence"), 
            ("2 + 2 equals", "5", "low_confidence")
        ]
        
        results = []
        
        for i, (context, completion, expected) in enumerate(test_cases):
            print(f"\n🧪 Test {i+1}: {context} -> {completion}")
            
            # Tokenize
            context_tokens = tokenizer.encode(context)
            completion_token = tokenizer.encode(' ' + completion)[0]  # First token of completion
            
            # Get model prediction
            with torch.no_grad():
                input_ids = torch.tensor([context_tokens])
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last position logits
                probs = torch.softmax(logits, dim=-1)
                
                target_prob = probs[completion_token].item()
                target_rank = (probs >= target_prob).sum().item()
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                
                result = {
                    'context': context,
                    'completion': completion,
                    'expected': expected,
                    'probability': float(target_prob),
                    'rank': int(target_rank),
                    'entropy': float(entropy)
                }
                results.append(result)
                
                print(f"   📊 Probability: {target_prob:.4f}")
                print(f"   🎯 Rank: {target_rank}/{model.config.vocab_size}")
                print(f"   🧮 Entropy: {entropy:.3f}")
        
        # Analyze discrimination
        high_conf_probs = [r['probability'] for r in results if r['expected'] == 'high_confidence']
        low_conf_probs = [r['probability'] for r in results if r['expected'] == 'low_confidence']
        
        if high_conf_probs and low_conf_probs:
            high_avg = np.mean(high_conf_probs)
            low_avg = np.mean(low_conf_probs)
            discrimination_ratio = high_avg / low_avg
            
            print(f"\n🎯 DISCRIMINATION ANALYSIS:")
            print(f"   ✅ High confidence average: {high_avg:.4f}")
            print(f"   ❌ Low confidence average: {low_avg:.4f}")
            print(f"   📊 Discrimination ratio: {discrimination_ratio:.2f}x")
            
            # Test ensemble-style computation
            print(f"\n🧮 ENSEMBLE COMPATIBILITY TEST:")
            
            # Create two probability distributions
            input_ids = torch.tensor([tokenizer.encode("The capital of France is Paris")])
            with torch.no_grad():
                outputs = model(input_ids)
                logits1 = outputs.logits[0, -1, :]
                probs1 = torch.softmax(logits1, dim=-1).detach().numpy()
                
                # Slightly perturbed version
                logits2 = logits1 + torch.randn_like(logits1) * 0.1
                probs2 = torch.softmax(logits2, dim=-1).detach().numpy()
            
            # Jensen-Shannon divergence (like our ensemble)
            M = 0.5 * (probs1 + probs2)
            js_div = 0.5 * np.sum(probs1 * np.log(probs1 / (M + 1e-12))) + 0.5 * np.sum(probs2 * np.log(probs2 / (M + 1e-12)))
            
            # KL divergence 
            kl_div = np.sum(probs1 * np.log(probs1 / (probs2 + 1e-12)))
            
            print(f"   📊 Distribution shape: {probs1.shape}")
            print(f"   ✅ JS divergence: {js_div:.4f}")
            print(f"   ✅ KL divergence: {kl_div:.4f}")
            print(f"   🧮 Combined (√(JS*KL)): {np.sqrt(js_div * kl_div):.4f}")
            
            # Success criteria
            if discrimination_ratio > 1.2 and js_div > 0 and kl_div > 0:
                status = "SUCCESS"
                print(f"\n🎉 PHASE 1: SUCCESS!")
                print("   ✅ Model shows discrimination between high/low confidence")
                print("   ✅ Ensemble mathematics work with real distributions")
                print("   ✅ Ready for integration with Rust ensemble system")
            else:
                status = "PARTIAL"
                print(f"\n⚠️  PHASE 1: PARTIAL SUCCESS")
                print("   ✅ Basic functionality works")
                print("   ⚠️  Limited discrimination observed")
        else:
            status = "FAILED"
            print(f"\n❌ PHASE 1: FAILED - Insufficient results")
        
        # Save results
        validation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "model_used": "gpt2",
            "vocab_size": model.config.vocab_size,
            "test_results": results,
            "discrimination_ratio": float(discrimination_ratio) if 'discrimination_ratio' in locals() else 0.0,
            "ensemble_metrics": {
                "js_divergence": float(js_div) if 'js_div' in locals() else 0.0,
                "kl_divergence": float(kl_div) if 'kl_div' in locals() else 0.0,
            }
        }
        
        with open('quick_logit_validation.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\n📁 Results saved: quick_logit_validation.json")
        return status == "SUCCESS", validation_results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, {"error": str(e)}

if __name__ == "__main__":
    success, results = quick_validation()
    exit(0 if success else 1)