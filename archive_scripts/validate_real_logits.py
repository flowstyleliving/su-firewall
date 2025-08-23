#!/usr/bin/env python3
"""
Phase 1: Validate Real Model Logit Extraction
Critical test to ensure we get actual model probability distributions instead of word frequencies
"""

import sys
import os
sys.path.append('venv_eval/lib/python3.13/site-packages')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
import time
from typing import Dict, List, Tuple

def test_model_logit_extraction():
    """Test extraction of real model logits for hallucination vs truth examples"""
    
    print("🔥 PHASE 1: REAL MODEL LOGIT VALIDATION")
    print("=" * 60)
    
    # Test cases: clear truth vs hallucination examples
    test_cases = [
        {
            "name": "Geography_Truth",
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "expected_confidence": "high"
        },
        {
            "name": "Geography_Hallucination", 
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Lyon.",
            "expected_confidence": "low"
        },
        {
            "name": "Math_Truth",
            "prompt": "What is 2 + 2?",
            "output": "2 + 2 equals 4.",
            "expected_confidence": "high"
        },
        {
            "name": "Math_Hallucination",
            "prompt": "What is 2 + 2?", 
            "output": "2 + 2 equals 5.",
            "expected_confidence": "low"
        }
    ]
    
    print("📊 Loading model (this may take a moment)...")
    
    try:
        # Use a smaller model first to validate the approach
        model_name = "microsoft/DialoGPT-medium"  # From our config/models.json
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Loaded model: {model_name}")
        print(f"📊 Vocab size: {len(tokenizer.vocab)}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("💡 Falling back to distilgpt2 for testing...")
        
        # Fallback to a smaller model
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Loaded fallback model: {model_name}")
    
    # Test logit extraction for each case
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}/4: {test_case['name']}")
        
        try:
            # Prepare input
            full_text = f"{test_case['prompt']} {test_case['output']}"
            prompt_tokens = tokenizer.encode(test_case['prompt'])
            full_tokens = tokenizer.encode(full_text)
            output_tokens = full_tokens[len(prompt_tokens):]
            
            if len(output_tokens) == 0:
                print("⚠️  No output tokens found, skipping...")
                continue
            
            # Get model logits
            with torch.no_grad():
                # Prepare input tensors
                input_ids = torch.tensor([full_tokens[:-1]])  # Input without last token
                target_ids = torch.tensor([full_tokens[-1:]])  # Target: last token
                
                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last position logits
                
                # Convert to probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get probability of the actual next token
                target_token_id = full_tokens[-1]
                target_probability = probabilities[target_token_id].item()
                
                # Get top-5 most likely tokens for comparison
                top_probs, top_indices = torch.topk(probabilities, 5)
                
                result = {
                    'test_case': test_case['name'],
                    'expected_confidence': test_case['expected_confidence'],
                    'target_token': tokenizer.decode([target_token_id]),
                    'target_probability': target_probability,
                    'rank_of_target': (probabilities >= target_probability).sum().item(),
                    'entropy': -torch.sum(probabilities * torch.log(probabilities + 1e-12)).item(),
                    'top_5_tokens': [tokenizer.decode([idx.item()]) for idx in top_indices],
                    'top_5_probs': [prob.item() for prob in top_probs]
                }
                
                results.append(result)
                
                print(f"   🎯 Target token: '{result['target_token']}'")
                print(f"   📊 Probability: {result['target_probability']:.4f}")
                print(f"   📈 Rank: {result['rank_of_target']}")
                print(f"   🧮 Entropy: {result['entropy']:.3f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'test_case': test_case['name'],
                'error': str(e)
            })
    
    # Analyze results
    print(f"\n🎯 LOGIT EXTRACTION VALIDATION RESULTS")
    print("=" * 60)
    
    successful_tests = [r for r in results if 'target_probability' in r]
    
    if len(successful_tests) >= 2:
        truth_cases = [r for r in successful_tests if 'Truth' in r['test_case']]
        halluc_cases = [r for r in successful_tests if 'Hallucination' in r['test_case']]
        
        if truth_cases and halluc_cases:
            truth_avg_prob = np.mean([r['target_probability'] for r in truth_cases])
            halluc_avg_prob = np.mean([r['target_probability'] for r in halluc_cases])
            
            print(f"✅ Truth cases average probability: {truth_avg_prob:.4f}")
            print(f"✅ Hallucination cases average probability: {halluc_avg_prob:.4f}")
            print(f"📊 Discrimination ratio: {truth_avg_prob / halluc_avg_prob:.2f}x")
            
            if truth_avg_prob > halluc_avg_prob * 1.5:
                print("🎉 SUCCESS: Model shows clear discrimination between truth and hallucination!")
                validation_status = "PASSED"
            else:
                print("⚠️  UNCLEAR: Limited discrimination observed")
                validation_status = "MARGINAL"
        else:
            print("⚠️  Insufficient data for comparison")
            validation_status = "INSUFFICIENT_DATA"
    else:
        print("❌ Most tests failed - unable to validate logit extraction")
        validation_status = "FAILED"
    
    # Save detailed results
    evaluation_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "validation_status": validation_status,
        "successful_tests": len(successful_tests),
        "total_tests": len(test_cases),
        "detailed_results": results
    }
    
    with open('logit_validation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n📁 Results saved to: logit_validation_results.json")
    
    return validation_status == "PASSED", evaluation_results

def validate_ensemble_requirements():
    """Validate we can compute distributions suitable for ensemble methods"""
    
    print(f"\n🧮 ENSEMBLE REQUIREMENTS VALIDATION")
    print("-" * 40)
    
    try:
        # Test creating probability distributions like our ensemble methods expect
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        test_text = "The capital of France is Paris"
        tokens = tokenizer.encode(test_text)
        
        with torch.no_grad():
            input_ids = torch.tensor([tokens])
            outputs = model(input_ids)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Get final token logits
            final_logits = logits[-1, :]
            probabilities = torch.softmax(final_logits, dim=-1)
            
            # Convert to numpy for ensemble processing
            prob_dist = probabilities.cpu().numpy().astype(np.float64)
            
            print(f"✅ Generated probability distribution")
            print(f"   📊 Shape: {prob_dist.shape}")  
            print(f"   🎯 Sum: {prob_dist.sum():.6f}")
            print(f"   📈 Max prob: {prob_dist.max():.4f}")
            print(f"   📉 Min prob: {prob_dist.min():.8f}")
            print(f"   🧮 Entropy: {-np.sum(prob_dist * np.log(prob_dist + 1e-12)):.3f}")
            
            # Test Jensen-Shannon divergence calculation (like our ensemble)
            # Create a second distribution for comparison
            shifted_logits = final_logits + torch.randn_like(final_logits) * 0.1
            shifted_probs = torch.softmax(shifted_logits, dim=-1).cpu().numpy().astype(np.float64)
            
            # Jensen-Shannon divergence
            M = 0.5 * (prob_dist + shifted_probs)
            js_div = 0.5 * np.sum(prob_dist * np.log(prob_dist / M)) + 0.5 * np.sum(shifted_probs * np.log(shifted_probs / M))
            
            print(f"   🔄 Jensen-Shannon divergence test: {js_div:.4f}")
            
            if js_div > 0 and js_div < 1:
                print("✅ ENSEMBLE COMPATIBILITY: PASSED")
                return True
            else:
                print("⚠️  JS divergence out of expected range")
                return False
            
    except Exception as e:
        print(f"❌ Ensemble validation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING PHASE 1: REAL MODEL LOGIT VALIDATION")
    print("🎯 Goal: Validate we can extract meaningful probability distributions")
    print("🧮 This replaces word frequency analysis with actual model confidence")
    
    # Run validation
    logit_success, logit_results = test_model_logit_extraction()
    ensemble_success = validate_ensemble_requirements()
    
    print(f"\n" + "=" * 60)
    print("🏆 PHASE 1 SUMMARY")
    print("=" * 60)
    
    if logit_success and ensemble_success:
        print("🎉 PHASE 1: SUCCESS!")
        print("   ✅ Real model logit extraction validated")  
        print("   ✅ Ensemble method compatibility confirmed")
        print("   ✅ Ready to proceed to Phase 2: Dataset preparation")
        exit_code = 0
    else:
        print("⚠️  PHASE 1: ISSUES IDENTIFIED")
        if not logit_success:
            print("   ❌ Logit extraction needs improvement")
        if not ensemble_success:
            print("   ❌ Ensemble compatibility issues")
        print("   🔧 Addressing these issues before proceeding...")
        exit_code = 1
    
    exit(exit_code)