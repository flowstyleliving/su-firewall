#!/usr/bin/env python3
"""
🎯 TRUTHFULQA GENERATION EVALUATION
Test real-time hallucination detection by generating responses to TruthfulQA questions
"""

import sys
sys.path.append('.')
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from scripts.world_class_benchmark_runner import WorldClassBenchmarkRunner
import torch

class TruthfulQAGenerationEvaluator:
    """Evaluate hallucination detection on generated responses to TruthfulQA questions."""
    
    def __init__(self):
        self.runner = WorldClassBenchmarkRunner()
        self.load_truthfulqa_questions()
        self.setup_generation_model()
        
    def load_truthfulqa_questions(self):
        """Load TruthfulQA questions for generation testing."""
        truthfulqa_file = Path("authentic_datasets/truthfulqa_data.json")
        with open(truthfulqa_file, 'r') as f:
            data = json.load(f)
        
        self.questions = data["validation"]
        print(f"✅ Loaded {len(self.questions)} TruthfulQA questions for generation")
    
    def setup_generation_model(self):
        """Setup model for text generation."""
        # We'll use the same GPT-2 model that's already loaded for consistency
        print(f"🤖 Using GPT-2 for generation and uncertainty analysis")
        print(f"   Model: {self.runner.model}")
        print(f"   Tokenizer: {self.runner.tokenizer}")
    
    def generate_response(self, question: str, max_length: int = 100) -> Optional[str]:
        """Generate a response to a question using GPT-2."""
        
        try:
            # Tokenize the question with attention mask
            encoded = self.runner.tokenizer(question, return_tensors='pt', padding=True)
            inputs = encoded['input_ids']
            attention_mask = encoded.get('attention_mask', torch.ones_like(inputs))
            
            # Generate response
            with torch.no_grad():
                # Use sampling for more natural responses that might contain hallucinations
                outputs = self.runner.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.8,  # Some randomness to allow hallucinations
                    top_p=0.9,        # Nucleus sampling
                    do_sample=True,   # Enable sampling
                    pad_token_id=self.runner.tokenizer.eos_token_id,
                    eos_token_id=self.runner.tokenizer.eos_token_id
                )
            
            # Decode the generated response (skip the input tokens)
            generated_tokens = outputs[0][inputs.shape[1]:]
            generated_text = self.runner.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the response
            generated_text = generated_text.strip()
            
            # Stop at sentence boundaries for cleaner responses
            if '.' in generated_text:
                sentences = generated_text.split('.')
                generated_text = sentences[0] + '.'
            
            return generated_text if generated_text else None
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return None
    
    def evaluate_generated_response(self, question: str, generated_response: str, 
                                  correct_answers: List[str], incorrect_answers: List[str]) -> Dict:
        """Evaluate if a generated response is factually correct or a hallucination."""
        
        # Check if generated response matches correct patterns
        is_factually_correct = self.check_factual_accuracy(generated_response, correct_answers, incorrect_answers)
        
        # Get uncertainty metrics from our system
        response_logits = self.runner.get_real_logits(generated_response)
        if not response_logits:
            return None
        
        # Calculate ℏₛ combinations
        uncertainty_combinations = self.runner.calculate_semantic_uncertainty_combinations(response_logits)
        
        # Get model parameters for P(fail) calculation
        model = self.runner.models_config["models"][0]  # Mixtral
        lambda_param = model["failure_law"]["lambda"]
        tau_param = model["failure_law"]["tau"]
        
        # Calculate uncertainty scores
        results = {
            "question": question,
            "generated_response": generated_response,
            "is_factually_correct": is_factually_correct,
            "response_entropy": response_logits["entropy"],
            "uncertainty_combinations": uncertainty_combinations,
            "uncertainty_scores": {},
            "pfail_scores": {}
        }
        
        # Calculate uncertainty and P(fail) for each combination
        for combo_name, hbar_value in uncertainty_combinations.items():
            if combo_name != "components":
                results["uncertainty_scores"][combo_name] = hbar_value
                
                # Calculate P(fail)
                pfail = 1 / (1 + np.exp(-lambda_param * (hbar_value - tau_param)))
                results["pfail_scores"][combo_name] = pfail
        
        return results
    
    def check_factual_accuracy(self, generated_response: str, correct_answers: List[str], 
                             incorrect_answers: List[str]) -> bool:
        """Check if generated response aligns with correct answers or incorrect answers."""
        
        generated_lower = generated_response.lower().strip()
        
        # Check for key phrases in correct answers
        correct_score = 0
        for correct_answer in correct_answers:
            if isinstance(correct_answer, str):
                correct_lower = correct_answer.lower()
                # Look for key words/phrases
                words = correct_lower.split()
                for word in words:
                    if len(word) > 3 and word in generated_lower:  # Skip short words
                        correct_score += 1
        
        # Check for key phrases in incorrect answers  
        incorrect_score = 0
        for incorrect_answer in incorrect_answers:
            if isinstance(incorrect_answer, str):
                incorrect_lower = incorrect_answer.lower()
                words = incorrect_lower.split()
                for word in words:
                    if len(word) > 3 and word in generated_lower:
                        incorrect_score += 1
        
        # Simple heuristic: more overlap with correct answers = factually correct
        return correct_score > incorrect_score
    
    def run_generation_evaluation(self, num_questions: int = 50) -> Dict:
        """Run complete generation-based evaluation."""
        
        print(f"🚀 TRUTHFULQA GENERATION EVALUATION")
        print(f"Testing {num_questions} questions with real-time generation")
        print("=" * 70)
        
        results = []
        factual_correct_count = 0
        factual_incorrect_count = 0
        
        # Uncertainty detection results
        detection_results = {combo: {"correct_high": 0, "incorrect_high": 0, "total_correct": 0, "total_incorrect": 0} 
                           for combo in ["basic", "hash_fisher", "hash_hash", "entropy_fisher", "hash_variance", "entropy_hash"]}
        
        for i in range(min(num_questions, len(self.questions))):
            question_data = self.questions[i]
            question = question_data["Question"]
            
            print(f"\n📝 Question {i+1}/{num_questions}")
            print(f"Q: {question[:80]}...")
            
            # Generate response
            generated_response = self.generate_response(question, max_length=50)
            if not generated_response:
                print("❌ Generation failed")
                continue
            
            print(f"A: {generated_response}")
            
            # Evaluate the generated response
            evaluation = self.evaluate_generated_response(
                question, 
                generated_response,
                question_data.get("Correct Answers", "").split("; ") if isinstance(question_data.get("Correct Answers"), str) else [],
                question_data.get("Incorrect Answers", "").split("; ") if isinstance(question_data.get("Incorrect Answers"), str) else []
            )
            
            if not evaluation:
                continue
            
            # Track factual accuracy
            if evaluation["is_factually_correct"]:
                factual_correct_count += 1
                print("✅ Factually correct")
            else:
                factual_incorrect_count += 1
                print("❌ Contains hallucination/error")
            
            # Track uncertainty detection performance
            for combo_name, uncertainty_value in evaluation["uncertainty_scores"].items():
                if combo_name in detection_results:
                    if evaluation["is_factually_correct"]:
                        detection_results[combo_name]["total_correct"] += 1
                        # For correct responses, we want LOW uncertainty (uncertainty < median)
                    else:
                        detection_results[combo_name]["total_incorrect"] += 1
                        # For incorrect responses, we want HIGH uncertainty (uncertainty > median)
            
            results.append(evaluation)
            
            # Show top uncertainty scores
            sorted_uncertainties = sorted(evaluation["uncertainty_scores"].items(), 
                                       key=lambda x: x[1], reverse=True)
            print(f"🔍 Top uncertainty: {sorted_uncertainties[0][0]} = {sorted_uncertainties[0][1]:.4f}")
        
        # Calculate detection accuracy
        print(f"\n🏆 GENERATION EVALUATION RESULTS")
        print("=" * 70)
        
        total_responses = len(results)
        if total_responses == 0:
            print("❌ No valid responses generated")
            return {}
        
        print(f"📊 Response Quality:")
        print(f"   Factually Correct: {factual_correct_count}/{total_responses} = {factual_correct_count/total_responses:.1%}")
        print(f"   Contains Hallucinations: {factual_incorrect_count}/{total_responses} = {factual_incorrect_count/total_responses:.1%}")
        
        # Calculate uncertainty thresholds and detection accuracy
        print(f"\n🎯 HALLUCINATION DETECTION PERFORMANCE:")
        
        detection_accuracies = {}
        
        for combo_name, combo_data in detection_results.items():
            if combo_data["total_correct"] + combo_data["total_incorrect"] == 0:
                continue
            
            # Get all uncertainty values for this combination
            correct_uncertainties = [r["uncertainty_scores"][combo_name] for r in results if r["is_factually_correct"] and combo_name in r["uncertainty_scores"]]
            incorrect_uncertainties = [r["uncertainty_scores"][combo_name] for r in results if not r["is_factually_correct"] and combo_name in r["uncertainty_scores"]]
            
            if not correct_uncertainties or not incorrect_uncertainties:
                continue
            
            # Simple threshold: median of all uncertainties
            all_uncertainties = correct_uncertainties + incorrect_uncertainties
            threshold = np.median(all_uncertainties)
            
            # Count correct classifications
            correct_classified = sum(1 for u in correct_uncertainties if u <= threshold)  # Correct should have low uncertainty
            incorrect_classified = sum(1 for u in incorrect_uncertainties if u > threshold)  # Incorrect should have high uncertainty
            
            total_classifications = len(correct_uncertainties) + len(incorrect_uncertainties)
            accuracy = (correct_classified + incorrect_classified) / total_classifications if total_classifications > 0 else 0
            
            detection_accuracies[combo_name] = accuracy
            
            emoji = "🎯" if accuracy >= 0.80 else "📈" if accuracy >= 0.60 else "📉"
            print(f"   {combo_name:15s}: {accuracy:.1%} {emoji} (threshold: {threshold:.4f})")
        
        # Find best detection method
        if detection_accuracies:
            best_combo = max(detection_accuracies.keys(), key=lambda x: detection_accuracies[x])
            best_accuracy = detection_accuracies[best_combo]
            
            print(f"\n🏆 BEST DETECTION METHOD: {best_combo} at {best_accuracy:.1%}")
            
            if best_accuracy >= 0.80:
                print("🎉 EXCELLENT: 80%+ hallucination detection achieved!")
            elif best_accuracy >= 0.60:
                print("📈 GOOD: 60%+ hallucination detection on generated responses")
            else:
                print("⚠️ NEEDS IMPROVEMENT: Detection accuracy below 60%")
        
        return {
            "total_responses": total_responses,
            "factual_accuracy_rate": factual_correct_count / total_responses,
            "hallucination_rate": factual_incorrect_count / total_responses,
            "detection_accuracies": detection_accuracies,
            "best_detection_method": best_combo if detection_accuracies else None,
            "best_detection_accuracy": best_accuracy if detection_accuracies else 0,
            "detailed_results": results
        }

def main():
    """Run TruthfulQA generation evaluation."""
    
    print("🎯 TRUTHFULQA REAL-TIME GENERATION EVALUATION")
    print("Testing hallucination detection on generated responses")
    print("=" * 70)
    
    try:
        evaluator = TruthfulQAGenerationEvaluator()
        results = evaluator.run_generation_evaluation(num_questions=30)
        
        if not results:
            print("❌ Evaluation failed")
            return
        
        # Save results
        output_file = "truthfulqa_generation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")
        
        # Final summary
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"Model generates hallucinations: {results['hallucination_rate']:.1%} of the time")
        if results.get('best_detection_accuracy', 0) >= 0.80:
            print("✅ Our system can detect these hallucinations with 80%+ accuracy!")
        else:
            print(f"📊 Our system detects hallucinations with {results.get('best_detection_accuracy', 0):.1%} accuracy")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()