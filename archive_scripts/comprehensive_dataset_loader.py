#!/usr/bin/env python3
"""
Comprehensive dataset loader for TruthfulQA and HaluEval datasets.
Loads from local authentic_datasets directory.
"""

import json
import random
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class EvaluationPair:
    prompt: str
    correct_answer: str
    hallucinated_answer: str
    source: str
    metadata: dict = None

def load_truthfulqa_fixed(max_samples: Optional[int] = None, seed: int = 42) -> List[EvaluationPair]:
    """
    Load TruthfulQA dataset from local authentic_datasets directory.
    
    Args:
        max_samples: Maximum number of samples to return
        seed: Random seed for sampling
        
    Returns:
        List of EvaluationPair objects
    """
    try:
        truthfulqa_path = Path("authentic_datasets/truthfulqa_data.json")
        
        if not truthfulqa_path.exists():
            print(f"⚠️ TruthfulQA data file not found at {truthfulqa_path}")
            return []
        
        with open(truthfulqa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pairs = []
        
        # Handle different data structures
        if isinstance(data, dict):
            if 'data' in data:
                entries = data['data']
            elif 'validation' in data:
                entries = data['validation']
            else:
                # Assume data is the root level
                entries = [data] if not isinstance(data.get('question'), list) else data
        elif isinstance(data, list):
            entries = data
        else:
            print("⚠️ Unexpected TruthfulQA data structure")
            return []
        
        for entry in entries:
            if isinstance(entry, dict):
                question = entry.get('Question', entry.get('question', ''))
                
                # Get correct answers (handle both string and array formats)
                best_answer = entry.get('Best Answer', entry.get('best_answer', ''))
                correct_answers = entry.get('Correct Answers', entry.get('correct_answers', []))
                
                # Get incorrect answers
                best_incorrect = entry.get('Best Incorrect Answer', entry.get('best_incorrect_answer', ''))
                incorrect_answers = entry.get('Incorrect Answers', entry.get('incorrect_answers', []))
                
                # Parse string format answers (semicolon separated)
                if isinstance(correct_answers, str):
                    correct_answers = [a.strip() for a in correct_answers.split(';') if a.strip()]
                if isinstance(incorrect_answers, str):
                    incorrect_answers = [a.strip() for a in incorrect_answers.split(';') if a.strip()]
                
                # Use best answer if available, otherwise first correct answer
                correct = best_answer if best_answer else (correct_answers[0] if correct_answers else '')
                
                # Use best incorrect if available, otherwise first incorrect answer
                hallucinated = best_incorrect if best_incorrect else (incorrect_answers[0] if incorrect_answers else '')
                
                if question and correct and hallucinated:
                    pairs.append(EvaluationPair(
                        prompt=question,
                        correct_answer=correct,
                        hallucinated_answer=hallucinated,
                        source="truthfulqa",
                        metadata={
                            "category": entry.get('category', 'unknown'),
                            "correct_count": len(correct_answers),
                            "incorrect_count": len(incorrect_answers)
                        }
                    ))
        
        # Shuffle and sample
        random.Random(seed).shuffle(pairs)
        
        if max_samples is not None and max_samples > 0:
            pairs = pairs[:max_samples]
        
        print(f"✅ Loaded {len(pairs)} TruthfulQA pairs")
        return pairs
        
    except Exception as e:
        print(f"❌ Error loading TruthfulQA: {e}")
        return []

def load_halueval_fixed(task: str = "qa", max_samples: Optional[int] = None, seed: int = 42) -> List[EvaluationPair]:
    """
    Load HaluEval dataset from local authentic_datasets directory.
    
    Args:
        task: HaluEval task type (qa, dialogue, summarization, general)
        max_samples: Maximum number of samples to return
        seed: Random seed for sampling
        
    Returns:
        List of EvaluationPair objects
    """
    try:
        # Map task names to file names
        task_files = {
            "qa": "halueval_qa_data.json",
            "dialogue": "halueval_dialogue_data.json", 
            "summarization": "halueval_summarization_data.json",
            "general": "halueval_general_data.json"
        }
        
        if task not in task_files:
            print(f"⚠️ Unknown HaluEval task: {task}")
            return []
        
        halueval_path = Path(f"authentic_datasets/{task_files[task]}")
        
        if not halueval_path.exists():
            print(f"⚠️ HaluEval {task} data file not found at {halueval_path}")
            return []
        
        pairs = []
        
        # Handle JSONL format (one JSON object per line)
        with open(halueval_path, 'r', encoding='utf-8') as f:
            entries = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        for entry in entries:
            if isinstance(entry, dict):
                # Extract prompt (various field names used)
                prompt = (
                    entry.get('question') or 
                    entry.get('input') or 
                    entry.get('prompt') or 
                    entry.get('context') or
                    entry.get('dialogue') or
                    entry.get('text') or
                    ''
                )
                
                # Extract correct answer
                correct = (
                    entry.get('right_answer') or
                    entry.get('answer') or
                    entry.get('correct') or
                    entry.get('reference') or
                    entry.get('target') or
                    ''
                )
                
                # Extract hallucinated/incorrect answer  
                hallucinated = (
                    entry.get('hallucinated_answer') or
                    entry.get('hallucination') or
                    entry.get('incorrect') or
                    entry.get('wrong_answer') or
                    entry.get('negative') or
                    entry.get('fake') or
                    ''
                )
                
                # Handle cases where answers are in lists
                if isinstance(correct, list):
                    correct = correct[0] if correct else ''
                if isinstance(hallucinated, list):
                    hallucinated = hallucinated[0] if hallucinated else ''
                
                # Convert to strings
                prompt = str(prompt) if prompt else ''
                correct = str(correct) if correct else ''
                hallucinated = str(hallucinated) if hallucinated else ''
                
                if prompt and correct and hallucinated:
                    pairs.append(EvaluationPair(
                        prompt=prompt,
                        correct_answer=correct,
                        hallucinated_answer=hallucinated,
                        source=f"halueval_{task}",
                        metadata={
                            "task": task,
                            "domain": entry.get('domain', 'unknown'),
                            "difficulty": entry.get('difficulty', 'unknown')
                        }
                    ))
        
        # Shuffle and sample
        random.Random(seed).shuffle(pairs)
        
        if max_samples is not None and max_samples > 0:
            pairs = pairs[:max_samples]
        
        print(f"✅ Loaded {len(pairs)} HaluEval {task} pairs")
        return pairs
        
    except Exception as e:
        print(f"❌ Error loading HaluEval {task}: {e}")
        return []

def test_loaders():
    """Test the dataset loaders"""
    print("🧪 Testing dataset loaders...")
    
    # Test TruthfulQA
    print("\n📊 Testing TruthfulQA loader...")
    truthful_pairs = load_truthfulqa_fixed(max_samples=5)
    
    for i, pair in enumerate(truthful_pairs[:3]):
        print(f"  Example {i+1}:")
        print(f"    Prompt: {pair.prompt[:100]}...")
        print(f"    Correct: {pair.correct_answer[:100]}...")
        print(f"    Hallucinated: {pair.hallucinated_answer[:100]}...")
        print()
    
    # Test HaluEval QA
    print("📊 Testing HaluEval QA loader...")
    halueval_pairs = load_halueval_fixed(task="qa", max_samples=5)
    
    for i, pair in enumerate(halueval_pairs[:3]):
        print(f"  Example {i+1}:")
        print(f"    Prompt: {pair.prompt[:100]}...")
        print(f"    Correct: {pair.correct_answer[:100]}...")
        print(f"    Hallucinated: {pair.hallucinated_answer[:100]}...")
        print()

if __name__ == "__main__":
    test_loaders()