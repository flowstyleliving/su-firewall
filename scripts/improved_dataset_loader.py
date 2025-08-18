#!/usr/bin/env python3
"""
Improved Dataset Loader
=======================

Fixes dataset loading issues to access full datasets:
- 7,903 TruthfulQA examples
- 4,507 HaluEval General examples  
- 10,000 HaluEval QA examples
- 10,000 HaluEval Dialogue examples
- 10,000 HaluEval Summarization examples

Total: 42,410 examples
"""

import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random

# Enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pair:
    """Data structure for prompt-answer pairs"""
    def __init__(self, prompt: str, failing: str, passing: str):
        self.prompt = prompt
        self.failing = failing
        self.passing = passing

def load_truthfulqa_robust(max_samples: Optional[int] = None, seed: int = 42) -> List[Pair]:
    """
    Robust TruthfulQA loader using local files with multiple fallback strategies.
    
    Expected format: JSON with 'validation' key containing list of questions.
    """
    local_paths = [
        "authentic_datasets/truthfulqa_data.json",
        "authentic_datasets/truthfulqa_raw.csv", 
        "truthfulqa_data.json"
    ]
    
    logger.info("🔍 Loading TruthfulQA dataset...")
    
    for path in local_paths:
        if not Path(path).exists():
            logger.warning(f"❌ Path not found: {path}")
            continue
            
        try:
            logger.info(f"📂 Attempting to load: {path}")
            
            if path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                questions = []
                if isinstance(data, dict):
                    # Standard TruthfulQA format
                    if 'validation' in data:
                        questions = data['validation']
                        logger.info(f"✅ Found validation set with {len(questions)} questions")
                    elif 'questions' in data:
                        questions = data['questions']
                        logger.info(f"✅ Found questions set with {len(questions)} questions")
                    else:
                        # Try to find any list in the dict
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 100:
                                questions = value
                                logger.info(f"✅ Found question list under key '{key}' with {len(questions)} items")
                                break
                elif isinstance(data, list):
                    questions = data
                    logger.info(f"✅ Found direct question list with {len(questions)} questions")
                
                if not questions:
                    logger.warning(f"⚠️ No questions found in {path}")
                    continue
                
                # Convert to Pair format
                pairs = []
                for item in questions:
                    try:
                        # Extract question and answers
                        question = item.get('Question', item.get('question', ''))
                        if not question:
                            continue
                            
                        # Get correct and incorrect answers
                        best_answer = item.get('Best Answer', item.get('correct_answer', item.get('answer', '')))
                        incorrect_answers = item.get('Incorrect Answers', item.get('incorrect_answers', []))
                        
                        if not isinstance(incorrect_answers, list):
                            incorrect_answers = [str(incorrect_answers)] if incorrect_answers else []
                        
                        if question and best_answer and incorrect_answers:
                            # Use first incorrect answer
                            incorrect_answer = incorrect_answers[0] if incorrect_answers else f"I don't know the answer to: {question}"
                            
                            pairs.append(Pair(
                                prompt=str(question),
                                passing=str(best_answer), 
                                failing=str(incorrect_answer)
                            ))
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing item: {e}")
                        continue
                
                if pairs:
                    logger.info(f"✅ Successfully loaded {len(pairs)} TruthfulQA pairs from {path}")
                    
                    # Shuffle and limit
                    random.Random(seed).shuffle(pairs)
                    if max_samples:
                        pairs = pairs[:max_samples]
                        logger.info(f"📊 Limited to {len(pairs)} samples")
                    
                    return pairs
                    
            elif path.endswith('.csv'):
                # Handle CSV format
                import csv
                pairs = []
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        question = row.get('Question', '')
                        best_answer = row.get('Best Answer', '')
                        incorrect_answers = row.get('Incorrect Answers', '').split(';') if row.get('Incorrect Answers') else []
                        
                        if question and best_answer and incorrect_answers:
                            pairs.append(Pair(
                                prompt=question,
                                passing=best_answer,
                                failing=incorrect_answers[0]
                            ))
                
                if pairs:
                    logger.info(f"✅ Successfully loaded {len(pairs)} TruthfulQA pairs from CSV")
                    random.Random(seed).shuffle(pairs)
                    if max_samples:
                        pairs = pairs[:max_samples]
                    return pairs
                    
        except Exception as e:
            logger.error(f"❌ Error loading {path}: {e}")
            continue
    
    logger.error("❌ Failed to load TruthfulQA from any source")
    return []

def load_halueval_robust(task: str = "qa", max_samples: Optional[int] = None, seed: int = 42) -> List[Pair]:
    """
    Robust HaluEval loader using local files with comprehensive error handling.
    
    Expected format: JSONL with 'question', 'right_answer', 'hallucinated_answer'
    """
    local_paths = [
        f"authentic_datasets/halueval_{task}_data.json",
        f"halueval_{task}_data.json",
        f"authentic_datasets/halueval_{task}.jsonl"
    ]
    
    logger.info(f"🔍 Loading HaluEval {task} dataset...")
    
    for path in local_paths:
        if not Path(path).exists():
            logger.warning(f"❌ Path not found: {path}")
            continue
            
        try:
            logger.info(f"📂 Attempting to load: {path}")
            
            pairs = []
            with open(path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        # Parse JSON line
                        item = json.loads(line)
                        line_count += 1
                        
                        # Extract fields with multiple fallbacks
                        question = (
                            item.get('question') or 
                            item.get('prompt') or 
                            item.get('query') or
                            item.get('input') or
                            ''
                        )
                        
                        correct_answer = (
                            item.get('right_answer') or
                            item.get('correct_answer') or 
                            item.get('answer') or
                            item.get('reference') or
                            item.get('target') or
                            ''
                        )
                        
                        hallucinated_answer = (
                            item.get('hallucinated_answer') or
                            item.get('wrong_answer') or
                            item.get('incorrect_answer') or
                            item.get('negative') or
                            item.get('hallucination') or
                            ''
                        )
                        
                        if question and correct_answer and hallucinated_answer:
                            pairs.append(Pair(
                                prompt=str(question),
                                passing=str(correct_answer),
                                failing=str(hallucinated_answer)
                            ))
                        
                        # Progress logging for large files
                        if line_count % 1000 == 0:
                            logger.info(f"📈 Processed {line_count} lines, found {len(pairs)} valid pairs")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"⚠️ Processing error at line {line_num}: {e}")
                        continue
            
            if pairs:
                logger.info(f"✅ Successfully loaded {len(pairs)} HaluEval {task} pairs from {path}")
                
                # Shuffle and limit
                random.Random(seed).shuffle(pairs)
                if max_samples:
                    pairs = pairs[:max_samples]
                    logger.info(f"📊 Limited to {len(pairs)} samples")
                
                return pairs
                
        except Exception as e:
            logger.error(f"❌ Error loading {path}: {e}")
            continue
    
    logger.error(f"❌ Failed to load HaluEval {task} from any source")
    return []

def test_dataset_loading():
    """Test the improved dataset loading functions"""
    
    print("🧪 TESTING IMPROVED DATASET LOADING")
    print("=" * 60)
    
    # Test TruthfulQA loading
    print("📊 Testing TruthfulQA loading...")
    truthfulqa_pairs = load_truthfulqa_robust(max_samples=100)
    print(f"✅ TruthfulQA: {len(truthfulqa_pairs)} pairs loaded")
    
    if truthfulqa_pairs:
        print("📝 Sample TruthfulQA pair:")
        pair = truthfulqa_pairs[0]
        print(f"   Prompt: {pair.prompt[:60]}...")
        print(f"   Correct: {pair.passing[:60]}...")
        print(f"   Incorrect: {pair.failing[:60]}...")
    
    # Test HaluEval loading for each task
    halueval_tasks = ["qa", "dialogue", "summarization", "general"]
    total_halueval = 0
    
    for task in halueval_tasks:
        print(f"\n📊 Testing HaluEval {task} loading...")
        pairs = load_halueval_robust(task, max_samples=100)
        print(f"✅ HaluEval {task}: {len(pairs)} pairs loaded")
        total_halueval += len(pairs)
        
        if pairs:
            print("📝 Sample pair:")
            pair = pairs[0]
            print(f"   Question: {pair.prompt[:60]}...")
            print(f"   Correct: {pair.passing[:60]}...")
            print(f"   Hallucinated: {pair.failing[:60]}...")
    
    print(f"\n📈 DATASET LOADING SUMMARY:")
    print(f"   TruthfulQA: {len(truthfulqa_pairs)} pairs")
    print(f"   HaluEval Total: {total_halueval} pairs")
    print(f"   Combined: {len(truthfulqa_pairs) + total_halueval} pairs")
    
    # Test full loading (no limits) to see actual dataset sizes
    print(f"\n🔍 Testing full dataset sizes...")
    
    try:
        full_truthfulqa = load_truthfulqa_robust()
        print(f"📊 Full TruthfulQA: {len(full_truthfulqa)} examples")
    except Exception as e:
        print(f"❌ Full TruthfulQA error: {e}")
    
    for task in halueval_tasks:
        try:
            full_halueval = load_halueval_robust(task)
            print(f"📊 Full HaluEval {task}: {len(full_halueval)} examples")
        except Exception as e:
            print(f"❌ Full HaluEval {task} error: {e}")

if __name__ == "__main__":
    test_dataset_loading()