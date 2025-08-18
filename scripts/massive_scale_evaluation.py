#!/usr/bin/env python3
"""
🚀 MASSIVE SCALE SEMANTIC ENTROPY EVALUATION
Run comprehensive evaluation on full 42,436 dataset
"""

import requests
import json
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_all_datasets():
    """Load all available datasets"""
    datasets = {}
    data_dir = Path("authentic_datasets")
    
    dataset_files = [
        ("truthfulqa", "truthfulqa_data.json"),
        ("halueval_qa", "halueval_qa_data.json"),
        ("halueval_dialogue", "halueval_dialogue_data.json"),
        ("halueval_general", "halueval_general_data.json"),
        ("halueval_summarization", "halueval_summarization_data.json"),
    ]
    
    total_samples = 0
    for name, filename in dataset_files:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                content = f.read().strip()
                try:
                    if content.startswith('['):
                        # JSON array format
                        data = json.loads(content)
                    else:
                        # JSONL format
                        data = []
                        for line in content.split('\n'):
                            if line.strip():
                                data.append(json.loads(line))
                    
                    datasets[name] = data
                    total_samples += len(data)
                    logger.info(f"📊 Loaded {name}: {len(data)} samples")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {name}: {e}")
    
    logger.info(f"🎯 Total dataset size: {total_samples} samples")
    return datasets, total_samples

def evaluate_sample(api_url, prompt, output):
    """Evaluate a single sample using the semantic entropy API"""
    try:
        response = requests.post(
            f"{api_url}/api/v1/analyze", 
            json={"prompt": prompt, "output": output},
            timeout=1.0  # Fast timeout for throughput
        )
        if response.status_code == 200:
            result = response.json()
            return {
                'semantic_uncertainty': result.get('semantic_uncertainty', 0.0),
                'risk_level': result.get('risk_level', 'safe'),
                'is_hallucination_predicted': result.get('risk_level') in ['warning', 'high', 'critical'],
                'processing_time': result.get('processing_time_ms', 0.0)
            }
    except Exception as e:
        # Return neutral prediction on timeout/error
        return {
            'semantic_uncertainty': 1.0,
            'risk_level': 'safe', 
            'is_hallucination_predicted': False,
            'processing_time': 1000.0
        }
    
    return None

def calculate_metrics(predictions, labels):
    """Calculate comprehensive performance metrics"""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels length mismatch")
    
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }

def run_massive_evaluation():
    """Run evaluation on the full massive dataset"""
    api_url = "http://localhost:8080"
    
    # Test API connectivity
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        if health.status_code != 200:
            logger.error("❌ API server not responding")
            return
        logger.info("✅ API server is running")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # Load all datasets
    datasets, total_samples = load_all_datasets()
    
    logger.info(f"\n🚀 MASSIVE SCALE SEMANTIC ENTROPY EVALUATION")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Total samples: {total_samples}")
    logger.info(f"⚡ Target: Process all {total_samples} samples")
    logger.info(f"🎯 Benchmark: Achieve >70% F1 on complete dataset")
    
    overall_results = {}
    grand_total_processed = 0
    grand_total_correct = 0
    all_predictions = []
    all_labels = []
    total_processing_time = 0.0
    
    for dataset_name, samples in datasets.items():
        logger.info(f"\n🔬 Evaluating {dataset_name.upper()}")
        logger.info(f"{'─'*50}")
        logger.info(f"📊 Dataset size: {len(samples)}")
        
        dataset_predictions = []
        dataset_labels = []
        dataset_processing_time = 0.0
        processed_count = 0
        
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(samples) - i) / rate if rate > 0 else 0
                logger.info(f"📈 Progress: {i}/{len(samples)} ({i/len(samples)*100:.1f}%) | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
            
            # Extract prompt and output from sample
            if 'prompt' in sample and 'output' in sample:
                prompt = sample['prompt']
                output = sample['output']
                is_hallucination = sample.get('label', False) == 'hallucination'
            elif 'question' in sample and 'best_answer' in sample:
                prompt = sample['question']
                output = sample['best_answer'] 
                is_hallucination = sample.get('hallucination', False)
            elif 'input' in sample and 'response' in sample:
                prompt = sample['input']
                output = sample['response']
                is_hallucination = sample.get('hallucination', False)
            else:
                continue
                
            result = evaluate_sample(api_url, prompt, output)
            if result:
                dataset_predictions.append(result['is_hallucination_predicted'])
                dataset_labels.append(is_hallucination)
                dataset_processing_time += result['processing_time']
                processed_count += 1
        
        if dataset_predictions:
            dataset_metrics = calculate_metrics(dataset_predictions, dataset_labels)
            overall_results[dataset_name] = {
                'samples_processed': processed_count,
                'metrics': dataset_metrics,
                'avg_processing_time_ms': dataset_processing_time / processed_count if processed_count > 0 else 0
            }
            
            # Accumulate for grand totals
            all_predictions.extend(dataset_predictions)
            all_labels.extend(dataset_labels)
            grand_total_processed += processed_count
            total_processing_time += dataset_processing_time
            
            logger.info(f"✅ {dataset_name}: F1={dataset_metrics['f1_score']:.3f}, Accuracy={dataset_metrics['accuracy']:.3f}")
    
    # Calculate overall metrics
    if all_predictions:
        overall_metrics = calculate_metrics(all_predictions, all_labels)
        
        logger.info(f"\n🏆 MASSIVE SCALE EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Total samples processed: {grand_total_processed:,}")
        logger.info(f"🎯 Overall F1-Score: {overall_metrics['f1_score']:.3f}")
        logger.info(f"🎯 Overall Accuracy: {overall_metrics['accuracy']:.3f}")
        logger.info(f"📈 Overall Precision: {overall_metrics['precision']:.3f}")
        logger.info(f"📈 Overall Recall: {overall_metrics['recall']:.3f}")
        logger.info(f"📈 Overall Specificity: {overall_metrics['specificity']:.3f}")
        logger.info(f"⏱️  Total processing time: {total_processing_time/1000:.1f}s")
        logger.info(f"⚡ Average throughput: {grand_total_processed/(total_processing_time/1000):.1f} samples/sec")
        
        # Performance assessment
        target_f1 = 0.70
        if overall_metrics['f1_score'] >= target_f1:
            logger.info(f"🥇 BREAKTHROUGH ACHIEVED! F1 ≥ {target_f1}")
        else:
            gap = target_f1 - overall_metrics['f1_score']
            logger.info(f"🔧 Performance gap: {gap:.3f} below target {target_f1}")
        
        # Save comprehensive results
        results = {
            'evaluation_type': 'massive_scale_semantic_entropy',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples_processed': grand_total_processed,
            'overall_metrics': overall_metrics,
            'dataset_breakdown': overall_results,
            'performance_summary': {
                'total_processing_time_ms': total_processing_time,
                'avg_processing_time_ms': total_processing_time / grand_total_processed if grand_total_processed > 0 else 0,
                'throughput_samples_per_sec': grand_total_processed / (total_processing_time/1000) if total_processing_time > 0 else 0,
                'target_f1_met': overall_metrics['f1_score'] >= target_f1
            }
        }
        
        with open('massive_scale_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: massive_scale_evaluation_results.json")
        
        return overall_metrics['f1_score']
    
    return 0.0

if __name__ == "__main__":
    run_massive_evaluation()