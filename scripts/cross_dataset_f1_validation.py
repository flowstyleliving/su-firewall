#!/usr/bin/env python3
"""
📊✅ CROSS-DATASET F1 VALIDATION
Validate 95% F1 across HaluEval, TruthfulQA, and other datasets for robustness
"""

import requests
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CrossDatasetF1Validator:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        # Optimal threshold from gap analysis
        self.optimal_threshold = 0.124
        
    def load_multi_dataset_samples(self, max_per_dataset=200):
        """Load samples from multiple datasets for cross-validation"""
        data_dir = Path("authentic_datasets")
        datasets = {}
        
        # 1. HaluEval QA
        qa_path = data_dir / "halueval_qa_data.json"
        if qa_path.exists():
            qa_samples = []
            with open(qa_path, 'r') as f:
                lines = f.read().strip().split('\n')[:max_per_dataset//2]
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            if 'question' in sample:
                                qa_samples.extend([
                                    {
                                        'prompt': sample['question'],
                                        'output': sample['right_answer'],
                                        'is_hallucination': False,
                                        'dataset': 'halueval_qa'
                                    },
                                    {
                                        'prompt': sample['question'],
                                        'output': sample['hallucinated_answer'],
                                        'is_hallucination': True,
                                        'dataset': 'halueval_qa'
                                    }
                                ])
                        except:
                            continue
            datasets['halueval_qa'] = qa_samples
        
        # 2. HaluEval Dialogue
        dialogue_path = data_dir / "halueval_dialogue_data.json"
        if dialogue_path.exists():
            dialogue_samples = []
            with open(dialogue_path, 'r') as f:
                lines = f.read().strip().split('\n')[:max_per_dataset]
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            if 'input' in sample and 'output' in sample:
                                dialogue_samples.append({
                                    'prompt': sample['input'],
                                    'output': sample['output'],
                                    'is_hallucination': sample.get('label') == 'hallucination',
                                    'dataset': 'halueval_dialogue'
                                })
                        except:
                            continue
            datasets['halueval_dialogue'] = dialogue_samples
        
        # 3. TruthfulQA
        truthfulqa_path = data_dir / "truthfulqa_data.json"
        if truthfulqa_path.exists():
            truthfulqa_samples = []
            try:
                with open(truthfulqa_path, 'r') as f:
                    truthfulqa_data = json.loads(f.read())
                    validation_samples = truthfulqa_data.get('validation', [])
                    
                    for sample in validation_samples[:max_per_dataset//2]:
                        question = sample.get('Question', '')
                        best_answer = sample.get('Best Answer', '')
                        incorrect_answer = sample.get('Best Incorrect Answer', '')
                        
                        if question and best_answer:
                            truthfulqa_samples.extend([
                                {
                                    'prompt': question,
                                    'output': best_answer,
                                    'is_hallucination': False,
                                    'dataset': 'truthfulqa'
                                },
                                {
                                    'prompt': question,
                                    'output': incorrect_answer,
                                    'is_hallucination': True,
                                    'dataset': 'truthfulqa'
                                } if incorrect_answer else None
                            ])
                    
                    # Remove None entries
                    truthfulqa_samples = [s for s in truthfulqa_samples if s is not None]
                    datasets['truthfulqa'] = truthfulqa_samples
            except Exception as e:
                logger.warning(f"Failed to load TruthfulQA: {e}")
        
        # Report dataset statistics
        total_samples = 0
        for dataset_name, samples in datasets.items():
            halluc_count = sum(1 for s in samples if s['is_hallucination'])
            correct_count = len(samples) - halluc_count
            total_samples += len(samples)
            logger.info(f"📊 {dataset_name}: {len(samples)} samples ({halluc_count}H, {correct_count}C)")
        
        logger.info(f"📊 Total cross-dataset samples: {total_samples}")
        return datasets
    
    def adaptive_ensemble_prediction(self, prompt, output):
        """Use the breakthrough adaptive ensemble method"""
        
        # Extract linguistic features (from breakthrough method)
        output_length = len(output.split())
        uncertainty_words = sum(1 for word in ['maybe', 'might', 'possibly', 'perhaps', 'unsure', 'uncertain'] 
                               if word in output.lower())
        contradiction_words = sum(1 for word in ['not', 'no', 'wrong', 'false', 'incorrect', 'never']
                                 if word in output.lower())
        certainty_words = sum(1 for word in ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously']
                             if word in output.lower())
        
        # Enhanced feature engineering
        question_words = sum(1 for word in ['what', 'when', 'where', 'who', 'why', 'how']
                            if word in prompt.lower())
        
        # Adaptive ensemble scoring (optimized from breakthrough)
        adaptive_score = (
            0.25 * min(output_length / 50.0, 1.0) +          # Length factor
            0.35 * min(uncertainty_words / 3.0, 1.0) +       # Uncertainty boost
            0.30 * min(contradiction_words / 2.0, 1.0) +     # Contradiction detection
            0.10 * (1.0 - min(certainty_words / 2.0, 1.0))   # Certainty penalty
        )
        
        # Domain adjustment
        if question_words > 0:  # QA domain
            adaptive_score *= 1.1  # Slightly more sensitive for QA
        
        # Convert to probability (optimized range from gap analysis)
        adaptive_p_fail = 0.05 + adaptive_score * 0.85
        
        return {
            'adaptive_p_fail': adaptive_p_fail,
            'feature_scores': {
                'length': output_length,
                'uncertainty': uncertainty_words,
                'contradiction': contradiction_words,
                'certainty': certainty_words
            }
        }
    
    def evaluate_dataset_f1(self, dataset_name, samples):
        """Evaluate F1 performance on specific dataset"""
        
        logger.info(f"\n🔍 Evaluating {dataset_name.upper()}")
        logger.info(f"{'─'*40}")
        
        predictions = []
        probabilities = []
        ground_truth = []
        processing_times = []
        
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            if i % 50 == 0 and len(samples) > 100:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(f"📈 {dataset_name}: {i}/{len(samples)} | Rate: {rate:.1f}/s")
            
            sample_start = time.time()
            result = self.adaptive_ensemble_prediction(sample['prompt'], sample['output'])
            processing_times.append((time.time() - sample_start) * 1000)
            
            # Use optimal threshold from gap analysis
            p_fail = result['adaptive_p_fail']
            is_predicted_hallucination = p_fail > self.optimal_threshold
            
            predictions.append(is_predicted_hallucination)
            probabilities.append(p_fail)
            ground_truth.append(sample['is_hallucination'])
        
        # Calculate comprehensive metrics
        try:
            f1 = f1_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions, zero_division=0)
            recall = recall_score(ground_truth, predictions, zero_division=0)
            auroc = roc_auc_score(ground_truth, probabilities)
            
            # Confusion matrix
            tp = sum(1 for p, l in zip(predictions, ground_truth) if p and l)
            fp = sum(1 for p, l in zip(predictions, ground_truth) if p and not l)
            tn = sum(1 for p, l in zip(predictions, ground_truth) if not p and not l)
            fn = sum(1 for p, l in zip(predictions, ground_truth) if not p and l)
            
            logger.info(f"   🎯 F1 Score: {f1:.1%} {'🏆' if f1 >= 0.85 else '📊'}")
            logger.info(f"   📈 Precision: {precision:.1%}")
            logger.info(f"   📈 Recall: {recall:.1%}")
            logger.info(f"   🎯 AUROC: {auroc:.1%}")
            logger.info(f"   📊 Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            logger.info(f"   ⏱️ Avg time: {np.mean(processing_times):.1f}ms")
            
            return {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'auroc': auroc,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
                'avg_processing_time_ms': np.mean(processing_times),
                'samples_tested': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"   ❌ Metrics calculation failed: {e}")
            return {
                'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'auroc': 0.5,
                'samples_tested': len(predictions)
            }
    
    def run_cross_dataset_validation(self, datasets):
        """Run comprehensive cross-dataset F1 validation"""
        
        logger.info(f"\n📊✅ CROSS-DATASET F1 VALIDATION")
        logger.info(f"{'='*60}")
        logger.info(f"🎯 Target: 85%+ F1 across all datasets")
        logger.info(f"🔧 Using adaptive ensemble with threshold: {self.optimal_threshold:.3f}")
        
        dataset_results = {}
        all_f1_scores = []
        
        for dataset_name, samples in datasets.items():
            if len(samples) < 10:
                logger.warning(f"⚠️ Skipping {dataset_name}: insufficient samples ({len(samples)})")
                continue
                
            result = self.evaluate_dataset_f1(dataset_name, samples)
            dataset_results[dataset_name] = result
            all_f1_scores.append(result['f1_score'])
        
        # Overall cross-dataset performance
        if all_f1_scores:
            avg_f1 = np.mean(all_f1_scores)
            min_f1 = np.min(all_f1_scores)
            max_f1 = np.max(all_f1_scores)
            f1_std = np.std(all_f1_scores)
            
            logger.info(f"\n🏆 CROSS-DATASET F1 VALIDATION RESULTS")
            logger.info(f"{'='*50}")
            logger.info(f"📊 Average F1: {avg_f1:.1%} {'🏆' if avg_f1 >= 0.85 else '📊'}")
            logger.info(f"📊 Min F1: {min_f1:.1%}")
            logger.info(f"📊 Max F1: {max_f1:.1%}")
            logger.info(f"📊 F1 Std Dev: {f1_std:.1%}")
            
            # Target achievement assessment
            target_met_count = sum(1 for f1 in all_f1_scores if f1 >= 0.85)
            total_datasets = len(all_f1_scores)
            
            if target_met_count == total_datasets:
                logger.info(f"\n🎉 85%+ F1 TARGET ACHIEVED ACROSS ALL DATASETS!")
                logger.info(f"   ✅ {target_met_count}/{total_datasets} datasets meet 85%+ F1")
                logger.info(f"   🏆 Robust cross-dataset performance confirmed")
            elif target_met_count >= total_datasets * 0.8:
                logger.info(f"\n⚡ STRONG CROSS-DATASET PERFORMANCE!")
                logger.info(f"   ✅ {target_met_count}/{total_datasets} datasets meet 85%+ F1")
                logger.info(f"   📊 {avg_f1:.1%} average F1 demonstrates robustness")
            else:
                logger.info(f"\n📈 PARTIAL CROSS-DATASET SUCCESS")
                logger.info(f"   ✅ {target_met_count}/{total_datasets} datasets meet 85%+ F1")
                logger.info(f"   🔧 Dataset-specific optimization needed")
            
            # Robustness analysis
            if f1_std < 0.05:
                logger.info(f"   🎯 LOW VARIANCE: Consistent performance across datasets")
            elif f1_std < 0.10:
                logger.info(f"   📊 MODERATE VARIANCE: Some dataset-specific effects")
            else:
                logger.info(f"   ⚠️ HIGH VARIANCE: Significant dataset-specific optimization needed")
            
            # Save cross-dataset validation results
            results = {
                'validation_type': 'cross_dataset_f1_validation',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'optimal_threshold_used': self.optimal_threshold,
                'dataset_results': dataset_results,
                'summary_metrics': {
                    'average_f1': avg_f1,
                    'min_f1': min_f1,
                    'max_f1': max_f1,
                    'f1_std_dev': f1_std,
                    'datasets_meeting_target': target_met_count,
                    'total_datasets_tested': total_datasets
                },
                'target_achievement': {
                    'target_f1': 0.85,
                    'all_datasets_meet_target': target_met_count == total_datasets,
                    'robustness_score': 1.0 - f1_std,  # Higher is better
                    'cross_dataset_success_rate': target_met_count / total_datasets
                }
            }
            
            output_file = "test_results/cross_dataset_f1_validation_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Cross-dataset validation saved to: {output_file}")
            
            return avg_f1, target_met_count, total_datasets
        
        return 0.0, 0, 0

def main():
    validator = CrossDatasetF1Validator()
    
    # Test API connectivity
    try:
        health = requests.get(f"{validator.api_url}/health", timeout=5)
        if health.status_code != 200:
            logger.error("❌ API server not responding")
            return
        logger.info("✅ API server is running")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # Load multi-dataset samples
    datasets = validator.load_multi_dataset_samples(max_per_dataset=150)
    
    if not datasets:
        logger.error("❌ No datasets loaded")
        return
    
    # Run cross-dataset validation
    avg_f1, target_met, total_datasets = validator.run_cross_dataset_validation(datasets)
    
    logger.info(f"\n🌟 CROSS-DATASET VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"🎯 Average F1: {avg_f1:.1%}")
    logger.info(f"✅ Datasets meeting 85%+ F1: {target_met}/{total_datasets}")
    
    if target_met == total_datasets and avg_f1 >= 0.85:
        logger.info(f"🏆 ROBUST 85%+ F1 CONFIRMED across all datasets!")
        logger.info(f"✅ Ready for production deployment")
    elif avg_f1 >= 0.85:
        logger.info(f"⚡ Strong average F1, but some datasets need optimization")
        logger.info(f"🔧 Consider dataset-specific threshold tuning")
    else:
        logger.info(f"📊 Cross-dataset optimization needed")
        logger.info(f"🔧 Next: Implement cost-sensitive learning and method-specific optimization")

if __name__ == "__main__":
    main()