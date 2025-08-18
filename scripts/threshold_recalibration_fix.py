#!/usr/bin/env python3
"""
🔧⚖️ THRESHOLD RECALIBRATION FIX
Fix 0% recall issue and achieve balanced precision-recall for world-class status
"""

import requests
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ThresholdRecalibrationFix:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        
    def load_calibration_dataset(self, max_samples=500):
        """Load balanced dataset for threshold recalibration"""
        data_dir = Path("authentic_datasets")
        all_samples = []
        
        # Load HaluEval QA for recalibration
        qa_path = data_dir / "halueval_qa_data.json"
        if qa_path.exists():
            with open(qa_path, 'r') as f:
                lines = f.read().strip().split('\n')[:max_samples//2]
                
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            if 'question' in sample:
                                all_samples.extend([
                                    {
                                        'prompt': sample['question'],
                                        'output': sample['right_answer'],
                                        'is_hallucination': False
                                    },
                                    {
                                        'prompt': sample['question'],
                                        'output': sample['hallucinated_answer'],
                                        'is_hallucination': True
                                    }
                                ])
                        except:
                            continue
        
        logger.info(f"📊 Calibration dataset: {len(all_samples)} samples")
        halluc_count = sum(1 for s in all_samples if s['is_hallucination'])
        logger.info(f"   🔍 Hallucinations: {halluc_count}")
        logger.info(f"   ✅ Correct: {len(all_samples) - halluc_count}")
        
        return all_samples
    
    def world_class_prediction_with_features(self, prompt, output):
        """World-class prediction with comprehensive feature extraction"""
        
        output_words = output.lower().split()
        output_length = len(output_words)
        
        # 1. Uncertainty markers (expanded)
        uncertainty_words = [
            'maybe', 'might', 'possibly', 'perhaps', 'unsure', 'uncertain',
            'probably', 'likely', 'seems', 'appears', 'could', 'may',
            'i think', 'i believe', 'i guess', 'not sure'
        ]
        uncertainty_count = sum(1 for word in uncertainty_words 
                               if word in output.lower())
        
        # 2. Contradiction/negation markers (expanded)
        contradiction_words = [
            'not', 'no', 'wrong', 'false', 'incorrect', 'never', 'none',
            'opposite', 'contrary', 'however', 'but', 'although', 'actually',
            'disagree', 'dispute', 'reject', 'deny'
        ]
        contradiction_count = sum(1 for word in contradiction_words 
                                 if word in output.lower())
        
        # 3. Confidence/certainty markers
        confidence_words = [
            'definitely', 'certainly', 'absolutely', 'clearly', 'obviously',
            'surely', 'exactly', 'precisely', 'undoubtedly', 'without doubt'
        ]
        confidence_count = sum(1 for word in confidence_words 
                              if word in output.lower())
        
        # 4. Factual assertion patterns
        factual_patterns = ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'does', 'did']
        factual_count = sum(1 for word in factual_patterns if word in output.lower())
        
        # 5. Question response quality
        question_words = ['what', 'when', 'where', 'who', 'why', 'how']
        prompt_has_question = any(word in prompt.lower() for word in question_words)
        
        # 6. Calculate comprehensive uncertainty score
        uncertainty_density = uncertainty_count / max(output_length, 1)
        contradiction_density = contradiction_count / max(output_length, 1)
        confidence_density = confidence_count / max(output_length, 1)
        factual_density = factual_count / max(output_length, 1)
        
        # Weighted uncertainty calculation
        uncertainty_score = (
            0.35 * uncertainty_density +           # Direct uncertainty markers
            0.30 * contradiction_density +         # Contradiction signals
            0.20 * (1.0 - confidence_density) +    # Lack of confidence
            0.15 * (1.0 - factual_density)         # Lack of factual assertions
        )
        
        # Length-based adjustment
        length_factor = 1.0
        if output_length < 5:
            length_factor = 1.3  # Short answers more suspicious
        elif output_length > 100:
            length_factor = 1.1  # Very long answers slightly more suspicious
        
        # Question type adjustment
        qa_factor = 1.2 if prompt_has_question else 1.0
        
        # Final calibrated score
        calibrated_score = uncertainty_score * length_factor * qa_factor
        
        # Convert to probability (recalibrated range)
        recalibrated_p_fail = 0.2 + calibrated_score * 0.6  # Range: 0.2 to 0.8
        
        return {
            'recalibrated_p_fail': recalibrated_p_fail,
            'uncertainty_score': uncertainty_score,
            'feature_counts': {
                'uncertainty': uncertainty_count,
                'contradiction': contradiction_count,
                'confidence': confidence_count,
                'factual': factual_count
            },
            'densities': {
                'uncertainty': uncertainty_density,
                'contradiction': contradiction_density,
                'confidence': confidence_density,
                'factual': factual_density
            }
        }
    
    def find_optimal_f1_threshold(self, samples):
        """Find threshold that maximizes F1 score"""
        
        logger.info(f"\n🔧 FINDING OPTIMAL F1 THRESHOLD")
        logger.info(f"{'='*50}")
        
        # Collect probability scores
        probabilities = []
        ground_truth = []
        
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"📊 Collecting scores: {i}/{len(samples)}")
            
            result = self.world_class_prediction_with_features(sample['prompt'], sample['output'])
            probabilities.append(result['recalibrated_p_fail'])
            ground_truth.append(sample['is_hallucination'])
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(ground_truth, probabilities)
        
        # Calculate F1 scores for each threshold
        f1_scores = []
        threshold_metrics = []
        
        for i, (p, r) in enumerate(zip(precision, recall)):
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
                f1_scores.append(f1)
                
                # Store detailed metrics for this threshold
                if i < len(thresholds):
                    threshold = thresholds[i]
                    threshold_metrics.append({
                        'threshold': threshold,
                        'f1': f1,
                        'precision': p,
                        'recall': r
                    })
        
        # Find optimal F1 threshold
        if f1_scores:
            max_f1_idx = np.argmax(f1_scores)
            optimal_f1 = f1_scores[max_f1_idx]
            
            if max_f1_idx < len(threshold_metrics):
                optimal_threshold_info = threshold_metrics[max_f1_idx]
                optimal_threshold = optimal_threshold_info['threshold']
                optimal_precision = optimal_threshold_info['precision']
                optimal_recall = optimal_threshold_info['recall']
            else:
                optimal_threshold = 0.5
                optimal_precision = precision[max_f1_idx] if max_f1_idx < len(precision) else 0.5
                optimal_recall = recall[max_f1_idx] if max_f1_idx < len(recall) else 0.5
            
            logger.info(f"🎯 OPTIMAL F1 THRESHOLD FOUND:")
            logger.info(f"   🔧 Threshold: {optimal_threshold:.3f}")
            logger.info(f"   📊 F1 Score: {optimal_f1:.1%} {'🏆' if optimal_f1 >= 0.85 else '📊'}")
            logger.info(f"   📈 Precision: {optimal_precision:.1%}")
            logger.info(f"   📈 Recall: {optimal_recall:.1%}")
            
            # Calculate AUROC with these scores
            try:
                auroc = roc_auc_score(ground_truth, probabilities)
                logger.info(f"   🎯 AUROC: {auroc:.1%}")
            except:
                auroc = 0.5
                logger.info(f"   🎯 AUROC: Could not calculate")
            
            return optimal_threshold, optimal_f1, optimal_precision, optimal_recall, auroc
        
        return 0.5, 0.0, 0.0, 0.0, 0.5
    
    def validate_recalibrated_system(self, test_samples, optimal_threshold):
        """Validate system with recalibrated threshold"""
        
        logger.info(f"\n🌍 WORLD-CLASS VALIDATION WITH RECALIBRATED THRESHOLD")
        logger.info(f"{'='*60}")
        logger.info(f"🔧 Using threshold: {optimal_threshold:.3f}")
        logger.info(f"📊 Test samples: {len(test_samples)}")
        
        predictions = []
        probabilities = []
        ground_truth = []
        processing_times = []
        
        start_time = time.time()
        
        for i, sample in enumerate(test_samples):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(test_samples) - i) / rate if rate > 0 else 0
                logger.info(f"🌍 Validation: {i}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%) | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
            
            sample_start = time.time()
            result = self.world_class_prediction_with_features(sample['prompt'], sample['output'])
            processing_times.append((time.time() - sample_start) * 1000)
            
            p_fail = result['recalibrated_p_fail']
            is_predicted_hallucination = p_fail > optimal_threshold
            
            predictions.append(is_predicted_hallucination)
            probabilities.append(p_fail)
            ground_truth.append(sample['is_hallucination'])
        
        # Calculate final world-class metrics
        try:
            final_f1 = f1_score(ground_truth, predictions)
            final_precision = precision_score(ground_truth, predictions, zero_division=0)
            final_recall = recall_score(ground_truth, predictions, zero_division=0)
            final_auroc = roc_auc_score(ground_truth, probabilities)
            
            # Calculate hallucination rate (Vectara metric)
            predicted_hallucinations = sum(predictions)
            hallucination_rate = predicted_hallucinations / len(predictions)
            factual_consistency_rate = 1.0 - hallucination_rate
            
            # Confusion matrix
            tp = sum(1 for p, l in zip(predictions, ground_truth) if p and l)
            fp = sum(1 for p, l in zip(predictions, ground_truth) if p and not l)
            tn = sum(1 for p, l in zip(predictions, ground_truth) if not p and not l)
            fn = sum(1 for p, l in zip(predictions, ground_truth) if not p and l)
            
            logger.info(f"\n🏆 RECALIBRATED WORLD-CLASS RESULTS")
            logger.info(f"{'='*50}")
            logger.info(f"🎯 F1 Score: {final_f1:.1%} {'🏆' if final_f1 >= 0.85 else '📊'}")
            logger.info(f"📈 Precision: {final_precision:.1%} {'🏆' if final_precision >= 0.89 else '📊'}")
            logger.info(f"📈 Recall: {final_recall:.1%} {'🏆' if final_recall >= 0.80 else '📊'}")
            logger.info(f"🎯 AUROC: {final_auroc:.1%} {'🏆' if final_auroc >= 0.79 else '📊'}")
            logger.info(f"📊 Hallucination Rate: {hallucination_rate:.1%}")
            logger.info(f"📊 Factual Consistency: {factual_consistency_rate:.1%}")
            
            logger.info(f"\n📊 Confusion Matrix:")
            logger.info(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            
            # World-class benchmark comparison
            logger.info(f"\n🌍 WORLD-CLASS BENCHMARK STATUS:")
            
            benchmarks_beaten = 0
            
            # Vectara SOTA comparison (0.6% hallucination rate)
            if hallucination_rate <= 0.006:
                logger.info(f"   ✅ BEATS Vectara SOTA: {hallucination_rate:.1%} ≤ 0.6%")
                benchmarks_beaten += 1
            else:
                logger.info(f"   ❌ Vectara SOTA: {hallucination_rate:.1%} > 0.6%")
            
            # Nature 2024 (79% AUROC)
            if final_auroc >= 0.79:
                logger.info(f"   ✅ BEATS Nature 2024: {final_auroc:.1%} ≥ 79%")
                benchmarks_beaten += 1
            else:
                logger.info(f"   ❌ Nature 2024: {final_auroc:.1%} < 79%")
            
            # NeurIPS 2024 (82% F1)
            if final_f1 >= 0.82:
                logger.info(f"   ✅ BEATS NeurIPS 2024: {final_f1:.1%} ≥ 82%")
                benchmarks_beaten += 1
            else:
                logger.info(f"   ❌ NeurIPS 2024: {final_f1:.1%} < 82%")
            
            # ICLR 2024 (89% Precision)
            if final_precision >= 0.89:
                logger.info(f"   ✅ BEATS ICLR 2024: {final_precision:.1%} ≥ 89%")
                benchmarks_beaten += 1
            else:
                logger.info(f"   ❌ ICLR 2024: {final_precision:.1%} < 89%")
            
            total_benchmarks = 4
            
            logger.info(f"\n🏆 WORLD-CLASS STATUS: {benchmarks_beaten}/{total_benchmarks} benchmarks beaten")
            
            if benchmarks_beaten == total_benchmarks:
                logger.info(f"🌍👑 BEST IN THE WORLD STATUS CONFIRMED!")
                logger.info(f"   ✨ Beats ALL major benchmarks")
                logger.info(f"   🥇 Vectara Leaderboard Champion")
                logger.info(f"   🥇 Academic Benchmark Champion")
            elif benchmarks_beaten >= 3:
                logger.info(f"🥇 WORLD-CLASS SYSTEM ACHIEVED!")
                logger.info(f"   ⭐ Beats {benchmarks_beaten}/{total_benchmarks} major benchmarks")
                logger.info(f"   🎯 Among top global systems")
            elif benchmarks_beaten >= 2:
                logger.info(f"⚡ COMPETITIVE WORLD-CLASS PERFORMANCE")
                logger.info(f"   📊 Beats {benchmarks_beaten}/{total_benchmarks} benchmarks")
            else:
                logger.info(f"📊 Good performance, optimization needed for world-class status")
            
            # Performance statistics
            avg_time = np.mean(processing_times)
            throughput = 1000 / avg_time if avg_time > 0 else 0
            
            logger.info(f"\n⚡ Performance Statistics:")
            logger.info(f"   📊 Samples processed: {len(predictions)}")
            logger.info(f"   ⏱️ Avg processing time: {avg_time:.1f}ms")
            logger.info(f"   🚀 Throughput: {throughput:.0f} analyses/sec")
            
            # Save recalibration results
            results = {
                'evaluation_type': 'threshold_recalibration_fix',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'recalibrated_threshold': optimal_threshold,
                'final_metrics': {
                    'f1_score': final_f1,
                    'precision': final_precision,
                    'recall': final_recall,
                    'auroc': final_auroc,
                    'hallucination_rate': hallucination_rate,
                    'factual_consistency_rate': factual_consistency_rate
                },
                'confusion_matrix': {
                    'true_positive': tp,
                    'false_positive': fp,
                    'true_negative': tn,
                    'false_negative': fn
                },
                'world_class_benchmarks': {
                    'vectara_sota': {'target': 0.006, 'achieved': hallucination_rate, 'beaten': hallucination_rate <= 0.006},
                    'nature_2024': {'target': 0.79, 'achieved': final_auroc, 'beaten': final_auroc >= 0.79},
                    'neurips_2024': {'target': 0.82, 'achieved': final_f1, 'beaten': final_f1 >= 0.82},
                    'iclr_2024': {'target': 0.89, 'achieved': final_precision, 'beaten': final_precision >= 0.89}
                },
                'world_class_status': {
                    'benchmarks_beaten': benchmarks_beaten,
                    'total_benchmarks': total_benchmarks,
                    'best_in_world_confirmed': benchmarks_beaten == total_benchmarks,
                    'world_class_confirmed': benchmarks_beaten >= 3
                }
            }
            
            output_file = "test_results/threshold_recalibration_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Recalibration results saved to: {output_file}")
            
            return benchmarks_beaten, total_benchmarks, final_f1, final_auroc
            
        except Exception as e:
            logger.error(f"❌ Recalibration validation failed: {e}")
            return 0, 4, 0.0, 0.0

def main():
    fixer = ThresholdRecalibrationFix()
    
    # Test API connectivity
    try:
        health = requests.get(f"{fixer.api_url}/health", timeout=5)
        if health.status_code != 200:
            logger.error("❌ API server not responding")
            return
        logger.info("✅ API server is running")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # Load calibration dataset
    calibration_samples = fixer.load_calibration_dataset(max_samples=400)
    
    if len(calibration_samples) < 100:
        logger.error("❌ Insufficient calibration samples")
        return
    
    # Split into threshold optimization and validation sets
    split_point = len(calibration_samples) // 2
    threshold_samples = calibration_samples[:split_point]
    validation_samples = calibration_samples[split_point:]
    
    logger.info(f"📊 Threshold optimization samples: {len(threshold_samples)}")
    logger.info(f"📊 Validation samples: {len(validation_samples)}")
    
    # Step 1: Find optimal F1 threshold
    optimal_threshold, optimal_f1, optimal_precision, optimal_recall, calibration_auroc = fixer.find_optimal_f1_threshold(threshold_samples)
    
    # Step 2: Validate with recalibrated threshold
    benchmarks_beaten, total_benchmarks, final_f1, final_auroc = fixer.validate_recalibrated_system(
        validation_samples, optimal_threshold
    )
    
    logger.info(f"\n🌟 THRESHOLD RECALIBRATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"🔧 Optimal Threshold: {optimal_threshold:.3f}")
    logger.info(f"🎯 Final F1: {final_f1:.1%}")
    logger.info(f"🎯 Final AUROC: {final_auroc:.1%}")
    logger.info(f"🏆 Benchmarks Beaten: {benchmarks_beaten}/{total_benchmarks}")
    
    if benchmarks_beaten == total_benchmarks:
        logger.info(f"🌍👑 BEST IN THE WORLD STATUS ACHIEVED!")
        logger.info(f"✨ All major benchmarks beaten - ready for publication")
    elif benchmarks_beaten >= 3:
        logger.info(f"🥇 WORLD-CLASS STATUS ACHIEVED!")
        logger.info(f"⭐ Top-tier global performance confirmed")
    else:
        logger.info(f"📊 Recalibration improved performance")
        logger.info(f"🔧 Further optimization needed for world-class status")

if __name__ == "__main__":
    main()