#!/usr/bin/env python3
"""
📊🎯 CALIBRATION BOUNDARY OPTIMIZATION
Fine-tune decision boundaries and similarity thresholds for 79%+ AUROC
"""

import requests
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CalibrationBoundaryOptimizer:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        
    def load_halueval_samples(self, max_samples=1000):
        """Load HaluEval samples with known hallucination labels"""
        data_dir = Path("authentic_datasets")
        all_samples = []
        
        # Focus on HaluEval QA which has clear hallucination labels
        filepath = data_dir / "halueval_qa_data.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                content = f.read().strip()
                lines = content.split('\n')[:max_samples]
                
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            # HaluEval QA format: question, right_answer, hallucinated_answer
                            if 'question' in sample and 'right_answer' in sample and 'hallucinated_answer' in sample:
                                # Add correct sample
                                all_samples.append({
                                    'prompt': sample['question'],
                                    'output': sample['right_answer'],
                                    'is_hallucination': False,
                                    'source': 'halueval_qa'
                                })
                                
                                # Add hallucinated sample
                                all_samples.append({
                                    'prompt': sample['question'],
                                    'output': sample['hallucinated_answer'],
                                    'is_hallucination': True,
                                    'source': 'halueval_qa'
                                })
                        except:
                            continue
                            
        logger.info(f"📊 Loaded {len(all_samples)} HaluEval QA samples")
        
        # Check label distribution
        hallucination_count = sum(1 for s in all_samples if s['is_hallucination'])
        correct_count = len(all_samples) - hallucination_count
        logger.info(f"   🔍 Hallucinations: {hallucination_count}")
        logger.info(f"   ✅ Correct: {correct_count}")
        
        return all_samples
    
    def analyze_with_custom_params(self, prompt, output, similarity_threshold=0.5, power_factor=2.0):
        """Analyze with custom similarity threshold and power factor"""
        
        # Generate semantic candidates
        candidates = [
            output,
            f"Actually, {output[:50]}... is incorrect",
            "I'm not certain about this information",
            f"The opposite of '{output[:30]}...' is true",
            "This seems questionable and unreliable"
        ]
        
        request_data = {
            "topk_indices": [1, 2, 3, 4, 5],
            "topk_probs": [0.4, 0.25, 0.2, 0.1, 0.05],
            "method": "semantic_entropy",
            "model_id": "mistral-7b",
            "answer_candidates": candidates,
            "candidate_probabilities": [0.4, 0.25, 0.2, 0.1, 0.05],
            "similarity_threshold": similarity_threshold,
            "power_factor": power_factor
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/analyze_topk_compact",
                json=request_data,
                timeout=3
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'semantic_entropy': result.get('semantic_entropy', 0.0),
                    'p_fail': result.get('p_fail', 0.5),
                    'hbar_s': result.get('hbar_s', 1.0),
                    'semantic_clusters': result.get('semantic_clusters', 1),
                    'success': True
                }
            else:
                return {'success': False}
                
        except Exception as e:
            return {'success': False}
    
    def evaluate_parameter_configuration(self, samples, similarity_threshold, power_factor, pfail_threshold=0.5):
        """Evaluate a specific parameter configuration"""
        
        predictions = []
        ground_truth = []
        semantic_entropies = []
        p_fails = []
        
        for sample in samples:
            result = self.analyze_with_custom_params(
                sample['prompt'], 
                sample['output'],
                similarity_threshold,
                power_factor
            )
            
            if result['success']:
                # Use P(fail) for binary prediction
                is_predicted_hallucination = result['p_fail'] > pfail_threshold
                predictions.append(is_predicted_hallucination)
                ground_truth.append(sample['is_hallucination'])
                semantic_entropies.append(result['semantic_entropy'])
                p_fails.append(result['p_fail'])
            else:
                # Default prediction on failure
                predictions.append(False)
                ground_truth.append(sample['is_hallucination'])
                semantic_entropies.append(0.0)
                p_fails.append(0.5)
        
        # Calculate metrics
        try:
            auroc = roc_auc_score(ground_truth, p_fails)
            f1 = f1_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions, zero_division=0)
            recall = recall_score(ground_truth, predictions, zero_division=0)
            
            return {
                'auroc': auroc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'avg_semantic_entropy': np.mean(semantic_entropies),
                'avg_p_fail': np.mean(p_fails)
            }
        except Exception as e:
            logger.debug(f"Metrics calculation failed: {e}")
            return {
                'auroc': 0.5,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'avg_semantic_entropy': 0.0,
                'avg_p_fail': 0.5
            }
    
    def run_parameter_grid_search(self, validation_samples):
        """Run grid search to find optimal parameters"""
        
        logger.info(f"\n🔧 PARAMETER GRID SEARCH OPTIMIZATION")
        logger.info(f"{'='*60}")
        
        # Define parameter grid
        param_grid = {
            'similarity_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
            'power_factor': [1.5, 2.0, 2.5, 3.0],
            'pfail_threshold': [0.4, 0.5, 0.6, 0.7]
        }
        
        grid = list(ParameterGrid(param_grid))
        logger.info(f"📊 Testing {len(grid)} parameter combinations")
        
        best_config = None
        best_auroc = 0.0
        results = []
        
        # Use subset for faster optimization
        optimization_samples = validation_samples[:200]
        
        for i, params in enumerate(grid):
            if i % 10 == 0:
                logger.info(f"📈 Progress: {i}/{len(grid)} configurations tested")
            
            metrics = self.evaluate_parameter_configuration(
                optimization_samples,
                params['similarity_threshold'],
                params['power_factor'],
                params['pfail_threshold']
            )
            
            results.append({
                'params': params,
                'metrics': metrics
            })
            
            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                best_config = params
                logger.info(f"🎯 New best AUROC: {best_auroc:.1%} with {params}")
        
        logger.info(f"\n🏆 GRID SEARCH RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"🎯 Best AUROC: {best_auroc:.1%}")
        logger.info(f"🔧 Best parameters:")
        for key, value in best_config.items():
            logger.info(f"   {key}: {value}")
        
        return best_config, best_auroc, results
    
    def run_full_evaluation_with_optimal_params(self, test_samples, optimal_params):
        """Run full evaluation with optimal parameters"""
        
        logger.info(f"\n📊 FULL EVALUATION WITH OPTIMAL PARAMETERS")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Test samples: {len(test_samples)}")
        
        predictions = []
        ground_truth = []
        p_fail_scores = []
        semantic_entropies = []
        processing_times = []
        
        start_time = time.time()
        
        for i, sample in enumerate(test_samples):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(test_samples) - i) / rate if rate > 0 else 0
                logger.info(f"📈 Progress: {i}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%) | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
            
            sample_start = time.time()
            result = self.analyze_with_custom_params(
                sample['prompt'],
                sample['output'],
                optimal_params['similarity_threshold'],
                optimal_params['power_factor']
            )
            processing_times.append((time.time() - sample_start) * 1000)
            
            if result['success']:
                p_fail = result['p_fail']
                is_predicted_hallucination = p_fail > optimal_params['pfail_threshold']
                
                predictions.append(is_predicted_hallucination)
                ground_truth.append(sample['is_hallucination'])
                p_fail_scores.append(p_fail)
                semantic_entropies.append(result['semantic_entropy'])
            else:
                predictions.append(False)
                ground_truth.append(sample['is_hallucination'])
                p_fail_scores.append(0.5)
                semantic_entropies.append(0.0)
        
        # Calculate final metrics
        try:
            final_auroc = roc_auc_score(ground_truth, p_fail_scores)
            final_f1 = f1_score(ground_truth, predictions)
            final_precision = precision_score(ground_truth, predictions, zero_division=0)
            final_recall = recall_score(ground_truth, predictions, zero_division=0)
            
            logger.info(f"\n🏆 OPTIMIZED CALIBRATION RESULTS")
            logger.info(f"{'='*50}")
            logger.info(f"🎯 Final AUROC: {final_auroc:.1%} {'🏆' if final_auroc >= 0.79 else '📊'}")
            logger.info(f"📊 Final F1: {final_f1:.3f}")
            logger.info(f"📈 Precision: {final_precision:.3f}")
            logger.info(f"📈 Recall: {final_recall:.3f}")
            
            # Nature 2024 target assessment
            if final_auroc >= 0.79:
                logger.info(f"\n🎉 NATURE 2024 TARGET ACHIEVED!")
                logger.info(f"   🏆 Calibrated AUROC: {final_auroc:.1%} ≥ 79%")
                logger.info(f"   🔧 Optimal parameters enabled breakthrough")
            else:
                gap = 0.79 - final_auroc
                logger.info(f"\n📈 Progress toward 79% AUROC:")
                logger.info(f"   Current: {final_auroc:.1%}")
                logger.info(f"   Gap: {gap:.1%} ({gap*100:.1f} percentage points)")
            
            # Performance stats
            avg_time = np.mean(processing_times)
            throughput = 1000 / avg_time if avg_time > 0 else 0
            
            logger.info(f"\n⚡ Performance Statistics:")
            logger.info(f"   📊 Samples processed: {len(predictions)}")
            logger.info(f"   ⏱️  Avg processing time: {avg_time:.1f}ms")
            logger.info(f"   🚀 Throughput: {throughput:.0f} analyses/sec")
            logger.info(f"   🌊 Avg semantic entropy: {np.mean(semantic_entropies):.3f}")
            logger.info(f"   📊 Avg P(fail): {np.mean(p_fail_scores):.3f}")
            
            # Save results
            results = {
                'evaluation_type': 'calibration_boundary_optimization',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'optimal_parameters': optimal_params,
                'final_metrics': {
                    'auroc': final_auroc,
                    'f1_score': final_f1,
                    'precision': final_precision,
                    'recall': final_recall
                },
                'target_achievement': {
                    'target_auroc': 0.79,
                    'achieved': final_auroc >= 0.79,
                    'gap_percentage_points': max(0, 79 - final_auroc*100)
                },
                'processing_stats': {
                    'samples_processed': len(predictions),
                    'avg_processing_time_ms': avg_time,
                    'throughput_analyses_per_sec': throughput,
                    'avg_semantic_entropy': np.mean(semantic_entropies),
                    'avg_p_fail': np.mean(p_fail_scores)
                }
            }
            
            output_file = "calibration_boundary_optimization_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Results saved to: {output_file}")
            
            return final_auroc
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return 0.0

def main():
    optimizer = CalibrationBoundaryOptimizer()
    
    # Test API connectivity
    try:
        health = requests.get(f"{optimizer.api_url}/health", timeout=5)
        if health.status_code != 200:
            logger.error("❌ API server not responding")
            return
        logger.info("✅ API server is running")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # Load validation samples
    validation_samples = optimizer.load_halueval_samples(max_samples=800)
    
    if len(validation_samples) < 100:
        logger.error("❌ Insufficient validation samples")
        return
    
    # Split into optimization and test sets
    split_point = len(validation_samples) // 2
    train_samples = validation_samples[:split_point]
    test_samples = validation_samples[split_point:]
    
    logger.info(f"📊 Train samples (optimization): {len(train_samples)}")
    logger.info(f"📊 Test samples (evaluation): {len(test_samples)}")
    
    # Step 1: Parameter grid search
    optimal_params, best_auroc, all_results = optimizer.run_parameter_grid_search(train_samples)
    
    # Step 2: Full evaluation with optimal parameters
    final_auroc = optimizer.run_full_evaluation_with_optimal_params(test_samples, optimal_params)
    
    logger.info(f"\n🌟 CALIBRATION OPTIMIZATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"🎯 Training AUROC: {best_auroc:.1%}")
    logger.info(f"🎯 Test AUROC: {final_auroc:.1%}")
    logger.info(f"🔧 Optimal similarity threshold: {optimal_params['similarity_threshold']}")
    logger.info(f"🔧 Optimal power factor: {optimal_params['power_factor']}")
    logger.info(f"🔧 Optimal P(fail) threshold: {optimal_params['pfail_threshold']}")
    
    if final_auroc >= 0.79:
        logger.info(f"🏆 SUCCESS! Nature 2024 target achieved via calibration optimization")
    else:
        logger.info(f"📈 {final_auroc:.1%} toward 79% target (gap: {79-final_auroc*100:.1f}pp)")
        
        # Next optimization suggestions
        if final_auroc >= 0.75:
            logger.info(f"🔥 VERY CLOSE! Consider contradiction detection clustering")
        elif final_auroc >= 0.70:
            logger.info(f"⚡ Getting close! Try ensemble with Fisher information")
        else:
            logger.info(f"🔧 Need deeper optimization - try adaptive learning methods")

if __name__ == "__main__":
    main()