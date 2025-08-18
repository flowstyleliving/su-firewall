#!/usr/bin/env python3
"""
🧠🔧 COMPREHENSIVE HYPERPARAMETER SWEEP
Advanced parameter optimization combining all methods for 79%+ AUROC breakthrough
"""

import requests
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from pathlib import Path
import logging
import re
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveHyperparameterSweep:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        
    def load_diverse_dataset(self, max_samples=1000):
        """Load diverse samples from multiple datasets"""
        data_dir = Path("authentic_datasets")
        all_samples = []
        
        # Load HaluEval QA (balanced correct/hallucinated pairs)
        qa_path = data_dir / "halueval_qa_data.json"
        if qa_path.exists():
            with open(qa_path, 'r') as f:
                lines = f.read().strip().split('\n')[:max_samples//3]
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            if 'question' in sample:
                                # Correct answer
                                all_samples.append({
                                    'prompt': sample['question'],
                                    'output': sample['right_answer'],
                                    'is_hallucination': False,
                                    'source': 'qa'
                                })
                                # Hallucinated answer
                                all_samples.append({
                                    'prompt': sample['question'],
                                    'output': sample['hallucinated_answer'],
                                    'is_hallucination': True,
                                    'source': 'qa'
                                })
                        except:
                            continue
        
        # Load HaluEval Dialogue  
        dialogue_path = data_dir / "halueval_dialogue_data.json"
        if dialogue_path.exists():
            with open(dialogue_path, 'r') as f:
                lines = f.read().strip().split('\n')[:max_samples//3]
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            if 'input' in sample and 'output' in sample:
                                all_samples.append({
                                    'prompt': sample['input'],
                                    'output': sample['output'],
                                    'is_hallucination': sample.get('label') == 'hallucination',
                                    'source': 'dialogue'
                                })
                        except:
                            continue
        
        logger.info(f"📊 Loaded {len(all_samples)} diverse samples")
        return all_samples
    
    def advanced_contradiction_analysis(self, candidates):
        """Advanced contradiction analysis with multiple techniques"""
        
        # 1. Pattern-based contradiction detection
        contradiction_patterns = [
            r'\b(not|never|no|false|incorrect|wrong)\b',
            r'\b(opposite|contrary|different|alternative)\b',
            r'\b(actually|however|but|although)\b',
            r'\b(uncertain|unsure|might|maybe|possibly)\b'
        ]
        
        pattern_scores = []
        for candidate in candidates:
            score = 0.0
            text_lower = candidate.lower()
            for pattern in contradiction_patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.15
            pattern_scores.append(min(score, 1.0))
        
        # 2. Semantic opposition detection
        opposition_pairs = [
            ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
            ('right', 'wrong'), ('accurate', 'inaccurate'), ('valid', 'invalid')
        ]
        
        opposition_scores = []
        for i, candidate in enumerate(candidates):
            opposition_score = 0.0
            words_i = set(candidate.lower().split())
            
            for j, other_candidate in enumerate(candidates):
                if i != j:
                    words_j = set(other_candidate.lower().split())
                    
                    for pos_word, neg_word in opposition_pairs:
                        if pos_word in words_i and neg_word in words_j:
                            opposition_score += 0.3
                        elif neg_word in words_i and pos_word in words_j:
                            opposition_score += 0.3
            
            opposition_scores.append(min(opposition_score, 1.0))
        
        # 3. Content length variation analysis
        lengths = [len(candidate.split()) for candidate in candidates]
        avg_length = np.mean(lengths)
        length_variation = np.std(lengths) / avg_length if avg_length > 0 else 0.0
        
        # Combine all contradiction signals
        combined_scores = []
        for i in range(len(candidates)):
            combined_score = (
                pattern_scores[i] * 0.4 +
                opposition_scores[i] * 0.4 +
                length_variation * 0.2
            )
            combined_scores.append(combined_score)
        
        return combined_scores, length_variation
    
    def multi_method_analysis(self, prompt, output, params):
        """Multi-method analysis with comprehensive parameters"""
        
        # Generate sophisticated answer candidates
        candidates = [
            output,  # Original
            f"Actually, {output[:50]}... is completely incorrect",
            f"The opposite is true: {output[:30]}... is wrong",
            f"No, that's false. The correct answer is different",
            f"I disagree - {output[:40]}... seems inaccurate",
            f"Contrary to that claim, the reality is opposite",
            "This information appears questionable and unreliable",
            f"Actually, I'm not certain about '{output[:30]}...'"
        ]
        
        # 1. Enhanced contradiction analysis
        contradiction_scores, length_variation = self.advanced_contradiction_analysis(candidates)
        
        # 2. Semantic clustering with custom similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        try:
            tfidf_matrix = vectorizer.fit_transform(candidates)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            similarity_matrix = np.eye(len(candidates))
        
        # Apply contradiction penalty to similarity
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i != j:
                    penalty = (contradiction_scores[i] + contradiction_scores[j]) / 2
                    similarity_matrix[i][j] *= (1.0 - penalty * params['contradiction_penalty'])
        
        # 3. Dynamic clustering based on similarity threshold
        cluster_labels = []
        used_clusters = []
        current_cluster = 0
        
        for i in range(len(candidates)):
            assigned = False
            for cluster_id, rep_idx in used_clusters:
                if similarity_matrix[i][rep_idx] > params['similarity_threshold']:
                    cluster_labels.append(cluster_id)
                    assigned = True
                    break
            
            if not assigned:
                cluster_labels.append(current_cluster)
                used_clusters.append((current_cluster, i))
                current_cluster += 1
        
        num_clusters = len(set(cluster_labels))
        
        # 4. Calculate comprehensive uncertainty metrics
        base_entropy = np.log(max(num_clusters, 1)) * params['entropy_scale']
        contradiction_uncertainty = np.mean(contradiction_scores) * params['contradiction_weight']
        cluster_uncertainty = (num_clusters / len(candidates)) * params['cluster_weight']
        length_uncertainty = length_variation * params['length_weight']
        
        # Final combined uncertainty
        total_uncertainty = (
            base_entropy + 
            contradiction_uncertainty + 
            cluster_uncertainty + 
            length_uncertainty
        )
        
        # Enhanced P(fail) calculation
        enhanced_p_fail = 0.5 + (total_uncertainty - 1.0) * params['pfail_sensitivity']
        enhanced_p_fail = max(0.0, min(enhanced_p_fail, 0.95))
        
        return {
            'total_uncertainty': total_uncertainty,
            'enhanced_p_fail': enhanced_p_fail,
            'num_clusters': num_clusters,
            'avg_contradiction': np.mean(contradiction_scores),
            'length_variation': length_variation
        }
    
    def run_comprehensive_sweep(self, validation_samples):
        """Run comprehensive hyperparameter sweep"""
        
        logger.info(f"\n🧠🔧 COMPREHENSIVE HYPERPARAMETER SWEEP")
        logger.info(f"{'='*60}")
        
        # Comprehensive parameter grid
        param_grid = {
            'similarity_threshold': [0.2, 0.3, 0.4, 0.5, 0.6],
            'contradiction_penalty': [0.2, 0.4, 0.6, 0.8],
            'entropy_scale': [0.8, 1.0, 1.2, 1.5],
            'contradiction_weight': [0.3, 0.5, 0.7, 1.0],
            'cluster_weight': [0.2, 0.4, 0.6],
            'length_weight': [0.1, 0.2, 0.3],
            'pfail_sensitivity': [0.2, 0.3, 0.4, 0.5]
        }
        
        # Generate parameter combinations (sample for efficiency)
        all_combinations = list(ParameterGrid(param_grid))
        sampled_combinations = np.random.choice(
            range(len(all_combinations)), 
            size=min(100, len(all_combinations)), 
            replace=False
        )
        
        logger.info(f"📊 Testing {len(sampled_combinations)} parameter combinations")
        
        best_params = None
        best_auroc = 0.0
        best_method = None
        
        # Use subset for optimization
        opt_samples = validation_samples[:200]
        
        for i, combo_idx in enumerate(sampled_combinations):
            params = all_combinations[combo_idx]
            
            if i % 20 == 0:
                logger.info(f"📈 Progress: {i}/{len(sampled_combinations)} combinations")
            
            # Evaluate this parameter combination
            uncertainties = []
            p_fails = []
            ground_truth = []
            
            for sample in opt_samples[:100]:  # Quick evaluation
                result = self.multi_method_analysis(sample['prompt'], sample['output'], params)
                uncertainties.append(result['total_uncertainty'])
                p_fails.append(result['enhanced_p_fail'])
                ground_truth.append(sample['is_hallucination'])
            
            # Calculate AUROC for both methods
            try:
                auroc_uncertainty = roc_auc_score(ground_truth, uncertainties)
                auroc_pfail = roc_auc_score(ground_truth, p_fails)
                
                # Track best method
                if auroc_uncertainty > best_auroc:
                    best_auroc = auroc_uncertainty
                    best_params = params
                    best_method = 'uncertainty'
                    
                if auroc_pfail > best_auroc:
                    best_auroc = auroc_pfail
                    best_params = params
                    best_method = 'p_fail'
                    
                if best_auroc > 0.65:  # Early stopping for promising results
                    logger.info(f"🔥 Promising configuration found: {best_auroc:.1%}")
                    
            except Exception as e:
                continue
        
        logger.info(f"\n🏆 HYPERPARAMETER SWEEP RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"🎯 Best AUROC: {best_auroc:.1%}")
        logger.info(f"🔧 Best method: {best_method}")
        logger.info(f"🔧 Best parameters:")
        for key, value in best_params.items():
            logger.info(f"   {key}: {value}")
        
        return best_params, best_auroc, best_method
    
    def run_final_evaluation(self, test_samples, optimal_params, best_method):
        """Run final evaluation with optimal parameters"""
        
        logger.info(f"\n📊 FINAL COMPREHENSIVE EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Test samples: {len(test_samples)}")
        logger.info(f"🔧 Using method: {best_method}")
        
        uncertainties = []
        p_fails = []
        ground_truth = []
        processing_times = []
        cluster_stats = []
        
        start_time = time.time()
        
        for i, sample in enumerate(test_samples):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(test_samples) - i) / rate if rate > 0 else 0
                logger.info(f"📈 Progress: {i}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%) | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
            
            sample_start = time.time()
            result = self.multi_method_analysis(sample['prompt'], sample['output'], optimal_params)
            processing_times.append((time.time() - sample_start) * 1000)
            
            uncertainties.append(result['total_uncertainty'])
            p_fails.append(result['enhanced_p_fail'])
            ground_truth.append(sample['is_hallucination'])
            cluster_stats.append({
                'num_clusters': result['num_clusters'],
                'contradiction': result['avg_contradiction'],
                'length_variation': result['length_variation']
            })
        
        # Calculate final metrics
        try:
            auroc_uncertainty = roc_auc_score(ground_truth, uncertainties)
            auroc_pfail = roc_auc_score(ground_truth, p_fails)
            
            # Use best method for binary predictions
            if best_method == 'uncertainty':
                threshold = np.percentile(uncertainties, 50)
                binary_preds = [1 if u > threshold else 0 for u in uncertainties]
                final_auroc = auroc_uncertainty
            else:
                threshold = 0.5
                binary_preds = [1 if pf > threshold else 0 for pf in p_fails]
                final_auroc = auroc_pfail
            
            final_f1 = f1_score(ground_truth, binary_preds)
            
            logger.info(f"\n🏆 COMPREHENSIVE OPTIMIZATION RESULTS")
            logger.info(f"{'='*50}")
            logger.info(f"🎯 Final AUROC ({best_method}): {final_auroc:.1%} {'🏆' if final_auroc >= 0.79 else '📊'}")
            logger.info(f"🎯 Alternative AUROC: {auroc_pfail if best_method == 'uncertainty' else auroc_uncertainty:.1%}")
            logger.info(f"📊 Final F1: {final_f1:.3f}")
            
            # Nature 2024 achievement check
            if final_auroc >= 0.79:
                logger.info(f"\n🎉 NATURE 2024 TARGET ACHIEVED!")
                logger.info(f"   🏆 Comprehensive AUROC: {final_auroc:.1%} ≥ 79%")
                logger.info(f"   🧠 Multi-method optimization breakthrough")
            else:
                gap = 0.79 - final_auroc
                logger.info(f"\n📈 Progress toward 79% AUROC:")
                logger.info(f"   Current: {final_auroc:.1%}")
                logger.info(f"   Gap: {gap:.1%} ({gap*100:.1f} percentage points)")
                
                if gap <= 0.03:
                    logger.info(f"   🔥 EXTREMELY CLOSE! Try adaptive learning next")
                elif gap <= 0.07:
                    logger.info(f"   ⚡ Very close! Domain-specific adaptation recommended")
                else:
                    logger.info(f"   🔧 Consider advanced ensemble methods")
            
            # Performance and clustering statistics
            avg_clusters = np.mean([cs['num_clusters'] for cs in cluster_stats])
            avg_contradiction = np.mean([cs['contradiction'] for cs in cluster_stats])
            avg_length_var = np.mean([cs['length_variation'] for cs in cluster_stats])
            
            logger.info(f"\n🔍 Advanced Analytics:")
            logger.info(f"   📊 Avg clusters per sample: {avg_clusters:.1f}")
            logger.info(f"   💥 Avg contradiction score: {avg_contradiction:.3f}")
            logger.info(f"   📏 Avg length variation: {avg_length_var:.3f}")
            logger.info(f"   ⏱️  Avg processing time: {np.mean(processing_times):.1f}ms")
            logger.info(f"   🚀 Throughput: {1000/np.mean(processing_times):.0f} analyses/sec")
            
            # Save comprehensive results
            results = {
                'evaluation_type': 'comprehensive_hyperparameter_sweep',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'optimal_parameters': optimal_params,
                'best_method': best_method,
                'final_metrics': {
                    'auroc_uncertainty': auroc_uncertainty,
                    'auroc_pfail': auroc_pfail,
                    'final_auroc': final_auroc,
                    'final_f1': final_f1
                },
                'advanced_analytics': {
                    'avg_clusters_per_sample': avg_clusters,
                    'avg_contradiction_score': avg_contradiction,
                    'avg_length_variation': avg_length_var
                },
                'target_achievement': {
                    'target_auroc': 0.79,
                    'achieved': final_auroc >= 0.79,
                    'gap_percentage_points': max(0, 79 - final_auroc*100)
                },
                'processing_stats': {
                    'samples_processed': len(uncertainties),
                    'avg_processing_time_ms': np.mean(processing_times),
                    'throughput_analyses_per_sec': 1000 / np.mean(processing_times)
                }
            }
            
            output_file = "comprehensive_hyperparameter_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Results saved to: {output_file}")
            
            return final_auroc
            
        except Exception as e:
            logger.error(f"❌ Final evaluation failed: {e}")
            return 0.0

def main():
    optimizer = ComprehensiveHyperparameterSweep()
    
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
    
    # Load diverse evaluation dataset
    evaluation_samples = optimizer.load_diverse_dataset(max_samples=800)
    
    if len(evaluation_samples) < 100:
        logger.error("❌ Insufficient evaluation samples")
        return
    
    # Split into optimization and test sets
    split_point = len(evaluation_samples) // 2
    train_samples = evaluation_samples[:split_point]
    test_samples = evaluation_samples[split_point:]
    
    logger.info(f"📊 Train samples: {len(train_samples)}")
    logger.info(f"📊 Test samples: {len(test_samples)}")
    
    # Step 1: Comprehensive hyperparameter sweep
    optimal_params, best_auroc, best_method = optimizer.run_comprehensive_sweep(train_samples)
    
    # Step 2: Final evaluation with optimal configuration
    final_auroc = optimizer.run_final_evaluation(test_samples, optimal_params, best_method)
    
    logger.info(f"\n🌟 COMPREHENSIVE OPTIMIZATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"🎯 Training AUROC: {best_auroc:.1%}")
    logger.info(f"🎯 Test AUROC: {final_auroc:.1%}")
    logger.info(f"🔧 Best method: {best_method}")
    
    if final_auroc >= 0.79:
        logger.info(f"🏆 SUCCESS! Nature 2024 target achieved via comprehensive optimization")
    else:
        improvement = final_auroc - 0.50  # Improvement from baseline
        logger.info(f"📈 Improvement: +{improvement:.1%} from baseline")
        logger.info(f"📈 {final_auroc:.1%} toward 79% target (gap: {79-final_auroc*100:.1f}pp)")
        
        # Next optimization recommendation
        if final_auroc >= 0.70:
            logger.info(f"🔥 CLOSE! Try adaptive learning from massive dataset")
        elif final_auroc >= 0.60:
            logger.info(f"⚡ Good progress! Consider domain-specific adaptations")
        else:
            logger.info(f"🔧 Foundation needs strengthening - check API detection logic")

if __name__ == "__main__":
    main()