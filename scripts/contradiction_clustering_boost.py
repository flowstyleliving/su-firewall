#!/usr/bin/env python3
"""
🔍💥 CONTRADICTION DETECTION CLUSTERING BOOST
Advanced semantic clustering with contradiction detection for 79%+ AUROC
"""

import requests
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import time
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ContradictionClusteringBoost:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        self.contradiction_patterns = [
            # Direct contradictions
            r'(not|never|no|false|incorrect|wrong)',
            r'(opposite|contrary|different|alternative)',
            r'(actually|however|but|although)',
            
            # Uncertainty markers
            r'(uncertain|unsure|might|maybe|possibly|perhaps)',
            r'(seems|appears|could be|may be)',
            
            # Contradiction indicators
            r'(instead|rather|alternatively|on the contrary)',
            r'(contradicts|conflicts|disputes|challenges)'
        ]
        
    def load_evaluation_dataset(self, max_samples=1200):
        """Load evaluation dataset for contradiction clustering"""
        data_dir = Path("authentic_datasets")
        all_samples = []
        
        # Load HaluEval QA with balanced correct/incorrect pairs
        filepath = data_dir / "halueval_qa_data.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                content = f.read().strip()
                lines = content.split('\n')[:max_samples//2]  # Limit base samples
                
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
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
                            
        logger.info(f"📊 Loaded {len(all_samples)} samples")
        
        # Check label distribution
        hallucination_count = sum(1 for s in all_samples if s['is_hallucination'])
        correct_count = len(all_samples) - hallucination_count
        logger.info(f"   🔍 Hallucinations: {hallucination_count}")
        logger.info(f"   ✅ Correct: {correct_count}")
        
        return all_samples
    
    def detect_contradictions(self, candidates):
        """Detect contradictions between answer candidates"""
        contradiction_scores = []
        
        for i, candidate in enumerate(candidates):
            contradiction_score = 0.0
            
            # Pattern-based contradiction detection
            text_lower = candidate.lower()
            for pattern in self.contradiction_patterns:
                if re.search(pattern, text_lower):
                    contradiction_score += 0.2
            
            # Cross-candidate contradiction detection
            for j, other_candidate in enumerate(candidates):
                if i != j:
                    # Simple negation detection
                    if any(word in candidate.lower() and f"not {word}" in other_candidate.lower() 
                           for word in ['is', 'are', 'was', 'were', 'can', 'will', 'should']):
                        contradiction_score += 0.3
                    
                    # Semantic opposition detection
                    positive_words = ['yes', 'true', 'correct', 'accurate', 'right']
                    negative_words = ['no', 'false', 'incorrect', 'wrong', 'inaccurate']
                    
                    candidate_words = set(candidate.lower().split())
                    other_words = set(other_candidate.lower().split())
                    
                    if (any(w in candidate_words for w in positive_words) and 
                        any(w in other_words for w in negative_words)):
                        contradiction_score += 0.4
            
            contradiction_scores.append(min(contradiction_score, 1.0))  # Cap at 1.0
            
        return contradiction_scores
    
    def enhanced_semantic_clustering(self, candidates):
        """Enhanced clustering with contradiction detection"""
        if len(candidates) < 2:
            return 1, [0.0] * len(candidates)
        
        # Create TF-IDF vectors for semantic similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(candidates)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            # Fallback if TF-IDF fails
            return len(candidates), [0.0] * len(candidates)
        
        # Detect contradictions
        contradiction_scores = self.detect_contradictions(candidates)
        
        # Enhanced clustering with contradiction weighting
        # Reduce similarity for contradictory pairs
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i != j:
                    contradiction_penalty = (contradiction_scores[i] + contradiction_scores[j]) / 2
                    similarity_matrix[i][j] *= (1.0 - contradiction_penalty * 0.5)
        
        # Simple clustering based on similarity threshold
        cluster_labels = []
        used_clusters = []
        current_cluster = 0
        
        for i in range(len(candidates)):
            assigned = False
            # Check if this candidate is similar to any existing cluster representative
            for cluster_id, representative_idx in used_clusters:
                if similarity_matrix[i][representative_idx] > 0.7:  # Similarity threshold
                    cluster_labels.append(cluster_id)
                    assigned = True
                    break
            
            if not assigned:
                # Create new cluster
                cluster_labels.append(current_cluster)
                used_clusters.append((current_cluster, i))
                current_cluster += 1
        
        # Count unique clusters
        unique_clusters = len(set(cluster_labels))
        if unique_clusters == 0:
            unique_clusters = len(candidates)  # Each is its own cluster
        
        # Calculate cluster coherence scores
        cluster_scores = []
        for i, candidate in enumerate(candidates):
            cluster_id = cluster_labels[i]
            
            # Calculate intra-cluster similarity
            same_cluster_indices = [j for j, cid in enumerate(cluster_labels) if cid == cluster_id]
            if len(same_cluster_indices) > 1:
                similarities = [similarity_matrix[i][j] for j in same_cluster_indices if j != i]
                avg_similarity = np.mean(similarities) if similarities else 0.0
                # Add contradiction boost
                contradiction_boost = contradiction_scores[i] * 0.3
                cluster_scores.append(max(0.0, 1.0 - avg_similarity + contradiction_boost))
            else:
                # Single-member cluster gets high uncertainty + contradiction score
                cluster_scores.append(0.5 + contradiction_scores[i] * 0.3)
        
        return unique_clusters, cluster_scores
    
    def analyze_with_enhanced_clustering(self, prompt, output):
        """Analyze with enhanced contradiction-aware clustering"""
        
        # Generate enhanced candidates with more semantic diversity
        candidates = [
            output,
            f"Actually, {output[:50]}... is completely wrong",
            f"The correct answer is the opposite of '{output[:30]}...'",
            "I'm not sure about this - it seems incorrect",
            f"No, {output[:40]}... is false information",
            f"Contrary to that claim, the truth is different",
            "This appears to be inaccurate or misleading"
        ]
        
        # Enhanced clustering
        num_clusters, cluster_scores = self.enhanced_semantic_clustering(candidates)
        
        # Calculate enhanced semantic entropy
        # Higher cluster count + contradiction signals = higher uncertainty
        base_entropy = np.log(max(num_clusters, 1))
        contradiction_boost = np.mean(cluster_scores) * 0.5
        enhanced_entropy = base_entropy + contradiction_boost
        
        # Enhanced P(fail) calculation
        cluster_penalty = 1.0 - (num_clusters / len(candidates))  # More clusters = higher uncertainty
        contradiction_penalty = np.mean(cluster_scores)
        enhanced_p_fail = 0.5 + (cluster_penalty + contradiction_penalty) * 0.25
        enhanced_p_fail = min(enhanced_p_fail, 0.95)  # Cap at 95%
        
        return {
            'enhanced_semantic_entropy': enhanced_entropy,
            'enhanced_p_fail': enhanced_p_fail,
            'num_clusters': num_clusters,
            'avg_contradiction_score': np.mean(cluster_scores),
            'cluster_scores': cluster_scores
        }
    
    def run_contradiction_clustering_evaluation(self, samples):
        """Run evaluation with enhanced contradiction clustering"""
        
        logger.info(f"\n🔍💥 CONTRADICTION CLUSTERING EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Samples: {len(samples)}")
        
        enhanced_entropies = []
        enhanced_p_fails = []
        ground_truth = []
        processing_times = []
        cluster_stats = []
        
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(samples) - i) / rate if rate > 0 else 0
                logger.info(f"📈 Progress: {i}/{len(samples)} ({i/len(samples)*100:.1f}%) | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
            
            sample_start = time.time()
            result = self.analyze_with_enhanced_clustering(sample['prompt'], sample['output'])
            processing_times.append((time.time() - sample_start) * 1000)
            
            enhanced_entropies.append(result['enhanced_semantic_entropy'])
            enhanced_p_fails.append(result['enhanced_p_fail'])
            ground_truth.append(sample['is_hallucination'])
            cluster_stats.append({
                'num_clusters': result['num_clusters'],
                'avg_contradiction': result['avg_contradiction_score']
            })
        
        # Calculate performance metrics
        try:
            # AUROC using enhanced P(fail)
            auroc_enhanced_pfail = roc_auc_score(ground_truth, enhanced_p_fails)
            auroc_enhanced_entropy = roc_auc_score(ground_truth, enhanced_entropies)
            
            # Binary predictions
            pfail_median = np.median(enhanced_p_fails)
            entropy_median = np.median(enhanced_entropies)
            
            pfail_binary = [1 if pf > pfail_median else 0 for pf in enhanced_p_fails]
            entropy_binary = [1 if se > entropy_median else 0 for se in enhanced_entropies]
            
            f1_pfail = f1_score(ground_truth, pfail_binary)
            f1_entropy = f1_score(ground_truth, entropy_binary)
            
            logger.info(f"\n🏆 CONTRADICTION CLUSTERING RESULTS")
            logger.info(f"{'='*50}")
            logger.info(f"🎯 AUROC Scores:")
            logger.info(f"   🔍 Enhanced P(fail): {auroc_enhanced_pfail:.1%} {'🏆' if auroc_enhanced_pfail >= 0.79 else '📊'}")
            logger.info(f"   💥 Enhanced Entropy: {auroc_enhanced_entropy:.1%} {'🏆' if auroc_enhanced_entropy >= 0.79 else '📊'}")
            
            logger.info(f"\n📊 F1 Scores:")
            logger.info(f"   🔍 Enhanced P(fail) F1: {f1_pfail:.3f}")
            logger.info(f"   💥 Enhanced Entropy F1: {f1_entropy:.3f}")
            
            # Best performance indicator
            best_auroc = max(auroc_enhanced_pfail, auroc_enhanced_entropy)
            
            if best_auroc >= 0.79:
                logger.info(f"\n🎉 NATURE 2024 TARGET ACHIEVED!")
                logger.info(f"   🏆 Best AUROC: {best_auroc:.1%} ≥ 79%")
                logger.info(f"   🔍 Contradiction clustering breakthrough confirmed")
            else:
                gap = 0.79 - best_auroc
                logger.info(f"\n📈 Progress toward 79% AUROC:")
                logger.info(f"   Current best: {best_auroc:.1%}")
                logger.info(f"   Gap: {gap:.1%} ({gap*100:.1f} percentage points)")
                
                if gap <= 0.05:
                    logger.info(f"   🔥 VERY CLOSE! Try hyperparameter sweep next")
                elif gap <= 0.10:
                    logger.info(f"   ⚡ Getting close! Consider adaptive learning")
            
            # Clustering statistics
            avg_clusters = np.mean([cs['num_clusters'] for cs in cluster_stats])
            avg_contradictions = np.mean([cs['avg_contradiction'] for cs in cluster_stats])
            
            logger.info(f"\n🔍 Clustering Statistics:")
            logger.info(f"   📊 Avg clusters per sample: {avg_clusters:.1f}")
            logger.info(f"   💥 Avg contradiction score: {avg_contradictions:.3f}")
            logger.info(f"   ⏱️  Avg processing time: {np.mean(processing_times):.1f}ms")
            logger.info(f"   🚀 Throughput: {1000/np.mean(processing_times):.0f} analyses/sec")
            
            # Save results
            results = {
                'evaluation_type': 'contradiction_clustering_boost',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'performance_metrics': {
                    'auroc_enhanced_pfail': auroc_enhanced_pfail,
                    'auroc_enhanced_entropy': auroc_enhanced_entropy,
                    'f1_enhanced_pfail': f1_pfail,
                    'f1_enhanced_entropy': f1_entropy,
                    'best_auroc': best_auroc
                },
                'clustering_stats': {
                    'avg_clusters_per_sample': avg_clusters,
                    'avg_contradiction_score': avg_contradictions
                },
                'target_achievement': {
                    'target_auroc': 0.79,
                    'achieved': best_auroc >= 0.79,
                    'gap_percentage_points': max(0, 79 - best_auroc*100)
                },
                'processing_stats': {
                    'samples_processed': len(enhanced_p_fails),
                    'avg_processing_time_ms': np.mean(processing_times),
                    'throughput_analyses_per_sec': 1000 / np.mean(processing_times)
                }
            }
            
            output_file = "contradiction_clustering_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Results saved to: {output_file}")
            
            return best_auroc
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return 0.0

def main():
    booster = ContradictionClusteringBoost()
    
    # Test API connectivity
    try:
        health = requests.get(f"{booster.api_url}/health", timeout=5)
        if health.status_code != 200:
            logger.error("❌ API server not responding")
            return
        logger.info("✅ API server is running")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # Load evaluation samples
    evaluation_samples = booster.load_evaluation_dataset(max_samples=800)
    
    if len(evaluation_samples) < 100:
        logger.error("❌ Insufficient evaluation samples")
        return
    
    # Run contradiction clustering evaluation
    final_auroc = booster.run_contradiction_clustering_evaluation(evaluation_samples)
    
    logger.info(f"\n🌟 CONTRADICTION CLUSTERING SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"🎯 Achieved AUROC: {final_auroc:.1%}")
    
    if final_auroc >= 0.79:
        logger.info(f"🏆 SUCCESS! Nature 2024 target achieved via contradiction clustering")
    else:
        logger.info(f"📈 {final_auroc:.1%} toward 79% target (gap: {79-final_auroc*100:.1f}pp)")
        
        # Next steps based on performance
        if final_auroc >= 0.75:
            logger.info(f"🔥 VERY CLOSE! Try hyperparameter sweep + adaptive learning")
        elif final_auroc >= 0.65:
            logger.info(f"⚡ Good progress! Combine with ensemble methods")
        else:
            logger.info(f"🔧 Need foundational improvements to API detection logic")

if __name__ == "__main__":
    main()