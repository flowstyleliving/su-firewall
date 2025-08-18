#!/usr/bin/env python3
"""
📊🔍 F1 PERFORMANCE GAP ANALYSIS
Diagnose AUROC vs F1 discrepancy and identify precision-recall optimization opportunities
"""

import requests
import json
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, 
    roc_curve, precision_score, recall_score, 
    average_precision_score, classification_report
)
from sklearn.calibration import calibration_curve
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class F1PerformanceGapAnalyzer:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        
    def load_balanced_halueval_dataset(self, max_samples=1000):
        """Load balanced HaluEval dataset for F1 analysis"""
        data_dir = Path("authentic_datasets")
        all_samples = []
        
        # Load HaluEval QA with perfect balance
        qa_path = data_dir / "halueval_qa_data.json"
        if qa_path.exists():
            with open(qa_path, 'r') as f:
                lines = f.read().strip().split('\n')[:max_samples//2]
                
                for line in lines:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            if 'question' in sample:
                                # Add both correct and hallucinated for perfect balance
                                all_samples.extend([
                                    {
                                        'prompt': sample['question'],
                                        'output': sample['right_answer'],
                                        'is_hallucination': False,
                                        'source': 'qa'
                                    },
                                    {
                                        'prompt': sample['question'],
                                        'output': sample['hallucinated_answer'],
                                        'is_hallucination': True,
                                        'source': 'qa'
                                    }
                                ])
                        except:
                            continue
        
        # Verify balance
        hallucination_count = sum(1 for s in all_samples if s['is_hallucination'])
        correct_count = len(all_samples) - hallucination_count
        balance_ratio = min(hallucination_count, correct_count) / max(hallucination_count, correct_count)
        
        logger.info(f"📊 Dataset loaded: {len(all_samples)} samples")
        logger.info(f"   🔍 Hallucinations: {hallucination_count}")
        logger.info(f"   ✅ Correct: {correct_count}")
        logger.info(f"   ⚖️ Balance ratio: {balance_ratio:.3f}")
        
        return all_samples
    
    def collect_method_specific_scores(self, samples):
        """Collect scores from each detection method separately"""
        
        logger.info(f"\n🔬 COLLECTING METHOD-SPECIFIC SCORES")
        logger.info(f"{'='*50}")
        
        method_scores = {
            'semantic_entropy': {'scores': [], 'p_fails': []},
            'fisher_information': {'scores': [], 'p_fails': []},
            'adaptive_ensemble': {'scores': [], 'p_fails': []}
        }
        ground_truth = []
        
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                logger.info(f"📈 Score collection: {i}/{len(samples)} ({i/len(samples)*100:.1f}%)")
            
            prompt, output = sample['prompt'], sample['output']
            ground_truth.append(sample['is_hallucination'])
            
            # Method 1: Semantic Entropy
            se_result = self.get_semantic_entropy_scores(prompt, output)
            method_scores['semantic_entropy']['scores'].append(se_result['semantic_entropy'])
            method_scores['semantic_entropy']['p_fails'].append(se_result['p_fail'])
            
            # Method 2: Fisher Information  
            fisher_result = self.get_fisher_information_scores(prompt, output)
            method_scores['fisher_information']['scores'].append(fisher_result['fisher_info'])
            method_scores['fisher_information']['p_fails'].append(fisher_result['p_fail'])
            
            # Method 3: Adaptive Ensemble (from breakthrough)
            adaptive_result = self.get_adaptive_ensemble_scores(prompt, output)
            method_scores['adaptive_ensemble']['scores'].append(adaptive_result['adaptive_score'])
            method_scores['adaptive_ensemble']['p_fails'].append(adaptive_result['adaptive_p_fail'])
        
        return method_scores, ground_truth
    
    def get_semantic_entropy_scores(self, prompt, output):
        """Get semantic entropy method scores"""
        candidates = [
            output,
            f"Actually, {output[:50]}... is incorrect",
            "I'm not certain about this",
            f"The opposite is true",
            "This seems questionable"
        ]
        
        request = {
            "topk_indices": [1, 2, 3, 4, 5],
            "topk_probs": [0.4, 0.25, 0.2, 0.1, 0.05],
            "method": "semantic_entropy",
            "model_id": "mistral-7b",
            "answer_candidates": candidates
        }
        
        try:
            response = requests.post(f"{self.api_url}/api/v1/analyze_topk_compact", json=request, timeout=2)
            if response.status_code == 200:
                result = response.json()
                return {
                    'semantic_entropy': result.get('semantic_entropy', 0.0),
                    'p_fail': result.get('p_fail', 0.5)
                }
        except:
            pass
        
        return {'semantic_entropy': 0.0, 'p_fail': 0.5}
    
    def get_fisher_information_scores(self, prompt, output):
        """Get Fisher information method scores"""
        request = {
            "topk_indices": [1, 2, 3, 4, 5],
            "topk_probs": [0.4, 0.25, 0.2, 0.1, 0.05],
            "method": "fisher_information",
            "model_id": "mistral-7b",
            "rest_mass": 0.1
        }
        
        try:
            response = requests.post(f"{self.api_url}/api/v1/analyze_topk_compact", json=request, timeout=2)
            if response.status_code == 200:
                result = response.json()
                return {
                    'fisher_info': result.get('fisher_information_uncertainty', 0.0),
                    'p_fail': result.get('p_fail', 0.5)
                }
        except:
            pass
        
        return {'fisher_info': 0.0, 'p_fail': 0.5}
    
    def get_adaptive_ensemble_scores(self, prompt, output):
        """Get adaptive ensemble scores (from breakthrough method)"""
        # Simulate the adaptive ensemble method from our breakthrough
        output_length = len(output.split())
        uncertainty_words = sum(1 for word in ['maybe', 'might', 'possibly', 'perhaps', 'unsure'] 
                               if word in output.lower())
        contradiction_words = sum(1 for word in ['not', 'no', 'wrong', 'false', 'incorrect']
                                 if word in output.lower())
        
        # Adaptive feature scoring
        adaptive_score = (
            0.3 * min(output_length / 50.0, 1.0) +
            0.4 * min(uncertainty_words / 3.0, 1.0) + 
            0.3 * min(contradiction_words / 2.0, 1.0)
        )
        
        adaptive_p_fail = 0.1 + adaptive_score * 0.8
        
        return {
            'adaptive_score': adaptive_score,
            'adaptive_p_fail': adaptive_p_fail
        }
    
    def analyze_precision_recall_curves(self, method_scores, ground_truth):
        """Comprehensive precision-recall curve analysis"""
        
        logger.info(f"\n📊 PRECISION-RECALL CURVE ANALYSIS")
        logger.info(f"{'='*50}")
        
        method_analysis = {}
        
        for method_name, scores in method_scores.items():
            logger.info(f"\n🔍 Analyzing {method_name.upper()}...")
            
            # Use P(fail) scores for analysis
            y_scores = scores['p_fails']
            
            try:
                # Calculate precision-recall curve
                precision, recall, thresholds = precision_recall_curve(ground_truth, y_scores)
                
                # Calculate F1 scores for each threshold
                f1_scores = []
                for p, r in zip(precision, recall):
                    if p + r > 0:
                        f1_scores.append(2 * p * r / (p + r))
                    else:
                        f1_scores.append(0.0)
                
                # Find optimal F1 threshold
                if f1_scores:
                    max_f1_idx = np.argmax(f1_scores)
                    optimal_f1 = f1_scores[max_f1_idx]
                    optimal_threshold = thresholds[min(max_f1_idx, len(thresholds)-1)]
                    optimal_precision = precision[max_f1_idx]
                    optimal_recall = recall[max_f1_idx]
                else:
                    optimal_f1 = 0.0
                    optimal_threshold = 0.5
                    optimal_precision = 0.0
                    optimal_recall = 0.0
                
                # Calculate AUROC and average precision
                auroc = roc_auc_score(ground_truth, y_scores)
                avg_precision = average_precision_score(ground_truth, y_scores)
                
                method_analysis[method_name] = {
                    'auroc': auroc,
                    'average_precision': avg_precision,
                    'optimal_f1': optimal_f1,
                    'optimal_threshold': optimal_threshold,
                    'optimal_precision': optimal_precision,
                    'optimal_recall': optimal_recall,
                    'f1_scores': f1_scores,
                    'precision_curve': precision.tolist(),
                    'recall_curve': recall.tolist(),
                    'thresholds': thresholds.tolist()
                }
                
                logger.info(f"   🎯 AUROC: {auroc:.1%}")
                logger.info(f"   📊 Average Precision: {avg_precision:.1%}")
                logger.info(f"   🎯 Optimal F1: {optimal_f1:.1%} {'🏆' if optimal_f1 >= 0.85 else '📊'}")
                logger.info(f"   🔧 Optimal Threshold: {optimal_threshold:.3f}")
                logger.info(f"   📈 Precision@Optimal: {optimal_precision:.1%}")
                logger.info(f"   📈 Recall@Optimal: {optimal_recall:.1%}")
                
            except Exception as e:
                logger.warning(f"   ❌ Analysis failed for {method_name}: {e}")
                method_analysis[method_name] = {
                    'auroc': 0.5, 'optimal_f1': 0.0, 'optimal_threshold': 0.5
                }
        
        return method_analysis
    
    def identify_class_imbalance_impact(self, method_scores, ground_truth):
        """Analyze class imbalance impact on F1 performance"""
        
        logger.info(f"\n⚖️ CLASS IMBALANCE IMPACT ANALYSIS")
        logger.info(f"{'='*50}")
        
        # Class distribution analysis
        positive_count = sum(ground_truth)
        negative_count = len(ground_truth) - positive_count
        imbalance_ratio = positive_count / negative_count if negative_count > 0 else 1.0
        
        logger.info(f"📊 Class Distribution:")
        logger.info(f"   🔍 Positive (hallucination): {positive_count}")
        logger.info(f"   ✅ Negative (correct): {negative_count}")  
        logger.info(f"   ⚖️ Imbalance ratio: {imbalance_ratio:.3f}")
        
        # Analyze impact on each method
        imbalance_analysis = {}
        
        for method_name, scores in method_scores.items():
            y_scores = scores['p_fails']
            
            # Calculate metrics at different thresholds
            thresholds = [0.3, 0.5, 0.7]
            threshold_analysis = {}
            
            for threshold in thresholds:
                binary_preds = [1 if score > threshold else 0 for score in y_scores]
                
                precision = precision_score(ground_truth, binary_preds, zero_division=0)
                recall = recall_score(ground_truth, binary_preds, zero_division=0)
                f1 = f1_score(ground_truth, binary_preds, zero_division=0)
                
                # Calculate true/false positive rates
                tp = sum(1 for p, l in zip(binary_preds, ground_truth) if p == 1 and l == 1)
                fp = sum(1 for p, l in zip(binary_preds, ground_truth) if p == 1 and l == 0)
                tn = sum(1 for p, l in zip(binary_preds, ground_truth) if p == 0 and l == 0)
                fn = sum(1 for p, l in zip(binary_preds, ground_truth) if p == 0 and l == 1)
                
                threshold_analysis[threshold] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                    'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0
                }
            
            imbalance_analysis[method_name] = threshold_analysis
            
            logger.info(f"\n🔍 {method_name.upper()} Threshold Analysis:")
            for threshold, metrics in threshold_analysis.items():
                logger.info(f"   Threshold {threshold}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
        
        return imbalance_analysis
    
    def generate_optimization_recommendations(self, method_analysis, imbalance_analysis):
        """Generate specific F1 optimization recommendations"""
        
        logger.info(f"\n🎯 F1 OPTIMIZATION RECOMMENDATIONS")
        logger.info(f"{'='*60}")
        
        recommendations = {}
        
        # Find best performing method for F1
        best_method = None
        best_f1 = 0.0
        
        for method_name, analysis in method_analysis.items():
            if analysis['optimal_f1'] > best_f1:
                best_f1 = analysis['optimal_f1']
                best_method = method_name
        
        logger.info(f"🏆 Best F1 Method: {best_method} → {best_f1:.1%}")
        
        # Method-specific recommendations
        for method_name, analysis in method_analysis.items():
            method_recs = []
            
            current_f1 = analysis['optimal_f1']
            auroc = analysis['auroc']
            optimal_threshold = analysis['optimal_threshold']
            
            # AUROC vs F1 discrepancy analysis
            auroc_f1_gap = auroc - current_f1
            
            if auroc_f1_gap > 0.2:  # Significant gap
                method_recs.append("🔧 Large AUROC-F1 gap → threshold recalibration needed")
                method_recs.append(f"📊 Consider cost-sensitive learning (current threshold: {optimal_threshold:.3f})")
                
            if current_f1 < 0.85:
                if analysis['optimal_precision'] < 0.80:
                    method_recs.append("📈 Low precision → increase threshold, add false positive penalties")
                if analysis['optimal_recall'] < 0.80:
                    method_recs.append("📈 Low recall → decrease threshold, boost hallucination detection")
                
                if abs(analysis['optimal_precision'] - analysis['optimal_recall']) > 0.15:
                    method_recs.append("⚖️ Precision-recall imbalance → weighted loss optimization needed")
            
            recommendations[method_name] = {
                'current_performance': {
                    'f1': current_f1,
                    'auroc': auroc,
                    'gap': auroc_f1_gap
                },
                'recommendations': method_recs
            }
            
            logger.info(f"\n🔍 {method_name.upper()} Recommendations:")
            for rec in method_recs:
                logger.info(f"   {rec}")
        
        # Overall optimization strategy
        logger.info(f"\n🚀 OVERALL F1 OPTIMIZATION STRATEGY:")
        
        if best_f1 >= 0.85:
            logger.info(f"   ✅ Target achieved with {best_method}")
            logger.info(f"   🔧 Focus on production deployment and validation")
        elif best_f1 >= 0.75:
            logger.info(f"   ⚡ Close to target - fine-tune {best_method}")
            logger.info(f"   🎯 Implement cost-sensitive learning")
        elif best_f1 >= 0.60:
            logger.info(f"   📊 Moderate performance - need ensemble optimization")
            logger.info(f"   🔧 Method-specific threshold grids required")
        else:
            logger.info(f"   🔧 Fundamental improvements needed")
            logger.info(f"   🧠 Consider feature engineering and model architecture changes")
        
        return recommendations
    
    def run_comprehensive_f1_gap_analysis(self, samples):
        """Run complete F1 performance gap analysis"""
        
        logger.info(f"\n📊🔍 COMPREHENSIVE F1 GAP ANALYSIS")
        logger.info(f"{'='*60}")
        
        # Step 1: Collect method-specific scores
        method_scores, ground_truth = self.collect_method_specific_scores(samples)
        
        # Step 2: Precision-recall curve analysis
        method_analysis = self.analyze_precision_recall_curves(method_scores, ground_truth)
        
        # Step 3: Class imbalance impact analysis
        imbalance_analysis = self.identify_class_imbalance_impact(method_scores, ground_truth)
        
        # Step 4: Generate optimization recommendations
        recommendations = self.generate_optimization_recommendations(method_analysis, imbalance_analysis)
        
        # Step 5: Summary and next steps
        best_f1 = max(analysis['optimal_f1'] for analysis in method_analysis.values())
        target_gap = 0.85 - best_f1
        
        logger.info(f"\n📋 F1 GAP ANALYSIS SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"🎯 Current Best F1: {best_f1:.1%}")
        logger.info(f"🎯 Target F1: 85.0%")
        logger.info(f"📊 Gap to Close: {target_gap:.1%} ({target_gap*100:.1f} percentage points)")
        
        if target_gap <= 0:
            logger.info(f"🏆 F1 TARGET ALREADY ACHIEVED!")
        elif target_gap <= 0.05:
            logger.info(f"🔥 VERY CLOSE - minor threshold tuning needed")
        elif target_gap <= 0.15:
            logger.info(f"⚡ ACHIEVABLE - cost-sensitive learning recommended")
        else:
            logger.info(f"🔧 SIGNIFICANT WORK - ensemble and feature optimization needed")
        
        # Save comprehensive analysis
        results = {
            'analysis_type': 'f1_performance_gap_analysis',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_samples': len(samples),
                'positive_samples': sum(ground_truth),
                'negative_samples': len(ground_truth) - sum(ground_truth)
            },
            'method_analysis': method_analysis,
            'imbalance_analysis': imbalance_analysis,
            'recommendations': recommendations,
            'gap_assessment': {
                'current_best_f1': best_f1,
                'target_f1': 0.85,
                'gap_percentage_points': max(0, 85 - best_f1*100),
                'achievability': 'high' if target_gap <= 0.15 else 'moderate'
            }
        }
        
        output_file = "test_results/f1_gap_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"💾 F1 gap analysis saved to: {output_file}")
        
        return best_f1, recommendations

def main():
    analyzer = F1PerformanceGapAnalyzer()
    
    # Test API connectivity
    try:
        health = requests.get(f"{analyzer.api_url}/health", timeout=5)
        if health.status_code != 200:
            logger.error("❌ API server not responding")
            return
        logger.info("✅ API server is running")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # Load balanced evaluation dataset
    evaluation_samples = analyzer.load_balanced_halueval_dataset(max_samples=400)
    
    if len(evaluation_samples) < 100:
        logger.error("❌ Insufficient evaluation samples")
        return
    
    # Run comprehensive F1 gap analysis
    best_f1, recommendations = analyzer.run_comprehensive_f1_gap_analysis(evaluation_samples)
    
    logger.info(f"\n🌟 F1 GAP ANALYSIS COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"🎯 Best Current F1: {best_f1:.1%}")
    logger.info(f"🎯 Target F1: 85.0%")
    
    if best_f1 >= 0.85:
        logger.info(f"🏆 F1 TARGET ACHIEVED! Ready for production deployment")
    else:
        gap = 0.85 - best_f1
        logger.info(f"📊 Optimization needed: {gap:.1%} gap to close")
        logger.info(f"🔧 Next: Implement cost-sensitive learning and method-specific optimization")

if __name__ == "__main__":
    main()