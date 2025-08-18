#!/usr/bin/env python3
"""
⚖️🎯 COST-SENSITIVE F1 OPTIMIZER  
Implement weighted loss functions and precision-recall balancing for 85%+ F1
"""

import requests
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CostSensitiveF1Optimizer:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        self.cost_sensitive_models = {}
        
    def load_training_dataset(self, max_samples=800):
        """Load balanced training dataset for cost-sensitive learning"""
        data_dir = Path("authentic_datasets")
        all_samples = []
        
        # Load HaluEval QA for training
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
        
        logger.info(f"📊 Training dataset: {len(all_samples)} samples")
        halluc_count = sum(1 for s in all_samples if s['is_hallucination'])
        logger.info(f"   🔍 Hallucinations: {halluc_count}")
        logger.info(f"   ✅ Correct: {len(all_samples) - halluc_count}")
        
        return all_samples
    
    def extract_comprehensive_features(self, prompt, output):
        """Extract comprehensive features for cost-sensitive learning"""
        
        # 1. Length features
        output_length = len(output.split())
        prompt_length = len(prompt.split())
        length_ratio = output_length / max(prompt_length, 1)
        
        # 2. Uncertainty markers
        uncertainty_words = ['maybe', 'might', 'possibly', 'perhaps', 'unsure', 'uncertain', 'probably']
        uncertainty_count = sum(1 for word in uncertainty_words if word in output.lower())
        uncertainty_density = uncertainty_count / max(output_length, 1)
        
        # 3. Certainty markers
        certainty_words = ['definitely', 'certainly', 'absolutely', 'clearly', 'obviously', 'surely', 'exactly']
        certainty_count = sum(1 for word in certainty_words if word in output.lower())
        certainty_density = certainty_count / max(output_length, 1)
        
        # 4. Contradiction indicators
        contradiction_words = ['not', 'no', 'wrong', 'false', 'incorrect', 'never', 'opposite', 'contrary']
        contradiction_count = sum(1 for word in contradiction_words if word in output.lower())
        contradiction_density = contradiction_count / max(output_length, 1)
        
        # 5. Question type features
        question_words = ['what', 'when', 'where', 'who', 'why', 'how']
        question_type_count = sum(1 for word in question_words if word in prompt.lower())
        
        # 6. Hedging patterns
        hedge_phrases = ['i think', 'i believe', 'it seems', 'appears to', 'might be', 'could be']
        hedge_count = sum(1 for phrase in hedge_phrases if phrase in output.lower())
        
        # 7. Factual claim density
        factual_words = ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'does']
        factual_count = sum(1 for word in factual_words if word in output.lower())
        factual_density = factual_count / max(output_length, 1)
        
        # 8. Semantic diversity (approximate)
        unique_words = len(set(output.lower().split()))
        word_diversity = unique_words / max(output_length, 1)
        
        return np.array([
            output_length,           # 0
            length_ratio,           # 1
            uncertainty_count,      # 2
            uncertainty_density,    # 3
            certainty_count,        # 4
            certainty_density,      # 5
            contradiction_count,    # 6
            contradiction_density,  # 7
            question_type_count,    # 8
            hedge_count,           # 9
            factual_count,         # 10
            factual_density,       # 11
            word_diversity         # 12
        ])
    
    def train_cost_sensitive_models(self, training_samples):
        """Train cost-sensitive models optimized for F1 score"""
        
        logger.info(f"\n⚖️ TRAINING COST-SENSITIVE MODELS")
        logger.info(f"{'='*50}")
        
        # Extract features and labels
        features = []
        labels = []
        
        for sample in training_samples:
            feature_vector = self.extract_comprehensive_features(sample['prompt'], sample['output'])
            features.append(feature_vector)
            labels.append(sample['is_hallucination'])
        
        features = np.array(features)
        labels = np.array(labels)
        
        logger.info(f"📊 Training features shape: {features.shape}")
        logger.info(f"📊 Label distribution: {np.sum(labels)}/{len(labels)}")
        
        # Calculate class weights for cost-sensitive learning
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        logger.info(f"⚖️ Class weights: {class_weight_dict}")
        
        # Split for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.3, stratify=labels, random_state=42
        )
        
        # Train multiple cost-sensitive models
        models = {
            'logistic_balanced': LogisticRegression(
                class_weight='balanced', 
                max_iter=1000, 
                random_state=42
            ),
            'logistic_f1_optimized': LogisticRegression(
                class_weight={0: 1.0, 1: 2.0},  # Favor recall
                max_iter=1000,
                random_state=42
            ),
            'forest_balanced': RandomForestClassifier(
                class_weight='balanced',
                n_estimators=100,
                random_state=42
            ),
            'forest_f1_optimized': RandomForestClassifier(
                class_weight={0: 1.0, 1: 1.5},  # Moderate recall boost
                n_estimators=100,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        model_performance = {}
        
        for model_name, model in models.items():
            logger.info(f"\n🔧 Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Get probability predictions for F1 optimization
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_proba = model.decision_function(X_val)
                
                # Find optimal F1 threshold
                precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
                f1_scores = []
                for p, r in zip(precision, recall):
                    if p + r > 0:
                        f1_scores.append(2 * p * r / (p + r))
                    else:
                        f1_scores.append(0.0)
                
                if f1_scores:
                    max_f1_idx = np.argmax(f1_scores)
                    optimal_f1 = f1_scores[max_f1_idx]
                    optimal_threshold = thresholds[min(max_f1_idx, len(thresholds)-1)]
                    optimal_precision = precision[max_f1_idx]
                    optimal_recall = recall[max_f1_idx]
                    
                    # Calculate AUROC
                    auroc = roc_auc_score(y_val, y_proba)
                    
                    model_performance[model_name] = {
                        'model': model,
                        'optimal_f1': optimal_f1,
                        'optimal_threshold': optimal_threshold,
                        'optimal_precision': optimal_precision,
                        'optimal_recall': optimal_recall,
                        'auroc': auroc
                    }
                    
                    logger.info(f"   🎯 Optimal F1: {optimal_f1:.1%} {'🏆' if optimal_f1 >= 0.85 else '📊'}")
                    logger.info(f"   🔧 Optimal Threshold: {optimal_threshold:.3f}")
                    logger.info(f"   📈 Precision: {optimal_precision:.1%}")
                    logger.info(f"   📈 Recall: {optimal_recall:.1%}")
                    logger.info(f"   🎯 AUROC: {auroc:.1%}")
                    
            except Exception as e:
                logger.warning(f"   ❌ Training failed: {e}")
        
        # Select best model
        if model_performance:
            best_model_name = max(model_performance.keys(), 
                                key=lambda k: model_performance[k]['optimal_f1'])
            best_model_info = model_performance[best_model_name]
            
            self.cost_sensitive_models = model_performance
            
            logger.info(f"\n🏆 BEST COST-SENSITIVE MODEL: {best_model_name}")
            logger.info(f"   🎯 F1: {best_model_info['optimal_f1']:.1%}")
            logger.info(f"   🔧 Threshold: {best_model_info['optimal_threshold']:.3f}")
            
            return best_model_name, best_model_info
        
        return None, None
    
    def validate_cost_sensitive_f1(self, test_samples, best_model_name, best_model_info):
        """Validate cost-sensitive F1 optimization on test set"""
        
        logger.info(f"\n📊 COST-SENSITIVE F1 VALIDATION")
        logger.info(f"{'='*50}")
        logger.info(f"🔧 Using model: {best_model_name}")
        logger.info(f"🔧 Using threshold: {best_model_info['optimal_threshold']:.3f}")
        
        predictions = []
        probabilities = []
        ground_truth = []
        processing_times = []
        
        model = best_model_info['model']
        threshold = best_model_info['optimal_threshold']
        
        start_time = time.time()
        
        for i, sample in enumerate(test_samples):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(f"📈 Validation: {i}/{len(test_samples)} | Rate: {rate:.1f}/s")
            
            sample_start = time.time()
            
            # Extract features
            feature_vector = self.extract_comprehensive_features(sample['prompt'], sample['output'])
            feature_vector = feature_vector.reshape(1, -1)
            
            # Get model prediction
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(feature_vector)[0][1]
                else:
                    prob = model.decision_function(feature_vector)[0]
                    prob = 1 / (1 + np.exp(-prob))  # Sigmoid for probability
                
                is_predicted_hallucination = prob > threshold
                
                predictions.append(is_predicted_hallucination)
                probabilities.append(prob)
                ground_truth.append(sample['is_hallucination'])
                
            except Exception as e:
                # Fallback prediction
                predictions.append(False)
                probabilities.append(0.5)
                ground_truth.append(sample['is_hallucination'])
            
            processing_times.append((time.time() - sample_start) * 1000)
        
        # Calculate final metrics
        try:
            final_f1 = f1_score(ground_truth, predictions)
            final_precision = precision_score(ground_truth, predictions, zero_division=0)
            final_recall = recall_score(ground_truth, predictions, zero_division=0)
            final_auroc = roc_auc_score(ground_truth, probabilities)
            
            # Confusion matrix
            tp = sum(1 for p, l in zip(predictions, ground_truth) if p and l)
            fp = sum(1 for p, l in zip(predictions, ground_truth) if p and not l)
            tn = sum(1 for p, l in zip(predictions, ground_truth) if not p and not l)
            fn = sum(1 for p, l in zip(predictions, ground_truth) if not p and l)
            
            # Performance analysis
            precision_recall_balance = abs(final_precision - final_recall)
            
            logger.info(f"\n🏆 COST-SENSITIVE F1 OPTIMIZATION RESULTS")
            logger.info(f"{'='*50}")
            logger.info(f"🎯 Final F1 Score: {final_f1:.1%} {'🏆' if final_f1 >= 0.85 else '📊'}")
            logger.info(f"📈 Precision: {final_precision:.1%}")
            logger.info(f"📈 Recall: {final_recall:.1%}")
            logger.info(f"⚖️ Precision-Recall Balance: {precision_recall_balance:.1%} {'✅' if precision_recall_balance < 0.1 else '⚠️'}")
            logger.info(f"🎯 AUROC: {final_auroc:.1%}")
            
            logger.info(f"\n📊 Confusion Matrix:")
            logger.info(f"   TP: {tp}, FP: {fp}")
            logger.info(f"   TN: {tn}, FN: {fn}")
            
            # False positive/negative analysis
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            logger.info(f"\n📊 Error Analysis:")
            logger.info(f"   🚨 False Positive Rate: {fpr:.1%}")
            logger.info(f"   🚨 False Negative Rate: {fnr:.1%}")
            
            # Target achievement assessment
            if final_f1 >= 0.85:
                logger.info(f"\n🎉 85%+ F1 TARGET ACHIEVED!")
                logger.info(f"   🏆 Cost-sensitive F1: {final_f1:.1%} ≥ 85%")
                logger.info(f"   ⚖️ Balanced precision-recall optimization successful")
                
                if final_f1 >= 0.95:
                    logger.info(f"   🚀 EXCEPTIONAL F1 PERFORMANCE: {final_f1:.1%}")
                elif final_f1 >= 0.90:
                    logger.info(f"   ⭐ OUTSTANDING F1 PERFORMANCE: {final_f1:.1%}")
            else:
                gap = 0.85 - final_f1
                logger.info(f"\n📈 F1 Optimization Progress:")
                logger.info(f"   Current: {final_f1:.1%}")
                logger.info(f"   Target: 85.0%")
                logger.info(f"   Gap: {gap:.1%} ({gap*100:.1f} percentage points)")
                
                # Specific optimization recommendations
                if final_precision < 0.80:
                    logger.info(f"   🔧 LOW PRECISION: Increase threshold or add FP penalties")
                if final_recall < 0.80:
                    logger.info(f"   🔧 LOW RECALL: Decrease threshold or boost hallucination weights")
                if precision_recall_balance > 0.15:
                    logger.info(f"   ⚖️ IMBALANCED: Adjust class weights for better P-R balance")
            
            # Performance statistics
            avg_time = np.mean(processing_times)
            throughput = 1000 / avg_time if avg_time > 0 else 0
            
            logger.info(f"\n⚡ Performance Statistics:")
            logger.info(f"   📊 Samples processed: {len(predictions)}")
            logger.info(f"   ⏱️ Avg processing time: {avg_time:.1f}ms")
            logger.info(f"   🚀 Throughput: {throughput:.0f} analyses/sec")
            
            # Save cost-sensitive optimization results
            results = {
                'optimization_type': 'cost_sensitive_f1_optimization',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'best_model': best_model_name,
                'optimal_threshold': threshold,
                'final_metrics': {
                    'f1_score': final_f1,
                    'precision': final_precision,
                    'recall': final_recall,
                    'auroc': final_auroc,
                    'precision_recall_balance': precision_recall_balance
                },
                'confusion_matrix': {
                    'true_positive': tp,
                    'false_positive': fp,
                    'true_negative': tn,
                    'false_negative': fn,
                    'false_positive_rate': fpr,
                    'false_negative_rate': fnr
                },
                'target_achievement': {
                    'target_f1': 0.85,
                    'achieved': final_f1 >= 0.85,
                    'gap_percentage_points': max(0, 85 - final_f1*100),
                    'exceptional_performance': final_f1 >= 0.95
                },
                'processing_stats': {
                    'samples_processed': len(predictions),
                    'avg_processing_time_ms': avg_time,
                    'throughput_analyses_per_sec': throughput
                }
            }
            
            output_file = "test_results/cost_sensitive_f1_optimization_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Cost-sensitive results saved to: {output_file}")
            
            return final_f1
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return 0.0

def main():
    optimizer = CostSensitiveF1Optimizer()
    
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
    
    # Load training dataset
    training_samples = optimizer.load_training_dataset(max_samples=600)
    
    if len(training_samples) < 100:
        logger.error("❌ Insufficient training samples")
        return
    
    # Split into train and test
    split_point = len(training_samples) // 2
    train_samples = training_samples[:split_point]
    test_samples = training_samples[split_point:]
    
    logger.info(f"📊 Train samples: {len(train_samples)}")
    logger.info(f"📊 Test samples: {len(test_samples)}")
    
    # Step 1: Train cost-sensitive models
    best_model_name, best_model_info = optimizer.train_cost_sensitive_models(train_samples)
    
    if not best_model_info:
        logger.error("❌ Model training failed")
        return
    
    # Step 2: Validate on test set
    final_f1 = optimizer.validate_cost_sensitive_f1(test_samples, best_model_name, best_model_info)
    
    logger.info(f"\n🌟 COST-SENSITIVE F1 OPTIMIZATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"🎯 Final F1: {final_f1:.1%}")
    logger.info(f"🔧 Best Model: {best_model_name}")
    
    if final_f1 >= 0.85:
        logger.info(f"🏆 SUCCESS! 85%+ F1 achieved via cost-sensitive learning")
        logger.info(f"✅ Ready for method-specific optimization and production deployment")
    else:
        logger.info(f"📊 {final_f1:.1%} toward 85% target")
        logger.info(f"🔧 Next: Method-specific threshold optimization")

if __name__ == "__main__":
    main()