#!/usr/bin/env python3
"""
🔬 CROSS-DOMAIN VALIDATION FRAMEWORK - Phase 3
Train on multiple domains, test on held-out domains to prevent overfitting
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import sys
import os

# Import our universal physics features
sys.path.append(str(Path(__file__).parent))
from universal_physics_features import UniversalPhysicsFeatures

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CrossDomainValidator:
    """
    Robust cross-domain validation to prevent overfitting
    """
    
    def __init__(self):
        self.feature_extractor = UniversalPhysicsFeatures()
        self.domains = ["medical", "legal", "technical", "creative", "conversational"]
        self.target_consistency = 0.20  # Max 20% F1 variation across domains
        self.min_performance_threshold = 0.50  # Minimum 50% F1 on any domain
        
    def load_multi_domain_data(self) -> Dict[str, List[Dict]]:
        """Load the multi-domain datasets"""
        
        logger.info("📂 LOADING MULTI-DOMAIN DATA")
        logger.info("="*50)
        
        datasets = {}
        data_dir = Path("multi_domain_datasets")
        
        if not data_dir.exists():
            logger.error("❌ Multi-domain datasets not found!")
            logger.error("   Run multi_domain_data_pipeline.py first")
            return {}
        
        for domain in self.domains:
            filename = data_dir / f"{domain}_dataset.json"
            if filename.exists():
                with open(filename, 'r') as f:
                    datasets[domain] = json.load(f)
                
                halluc_count = sum(1 for s in datasets[domain] if s['is_hallucination'])
                halluc_rate = halluc_count / len(datasets[domain])
                
                logger.info(f"✅ {domain}: {len(datasets[domain])} samples, {halluc_rate:.1%} hallucination rate")
            else:
                logger.warning(f"⚠️ {domain} dataset not found: {filename}")
        
        return datasets
    
    def extract_domain_features(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for a domain's samples"""
        
        features = []
        labels = []
        
        for sample in samples:
            # Extract universal physics features
            feature_vector = self.feature_extractor.extract_features(
                sample['prompt'], 
                sample['output']
            )
            
            features.append(feature_vector)
            labels.append(sample['is_hallucination'])
        
        return np.array(features), np.array(labels)
    
    def train_cross_domain_model(self, train_domains: List[str], datasets: Dict[str, List[Dict]]) -> RandomForestClassifier:
        """Train model on multiple domains"""
        
        logger.info(f"🏭 Training on domains: {', '.join(train_domains)}")
        
        # Combine training data from multiple domains
        all_features = []
        all_labels = []
        
        for domain in train_domains:
            if domain in datasets:
                domain_features, domain_labels = self.extract_domain_features(datasets[domain])
                all_features.extend(domain_features)
                all_labels.extend(domain_labels)
        
        X_train = np.array(all_features)
        y_train = np.array(all_labels)
        
        logger.info(f"   📊 Training samples: {len(X_train)}")
        logger.info(f"   📊 Hallucination rate: {np.mean(y_train):.1%}")
        
        # Train with conservative settings (avoid overfitting)
        model = RandomForestClassifier(
            n_estimators=100,        # Fewer trees to prevent overfitting
            max_depth=10,            # Shallower trees for generalization
            min_samples_split=10,    # Require more samples for splits
            min_samples_leaf=5,      # Larger leaf sizes for robustness
            max_features='sqrt',     # Feature subsampling for diversity
            class_weight='balanced', # Handle imbalanced data
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        logger.info("   ✅ Model trained with cross-domain data")
        
        return model
    
    def evaluate_on_domain(self, model: RandomForestClassifier, test_domain: str, 
                          test_data: List[Dict], threshold: float = 0.5) -> Dict:
        """Evaluate model performance on a specific domain"""
        
        # Extract features
        X_test, y_test = self.extract_domain_features(test_data)
        
        # Make predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        try:
            auroc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auroc = 0.5  # No positive samples
        
        # Production metrics
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        false_positive_rate = fp / len(y_test) if len(y_test) > 0 else 0
        false_negative_rate = fn / len(y_test) if len(y_test) > 0 else 0
        accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
        
        return {
            'domain': test_domain,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auroc': auroc,
            'accuracy': accuracy,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'samples': len(test_data),
            'threshold': threshold,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }
    
    def run_cross_domain_validation(self, datasets: Dict[str, List[Dict]]) -> Dict:
        """Run full cross-domain validation"""
        
        logger.info("\n🔬 CROSS-DOMAIN VALIDATION")
        logger.info("="*60)
        logger.info("✅ Train on N-1 domains, test on 1 held-out domain")
        logger.info("✅ Prevents domain-specific overfitting")
        logger.info("✅ Tests true generalization capability")
        
        # Test different thresholds
        thresholds = [0.3, 0.5, 0.7]
        validation_results = {}
        
        for threshold in thresholds:
            logger.info(f"\n🎯 THRESHOLD: {threshold}")
            logger.info("-"*40)
            
            domain_results = {}
            
            # Test each domain as held-out test set
            for test_domain in self.domains:
                if test_domain not in datasets:
                    continue
                
                # Training domains = all others
                train_domains = [d for d in self.domains if d != test_domain and d in datasets]
                
                if len(train_domains) == 0:
                    continue
                
                logger.info(f"\n📊 Testing {test_domain.upper()} (held-out)")
                logger.info(f"   Training on: {', '.join(train_domains)}")
                
                # Train model on other domains
                model = self.train_cross_domain_model(train_domains, datasets)
                
                # Test on held-out domain
                results = self.evaluate_on_domain(model, test_domain, datasets[test_domain], threshold)
                domain_results[test_domain] = results
                
                # Log results
                logger.info(f"   🎯 F1: {results['f1']:.1%} {'🏆' if results['f1'] >= 0.60 else '📊' if results['f1'] >= 0.40 else '❌'}")
                logger.info(f"   📈 Precision: {results['precision']:.1%} {'🏆' if results['precision'] >= 0.70 else '📊' if results['precision'] >= 0.50 else '❌'}")
                logger.info(f"   📈 Recall: {results['recall']:.1%} {'🏆' if results['recall'] >= 0.60 else '📊' if results['recall'] >= 0.40 else '❌'}")
                logger.info(f"   🎯 AUROC: {results['auroc']:.1%}")
                logger.info(f"   🔥 False Positive Rate: {results['false_positive_rate']:.1%}")
                logger.info(f"   📊 Accuracy: {results['accuracy']:.1%}")
            
            validation_results[threshold] = domain_results
        
        return validation_results
    
    def analyze_cross_domain_robustness(self, results: Dict) -> Dict:
        """Analyze robustness across domains"""
        
        logger.info(f"\n🔍 CROSS-DOMAIN ROBUSTNESS ANALYSIS")
        logger.info("="*60)
        
        analysis = {}
        
        for threshold, domain_results in results.items():
            logger.info(f"\nThreshold {threshold}:")
            
            if not domain_results:
                continue
            
            # Extract metrics
            f1_scores = [r['f1'] for r in domain_results.values()]
            precision_scores = [r['precision'] for r in domain_results.values()]
            recall_scores = [r['recall'] for r in domain_results.values()]
            fp_rates = [r['false_positive_rate'] for r in domain_results.values()]
            
            # Calculate statistics
            avg_f1 = np.mean(f1_scores)
            min_f1 = np.min(f1_scores)
            max_f1 = np.max(f1_scores)
            f1_consistency = max_f1 - min_f1  # Lower is better
            
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            avg_fp_rate = np.mean(fp_rates)
            max_fp_rate = np.max(fp_rates)
            
            # Robustness assessment
            passes_min_performance = min_f1 >= self.min_performance_threshold
            passes_consistency = f1_consistency <= self.target_consistency
            passes_production = max_fp_rate <= 0.15  # Max 15% false positive rate
            
            overall_robust = passes_min_performance and passes_consistency and passes_production
            
            logger.info(f"   📊 Average F1: {avg_f1:.1%}")
            logger.info(f"   📊 Min F1: {min_f1:.1%} {'✅' if passes_min_performance else '❌'}")
            logger.info(f"   📊 Max F1: {max_f1:.1%}")
            logger.info(f"   📊 F1 Consistency Gap: {f1_consistency:.1%} {'✅' if passes_consistency else '❌'} (target: ≤{self.target_consistency:.0%})")
            logger.info(f"   📊 Average Precision: {avg_precision:.1%}")
            logger.info(f"   📊 Average Recall: {avg_recall:.1%}")
            logger.info(f"   📊 Average FP Rate: {avg_fp_rate:.1%}")
            logger.info(f"   📊 Max FP Rate: {max_fp_rate:.1%} {'✅' if passes_production else '❌'}")
            
            # Overall assessment
            if overall_robust:
                logger.info(f"   🏆 ROBUST: Passes all cross-domain tests")
            elif passes_min_performance and passes_consistency:
                logger.info(f"   ⚠️ MODERATE: Good generalization, high FP rate")
            elif passes_min_performance:
                logger.info(f"   📊 INCONSISTENT: Varies significantly across domains")
            else:
                logger.info(f"   ❌ POOR: Fails minimum performance on some domains")
            
            analysis[threshold] = {
                'avg_f1': avg_f1,
                'min_f1': min_f1,
                'max_f1': max_f1,
                'f1_consistency': f1_consistency,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_fp_rate': avg_fp_rate,
                'max_fp_rate': max_fp_rate,
                'passes_min_performance': passes_min_performance,
                'passes_consistency': passes_consistency,
                'passes_production': passes_production,
                'overall_robust': overall_robust
            }
        
        return analysis
    
    def save_validation_results(self, results: Dict, analysis: Dict):
        """Save validation results"""
        
        output_data = {
            'validation_type': 'cross_domain_validation',
            'timestamp': '2025-08-18',
            'approach': 'domain_agnostic_physics_features',
            'feature_count': 6,  # Universal physics features
            'domains_tested': self.domains,
            'validation_results': results,
            'robustness_analysis': analysis,
            'target_consistency': self.target_consistency,
            'min_performance_threshold': self.min_performance_threshold
        }
        
        output_file = "cross_domain_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n💾 Cross-domain validation results saved to: {output_file}")

def main():
    """Run cross-domain validation"""
    
    validator = CrossDomainValidator()
    
    # Load multi-domain data
    datasets = validator.load_multi_domain_data()
    
    if not datasets:
        logger.error("❌ No datasets found. Run multi_domain_data_pipeline.py first.")
        return
    
    # Run cross-domain validation
    results = validator.run_cross_domain_validation(datasets)
    
    # Analyze robustness
    analysis = validator.analyze_cross_domain_robustness(results)
    
    # Save results
    validator.save_validation_results(results, analysis)
    
    # Final assessment
    logger.info(f"\n🏆 CROSS-DOMAIN VALIDATION SUMMARY")
    logger.info("="*60)
    
    # Find best threshold
    best_threshold = None
    best_score = 0
    
    for threshold, analysis_result in analysis.items():
        if analysis_result['overall_robust']:
            score = analysis_result['avg_f1'] * (1 - analysis_result['f1_consistency'])  # Balance performance and consistency
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    if best_threshold:
        logger.info(f"🎯 ROBUST SYSTEM FOUND!")
        logger.info(f"   ✅ Best threshold: {best_threshold}")
        logger.info(f"   ✅ Cross-domain consistency: {analysis[best_threshold]['f1_consistency']:.1%}")
        logger.info(f"   ✅ Average F1: {analysis[best_threshold]['avg_f1']:.1%}")
        logger.info(f"   ✅ Production ready: Low false positive rate")
    else:
        logger.info(f"⚠️ NO ROBUST CONFIGURATION FOUND")
        logger.info(f"   Need to improve universal features or model architecture")
        
        # Show best available option
        best_threshold = max(analysis.keys(), 
                           key=lambda t: analysis[t]['avg_f1'] * (1 - analysis[t]['f1_consistency']))
        logger.info(f"   📊 Best available: threshold {best_threshold}")
        logger.info(f"   📊 Average F1: {analysis[best_threshold]['avg_f1']:.1%}")
        logger.info(f"   📊 Consistency gap: {analysis[best_threshold]['f1_consistency']:.1%}")

if __name__ == "__main__":
    main()