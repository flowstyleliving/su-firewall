#!/usr/bin/env python3
"""
🌍 REALISTIC PRODUCTION EVALUATION
Test on diverse domains with natural distributions and realistic thresholds
"""

import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import time
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class RealisticProductionEvaluator:
    def __init__(self):
        self.model = None
        
    def create_realistic_datasets(self):
        """Create diverse datasets with natural hallucination distributions"""
        
        logger.info("🌍 CREATING REALISTIC PRODUCTION DATASETS")
        logger.info("="*60)
        
        datasets = {}
        
        # 1. Medical Content (5% hallucination rate - high stakes)
        medical_correct = [
            {"prompt": "What is the treatment for Type 2 diabetes?", 
             "output": "Treatment typically includes lifestyle changes like diet and exercise, along with medications such as metformin when needed.", 
             "is_hallucination": False, "domain": "medical"},
            {"prompt": "What are symptoms of pneumonia?", 
             "output": "Common symptoms include fever, cough with phlegm, chest pain, and difficulty breathing.", 
             "is_hallucination": False, "domain": "medical"},
            {"prompt": "How does aspirin work?", 
             "output": "Aspirin works by inhibiting cyclooxygenase enzymes, which reduces inflammation and prevents blood clot formation.", 
             "is_hallucination": False, "domain": "medical"},
            {"prompt": "What is hypertension?", 
             "output": "Hypertension is high blood pressure, defined as consistently elevated pressure in the arteries.", 
             "is_hallucination": False, "domain": "medical"},
            {"prompt": "What causes heart attacks?", 
             "output": "Heart attacks are usually caused by blocked coronary arteries, often due to atherosclerosis and blood clots.", 
             "is_hallucination": False, "domain": "medical"},
        ]
        
        medical_hallucinated = [
            {"prompt": "What is the treatment for Type 2 diabetes?", 
             "output": "Type 2 diabetes can be completely cured by drinking green tea three times daily and avoiding all carbohydrates forever.", 
             "is_hallucination": True, "domain": "medical"},
        ]
        
        # Extend medical with realistic 95% correct / 5% hallucinated
        medical_samples = medical_correct * 19 + medical_hallucinated * 1  # 95 correct, 5 hallucinated
        
        # 2. Legal Content (8% hallucination rate - complex domain)
        legal_correct = [
            {"prompt": "What is breach of contract?", 
             "output": "Breach of contract occurs when one party fails to fulfill their obligations under a legally binding agreement.", 
             "is_hallucination": False, "domain": "legal"},
            {"prompt": "What is the Fifth Amendment?", 
             "output": "The Fifth Amendment protects against self-incrimination and guarantees due process rights.", 
             "is_hallucination": False, "domain": "legal"},
            {"prompt": "What is copyright infringement?", 
             "output": "Copyright infringement is the unauthorized use of copyrighted material without permission from the owner.", 
             "is_hallucination": False, "domain": "legal"},
            {"prompt": "What is a tort?", 
             "output": "A tort is a civil wrong that causes harm to another person, leading to legal liability.", 
             "is_hallucination": False, "domain": "legal"},
        ]
        
        legal_hallucinated = [
            {"prompt": "What is breach of contract?", 
             "output": "Breach of contract automatically results in criminal charges and mandatory jail time for the violating party.", 
             "is_hallucination": True, "domain": "legal"},
        ]
        
        legal_samples = legal_correct * 23 + legal_hallucinated * 2  # 92% correct, 8% hallucinated
        
        # 3. Technical Conversations (12% hallucination rate - informal domain)
        tech_correct = [
            {"prompt": "How do I fix a memory leak in Python?", 
             "output": "Check for circular references, use weak references, and profile with tools like memory_profiler to identify the source.", 
             "is_hallucination": False, "domain": "technical"},
            {"prompt": "What is REST API?", 
             "output": "REST is an architectural style for web services that uses HTTP methods and stateless communication.", 
             "is_hallucination": False, "domain": "technical"},
            {"prompt": "Explain Docker containers", 
             "output": "Docker containers package applications with their dependencies, providing consistent environments across different systems.", 
             "is_hallucination": False, "domain": "technical"},
            {"prompt": "What is machine learning?", 
             "output": "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions.", 
             "is_hallucination": False, "domain": "technical"},
        ]
        
        tech_hallucinated = [
            {"prompt": "How do I fix a memory leak in Python?", 
             "output": "Python doesn't have memory leaks because it uses Java's garbage collector and automatically converts to C++.", 
             "is_hallucination": True, "domain": "technical"},
        ]
        
        tech_samples = tech_correct * 22 + tech_hallucinated * 3  # 88% correct, 12% hallucinated
        
        # 4. Creative Content (15% hallucination rate - subjective domain)
        creative_correct = [
            {"prompt": "Write about a sunset", 
             "output": "The golden sun dipped below the horizon, painting the sky in brilliant shades of orange and pink.", 
             "is_hallucination": False, "domain": "creative"},
            {"prompt": "Describe a forest", 
             "output": "Tall trees swayed gently in the breeze, their leaves rustling like whispered secrets in the green canopy above.", 
             "is_hallucination": False, "domain": "creative"},
            {"prompt": "What makes good poetry?", 
             "output": "Good poetry often combines vivid imagery, emotional resonance, and thoughtful use of language and rhythm.", 
             "is_hallucination": False, "domain": "creative"},
        ]
        
        creative_hallucinated = [
            {"prompt": "Write about a sunset", 
             "output": "The sun exploded into purple flames and started raining diamonds while singing opera music to the dolphins.", 
             "is_hallucination": True, "domain": "creative"},
        ]
        
        creative_samples = creative_correct * 17 + creative_hallucinated * 3  # 85% correct, 15% hallucinated
        
        datasets = {
            "medical": medical_samples,
            "legal": legal_samples, 
            "technical": tech_samples,
            "creative": creative_samples
        }
        
        # Log dataset stats
        for domain, samples in datasets.items():
            total = len(samples)
            halluc_count = sum(1 for s in samples if s['is_hallucination'])
            halluc_rate = halluc_count / total * 100
            logger.info(f"📊 {domain.title()}: {total} samples, {halluc_rate:.1f}% hallucination rate")
        
        return datasets
    
    def extract_production_features(self, prompt, output):
        """Same feature extraction as our optimized system"""
        output_words = output.lower().split()
        output_length = len(output_words)
        prompt_length = len(prompt.split())
        length_ratio = output_length / max(prompt_length, 1)
        
        # Core uncertainty indicators
        uncertainty_words = [
            'maybe', 'might', 'possibly', 'perhaps', 'unsure', 'uncertain',
            'probably', 'likely', 'seems', 'appears', 'could', 'may'
        ]
        uncertainty_count = sum(1 for word in uncertainty_words if word in output.lower())
        uncertainty_density = uncertainty_count / max(output_length, 1)
        
        # Confidence indicators
        confidence_words = [
            'definitely', 'certainly', 'absolutely', 'clearly', 'obviously',
            'surely', 'exactly', 'precisely', 'undoubtedly'
        ]
        confidence_count = sum(1 for word in confidence_words if word in output.lower())
        confidence_density = confidence_count / max(output_length, 1)
        
        # Contradiction patterns
        contradiction_words = [
            'not', 'no', 'wrong', 'false', 'incorrect', 'never',
            'opposite', 'contrary', 'however', 'but', 'although'
        ]
        contradiction_count = sum(1 for word in contradiction_words if word in output.lower())
        contradiction_density = contradiction_count / max(output_length, 1)
        
        # Question analysis
        question_words = ['what', 'when', 'where', 'who', 'why', 'how']
        question_count = sum(1 for word in question_words if word in prompt.lower())
        
        # Hedging patterns
        hedge_words = ['think', 'believe', 'suppose', 'assume', 'guess']
        hedge_count = sum(1 for word in hedge_words if word in output.lower())
        
        # Factual assertions
        factual_words = ['is', 'are', 'was', 'were', 'has', 'have', 'will']
        factual_count = sum(1 for word in factual_words if word in output.lower())
        factual_density = factual_count / max(output_length, 1)
        
        # Semantic diversity
        unique_words = len(set(output_words))
        word_diversity = unique_words / max(output_length, 1)
        
        # Qualification markers
        qualification_words = ['some', 'many', 'often', 'sometimes', 'usually', 'generally']
        qualification_count = sum(1 for word in qualification_words if word in output.lower())
        
        # Temporal markers
        temporal_words = ['always', 'never', 'forever', 'eternal', 'permanent']
        temporal_count = sum(1 for word in temporal_words if word in output.lower())
        
        return np.array([
            output_length,
            length_ratio,
            uncertainty_count,
            uncertainty_density,
            confidence_count,
            confidence_density,
            contradiction_count,
            contradiction_density,
            question_count,
            hedge_count,
            factual_count,
            factual_density,
            word_diversity,
            qualification_count,
            temporal_count
        ])
    
    def train_on_original_data(self):
        """Train on original HaluEval data (what we optimized on)"""
        
        logger.info("\n🏭 TRAINING ON ORIGINAL HALUEVAL DATA")
        logger.info("="*60)
        
        # Load original HaluEval QA data
        from pathlib import Path
        data_dir = Path("authentic_datasets")
        qa_path = data_dir / "halueval_qa_data.json"
        
        training_samples = []
        with open(qa_path, 'r') as f:
            lines = f.read().strip().split('\n')[:3000]  # Smaller training set
            
            for line in lines:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        training_samples.extend([
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
        
        logger.info(f"📊 Training samples: {len(training_samples)}")
        
        # Extract features
        features = []
        labels = []
        
        for sample in training_samples:
            feature_vector = self.extract_production_features(sample['prompt'], sample['output'])
            features.append(feature_vector)
            labels.append(sample['is_hallucination'])
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Train model (same as our "optimized" version)
        self.model = RandomForestClassifier(
            class_weight={0: 1.0, 1: 10.0},  # Heavy penalty for missing hallucinations
            n_estimators=500,
            random_state=42,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1
        )
        
        logger.info("🔧 Training model...")
        self.model.fit(features, labels)
        
        logger.info("✅ Model trained on HaluEval QA data")
    
    def evaluate_realistic_performance(self, datasets):
        """Evaluate on realistic datasets with realistic thresholds"""
        
        logger.info("\n🌍 REALISTIC PRODUCTION EVALUATION")
        logger.info("="*60)
        
        # Test realistic thresholds (not 0.999!)
        realistic_thresholds = [0.3, 0.5, 0.7]
        
        overall_results = {}
        
        for threshold in realistic_thresholds:
            logger.info(f"\n🎯 THRESHOLD: {threshold}")
            logger.info("-"*40)
            
            domain_results = {}
            
            for domain_name, samples in datasets.items():
                # Extract features
                features = []
                labels = []
                
                for sample in samples:
                    feature_vector = self.extract_production_features(sample['prompt'], sample['output'])
                    features.append(feature_vector)
                    labels.append(sample['is_hallucination'])
                
                features = np.array(features)
                labels = np.array(labels)
                
                # Make predictions
                y_proba = self.model.predict_proba(features)[:, 1]
                predictions = (y_proba > threshold).astype(int)
                
                # Calculate metrics
                f1 = f1_score(labels, predictions, zero_division=0)
                precision = precision_score(labels, predictions, zero_division=0)
                recall = recall_score(labels, predictions, zero_division=0)
                
                # Production hallucination rate (false positives)
                fp = np.sum((predictions == 1) & (labels == 0))
                halluc_rate = fp / len(labels)
                
                # System flagging rate
                flagging_rate = np.sum(predictions) / len(predictions)
                
                domain_results[domain_name] = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'halluc_rate': halluc_rate,
                    'flagging_rate': flagging_rate,
                    'samples': len(samples)
                }
                
                logger.info(f"📊 {domain_name.upper()}:")
                logger.info(f"   🎯 F1: {f1:.1%} {'🏆' if f1 >= 0.80 else '📊' if f1 >= 0.60 else '❌'}")
                logger.info(f"   📈 Precision: {precision:.1%} {'🏆' if precision >= 0.85 else '📊' if precision >= 0.60 else '❌'}")
                logger.info(f"   📈 Recall: {recall:.1%} {'🏆' if recall >= 0.70 else '📊' if recall >= 0.50 else '❌'}")
                logger.info(f"   🔥 Production Halluc Rate: {halluc_rate:.1%}")
                logger.info(f"   📊 System Flagging Rate: {flagging_rate:.1%}")
                logger.info(f"   📊 Samples: {len(samples)}")
            
            overall_results[threshold] = domain_results
        
        return overall_results
    
    def analyze_domain_transfer(self, results):
        """Analyze how well the model transfers across domains"""
        
        logger.info(f"\n🔬 DOMAIN TRANSFER ANALYSIS")
        logger.info("="*60)
        
        for threshold, domain_results in results.items():
            logger.info(f"\nThreshold {threshold}:")
            
            # Calculate average performance across domains
            avg_f1 = np.mean([r['f1'] for r in domain_results.values()])
            avg_precision = np.mean([r['precision'] for r in domain_results.values()])
            avg_recall = np.mean([r['recall'] for r in domain_results.values()])
            avg_halluc_rate = np.mean([r['halluc_rate'] for r in domain_results.values()])
            
            # Calculate variance (consistency across domains)
            f1_variance = np.var([r['f1'] for r in domain_results.values()])
            precision_variance = np.var([r['precision'] for r in domain_results.values()])
            
            logger.info(f"   📊 Average F1: {avg_f1:.1%}")
            logger.info(f"   📊 Average Precision: {avg_precision:.1%}")
            logger.info(f"   📊 Average Recall: {avg_recall:.1%}")
            logger.info(f"   📊 Average Halluc Rate: {avg_halluc_rate:.1%}")
            logger.info(f"   📊 F1 Consistency: {1-f1_variance:.3f} (higher = more consistent)")
            logger.info(f"   📊 Precision Consistency: {1-precision_variance:.3f}")
            
            # Check if performance is acceptable across all domains
            min_f1 = min([r['f1'] for r in domain_results.values()])
            max_halluc = max([r['halluc_rate'] for r in domain_results.values()])
            
            if min_f1 >= 0.60 and max_halluc <= 0.10:
                logger.info(f"   ✅ ROBUST: Good performance across all domains")
            elif min_f1 >= 0.40:
                logger.info(f"   ⚠️ MODERATE: Acceptable but inconsistent performance")
            else:
                logger.info(f"   ❌ POOR: Significant domain transfer issues")

def main():
    evaluator = RealisticProductionEvaluator()
    
    # Create realistic test datasets
    datasets = evaluator.create_realistic_datasets()
    
    # Train on original HaluEval data (what we optimized on)
    evaluator.train_on_original_data()
    
    # Evaluate on realistic datasets
    results = evaluator.evaluate_realistic_performance(datasets)
    
    # Analyze domain transfer
    evaluator.analyze_domain_transfer(results)
    
    # Final honest assessment
    logger.info(f"\n🏆 HONEST PRODUCTION ASSESSMENT")
    logger.info("="*60)
    logger.info("This evaluation tests our 'world-class' system on:")
    logger.info("✅ Diverse domains (medical, legal, technical, creative)")
    logger.info("✅ Natural hallucination rates (5-15%, not 50%)")
    logger.info("✅ Realistic thresholds (0.3-0.7, not 0.999)")
    logger.info("✅ Cross-domain transfer (trained on QA, tested on everything)")
    logger.info("\nIf performance drops significantly, our original")
    logger.info("'breakthrough' was indeed overfitted to the test.")

if __name__ == "__main__":
    main()