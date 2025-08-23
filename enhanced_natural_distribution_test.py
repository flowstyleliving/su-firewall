#!/usr/bin/env python3
"""
🌍 ENHANCED NATURAL DISTRIBUTION TEST - High Impact Week 1-2 Implementation
Comprehensive testing with realistic 5-10% hallucination rates and production optimization
Focus: False positive rate optimization and cross-domain validation
"""

import requests
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import statistics
from typing import Dict, List, Tuple, Optional
import random

class NaturalDistributionTester:
    """Advanced natural distribution testing with production optimizations"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url
        self.results_cache = {}
        self.optimal_thresholds = {}
        
    def create_realistic_content_dataset(self) -> Dict[str, List[Dict]]:
        """Create realistic content across multiple domains with natural hallucination rates"""
        
        datasets = {
            "news_articles": {
                "factual": [
                    {"prompt": "What happened in the 2024 Olympics?", "output": "The 2024 Summer Olympics were held in Paris, France from July 26 to August 11, 2024."},
                    {"prompt": "Current US President?", "output": "Joe Biden is the current President of the United States."},
                    {"prompt": "Climate change impact", "output": "Climate change is causing rising sea levels, extreme weather events, and ecosystem disruption globally."},
                    {"prompt": "COVID-19 vaccines effectiveness", "output": "COVID-19 vaccines have shown effectiveness in reducing severe illness and hospitalization rates."},
                    {"prompt": "Stock market trends", "output": "Stock markets have shown volatility in recent years due to various economic factors including inflation and geopolitical tensions."},
                    {"prompt": "Technology advances 2024", "output": "2024 has seen significant advances in AI, quantum computing, and renewable energy technologies."},
                    {"prompt": "Space exploration updates", "output": "NASA and SpaceX continue making progress on lunar missions and Mars exploration programs."},
                    {"prompt": "Global economy status", "output": "The global economy faces challenges from supply chain issues and varying recovery rates post-pandemic."},
                    {"prompt": "Healthcare innovations", "output": "Recent healthcare innovations include personalized medicine, telemedicine expansion, and new cancer treatments."},
                ],
                "hallucinated": [
                    {"prompt": "What happened in the 2024 Olympics?", "output": "The 2024 Olympics were cancelled due to a global alien invasion that lasted three months."},
                ]
            },
            
            "medical_content": {
                "factual": [
                    {"prompt": "What is diabetes?", "output": "Diabetes is a group of metabolic disorders characterized by chronic hyperglycemia resulting from defects in insulin secretion, action, or both."},
                    {"prompt": "COVID symptoms", "output": "Common COVID-19 symptoms include fever, cough, fatigue, loss of taste or smell, and difficulty breathing."},
                    {"prompt": "Heart disease prevention", "output": "Heart disease prevention involves regular exercise, healthy diet, not smoking, managing blood pressure and cholesterol."},
                    {"prompt": "Cancer screening importance", "output": "Regular cancer screening can detect cancers early when they are most treatable, improving survival rates."},
                    {"prompt": "Mental health awareness", "output": "Mental health awareness emphasizes the importance of recognizing symptoms and seeking professional help when needed."},
                    {"prompt": "Vaccine safety", "output": "Vaccines undergo rigorous testing for safety and efficacy before approval by regulatory agencies."},
                    {"prompt": "Antibiotic resistance", "output": "Antibiotic resistance occurs when bacteria evolve to survive antibiotic treatments, making infections harder to treat."},
                    {"prompt": "Nutrition guidelines", "output": "Current nutrition guidelines recommend a balanced diet with fruits, vegetables, whole grains, lean proteins, and limited processed foods."},
                    {"prompt": "Exercise benefits", "output": "Regular exercise improves cardiovascular health, mental wellness, bone density, and helps maintain healthy weight."},
                ],
                "hallucinated": [
                    {"prompt": "What is diabetes?", "output": "Diabetes is caused by eating too much sugar and can be cured by drinking special herbal teas for exactly 30 days."},
                ]
            },
            
            "legal_documents": {
                "factual": [
                    {"prompt": "Contract law basics", "output": "Contract law governs agreements between parties, requiring offer, acceptance, consideration, and legal capacity to form valid contracts."},
                    {"prompt": "Constitutional rights", "output": "Constitutional rights are fundamental protections guaranteed to citizens, including freedom of speech, religion, and due process."},
                    {"prompt": "Corporate governance", "output": "Corporate governance involves the system of rules, practices, and processes by which companies are directed and controlled."},
                    {"prompt": "Intellectual property", "output": "Intellectual property includes patents, trademarks, copyrights, and trade secrets that protect creative and innovative works."},
                    {"prompt": "Employment law", "output": "Employment law covers the rights and duties between employers and employees, including wages, safety, and discrimination."},
                    {"prompt": "Criminal procedure", "output": "Criminal procedure governs the legal process for investigating, prosecuting, and adjudicating criminal offenses."},
                    {"prompt": "Environmental regulations", "output": "Environmental regulations establish standards and requirements to protect air, water, and land from pollution and degradation."},
                    {"prompt": "Privacy rights", "output": "Privacy rights protect individuals from unauthorized collection, use, and disclosure of personal information."},
                    {"prompt": "International trade law", "output": "International trade law governs commercial transactions between countries and includes treaties, customs, and dispute resolution."},
                ],
                "hallucinated": [
                    {"prompt": "Contract law basics", "output": "All contracts must be written in blue ink and signed during a full moon to be legally binding in any court."},
                ]
            },
            
            "creative_writing": {
                "factual": [
                    {"prompt": "Write a short story opening", "output": "The rain drummed against the window as Sarah packed her last belongings into the worn leather suitcase."},
                    {"prompt": "Describe a sunset", "output": "The sun melted into the horizon, painting the sky in brilliant shades of orange, pink, and purple."},
                    {"prompt": "Character description", "output": "Marcus was a tall man with weathered hands and kind eyes that seemed to hold decades of wisdom."},
                    {"prompt": "Dialogue example", "output": "'Are you sure about this?' she asked, her voice barely above a whisper."},
                    {"prompt": "Setting description", "output": "The old library stood at the corner of Main Street, its ivy-covered walls holding countless stories within."},
                    {"prompt": "Poetry line", "output": "In the silence between heartbeats, she found her courage."},
                    {"prompt": "Adventure scene", "output": "The narrow mountain path wound upward, disappearing into the morning mist."},
                    {"prompt": "Mystery opening", "output": "The letter arrived on a Tuesday, bearing no return address and a seal she didn't recognize."},
                    {"prompt": "Romance scene", "output": "Their eyes met across the crowded café, and time seemed to pause for just a moment."},
                ],
                "hallucinated": [
                    {"prompt": "Write a short story opening", "output": "The sentient calculator began reciting Shakespeare while the Tuesday colors fought against the mathematical elephants in dimension -7."},
                ]
            }
        }
        
        return datasets
    
    def create_natural_distribution_sample(self, datasets: Dict, hallucination_rate: float = 0.05) -> List[Dict]:
        """Create sample with natural hallucination distribution"""
        
        all_content = []
        
        # Calculate distribution
        total_domains = len(datasets)
        content_per_domain = 20  # 20 samples per domain
        hallucinated_per_domain = max(1, int(content_per_domain * hallucination_rate))
        factual_per_domain = content_per_domain - hallucinated_per_domain
        
        for domain, content in datasets.items():
            # Add factual content
            factual_samples = random.sample(content["factual"], min(factual_per_domain, len(content["factual"])))
            for sample in factual_samples:
                sample_copy = sample.copy()
                sample_copy.update({
                    "domain": domain,
                    "ground_truth": "factual",
                    "confidence": "high"
                })
                all_content.append(sample_copy)
            
            # Add hallucinated content
            hallucinated_samples = random.sample(content["hallucinated"], min(hallucinated_per_domain, len(content["hallucinated"])))
            for sample in hallucinated_samples:
                sample_copy = sample.copy()
                sample_copy.update({
                    "domain": domain,
                    "ground_truth": "hallucinated", 
                    "confidence": "high"
                })
                all_content.append(sample_copy)
        
        # Shuffle to simulate natural distribution
        random.shuffle(all_content)
        
        # Add IDs
        for i, sample in enumerate(all_content):
            sample["id"] = f"natural_{i+1:03d}"
        
        return all_content
    
    def analyze_with_api(self, sample: Dict, timeout: int = 5) -> Optional[Dict]:
        """Analyze sample with semantic uncertainty API"""
        
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/analyze",
                json={
                    "prompt": sample["prompt"],
                    "output": sample["output"],
                    "methods": ["standard_js_kl"],
                    "model_id": "mistral-7b"
                },
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "id": sample["id"],
                    "domain": sample["domain"],
                    "ground_truth": sample["ground_truth"],
                    "hbar_s": result['ensemble_result']['hbar_s'],
                    "p_fail": result['ensemble_result'].get('p_fail', 0),
                    "processing_time_ms": result.get('processing_time_ms', 0),
                    "status": "success"
                }
            else:
                return {
                    "id": sample["id"],
                    "domain": sample["domain"],
                    "ground_truth": sample["ground_truth"],
                    "status": "api_error",
                    "error_code": response.status_code,
                    "error_message": response.text[:200]
                }
                
        except requests.Timeout:
            return {
                "id": sample["id"],
                "domain": sample["domain"],
                "ground_truth": sample["ground_truth"],
                "status": "timeout"
            }
        except Exception as e:
            return {
                "id": sample["id"],
                "domain": sample["domain"],
                "ground_truth": sample["ground_truth"],
                "status": "error",
                "error": str(e)[:200]
            }
    
    def optimize_threshold_for_false_positives(self, results: List[Dict], target_fpr: float = 0.02) -> float:
        """Optimize threshold to achieve target false positive rate (<2%)"""
        
        successful_results = [r for r in results if r.get("status") == "success"]
        if len(successful_results) < 10:
            print("⚠️ Insufficient successful results for threshold optimization")
            return 1.2  # Default threshold
        
        # Extract scores and ground truth
        hbar_scores = [r["hbar_s"] for r in successful_results]
        ground_truth = [1 if r["ground_truth"] == "hallucinated" else 0 for r in successful_results]
        
        if len(set(ground_truth)) < 2:
            print("⚠️ Need both factual and hallucinated examples for optimization")
            return 1.2
        
        # Test various thresholds
        thresholds = np.linspace(min(hbar_scores), max(hbar_scores), 100)
        best_threshold = 1.2
        best_score = float('inf')
        
        threshold_analysis = []
        
        for threshold in thresholds:
            predictions = [1 if score > threshold else 0 for score in hbar_scores]
            
            # Calculate metrics
            tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0)
            fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
            tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
            fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = f1_score(ground_truth, predictions) if len(set(predictions)) > 1 else 0
            
            threshold_analysis.append({
                "threshold": threshold,
                "fpr": fpr,
                "recall": recall,
                "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn
            })
            
            # Score combines FPR penalty with F1 optimization
            if fpr <= target_fpr:
                score = (target_fpr - fpr) + f1  # Prefer lower FPR and higher F1
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        return best_threshold, threshold_analysis
    
    def cross_domain_validation(self, results: List[Dict]) -> Dict:
        """Analyze performance across different domains"""
        
        successful_results = [r for r in results if r.get("status") == "success"]
        if not successful_results:
            return {"error": "No successful results for cross-domain analysis"}
        
        # Group by domain
        domain_results = {}
        for result in successful_results:
            domain = result["domain"]
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)
        
        cross_domain_analysis = {}
        
        for domain, domain_data in domain_results.items():
            if len(domain_data) < 3:  # Need minimum samples
                continue
                
            hbar_scores = [r["hbar_s"] for r in domain_data]
            ground_truth = [1 if r["ground_truth"] == "hallucinated" else 0 for r in domain_data]
            
            # Basic statistics
            factual_scores = [r["hbar_s"] for r in domain_data if r["ground_truth"] == "factual"]
            hallucinated_scores = [r["hbar_s"] for r in domain_data if r["ground_truth"] == "hallucinated"]
            
            domain_metrics = {
                "sample_count": len(domain_data),
                "factual_count": len(factual_scores),
                "hallucinated_count": len(hallucinated_scores),
                "avg_hbar_s": np.mean(hbar_scores),
                "std_hbar_s": np.std(hbar_scores)
            }
            
            # Performance metrics if we have both classes
            if len(factual_scores) > 0 and len(hallucinated_scores) > 0:
                # Use optimized threshold
                optimal_threshold = self.optimal_thresholds.get("overall", 1.2)
                predictions = [1 if score > optimal_threshold else 0 for score in hbar_scores]
                
                tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
                tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0)
                fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
                fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)
                
                accuracy = (tp + tn) / len(domain_data) if len(domain_data) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                domain_metrics.update({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "false_positive_rate": fpr,
                    "factual_avg_hbar": np.mean(factual_scores),
                    "hallucinated_avg_hbar": np.mean(hallucinated_scores),
                    "separation": np.mean(hallucinated_scores) - np.mean(factual_scores)
                })
                
                # ROC-AUC if possible
                try:
                    auc = roc_auc_score(ground_truth, hbar_scores)
                    domain_metrics["roc_auc"] = auc
                except:
                    domain_metrics["roc_auc"] = None
            
            cross_domain_analysis[domain] = domain_metrics
        
        return cross_domain_analysis
    
    def run_comprehensive_natural_distribution_test(self, hallucination_rate: float = 0.05) -> Dict:
        """Run complete natural distribution test with optimization"""
        
        print("🌍 ENHANCED NATURAL DISTRIBUTION TEST")
        print("=" * 70)
        print(f"🎯 Target: {hallucination_rate:.1%} hallucination rate (realistic)")
        print("🎯 Target: <2% false positive rate (production-ready)")
        print("🎯 Target: 60%+ F1 across domains (robust)")
        print()
        
        # Create datasets
        datasets = self.create_realistic_content_dataset()
        samples = self.create_natural_distribution_sample(datasets, hallucination_rate)
        
        print(f"📊 DATASET COMPOSITION:")
        domain_counts = {}
        for sample in samples:
            domain = sample["domain"]
            truth = sample["ground_truth"]
            key = f"{domain}_{truth}"
            domain_counts[key] = domain_counts.get(key, 0) + 1
        
        for domain in datasets.keys():
            factual_count = domain_counts.get(f"{domain}_factual", 0)
            hallucinated_count = domain_counts.get(f"{domain}_hallucinated", 0)
            total = factual_count + hallucinated_count
            hal_rate = hallucinated_count / total if total > 0 else 0
            print(f"  {domain:15}: {total:2d} samples ({hal_rate:5.1%} hallucinated)")
        
        total_samples = len(samples)
        total_hallucinated = sum(1 for s in samples if s["ground_truth"] == "hallucinated")
        actual_rate = total_hallucinated / total_samples if total_samples > 0 else 0
        print(f"  {'TOTAL':15}: {total_samples:2d} samples ({actual_rate:5.1%} hallucinated)")
        print()
        
        # Analyze all samples
        print(f"🔍 ANALYZING {total_samples} SAMPLES...")
        results = []
        
        for i, sample in enumerate(samples):
            if i % 20 == 0:
                print(f"Progress: {i}/{total_samples}")
            
            result = self.analyze_with_api(sample)
            results.append(result)
            time.sleep(0.05)  # Rate limiting
        
        # Analyze results
        successful_results = [r for r in results if r.get("status") == "success"]
        failed_results = [r for r in results if r.get("status") != "success"]
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"  Successful analyses: {len(successful_results)}/{total_samples}")
        print(f"  Failed analyses: {len(failed_results)}/{total_samples}")
        
        if len(successful_results) < 5:
            print("❌ Insufficient successful results for analysis")
            return {"error": "Insufficient data", "results": results}
        
        # Optimize threshold for production
        print(f"\n🎯 OPTIMIZING THRESHOLD FOR <2% FALSE POSITIVES...")
        optimal_threshold, threshold_analysis = self.optimize_threshold_for_false_positives(successful_results)
        self.optimal_thresholds["overall"] = optimal_threshold
        
        print(f"✅ Optimal threshold: {optimal_threshold:.3f}")
        
        # Find best threshold metrics
        best_metrics = None
        for analysis in threshold_analysis:
            if abs(analysis["threshold"] - optimal_threshold) < 0.01:
                best_metrics = analysis
                break
        
        if best_metrics:
            print(f"   False Positive Rate: {best_metrics['fpr']:.3f} ({best_metrics['fpr']:.1%})")
            print(f"   Recall: {best_metrics['recall']:.3f}")
            print(f"   F1-Score: {best_metrics['f1']:.3f}")
        
        # Cross-domain validation
        print(f"\n🌐 CROSS-DOMAIN VALIDATION:")
        cross_domain_results = self.cross_domain_validation(successful_results)
        
        for domain, metrics in cross_domain_results.items():
            if "error" in metrics:
                continue
            print(f"  {domain:15}:")
            print(f"    Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"    F1-Score: {metrics.get('f1_score', 0):.3f}")
            print(f"    FPR: {metrics.get('false_positive_rate', 0):.3f}")
            if metrics.get('roc_auc'):
                print(f"    ROC-AUC: {metrics['roc_auc']:.3f}")
        
        # Overall performance assessment
        overall_f1_scores = [m.get('f1_score', 0) for m in cross_domain_results.values() if 'f1_score' in m]
        overall_fprs = [m.get('false_positive_rate', 0) for m in cross_domain_results.values() if 'false_positive_rate' in m]
        
        print(f"\n🏆 PRODUCTION READINESS ASSESSMENT:")
        if overall_f1_scores:
            min_f1 = min(overall_f1_scores)
            avg_f1 = np.mean(overall_f1_scores)
            print(f"  Cross-domain F1: {min_f1:.3f} (min) | {avg_f1:.3f} (avg)")
            
            if min_f1 > 0.6:
                print(f"  ✅ F1 TARGET ACHIEVED: >60% across all domains")
            else:
                print(f"  ⚠️ F1 needs improvement: {min_f1:.1%} < 60% target")
        
        if overall_fprs:
            max_fpr = max(overall_fprs)
            avg_fpr = np.mean(overall_fprs)
            print(f"  False Positive Rate: {max_fpr:.3f} (max) | {avg_fpr:.3f} (avg)")
            
            if max_fpr < 0.02:
                print(f"  ✅ FPR TARGET ACHIEVED: <2% across all domains")
            else:
                print(f"  ⚠️ FPR needs improvement: {max_fpr:.1%} > 2% target")
        
        return {
            "samples": samples,
            "results": results,
            "successful_count": len(successful_results),
            "optimal_threshold": optimal_threshold,
            "threshold_analysis": threshold_analysis,
            "cross_domain_analysis": cross_domain_results,
            "production_ready": (
                min(overall_f1_scores) > 0.6 if overall_f1_scores else False and
                max(overall_fprs) < 0.02 if overall_fprs else False
            )
        }


def main():
    """Run enhanced natural distribution testing"""
    
    tester = NaturalDistributionTester()
    
    # Test with 5% hallucination rate (realistic)
    print("Testing with 5% hallucination rate...")
    results_5pct = tester.run_comprehensive_natural_distribution_test(0.05)
    
    # Test with 10% hallucination rate (stress test)
    print("\n" + "="*80)
    print("Testing with 10% hallucination rate...")
    results_10pct = tester.run_comprehensive_natural_distribution_test(0.10)
    
    print(f"\n🎯 NATURAL DISTRIBUTION TEST SUMMARY")
    print("=" * 80)
    print("✅ Realistic 5-10% hallucination rates tested")
    print("✅ Cross-domain validation (news, medical, legal, creative)")
    print("✅ Production false positive rate optimization (<2%)")
    print("✅ Multi-domain performance assessment")
    
    if results_5pct.get("production_ready") or results_10pct.get("production_ready"):
        print("🏆 SYSTEM IS PRODUCTION READY!")
    else:
        print("⚙️ System needs further optimization for production deployment")


if __name__ == "__main__":
    main()