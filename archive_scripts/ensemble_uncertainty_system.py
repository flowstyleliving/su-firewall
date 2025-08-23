#!/usr/bin/env python3
"""
🔄 Ensemble Uncertainty System - Multiple ℏₛ Calculations per Input

This system implements multiple methods for calculating semantic uncertainty (ℏₛ)
and combines them using various ensemble strategies for more robust predictions.

Architecture:
1. Multiple ℏₛ calculation methods (semantic diversity)
2. Ensemble aggregation strategies (voting, weighting, consensus)
3. Confidence estimation and uncertainty quantification
4. Golden scale calibration applied to ensemble results
"""

import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import statistics

class EnsembleMethod(Enum):
    """Different ensemble aggregation methods."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    ROBUST_MEAN = "robust_mean"
    CONSENSUS_VOTING = "consensus_voting"
    MAXIMUM_UNCERTAINTY = "max_uncertainty"
    MINIMUM_UNCERTAINTY = "min_uncertainty"
    CONFIDENCE_WEIGHTED = "confidence_weighted"

class UncertaintyCalculationMethod(Enum):
    """Different methods for calculating ℏₛ."""
    STANDARD_JS_KL = "js_kl_divergence"
    ENTROPY_BASED = "entropy_based"
    FISHER_INFORMATION = "fisher_information"
    BOOTSTRAP_SAMPLING = "bootstrap_sampling"
    PERTURBATION_ANALYSIS = "perturbation_analysis"
    BAYESIAN_UNCERTAINTY = "bayesian_uncertainty"
    MONTE_CARLO = "monte_carlo"
    ENSEMBLE_DISTILLATION = "ensemble_distillation"

@dataclass
class UncertaintyResult:
    """Result from a single uncertainty calculation method."""
    method: UncertaintyCalculationMethod
    hbar_s: float
    confidence: float
    delta_mu: float
    delta_sigma: float
    computation_time_ms: float
    metadata: Dict[str, Any]

@dataclass
class EnsembleUncertaintyResult:
    """Result from ensemble uncertainty calculation."""
    ensemble_hbar_s: float
    individual_results: List[UncertaintyResult]
    aggregation_method: EnsembleMethod
    consensus_score: float
    uncertainty_bounds: Tuple[float, float]
    reliability_score: float
    golden_scale_applied: bool
    final_p_fail: float
    computation_time_ms: float

class SemanticUncertaintyCalculator:
    """Individual uncertainty calculation methods."""
    
    def __init__(self):
        self.calculation_cache = {}
    
    def calculate_js_kl_standard(self, p_dist: List[float], q_dist: List[float]) -> UncertaintyResult:
        """Standard Jensen-Shannon + KL divergence approach (current method)."""
        start_time = datetime.now()
        
        p = np.array(p_dist)
        q = np.array(q_dist)
        
        # Jensen-Shannon divergence (precision/μ)
        m = 0.5 * (p + q)
        js_div = 0.5 * (self._kl_divergence(p, m) + self._kl_divergence(q, m))
        
        # KL divergence (flexibility/σ)
        kl_div = self._kl_divergence(p, q)
        
        # Semantic uncertainty: ℏₛ = √(Δμ × Δσ)
        hbar_s = np.sqrt(js_div * kl_div)
        confidence = self._calculate_confidence(js_div, kl_div)
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return UncertaintyResult(
            method=UncertaintyCalculationMethod.STANDARD_JS_KL,
            hbar_s=hbar_s,
            confidence=confidence,
            delta_mu=js_div,
            delta_sigma=kl_div,
            computation_time_ms=computation_time,
            metadata={"approach": "standard"}
        )
    
    def calculate_entropy_based(self, p_dist: List[float], q_dist: List[float]) -> UncertaintyResult:
        """Entropy-based uncertainty calculation."""
        start_time = datetime.now()
        
        p = np.array(p_dist)
        q = np.array(q_dist)
        
        # Shannon entropy difference
        h_p = -np.sum(p * np.log2(p + 1e-12))
        h_q = -np.sum(q * np.log2(q + 1e-12))
        
        # Cross-entropy
        cross_entropy = -np.sum(p * np.log2(q + 1e-12))
        
        # Entropy-based uncertainty
        entropy_diff = abs(h_p - h_q)
        excess_entropy = cross_entropy - h_p
        
        hbar_s = np.sqrt(entropy_diff * excess_entropy)
        confidence = 1.0 / (1.0 + entropy_diff)
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return UncertaintyResult(
            method=UncertaintyCalculationMethod.ENTROPY_BASED,
            hbar_s=hbar_s,
            confidence=confidence,
            delta_mu=entropy_diff,
            delta_sigma=excess_entropy,
            computation_time_ms=computation_time,
            metadata={"h_p": h_p, "h_q": h_q}
        )
    
    def calculate_bootstrap_sampling(self, p_dist: List[float], q_dist: List[float], 
                                   n_samples: int = 100) -> UncertaintyResult:
        """Bootstrap sampling for uncertainty estimation."""
        start_time = datetime.now()
        
        p = np.array(p_dist)
        q = np.array(q_dist)
        
        # Generate bootstrap samples
        uncertainties = []
        for _ in range(n_samples):
            # Add noise to distributions
            p_noisy = p + np.random.normal(0, 0.01, len(p))
            q_noisy = q + np.random.normal(0, 0.01, len(q))
            
            # Renormalize
            p_noisy = np.abs(p_noisy)
            q_noisy = np.abs(q_noisy)
            p_noisy = p_noisy / np.sum(p_noisy)
            q_noisy = q_noisy / np.sum(q_noisy)
            
            # Calculate uncertainty for this sample
            js_div = self._js_divergence(p_noisy, q_noisy)
            kl_div = self._kl_divergence(p_noisy, q_noisy)
            uncertainties.append(np.sqrt(js_div * kl_div))
        
        hbar_s = np.mean(uncertainties)
        uncertainty_std = np.std(uncertainties)
        confidence = 1.0 - (uncertainty_std / (hbar_s + 1e-12))
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return UncertaintyResult(
            method=UncertaintyCalculationMethod.BOOTSTRAP_SAMPLING,
            hbar_s=hbar_s,
            confidence=max(0.0, confidence),
            delta_mu=np.mean([self._js_divergence(p, q)]),
            delta_sigma=np.mean([self._kl_divergence(p, q)]),
            computation_time_ms=computation_time,
            metadata={"n_samples": n_samples, "std": uncertainty_std}
        )
    
    def calculate_perturbation_analysis(self, p_dist: List[float], q_dist: List[float]) -> UncertaintyResult:
        """Perturbation-based uncertainty analysis."""
        start_time = datetime.now()
        
        p = np.array(p_dist)
        q = np.array(q_dist)
        
        # Calculate baseline uncertainty
        baseline_js = self._js_divergence(p, q)
        baseline_kl = self._kl_divergence(p, q)
        baseline_uncertainty = np.sqrt(baseline_js * baseline_kl)
        
        # Apply various perturbations and measure sensitivity
        perturbation_levels = [0.001, 0.005, 0.01, 0.05]
        sensitivity_scores = []
        
        for level in perturbation_levels:
            perturbations = []
            for _ in range(10):  # Multiple random perturbations
                p_pert = p + np.random.uniform(-level, level, len(p))
                p_pert = np.abs(p_pert) / np.sum(np.abs(p_pert))
                
                js_div = self._js_divergence(p_pert, q)
                kl_div = self._kl_divergence(p_pert, q)
                uncertainty = np.sqrt(js_div * kl_div)
                perturbations.append(abs(uncertainty - baseline_uncertainty))
            
            sensitivity_scores.append(np.mean(perturbations))
        
        # Uncertainty increases with perturbation sensitivity
        perturbation_sensitivity = np.mean(sensitivity_scores)
        hbar_s = baseline_uncertainty * (1.0 + perturbation_sensitivity)
        confidence = 1.0 / (1.0 + perturbation_sensitivity)
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return UncertaintyResult(
            method=UncertaintyCalculationMethod.PERTURBATION_ANALYSIS,
            hbar_s=hbar_s,
            confidence=confidence,
            delta_mu=baseline_js,
            delta_sigma=baseline_kl,
            computation_time_ms=computation_time,
            metadata={"sensitivity": perturbation_sensitivity}
        )
    
    def calculate_bayesian_uncertainty(self, p_dist: List[float], q_dist: List[float]) -> UncertaintyResult:
        """Bayesian uncertainty estimation."""
        start_time = datetime.now()
        
        p = np.array(p_dist)
        q = np.array(q_dist)
        
        # Model uncertainty using Dirichlet distribution
        alpha_p = p * 10 + 0.1  # Prior concentration
        alpha_q = q * 10 + 0.1
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = np.sum(p * (1 - p))
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = np.sum((alpha_p - 1) / (np.sum(alpha_p) * (np.sum(alpha_p) + 1)))
        
        # Total uncertainty
        total_uncertainty = aleatoric + epistemic
        
        # Convert to semantic uncertainty scale
        hbar_s = np.sqrt(total_uncertainty * self._kl_divergence(p, q))
        confidence = epistemic / (aleatoric + epistemic + 1e-12)
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return UncertaintyResult(
            method=UncertaintyCalculationMethod.BAYESIAN_UNCERTAINTY,
            hbar_s=hbar_s,
            confidence=confidence,
            delta_mu=aleatoric,
            delta_sigma=epistemic,
            computation_time_ms=computation_time,
            metadata={"aleatoric": aleatoric, "epistemic": epistemic}
        )
    
    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence."""
        m = 0.5 * (p + q)
        return 0.5 * (self._kl_divergence(p, m) + self._kl_divergence(q, m))
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Kullback-Leibler divergence."""
        return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))
    
    def _calculate_confidence(self, js_div: float, kl_div: float) -> float:
        """Calculate confidence score for uncertainty estimate."""
        # Higher divergences suggest more reliable uncertainty estimates
        return min(1.0, (js_div + kl_div) / 2.0)

class EnsembleUncertaintySystem:
    """Main ensemble system coordinating multiple uncertainty calculations."""
    
    def __init__(self, golden_scale: float = 3.4):
        self.calculator = SemanticUncertaintyCalculator()
        self.golden_scale = golden_scale
        self.method_weights = self._initialize_method_weights()
    
    def _initialize_method_weights(self) -> Dict[UncertaintyCalculationMethod, float]:
        """Initialize weights for different calculation methods."""
        return {
            UncertaintyCalculationMethod.STANDARD_JS_KL: 1.0,
            UncertaintyCalculationMethod.ENTROPY_BASED: 0.8,
            UncertaintyCalculationMethod.BOOTSTRAP_SAMPLING: 0.9,
            UncertaintyCalculationMethod.PERTURBATION_ANALYSIS: 0.7,
            UncertaintyCalculationMethod.BAYESIAN_UNCERTAINTY: 0.85,
        }
    
    def calculate_ensemble_uncertainty(self, 
                                     p_dist: List[float], 
                                     q_dist: List[float],
                                     methods: List[UncertaintyCalculationMethod] = None,
                                     ensemble_method: EnsembleMethod = EnsembleMethod.CONFIDENCE_WEIGHTED) -> EnsembleUncertaintyResult:
        """Calculate ensemble uncertainty using multiple methods."""
        start_time = datetime.now()
        
        if methods is None:
            methods = [
                UncertaintyCalculationMethod.STANDARD_JS_KL,
                UncertaintyCalculationMethod.ENTROPY_BASED,
                UncertaintyCalculationMethod.BOOTSTRAP_SAMPLING,
                UncertaintyCalculationMethod.PERTURBATION_ANALYSIS,
                UncertaintyCalculationMethod.BAYESIAN_UNCERTAINTY,
            ]
        
        # Calculate uncertainty using each method
        individual_results = []
        for method in methods:
            try:
                if method == UncertaintyCalculationMethod.STANDARD_JS_KL:
                    result = self.calculator.calculate_js_kl_standard(p_dist, q_dist)
                elif method == UncertaintyCalculationMethod.ENTROPY_BASED:
                    result = self.calculator.calculate_entropy_based(p_dist, q_dist)
                elif method == UncertaintyCalculationMethod.BOOTSTRAP_SAMPLING:
                    result = self.calculator.calculate_bootstrap_sampling(p_dist, q_dist)
                elif method == UncertaintyCalculationMethod.PERTURBATION_ANALYSIS:
                    result = self.calculator.calculate_perturbation_analysis(p_dist, q_dist)
                elif method == UncertaintyCalculationMethod.BAYESIAN_UNCERTAINTY:
                    result = self.calculator.calculate_bayesian_uncertainty(p_dist, q_dist)
                else:
                    continue
                
                individual_results.append(result)
            except Exception as e:
                print(f"⚠️ Method {method.value} failed: {e}")
                continue
        
        if not individual_results:
            raise ValueError("No uncertainty calculations succeeded")
        
        # Aggregate results using specified ensemble method
        ensemble_hbar_s = self._aggregate_uncertainties(individual_results, ensemble_method)
        
        # Calculate ensemble statistics
        consensus_score = self._calculate_consensus_score(individual_results)
        uncertainty_bounds = self._calculate_uncertainty_bounds(individual_results)
        reliability_score = self._calculate_reliability_score(individual_results, consensus_score)
        
        # Apply golden scale calibration
        calibrated_hbar_s = ensemble_hbar_s * self.golden_scale
        
        # Calculate final failure probability (simplified)
        lambda_val, tau_val = 5.0, 2.0
        p_fail = 1.0 / (1.0 + np.exp(-lambda_val * (calibrated_hbar_s - tau_val)))
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return EnsembleUncertaintyResult(
            ensemble_hbar_s=calibrated_hbar_s,
            individual_results=individual_results,
            aggregation_method=ensemble_method,
            consensus_score=consensus_score,
            uncertainty_bounds=uncertainty_bounds,
            reliability_score=reliability_score,
            golden_scale_applied=True,
            final_p_fail=p_fail,
            computation_time_ms=computation_time
        )
    
    def _aggregate_uncertainties(self, results: List[UncertaintyResult], method: EnsembleMethod) -> float:
        """Aggregate individual uncertainty estimates."""
        values = [r.hbar_s for r in results]
        confidences = [r.confidence for r in results]
        
        if method == EnsembleMethod.SIMPLE_AVERAGE:
            return np.mean(values)
        
        elif method == EnsembleMethod.WEIGHTED_AVERAGE:
            weights = [self.method_weights.get(r.method, 1.0) for r in results]
            return np.average(values, weights=weights)
        
        elif method == EnsembleMethod.MEDIAN:
            return np.median(values)
        
        elif method == EnsembleMethod.ROBUST_MEAN:
            # Trim extreme values and take mean
            sorted_values = sorted(values)
            trim_size = max(1, len(values) // 4)
            trimmed = sorted_values[trim_size:-trim_size] if len(values) > 4 else sorted_values
            return np.mean(trimmed)
        
        elif method == EnsembleMethod.CONFIDENCE_WEIGHTED:
            # Weight by confidence scores
            total_confidence = sum(confidences)
            if total_confidence == 0:
                return np.mean(values)
            weights = [c / total_confidence for c in confidences]
            return np.average(values, weights=weights)
        
        elif method == EnsembleMethod.CONSENSUS_VOTING:
            # Bin values and vote
            bins = np.linspace(min(values), max(values), 10)
            bin_indices = np.digitize(values, bins)
            consensus_bin = statistics.mode(bin_indices)
            consensus_values = [v for v, b in zip(values, bin_indices) if b == consensus_bin]
            return np.mean(consensus_values)
        
        elif method == EnsembleMethod.MAXIMUM_UNCERTAINTY:
            return max(values)
        
        elif method == EnsembleMethod.MINIMUM_UNCERTAINTY:
            return min(values)
        
        else:
            return np.mean(values)
    
    def _calculate_consensus_score(self, results: List[UncertaintyResult]) -> float:
        """Calculate how much the different methods agree."""
        values = [r.hbar_s for r in results]
        if len(values) < 2:
            return 1.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        coefficient_of_variation = std_val / (mean_val + 1e-12)
        
        # Consensus score: higher when methods agree more
        return max(0.0, 1.0 - coefficient_of_variation)
    
    def _calculate_uncertainty_bounds(self, results: List[UncertaintyResult]) -> Tuple[float, float]:
        """Calculate confidence bounds for ensemble uncertainty."""
        values = [r.hbar_s for r in results]
        return (min(values), max(values))
    
    def _calculate_reliability_score(self, results: List[UncertaintyResult], consensus_score: float) -> float:
        """Calculate overall reliability of ensemble estimate."""
        # Combine consensus with individual method confidences
        avg_confidence = np.mean([r.confidence for r in results])
        method_diversity = len(set(r.method for r in results)) / len(UncertaintyCalculationMethod)
        
        reliability = 0.4 * consensus_score + 0.4 * avg_confidence + 0.2 * method_diversity
        return min(1.0, reliability)

def demonstrate_ensemble_uncertainty():
    """Demonstrate the ensemble uncertainty system."""
    print("🔄 ENSEMBLE UNCERTAINTY SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    ensemble_system = EnsembleUncertaintySystem(golden_scale=3.4)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "High Confidence (Low Uncertainty)",
            "p_dist": [0.8, 0.15, 0.03, 0.02],
            "q_dist": [0.75, 0.18, 0.04, 0.03],
            "expected": "Low ensemble uncertainty"
        },
        {
            "name": "Conflicting Distributions (High Uncertainty)", 
            "p_dist": [0.4, 0.3, 0.2, 0.1],
            "q_dist": [0.1, 0.2, 0.3, 0.4],
            "expected": "High ensemble uncertainty"
        },
        {
            "name": "Moderate Disagreement",
            "p_dist": [0.6, 0.25, 0.1, 0.05],
            "q_dist": [0.45, 0.35, 0.15, 0.05],
            "expected": "Medium ensemble uncertainty"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Expected: {scenario['expected']}")
        print("-" * 60)
        
        try:
            result = ensemble_system.calculate_ensemble_uncertainty(
                scenario['p_dist'], 
                scenario['q_dist'],
                ensemble_method=EnsembleMethod.CONFIDENCE_WEIGHTED
            )
            
            print(f"🎯 Ensemble ℏₛ: {result.ensemble_hbar_s:.6f}")
            print(f"📈 P(fail): {result.final_p_fail:.6f}")
            print(f"🤝 Consensus Score: {result.consensus_score:.3f}")
            print(f"🎚️ Uncertainty Bounds: [{result.uncertainty_bounds[0]:.4f}, {result.uncertainty_bounds[1]:.4f}]")
            print(f"🔒 Reliability Score: {result.reliability_score:.3f}")
            print(f"⏱️ Computation Time: {result.computation_time_ms:.1f}ms")
            
            print("\n   Individual Method Results:")
            for individual in result.individual_results:
                print(f"   • {individual.method.value:<20}: ℏₛ={individual.hbar_s:.6f}, confidence={individual.confidence:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True

if __name__ == "__main__":
    demonstrate_ensemble_uncertainty()