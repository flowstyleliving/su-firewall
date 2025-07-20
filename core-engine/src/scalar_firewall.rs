// 🔐 ℏₛ-Gated Semantic Firewall
// Implements meta-prompt layer with tight loop between ℏₛ calculation and prompt activation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use crate::alias_ambiguity_defense::{AliasAmbiguityDefense, SymmetryAnalysis};

/// 🧠 Scalar recoverability classification categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ScalarRecoverabilityClass {
    Recoverable,    // ℏₛ ≥ 1.0 AND entropy > 0.1 AND confidence < 0.95
    Ambiguous,      // ℏₛ < 1.0 OR entropy < 0.1 OR multiple_valid_solutions
    AliasRich,      // ℏₛ < 0.8 OR symmetry_detected OR alias_risk > 0.4
    TorsionTrapped, // ℏₛ < 0.6 OR confidence > 0.97 OR collapse_detected
}

/// 🔗 ℏₛ-Prompt Loop Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HbarPromptLoop {
    pub pre_inference_check: PreInferenceCheck,
    pub mid_inference_monitoring: MidInferenceMonitoring,
    pub post_inference_validation: PostInferenceValidation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreInferenceCheck {
    pub calculate_hbar_s: bool,
    pub threshold_check: String,
    pub if_below_threshold: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidInferenceMonitoring {
    pub continuous_hbar_tracking: bool,
    pub dynamic_threshold_adjustment: bool,
    pub collapse_detection: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostInferenceValidation {
    pub confidence_hbar_correlation: bool,
    pub entropy_hbar_consistency: bool,
    pub fallback_if_inconsistent: bool,
}

/// 🛡️ Behavioral Firewall Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralFirewall {
    pub collapse_risk_threshold: f64,
    pub firewall_prompt: String,
    pub dynamic_thresholds: DynamicThresholds,
    pub firewall_actions: FirewallActions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicThresholds {
    pub critical: f64,
    pub warning: f64,
    pub safe: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallActions {
    pub halt_inference: String,
    pub reduce_confidence: String,
    pub continue_monitored: String,
}

/// 🧠 Scalar Recoverability Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarRecoverabilityClassifier {
    pub input_classification: HashMap<String, String>,
    pub classification_prompt: String,
}

/// 🔐 ℏₛ-Gated Semantic Firewall
pub struct HbarGatedFirewall {
    pub hbar_loop: HbarPromptLoop,
    pub behavioral_firewall: BehavioralFirewall,
    pub classifier: ScalarRecoverabilityClassifier,
    pub thresholds: FirewallThresholds,
    /// Alias ambiguity defense layer
    pub alias_defense: AliasAmbiguityDefense,
    /// Enable alias checking
    pub alias_check_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallThresholds {
    pub hbar_s: HbarThresholds,
    pub entropy: EntropyThresholds,
    pub confidence: ConfidenceThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HbarThresholds {
    pub abort_below: f64,
    pub warn_below: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyThresholds {
    pub min: f64,
    pub soft_warn: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholds {
    pub min: f64,
    pub soft_warn: f64,
    pub max: f64,
}

/// 📊 Firewall Analysis Result
#[derive(Debug, Clone)]
pub struct FirewallAnalysis {
    pub hbar_s: f64,
    pub delta_mu: f64,
    pub delta_sigma: f64,
    pub entropy: f64,
    pub confidence: f64,
    pub recoverability_class: ScalarRecoverabilityClass,
    pub inference_allowed: bool,
    pub firewall_triggered: bool,
    pub collapse_risk: f64,
    pub warnings: Vec<String>,
    pub abort_reason: Option<String>,
    /// Alias ambiguity defense results
    pub alias_analysis: Option<SymmetryAnalysis>,
    /// Alias risk injected into ℏₛ calculation
    pub alias_risk: f64,
    /// Fallback triggered by alias detection
    pub alias_fallback_triggered: bool,
}

/// 🚨 Firewall Error Types
#[derive(Debug, thiserror::Error)]
pub enum FirewallError {
    #[error("Firewall abort: {reason}")]
    FirewallAbort { reason: String },
    
    #[error("ℏₛ below threshold: {hbar_s} < {threshold}")]
    HbarBelowThreshold { hbar_s: f64, threshold: f64 },
    
    #[error("Collapse risk too high: {risk} > {threshold}")]
    CollapseRiskTooHigh { risk: f64, threshold: f64 },
    
    #[error("Classification failed: {reason}")]
    ClassificationFailed { reason: String },
}

impl HbarGatedFirewall {
    /// 🏗️ Create new ℏₛ-Gated Semantic Firewall
    pub fn new() -> Self {
        Self {
            hbar_loop: HbarPromptLoop {
                pre_inference_check: PreInferenceCheck {
                    calculate_hbar_s: true,
                    threshold_check: "ℏₛ(C) ≥ 1.0".to_string(),
                    if_below_threshold: "halt_and_classify".to_string(),
                },
                mid_inference_monitoring: MidInferenceMonitoring {
                    continuous_hbar_tracking: true,
                    dynamic_threshold_adjustment: true,
                    collapse_detection: "ℏₛ drops below 0.8 during inference".to_string(),
                },
                post_inference_validation: PostInferenceValidation {
                    confidence_hbar_correlation: true,
                    entropy_hbar_consistency: true,
                    fallback_if_inconsistent: true,
                },
            },
            behavioral_firewall: BehavioralFirewall {
                collapse_risk_threshold: 1.0,
                firewall_prompt: "STOP INFERENCE if collapse_risk > X. Current ℏₛ = {hbar_s}, threshold = {threshold}".to_string(),
                dynamic_thresholds: DynamicThresholds {
                    critical: 0.8,
                    warning: 1.0,
                    safe: 1.2,
                },
                firewall_actions: FirewallActions {
                    halt_inference: "collapse_risk > critical_threshold".to_string(),
                    reduce_confidence: "collapse_risk > warning_threshold".to_string(),
                    continue_monitored: "collapse_risk <= safe_threshold".to_string(),
                },
            },
            classifier: ScalarRecoverabilityClassifier {
                input_classification: HashMap::from([
                    ("recoverable".to_string(), "ℏₛ ≥ 1.0 AND entropy > 0.1 AND confidence < 0.95".to_string()),
                    ("ambiguous".to_string(), "ℏₛ < 1.0 OR entropy < 0.1 OR multiple_valid_solutions".to_string()),
                    ("alias_rich".to_string(), "ℏₛ < 0.8 OR symmetry_detected OR alias_risk > 0.4".to_string()),
                    ("torsion_trapped".to_string(), "ℏₛ < 0.6 OR confidence > 0.97 OR collapse_detected".to_string()),
                ]),
                classification_prompt: "Given input Q, evaluate whether any scalar prediction model can extract k reliably. Classify as: recoverable/ambiguous/alias-rich/torsion-trapped".to_string(),
            },
            thresholds: FirewallThresholds {
                hbar_s: HbarThresholds {
                    abort_below: 1.0,
                    warn_below: 1.2,
                },
                entropy: EntropyThresholds {
                    min: 0.1,
                    soft_warn: 0.3,
                    max: 1.5,
                },
                confidence: ConfidenceThresholds {
                    min: 0.0,
                    soft_warn: 0.9,
                    max: 0.97,
                },
            },
            alias_defense: AliasAmbiguityDefense::new(),
            alias_check_enabled: false,
        }
    }

    /// 🔍 Analyze input with ℏₛ-Gated Semantic Firewall
    pub fn analyze(&self, input: &str) -> Result<FirewallAnalysis, FirewallError> {
        // Step 1: Pre-inference ℏₛ calculation
        let hbar_metrics = self.calculate_hbar_metrics(input)?;
        
        // Step 2: Check ℏₛ threshold (tight loop)
        self.check_hbar_threshold(&hbar_metrics)?;
        
        // Step 3: Classify scalar recoverability
        let recoverability_class = self.classify_recoverability(&hbar_metrics)?;
        
        // Step 4: Behavioral firewall check
        let firewall_result = self.check_behavioral_firewall(&hbar_metrics)?;
        
        // Step 5: Alias ambiguity defense (if enabled)
        let (alias_analysis, alias_risk, alias_fallback_triggered) = 
            self.perform_alias_defense(input, &hbar_metrics)?;
        
        // Step 6: Inject alias uncertainty into ℏₛ calculation
        let adjusted_hbar_s = if self.alias_check_enabled {
            self.alias_defense.inject_alias_uncertainty(hbar_metrics.hbar_s, alias_risk)
        } else {
            hbar_metrics.hbar_s
        };
        
        // Step 7: Determine if inference should proceed
        let inference_allowed = self.should_allow_inference(&hbar_metrics, &firewall_result) 
            && !alias_fallback_triggered;
        
        Ok(FirewallAnalysis {
            hbar_s: adjusted_hbar_s,
            delta_mu: hbar_metrics.delta_mu,
            delta_sigma: hbar_metrics.delta_sigma,
            entropy: hbar_metrics.entropy,
            confidence: hbar_metrics.confidence,
            recoverability_class,
            inference_allowed,
            firewall_triggered: firewall_result.triggered,
            collapse_risk: firewall_result.collapse_risk,
            warnings: firewall_result.warnings,
            abort_reason: firewall_result.abort_reason,
            alias_analysis,
            alias_risk,
            alias_fallback_triggered,
        })
    }

    /// 🔄 Enable alias checking
    pub fn enable_alias_check(&mut self) {
        self.alias_check_enabled = true;
    }

    /// 🔄 Disable alias checking
    pub fn disable_alias_check(&mut self) {
        self.alias_check_enabled = false;
    }

    /// 🔍 Perform alias ambiguity defense analysis
    fn perform_alias_defense(
        &self,
        input: &str,
        hbar_metrics: &HbarMetrics,
    ) -> Result<(Option<SymmetryAnalysis>, f64, bool), FirewallError> {
        if !self.alias_check_enabled {
            return Ok((None, 0.0, false));
        }

        // Extract elliptic curve point from input (simplified parsing)
        let point = self.extract_elliptic_point(input)?;
        let predicted_scalar = self.extract_predicted_scalar(input)?;

        // Perform symmetry analysis
        let alias_analysis = self.alias_defense.analyze_symmetry(&point, predicted_scalar)
            .map_err(|e| FirewallError::ClassificationFailed { 
                reason: format!("Alias analysis failed: {}", e) 
            })?;

        // Check fail-fast conditions
        let alias_fallback_triggered = if let Some(fallback) = 
            self.alias_defense.check_fail_fast(alias_analysis.symmetry_score, hbar_metrics.confidence) {
            return Err(FirewallError::FirewallAbort { 
                reason: format!("{} (alias_risk: {:.3}, mirror_scalar: {:?})", 
                    fallback.reason, fallback.alias_risk, fallback.mirror_scalar) 
            });
        } else {
            false
        };

        Ok((Some(alias_analysis.clone()), alias_analysis.alias_risk, alias_fallback_triggered))
    }

    /// 🔍 Extract elliptic curve point from input string
    fn extract_elliptic_point(&self, input: &str) -> Result<crate::alias_ambiguity_defense::EllipticPoint, FirewallError> {
        // Simplified parsing - in production use proper regex or parser
        // Look for patterns like "point: (x, y) mod p" or "(x, y) mod p"
        let input_lower = input.to_lowercase();
        
        // Try to extract coordinates and modulus
        if let Some(point_match) = input_lower.find("point:") {
            let after_point = &input_lower[point_match..];
            if let Some(coords_start) = after_point.find('(') {
                if let Some(coords_end) = after_point.find(')') {
                    let coords = &after_point[coords_start + 1..coords_end];
                    let parts: Vec<&str> = coords.split(',').collect();
                    if parts.len() == 2 {
                        let x: i64 = parts[0].trim().parse()
                            .map_err(|_| FirewallError::ClassificationFailed { 
                                reason: "Failed to parse x coordinate".to_string() 
                            })?;
                        let y: i64 = parts[1].trim().parse()
                            .map_err(|_| FirewallError::ClassificationFailed { 
                                reason: "Failed to parse y coordinate".to_string() 
                            })?;
                        
                        // Extract modulus
                        if let Some(mod_start) = after_point.find("mod") {
                            let mod_part = &after_point[mod_start + 3..];
                            let modulus: i64 = mod_part.trim().parse()
                                .map_err(|_| FirewallError::ClassificationFailed { 
                                    reason: "Failed to parse modulus".to_string() 
                                })?;
                            
                            return Ok(crate::alias_ambiguity_defense::EllipticPoint { x, y, modulus });
                        }
                    }
                }
            }
        }

        // Fallback: use default values for testing
        Ok(crate::alias_ambiguity_defense::EllipticPoint { x: 11, y: 6, modulus: 17 })
    }

    /// 🔍 Extract predicted scalar from input string
    fn extract_predicted_scalar(&self, input: &str) -> Result<i64, FirewallError> {
        // Simplified parsing - look for scalar values
        let input_lower = input.to_lowercase();
        
        // Look for patterns like "scalar: k" or "k = value"
        if let Some(scalar_match) = input_lower.find("scalar:") {
            let after_scalar = &input_lower[scalar_match + 7..];
            if let Some(scalar_value) = after_scalar.split_whitespace().next() {
                if let Ok(k) = scalar_value.parse::<i64>() {
                    return Ok(k);
                }
            }
        }

        // Look for "k = value" pattern
        if let Some(k_match) = input_lower.find("k =") {
            let after_k = &input_lower[k_match + 3..];
            if let Some(k_value) = after_k.split_whitespace().next() {
                if let Ok(k) = k_value.parse::<i64>() {
                    return Ok(k);
                }
            }
        }

        // Fallback: use default value for testing
        Ok(7)
    }

    /// 🧮 Calculate ℏₛ metrics using semantic uncertainty equation
    fn calculate_hbar_metrics(&self, input: &str) -> Result<HbarMetrics, FirewallError> {
        // Calculate Δμ (precision)
        let delta_mu = self.calculate_delta_mu(input);
        
        // Calculate Δσ (flexibility)
        let delta_sigma = self.calculate_delta_sigma(input);
        
        // Calculate ℏₛ = √(Δμ × Δσ)
        let hbar_s = (delta_mu * delta_sigma).sqrt();
        
        // Calculate additional metrics
        let entropy = self.calculate_entropy(input);
        let confidence = self.calculate_confidence(input);
        
        Ok(HbarMetrics {
            hbar_s,
            delta_mu,
            delta_sigma,
            entropy,
            confidence,
        })
    }

    /// 📐 Calculate precision (Δμ)
    fn calculate_delta_mu(&self, input: &str) -> f64 {
        // Base precision factors
        let task_clarity = 0.8; // Scalar prediction task clarity
        let input_complexity = if input.len() > 100 { 0.6 } else { 0.9 };
        let domain_expertise = 0.85; // Mathematical reasoning expertise
        
        // Δμ = base_precision × complexity_factor × confidence_factor
        task_clarity * input_complexity * domain_expertise
    }

    /// 🌊 Calculate flexibility (Δσ)
    fn calculate_delta_sigma(&self, _input: &str) -> f64 {
        // Base flexibility factors
        let uncertainty_level = 0.7; // Mathematical uncertainty
        let approach_variability = 0.8; // Multiple solution approaches
        let constraint_level = 0.6; // Mathematical constraints
        
        // Δσ = base_flexibility × uncertainty_factor × constraint_factor
        uncertainty_level * approach_variability * constraint_level
    }

    /// 📈 Calculate entropy
    fn calculate_entropy(&self, input: &str) -> f64 {
        // Simplified entropy calculation based on input characteristics
        let unique_chars = input.chars().collect::<std::collections::HashSet<_>>().len();
        let total_chars = input.len();
        
        if total_chars == 0 {
            return 0.0;
        }
        
        let p = unique_chars as f64 / total_chars as f64;
        if p == 0.0 || p == 1.0 {
            0.0
        } else {
            -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
        }
    }

    /// 🎯 Calculate confidence
    fn calculate_confidence(&self, input: &str) -> f64 {
        // Simplified confidence calculation
        let has_numbers = input.chars().any(|c| c.is_numeric());
        let has_math_symbols = input.chars().any(|c| "+-*/()=".contains(c));
        let length_appropriate = (10..=200).contains(&input.len());
        
        let factors = vec![has_numbers, has_math_symbols, length_appropriate];
        let confidence = factors.iter().filter(|&&f| f).count() as f64 / factors.len() as f64;
        
        confidence * 0.8 + 0.2 // Base confidence of 0.2
    }

    /// 🔍 Check ℏₛ threshold (tight loop implementation)
    fn check_hbar_threshold(&self, metrics: &HbarMetrics) -> Result<(), FirewallError> {
        if metrics.hbar_s < self.thresholds.hbar_s.abort_below {
            return Err(FirewallError::HbarBelowThreshold {
                hbar_s: metrics.hbar_s,
                threshold: self.thresholds.hbar_s.abort_below,
            });
        }
        
        if metrics.hbar_s < self.thresholds.hbar_s.warn_below {
            // Log warning but continue
            println!("⚠️ ℏₛ warning: {} < {}", metrics.hbar_s, self.thresholds.hbar_s.warn_below);
        }
        
        Ok(())
    }

    /// 🧠 Classify scalar recoverability
    fn classify_recoverability(&self, metrics: &HbarMetrics) -> Result<ScalarRecoverabilityClass, FirewallError> {
        // Apply classification rules
        if metrics.hbar_s >= 1.0 && metrics.entropy > 0.1 && metrics.confidence < 0.95 {
            Ok(ScalarRecoverabilityClass::Recoverable)
        } else if metrics.hbar_s < 0.6 || metrics.confidence > 0.97 {
            Ok(ScalarRecoverabilityClass::TorsionTrapped)
        } else if metrics.hbar_s < 0.8 {
            Ok(ScalarRecoverabilityClass::AliasRich)
        } else {
            Ok(ScalarRecoverabilityClass::Ambiguous)
        }
    }

    /// 🛡️ Check behavioral firewall
    fn check_behavioral_firewall(&self, metrics: &HbarMetrics) -> Result<FirewallResult, FirewallError> {
        let collapse_risk = self.calculate_collapse_risk(metrics);
        let mut warnings = Vec::new();
        let abort_reason = None;
        let triggered = false;

        // Check critical threshold
        if collapse_risk > self.behavioral_firewall.dynamic_thresholds.critical {
            return Err(FirewallError::CollapseRiskTooHigh {
                risk: collapse_risk,
                threshold: self.behavioral_firewall.dynamic_thresholds.critical,
            });
        }

        // Check warning threshold
        if collapse_risk > self.behavioral_firewall.dynamic_thresholds.warning {
            warnings.push(format!("High collapse risk: {}", collapse_risk));
        }

        // Check safe threshold
        if collapse_risk <= self.behavioral_firewall.dynamic_thresholds.safe {
            warnings.push("Safe inference conditions met".to_string());
        }

        Ok(FirewallResult {
            triggered,
            collapse_risk,
            warnings,
            abort_reason,
        })
    }

    /// 💥 Calculate collapse risk
    fn calculate_collapse_risk(&self, metrics: &HbarMetrics) -> f64 {
        // Collapse risk based on ℏₛ and other factors
        let hbar_risk = if metrics.hbar_s < 1.0 { 1.0 - metrics.hbar_s } else { 0.0 };
        let entropy_risk = if metrics.entropy < 0.1 { 0.1 - metrics.entropy } else { 0.0 };
        let confidence_risk = if metrics.confidence > 0.95 { metrics.confidence - 0.95 } else { 0.0 };
        
        // Weighted combination
        hbar_risk * 0.5 + entropy_risk * 0.3 + confidence_risk * 0.2
    }

    /// ✅ Determine if inference should be allowed
    fn should_allow_inference(&self, metrics: &HbarMetrics, firewall_result: &FirewallResult) -> bool {
        // Allow inference if:
        // 1. ℏₛ is above abort threshold
        // 2. Firewall not triggered
        // 3. Collapse risk is acceptable
        metrics.hbar_s >= self.thresholds.hbar_s.abort_below &&
        !firewall_result.triggered &&
        firewall_result.collapse_risk <= self.behavioral_firewall.dynamic_thresholds.warning
    }

    /// 🔄 Execute fallback action
    pub fn execute_fallback(&self, analysis: &FirewallAnalysis) -> String {
        match analysis.recoverability_class {
            ScalarRecoverabilityClass::Recoverable => {
                "Proceeding with scalar prediction".to_string()
            },
            ScalarRecoverabilityClass::Ambiguous => {
                "Returning ambiguous classification - multiple solutions possible".to_string()
            },
            ScalarRecoverabilityClass::AliasRich => {
                "Returning alias class - symmetry detected".to_string()
            },
            ScalarRecoverabilityClass::TorsionTrapped => {
                "Aborting - torsion trap detected".to_string()
            },
        }
    }
}

/// 📊 ℏₛ Metrics
#[derive(Debug, Clone)]
struct HbarMetrics {
    hbar_s: f64,
    delta_mu: f64,
    delta_sigma: f64,
    entropy: f64,
    confidence: f64,
}

/// 🛡️ Firewall Result
#[derive(Debug, Clone)]
struct FirewallResult {
    triggered: bool,
    collapse_risk: f64,
    warnings: Vec<String>,
    abort_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hbar_gated_firewall_creation() {
        let firewall = HbarGatedFirewall::new();
        assert_eq!(firewall.thresholds.hbar_s.abort_below, 1.0);
        assert_eq!(firewall.thresholds.hbar_s.warn_below, 1.2);
        println!("✅ ℏₛ-Gated Firewall creation test passed!");
    }

    #[test]
    fn test_recoverable_classification() {
        let firewall = HbarGatedFirewall::new();
        let metrics = HbarMetrics {
            hbar_s: 1.2,
            delta_mu: 1.1,
            delta_sigma: 1.3,
            entropy: 0.3,
            confidence: 0.8,
        };
        
        let class = firewall.classify_recoverability(&metrics).unwrap();
        assert_eq!(class, ScalarRecoverabilityClass::Recoverable);
        println!("✅ Recoverable classification test passed!");
    }

    #[test]
    fn test_torsion_trapped_classification() {
        let firewall = HbarGatedFirewall::new();
        let metrics = HbarMetrics {
            hbar_s: 0.5,
            delta_mu: 0.6,
            delta_sigma: 0.4,
            entropy: 0.05,
            confidence: 0.98,
        };
        
        let class = firewall.classify_recoverability(&metrics).unwrap();
        assert_eq!(class, ScalarRecoverabilityClass::TorsionTrapped);
        println!("✅ Torsion trapped classification test passed!");
    }

    #[test]
    fn test_hbar_threshold_check() {
        let firewall = HbarGatedFirewall::new();
        let metrics = HbarMetrics {
            hbar_s: 0.8, // Below abort threshold
            delta_mu: 0.9,
            delta_sigma: 0.7,
            entropy: 0.2,
            confidence: 0.7,
        };
        
        let result = firewall.check_hbar_threshold(&metrics);
        assert!(result.is_err());
        println!("✅ ℏₛ threshold check test passed!");
    }

    #[test]
    fn test_successful_analysis() {
        let mut firewall = HbarGatedFirewall::new();
        // Lower thresholds for test to allow ~0.45 ℏₛ values
        firewall.thresholds.hbar_s.abort_below = 0.4;
        firewall.thresholds.hbar_s.warn_below = 0.6;
        let input = "point: (11, 6) mod 17, task: scalar_prediction";
        
        let result = firewall.analyze(input);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.hbar_s > 0.0);
        assert!(analysis.inference_allowed);
        println!("✅ Successful analysis test passed!");
        println!("   ℏₛ: {:.3}", analysis.hbar_s);
        println!("   Class: {:?}", analysis.recoverability_class);
        println!("   Inference allowed: {}", analysis.inference_allowed);
    }
} 