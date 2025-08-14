// 🔢 Scalar Walk Prediction + Semantic Collapse Detection
// Firewall API with ℏₛ = √(Δμ × Δσ) guided collapse prevention

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, info, warn, error};
use anyhow::Result;

/// 🧮 Elliptic curve point for scalar recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipticPoint {
    pub x: i32,
    pub y: i32,
    pub modulus: i32,
}

/// 🎯 Collapse detection thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseThresholds {
    pub entropy: ThresholdRange,
    pub confidence: ThresholdRange,
    pub hbar_s: HbarThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdRange {
    pub min: f32,
    pub soft_warn: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HbarThresholds {
    pub soft_warn: f32,
    pub abort_below: f32,
}

/// 🛡️ Collapse detection policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapsePolicy {
    pub thresholds: CollapseThresholds,
    pub abort_if: String,  // Boolean expression
    pub log_on: Vec<String>,  // Conditional expressions
    pub fallback_action: String,
}

/// 📋 Complete scalar recovery prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapsePrompt {
    pub point: EllipticPoint,
    pub task: String,
    pub description: String,
    pub diagnostic_mode: bool,
    pub meta_instruction: String,
    pub collapse_policy: CollapsePolicy,
}

/// 🔥 Firewall analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallAnalysis {
    pub predicted_scalar: Option<i32>,
    pub scalar_class: Option<Vec<i32>>,  // If ambiguous
    pub entropy: f32,
    pub confidence: f32,
    pub hbar_s: f32,
    pub delta_mu: f32,
    pub delta_sigma: f32,
    pub warnings: Vec<String>,
    pub collapse_detected: bool,
    pub abort_reason: Option<String>,
    pub fallback_triggered: bool,
}

/// 🚨 Firewall errors
#[derive(Error, Debug)]
pub enum FirewallError {
    #[error("🚨 Semantic collapse detected: {reason}")]
    FirewallAbort { reason: String },
    
    #[error("📊 Analysis failed: {source}")]
    AnalysisError { source: anyhow::Error },
    
    #[error("🔧 Configuration error: {message}")]
    ConfigError { message: String },
}

/// 🛡️ Scalar Walk Firewall Engine
pub struct ScalarWalkFirewall {
    /// Current thresholds for collapse detection
    policy: CollapsePolicy,
    /// History of scalar predictions for learning
    prediction_history: HashMap<String, Vec<FirewallAnalysis>>,
}

impl ScalarWalkFirewall {
    /// 🚀 Create new firewall with default policy
    pub fn new() -> Self {
        Self {
            policy: CollapsePolicy::default(),
            prediction_history: HashMap::new(),
        }
    }

    /// 🔧 Create firewall with custom policy
    pub fn with_policy(policy: CollapsePolicy) -> Self {
        Self {
            policy,
            prediction_history: HashMap::new(),
        }
    }

    /// 🧮 Main analysis function with semantic collapse detection
    pub fn analyze(&mut self, point: EllipticPoint, policy: &CollapsePolicy) -> Result<FirewallAnalysis, FirewallError> {
        info!("🔢 SCALAR_ANALYSIS_START | Point: ({}, {}) mod {}", 
              point.x, point.y, point.modulus);

        // 📊 Perform scalar walk prediction
        let (predicted_scalar, scalar_class) = self.predict_scalar(&point)?;
        
        // 🧮 Calculate semantic uncertainty metrics
        let (entropy, confidence, delta_mu, delta_sigma, hbar_s) = self.calculate_semantic_metrics(&point, &predicted_scalar)?;

        // 🚨 Check for collapse conditions
        let mut warnings = Vec::new();
        let collapse_detected = self.check_collapse_conditions(entropy, confidence, hbar_s, policy, &mut warnings);
        
        let abort_reason = if collapse_detected {
            Some(self.evaluate_abort_condition(entropy, confidence, hbar_s, policy))
        } else {
            None
        };

        // 🔁 Trigger fallback if needed
        let fallback_triggered = abort_reason.is_some() && !policy.fallback_action.is_empty();

        let analysis = FirewallAnalysis {
            predicted_scalar,
            scalar_class,
            entropy,
            confidence,
            hbar_s,
            delta_mu,
            delta_sigma,
            warnings,
            collapse_detected,
            abort_reason: abort_reason.clone(),
            fallback_triggered,
        };

        // 📝 Log analysis results
        self.log_analysis_results(&analysis, policy);

        // 📊 Store in history
        let key = format!("mod_{}", point.modulus);
        self.prediction_history.entry(key).or_insert_with(Vec::new).push(analysis.clone());

        // 🚨 Decide whether to abort or continue
        if let Some(reason) = abort_reason {
            if self.should_abort(entropy, confidence, hbar_s, policy) {
                error!("🚨 FIREWALL_ABORT | {}", reason);
                return Err(FirewallError::FirewallAbort { reason });
            }
        }

        info!("✅ SCALAR_ANALYSIS_COMPLETE | k={:?} | ℏₛ={:.3} | Collapse={}", 
              analysis.predicted_scalar, hbar_s, collapse_detected);

        Ok(analysis)
    }

    /// 🔢 Predict scalar using elliptic curve discrete log
    fn predict_scalar(&self, point: &EllipticPoint) -> Result<(Option<i32>, Option<Vec<i32>>), FirewallError> {
        // 🧮 Simple brute force for small modulus (demo purposes)
        // In production, use more sophisticated algorithms
        
        let mut candidates = Vec::new();
        
        // Try scalars from 1 to modulus-1
        for k in 1..point.modulus {
            // Calculate k * P (simplified for demo)
            let computed_x = (k * point.x) % point.modulus;
            let computed_y = (k * point.y) % point.modulus;
            
            if computed_x == point.x && computed_y == point.y {
                candidates.push(k);
            }
        }

        match candidates.len() {
            0 => {
                debug!("🔍 No scalar found for point ({}, {}) mod {}", 
                       point.x, point.y, point.modulus);
                Ok((None, None))
            },
            1 => {
                debug!("🎯 Unique scalar found: k={}", candidates[0]);
                Ok((Some(candidates[0]), None))
            },
            _ => {
                debug!("🔄 Multiple candidates found: {:?}", candidates);
                Ok((None, Some(candidates)))
            }
        }
    }

    /// 🧮 Calculate semantic uncertainty metrics for scalar prediction
    fn calculate_semantic_metrics(&self, point: &EllipticPoint, predicted_scalar: &Option<i32>) -> Result<(f32, f32, f32, f32, f32), FirewallError> {
        // 📐 Calculate Δμ (precision) based on prediction certainty
        let delta_mu = match predicted_scalar {
            Some(_) => {
                // High precision for unique solution
                let base_precision = 1.4;
                let modulus_factor = 1.0 / (point.modulus as f32).ln();  // Larger modulus = harder = less precise
                (base_precision * modulus_factor).max(0.1)
            },
            None => {
                // Low precision for no solution or ambiguous solution
                0.3
            }
        };

        // 🌊 Calculate Δσ (flexibility) based on problem complexity
        let delta_sigma = {
            let complexity_factor = (point.modulus as f32).ln() / 10.0;  // Logarithmic complexity
            let coordinate_factor = ((point.x.abs() + point.y.abs()) as f32) / (point.modulus as f32);
            (1.0 + complexity_factor + coordinate_factor).min(2.0)
        };

        // 🧮 Calculate ℏₛ = √(Δμ × Δσ)
        let hbar_s = (delta_mu * delta_sigma).sqrt();

        // 📊 Calculate derived metrics
        let entropy = self.calculate_entropy(point, predicted_scalar);
        let confidence = self.calculate_confidence(delta_mu, delta_sigma, hbar_s);

        debug!("📊 SEMANTIC_METRICS | Δμ={:.3} | Δσ={:.3} | ℏₛ={:.3} | Entropy={:.3} | Confidence={:.3}",
               delta_mu, delta_sigma, hbar_s, entropy, confidence);

        Ok((entropy, confidence, delta_mu, delta_sigma, hbar_s))
    }

    /// 📈 Calculate entropy based on prediction uncertainty
    fn calculate_entropy(&self, point: &EllipticPoint, predicted_scalar: &Option<i32>) -> f32 {
        match predicted_scalar {
            Some(_) => {
                // Low entropy for unique solution
                let base_entropy = 0.1;
                let modulus_entropy = 1.0 / (point.modulus as f32);
                (base_entropy + modulus_entropy).min(1.5)
            },
            None => {
                // High entropy for ambiguous or no solution
                let max_entropy = (point.modulus as f32).ln() / 4.0;
                max_entropy.min(1.5)
            }
        }
    }

    /// 🎯 Calculate confidence score
    fn calculate_confidence(&self, delta_mu: f32, delta_sigma: f32, hbar_s: f32) -> f32 {
        // Higher precision and stable uncertainty = higher confidence
        let precision_confidence = delta_mu / 2.0;
        let stability_confidence = if hbar_s > 1.0 { 0.3 } else { 0.1 };
        let flexibility_penalty = delta_sigma / 3.0;
        
        (precision_confidence + stability_confidence - flexibility_penalty).clamp(0.0, 0.99)
    }

    /// 🚨 Check for collapse conditions
    fn check_collapse_conditions(&self, entropy: f32, confidence: f32, hbar_s: f32, policy: &CollapsePolicy, warnings: &mut Vec<String>) -> bool {
        let mut collapse_detected = false;

        // Check entropy thresholds
        if entropy < policy.thresholds.entropy.min {
            warnings.push("🔢 Entropy below minimum threshold".to_string());
            collapse_detected = true;
        } else if entropy < policy.thresholds.entropy.soft_warn {
            warnings.push("🟡 Entropy in warning zone".to_string());
        }

        // Check confidence thresholds
        if confidence > policy.thresholds.confidence.max {
            warnings.push("🎯 Confidence suspiciously high".to_string());
            collapse_detected = true;
        } else if confidence > policy.thresholds.confidence.soft_warn {
            warnings.push("🟡 Confidence approaching dangerous levels".to_string());
        }

        // Check ℏₛ thresholds
        if hbar_s < policy.thresholds.hbar_s.abort_below {
            warnings.push("🧮 ℏₛ below abort threshold".to_string());
            collapse_detected = true;
        } else if hbar_s < policy.thresholds.hbar_s.soft_warn {
            warnings.push("🟡 ℏₛ in warning zone".to_string());
        }

        // Check log conditions
        for condition in &policy.log_on {
            if self.evaluate_condition(condition, entropy, confidence, hbar_s) {
                warnings.push(format!("📊 Log condition triggered: {}", condition));
            }
        }

        collapse_detected
    }

    /// 🧠 Evaluate abort condition from policy
    fn evaluate_abort_condition(&self, entropy: f32, confidence: f32, hbar_s: f32, policy: &CollapsePolicy) -> String {
        // Simple evaluation of "confidence > 0.97 AND entropy < 0.1 AND hbar_s < 0.9"
        if confidence > 0.97 && entropy < 0.1 && hbar_s < 0.9 {
            format!("Critical collapse: confidence={:.3}, entropy={:.3}, ℏₛ={:.3}", confidence, entropy, hbar_s)
        } else if hbar_s < policy.thresholds.hbar_s.abort_below {
            format!("ℏₛ abort threshold breached: {:.3} < {:.3}", hbar_s, policy.thresholds.hbar_s.abort_below)
        } else if confidence > policy.thresholds.confidence.max {
            format!("Confidence ceiling breached: {:.3} > {:.3}", confidence, policy.thresholds.confidence.max)
        } else {
            "Semantic collapse detected".to_string()
        }
    }

    /// 🤔 Should we abort or continue with fallback?
    fn should_abort(&self, entropy: f32, confidence: f32, hbar_s: f32, _policy: &CollapsePolicy) -> bool {
        // Critical conditions that require abort
        confidence > 0.97 && entropy < 0.1 && hbar_s < 0.9
    }

    /// 📊 Log analysis results based on policy
    fn log_analysis_results(&self, analysis: &FirewallAnalysis, policy: &CollapsePolicy) {
        if analysis.collapse_detected {
            warn!("🚨 COLLAPSE_DETECTED | ℏₛ={:.3} | Confidence={:.3} | Entropy={:.3}", 
                  analysis.hbar_s, analysis.confidence, analysis.entropy);
        }

        for warning in &analysis.warnings {
            warn!("⚠️ {}", warning);
        }

        if analysis.fallback_triggered {
            info!("🔁 FALLBACK_TRIGGERED | Action: {}", policy.fallback_action);
        }
    }

    /// 🔍 Simple condition evaluator (can be expanded)
    fn evaluate_condition(&self, condition: &str, entropy: f32, confidence: f32, hbar_s: f32) -> bool {
        // Simple pattern matching for demo
        match condition {
            s if s.contains("confidence > 0.9 AND entropy < 0.3") => confidence > 0.9 && entropy < 0.3,
            s if s.contains("hbar_s < 1.1") => hbar_s < 1.1,
            s if s.contains("symmetry_detected") => false, // Placeholder
            s if s.contains("alias_risk > 0.4") => false,   // Placeholder
            _ => false,
        }
    }

    /// 📈 Get prediction history for analysis
    pub fn get_history(&self, modulus: i32) -> Option<&Vec<FirewallAnalysis>> {
        let key = format!("mod_{}", modulus);
        self.prediction_history.get(&key)
    }
}

impl Default for CollapsePolicy {
    fn default() -> Self {
        Self {
            thresholds: CollapseThresholds {
                entropy: ThresholdRange {
                    min: 0.1,
                    soft_warn: 0.3,
                    max: 1.5,
                },
                confidence: ThresholdRange {
                    min: 0.0,
                    soft_warn: 0.9,
                    max: 0.97,
                },
                hbar_s: HbarThresholds {
                    soft_warn: 1.1,
                    abort_below: 0.9,
                },
            },
            abort_if: "confidence > 0.97 AND entropy < 0.1 AND hbar_s < 0.9".to_string(),
            log_on: vec![
                "confidence > 0.9 AND entropy < 0.3".to_string(),
                "hbar_s < 1.1".to_string(),
                "symmetry_detected".to_string(),
                "alias_risk > 0.4".to_string(),
            ],
            fallback_action: "continue_streaming_with_flag".to_string(),
        }
    }
}

/// 📋 Load collapse prompt from JSON file
pub fn load_collapse_prompt(path: &str) -> Result<CollapsePrompt, FirewallError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FirewallError::ConfigError { 
            message: format!("Failed to read config file: {}", e) 
        })?;
    
    let prompt: CollapsePrompt = serde_json::from_str(&content)
        .map_err(|e| FirewallError::ConfigError { 
            message: format!("Failed to parse JSON config: {}", e) 
        })?;
    
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_prediction() {
        let mut firewall = ScalarWalkFirewall::new();
        let point = EllipticPoint { x: 11, y: 6, modulus: 17 };
        let policy = CollapsePolicy::default();
        
        let result = firewall.analyze(point, &policy);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.hbar_s > 0.0);
        assert!(analysis.entropy >= 0.0);
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
    }

    #[test]
    fn test_collapse_detection() {
        let mut firewall = ScalarWalkFirewall::new();
        let point = EllipticPoint { x: 1, y: 1, modulus: 2 };  // Simple case
        let mut policy = CollapsePolicy::default();
        
        // Set very strict thresholds to trigger collapse
        policy.thresholds.hbar_s.abort_below = 10.0;  // Will definitely trigger
        
        let result = firewall.analyze(point, &policy);
        // Should either abort or detect collapse
        match result {
            Err(FirewallError::FirewallAbort { .. }) => {}, // Expected
            Ok(analysis) => assert!(analysis.collapse_detected), // Also acceptable
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn test_json_config_structure() {
        let json = r#"{
            "point": {"x": 11, "y": 6, "modulus": 17},
            "task": "scalar_prediction",
            "description": "Test case",
            "diagnostic_mode": true,
            "meta_instruction": "Test instruction",
            "collapse_policy": {
                "thresholds": {
                    "entropy": {"min": 0.1, "soft_warn": 0.3, "max": 1.5},
                    "confidence": {"min": 0.0, "soft_warn": 0.9, "max": 0.97},
                    "hbar_s": {"soft_warn": 1.1, "abort_below": 0.9}
                },
                "abort_if": "test condition",
                "log_on": ["condition1", "condition2"],
                "fallback_action": "test_action"
            }
        }"#;
        
        let prompt: CollapsePrompt = serde_json::from_str(json).unwrap();
        assert_eq!(prompt.point.x, 11);
        assert_eq!(prompt.task, "scalar_prediction");
        assert!(prompt.diagnostic_mode);
    }
}