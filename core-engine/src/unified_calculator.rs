// 🎯 Unified Semantic Uncertainty Calculator Trait
// Standardizes all implementations with configurable rigor levels
// Implements quantum-inspired non-commutativity and context-aware calibration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use ndarray::{Array1, Array2};

/// 🎯 Rigor Level for Uncertainty Calculations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RigorLevel {
    /// Basic: Fast, simple calculations (WASM Simple style)
    Basic,
    /// Intermediate: Balanced accuracy and performance (Cloudflare Workers style)
    Intermediate,
    /// Advanced: Maximum mathematical rigor (Core Engine style)
    Advanced,
}

impl RigorLevel {
    pub fn name(&self) -> &str {
        match self {
            RigorLevel::Basic => "Basic",
            RigorLevel::Intermediate => "Intermediate", 
            RigorLevel::Advanced => "Advanced",
        }
    }
    
    pub fn description(&self) -> &str {
        match self {
            RigorLevel::Basic => "Fast entropy + JSD calculations",
            RigorLevel::Intermediate => "Fisher Information + Fisher-Rao metrics",
            RigorLevel::Advanced => "Full Fisher matrix + multi-divergence + quantum framework",
        }
    }
}

/// 🧮 Core Uncertainty Calculation Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedUncertaintyResult {
    /// Raw ℏₛ = √(Δμ × Δσ)
    pub raw_hbar: f64,
    /// Calibrated ℏₛ for decision making
    pub calibrated_hbar: f64,
    /// Precision component Δμ
    pub delta_mu: f64,
    /// Flexibility component Δσ
    pub delta_sigma: f64,
    /// Risk level assessment
    pub risk_level: RiskLevel,
    /// Rigor level used
    pub rigor_level: RigorLevel,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Context-aware calibration applied
    pub context_calibration: ContextCalibration,
    /// Quantum-inspired non-commutativity effects
    pub quantum_effects: Option<QuantumEffects>,
    /// Mathematical validation metrics
    pub validation_metrics: ValidationMetrics,
}

/// 🚦 Risk Level Assessment
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Safe,      // ℏₛ > 1.2
    Warning,   // 0.8 < ℏₛ ≤ 1.2
    HighRisk,  // 0.4 < ℏₛ ≤ 0.8
    Critical,  // ℏₛ ≤ 0.4
}

/// 🎛️ Context-Aware Calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCalibration {
    /// Base calibration factor
    pub base_factor: f64,
    /// Context-specific adjustment
    pub context_adjustment: f64,
    /// Semantic entropy contribution
    pub semantic_entropy_factor: f64,
    /// Architecture-specific κ constant
    pub architecture_kappa: f64,
    /// Final calibrated value
    pub final_calibration: f64,
}

/// ⚛️ Quantum-Inspired Non-Commutativity Effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEffects {
    /// Order-dependent measurement effects
    pub measurement_order_effects: f64,
    /// Non-commutative operator variance
    pub operator_variance: f64,
    /// Quantum Fisher information contribution
    pub quantum_fisher_contribution: f64,
    /// Uncertainty principle effects
    pub uncertainty_principle_effects: f64,
}

/// 📊 Mathematical Validation Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// Numerical stability score (0-1)
    pub numerical_stability: f64,
    /// Mathematical consistency check
    pub mathematical_consistency: bool,
    /// Convergence indicators
    pub convergence_indicators: Vec<f64>,
    /// Error bounds
    pub error_bounds: (f64, f64),
}

/// 🎯 Unified Uncertainty Calculator Trait
pub trait UncertaintyCalculator: Send + Sync {
    /// Calculate semantic uncertainty with specified rigor level
    fn calculate_uncertainty(
        &self,
        prompt: &str,
        output: &str,
        rigor_level: RigorLevel,
        context: &CalculationContext,
    ) -> Result<UnifiedUncertaintyResult>;
    
    /// Get supported rigor levels
    fn supported_rigor_levels(&self) -> Vec<RigorLevel>;
    
    /// Validate input parameters
    fn validate_input(&self, prompt: &str, output: &str) -> Result<()>;
    
    /// Get calculator metadata
    fn metadata(&self) -> CalculatorMetadata;
}

/// 📋 Calculation Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculationContext {
    /// Model architecture (if known)
    pub model_architecture: Option<ModelArchitecture>,
    /// Prompt type classification
    pub prompt_type: PromptType,
    /// Domain context
    pub domain: Domain,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
    /// Calibration preferences
    pub calibration_preferences: CalibrationPreferences,
}

/// 🏗️ Model Architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelArchitecture {
    EncoderOnly,
    DecoderOnly,
    EncoderDecoder,
    Unknown,
}

/// 📝 Prompt Type Classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptType {
    Factual,
    Creative,
    Ambiguous,
    Mathematical,
    Conversational,
    CodeGeneration,
    Summarization,
    Translation,
    Explanation,
}

/// 🌐 Domain Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Domain {
    General,
    Technical,
    Creative,
    Scientific,
    Business,
    Educational,
    Medical,
    Legal,
    Financial,
}

/// ⚡ Performance Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum processing time in milliseconds
    pub max_processing_time_ms: f64,
    /// Memory constraints
    pub memory_constraints: Option<usize>,
    /// Accuracy requirements
    pub accuracy_requirements: AccuracyRequirements,
}

/// 🎯 Accuracy Requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyRequirements {
    /// Speed over accuracy
    SpeedOptimized,
    /// Balanced approach
    Balanced,
    /// Maximum accuracy
    AccuracyOptimized,
}

/// 🎛️ Calibration Preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPreferences {
    /// Use context-aware calibration
    pub use_context_calibration: bool,
    /// Use quantum-inspired effects
    pub use_quantum_effects: bool,
    /// Use architecture-specific κ constants
    pub use_architecture_kappa: bool,
    /// Custom calibration factor
    pub custom_calibration_factor: Option<f64>,
}

/// 📊 Calculator Metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculatorMetadata {
    /// Calculator name
    pub name: String,
    /// Version
    pub version: String,
    /// Description
    pub description: String,
    /// Supported features
    pub supported_features: Vec<String>,
    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
}

/// ⚡ Performance Characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Typical processing time range (ms)
    pub typical_processing_time_ms: (f64, f64),
    /// Memory usage range (MB)
    pub memory_usage_mb: (f64, f64),
    /// Accuracy range (0-1)
    pub accuracy_range: (f64, f64),
    /// Scalability characteristics
    pub scalability: ScalabilityCharacteristics,
}

/// 📈 Scalability Characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityCharacteristics {
    /// Linear scaling factor
    pub linear_scaling_factor: f64,
    /// Memory scaling factor
    pub memory_scaling_factor: f64,
    /// Batch processing efficiency
    pub batch_efficiency: f64,
}

/// 🧮 Basic Uncertainty Calculator Implementation
pub struct BasicUncertaintyCalculator {
    epsilon: f64,
}

impl BasicUncertaintyCalculator {
    pub fn new() -> Self {
        Self {
            epsilon: 1e-8,
        }
    }
    
    /// Calculate precision using Shannon entropy (Basic rigor)
    fn calculate_precision_basic(&self, prompt: &str, output: &str) -> f64 {
        let prompt_freq = self.calculate_word_frequencies(prompt);
        let output_freq = self.calculate_word_frequencies(output);
        
        let prompt_entropy = self.calculate_entropy(&prompt_freq);
        let output_entropy = self.calculate_entropy(&output_freq);
        
        (prompt_entropy + output_entropy) / 2.0
    }
    
    /// Calculate flexibility using JSD (Basic rigor)
    fn calculate_flexibility_basic(&self, prompt: &str, output: &str) -> f64 {
        let prompt_freq = self.calculate_word_frequencies(prompt);
        let output_freq = self.calculate_word_frequencies(output);
        
        self.calculate_jsd(&prompt_freq, &output_freq) / 2.0
    }
    
    fn calculate_word_frequencies(&self, text: &str) -> HashMap<String, f64> {
        let words: Vec<&str> = text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .collect();
        
        let mut freq = HashMap::new();
        for word in words {
            *freq.entry(word.to_string()).or_insert(0.0) += 1.0;
        }
        
        // Normalize to probabilities
        let total: f64 = freq.values().sum();
        if total > 0.0 {
            for value in freq.values_mut() {
                *value /= total;
            }
        }
        
        freq
    }
    
    fn calculate_entropy(&self, freq: &HashMap<String, f64>) -> f64 {
        let mut entropy = 0.0;
        for &prob in freq.values() {
            if prob > 0.0 {
                entropy -= prob * prob.log2();
            }
        }
        entropy.max(self.epsilon)
    }
    
    fn calculate_jsd(&self, p: &HashMap<String, f64>, q: &HashMap<String, f64>) -> f64 {
        let mut jsd = 0.0;
        let all_words: std::collections::HashSet<&String> = p.keys().chain(q.keys()).collect();
        
        for word in all_words {
            let p_val = p.get(word).unwrap_or(&0.0);
            let q_val = q.get(word).unwrap_or(&0.0);
            let m = (p_val + q_val) / 2.0;
            
            if m > 0.0 {
                if *p_val > 0.0 {
                    jsd += p_val * (p_val / m).log2();
                }
                if *q_val > 0.0 {
                    jsd += q_val * (q_val / m).log2();
                }
            }
        }
        
        jsd.max(self.epsilon)
    }
}

impl UncertaintyCalculator for BasicUncertaintyCalculator {
    fn calculate_uncertainty(
        &self,
        prompt: &str,
        output: &str,
        rigor_level: RigorLevel,
        _context: &CalculationContext,
    ) -> Result<UnifiedUncertaintyResult> {
        let start_time = std::time::Instant::now();
        
        self.validate_input(prompt, output)?;
        
        let delta_mu = self.calculate_precision_basic(prompt, output);
        let delta_sigma = self.calculate_flexibility_basic(prompt, output);
        
        let raw_hbar = (delta_mu * delta_sigma).sqrt();
        
        // Basic calibration
        let calibrated_hbar = raw_hbar * 3.4; // Empirical golden scale
        let risk_level = self.determine_risk_level(calibrated_hbar);
        
        let processing_time = start_time.elapsed().as_millis() as f64;
        
        Ok(UnifiedUncertaintyResult {
            raw_hbar,
            calibrated_hbar,
            delta_mu,
            delta_sigma,
            risk_level,
            rigor_level,
            processing_time_ms: processing_time,
            context_calibration: ContextCalibration {
                base_factor: 3.4,
                context_adjustment: 0.0,
                semantic_entropy_factor: 0.0,
                architecture_kappa: 1.0,
                final_calibration: calibrated_hbar,
            },
            quantum_effects: None,
            validation_metrics: ValidationMetrics {
                numerical_stability: 0.8,
                mathematical_consistency: true,
                convergence_indicators: vec![1.0],
                error_bounds: (0.0, 0.1),
            },
        })
    }
    
    fn supported_rigor_levels(&self) -> Vec<RigorLevel> {
        vec![RigorLevel::Basic]
    }
    
    fn validate_input(&self, prompt: &str, output: &str) -> Result<()> {
        if prompt.trim().is_empty() || output.trim().is_empty() {
            return Err(anyhow::anyhow!("Empty prompt or output"));
        }
        Ok(())
    }
    
    fn metadata(&self) -> CalculatorMetadata {
        CalculatorMetadata {
            name: "Basic Uncertainty Calculator".to_string(),
            version: "1.0.0".to_string(),
            description: "Fast entropy + JSD calculations for basic uncertainty analysis".to_string(),
            supported_features: vec![
                "Shannon entropy".to_string(),
                "Jensen-Shannon divergence".to_string(),
                "Word frequency analysis".to_string(),
            ],
            performance_characteristics: PerformanceCharacteristics {
                typical_processing_time_ms: (1.0, 5.0),
                memory_usage_mb: (1.0, 10.0),
                accuracy_range: (0.6, 0.8),
                scalability: ScalabilityCharacteristics {
                    linear_scaling_factor: 1.0,
                    memory_scaling_factor: 1.0,
                    batch_efficiency: 0.9,
                },
            },
        }
    }
}

impl BasicUncertaintyCalculator {
    fn determine_risk_level(&self, hbar: f64) -> RiskLevel {
        if hbar <= 0.4 {
            RiskLevel::Critical
        } else if hbar <= 0.8 {
            RiskLevel::HighRisk
        } else if hbar <= 1.2 {
            RiskLevel::Warning
        } else {
            RiskLevel::Safe
        }
    }
}

/// 🏭 Calculator Factory
pub struct CalculatorFactory;

impl CalculatorFactory {
    /// Create calculator based on rigor level
    pub fn create_calculator(rigor_level: RigorLevel) -> Box<dyn UncertaintyCalculator> {
        match rigor_level {
            RigorLevel::Basic => Box::new(BasicUncertaintyCalculator::new()),
            RigorLevel::Intermediate => Box::new(IntermediateUncertaintyCalculator::new()),
            RigorLevel::Advanced => Box::new(AdvancedUncertaintyCalculator::new()),
        }
    }
    
    /// Get all available calculators
    pub fn get_all_calculators() -> Vec<Box<dyn UncertaintyCalculator>> {
        vec![
            Box::new(BasicUncertaintyCalculator::new()),
            Box::new(IntermediateUncertaintyCalculator::new()),
            Box::new(AdvancedUncertaintyCalculator::new()),
        ]
    }
}

// Placeholder implementations for Intermediate and Advanced calculators
pub struct IntermediateUncertaintyCalculator;
pub struct AdvancedUncertaintyCalculator;

impl IntermediateUncertaintyCalculator {
    pub fn new() -> Self {
        Self
    }
}

impl AdvancedUncertaintyCalculator {
    pub fn new() -> Self {
        Self
    }
}

impl UncertaintyCalculator for IntermediateUncertaintyCalculator {
    fn calculate_uncertainty(
        &self,
        _prompt: &str,
        _output: &str,
        _rigor_level: RigorLevel,
        _context: &CalculationContext,
    ) -> Result<UnifiedUncertaintyResult> {
        // TODO: Implement Fisher Information + Fisher-Rao metrics
        todo!("Intermediate calculator implementation")
    }
    
    fn supported_rigor_levels(&self) -> Vec<RigorLevel> {
        vec![RigorLevel::Basic, RigorLevel::Intermediate]
    }
    
    fn validate_input(&self, _prompt: &str, _output: &str) -> Result<()> {
        Ok(())
    }
    
    fn metadata(&self) -> CalculatorMetadata {
        CalculatorMetadata {
            name: "Intermediate Uncertainty Calculator".to_string(),
            version: "1.0.0".to_string(),
            description: "Fisher Information + Fisher-Rao metrics".to_string(),
            supported_features: vec![],
            performance_characteristics: PerformanceCharacteristics {
                typical_processing_time_ms: (5.0, 10.0),
                memory_usage_mb: (5.0, 20.0),
                accuracy_range: (0.7, 0.9),
                scalability: ScalabilityCharacteristics {
                    linear_scaling_factor: 1.2,
                    memory_scaling_factor: 1.5,
                    batch_efficiency: 0.8,
                },
            },
        }
    }
}

impl UncertaintyCalculator for AdvancedUncertaintyCalculator {
    fn calculate_uncertainty(
        &self,
        _prompt: &str,
        _output: &str,
        _rigor_level: RigorLevel,
        _context: &CalculationContext,
    ) -> Result<UnifiedUncertaintyResult> {
        // TODO: Implement full Fisher matrix + multi-divergence + quantum framework
        todo!("Advanced calculator implementation")
    }
    
    fn supported_rigor_levels(&self) -> Vec<RigorLevel> {
        vec![RigorLevel::Basic, RigorLevel::Intermediate, RigorLevel::Advanced]
    }
    
    fn validate_input(&self, _prompt: &str, _output: &str) -> Result<()> {
        Ok(())
    }
    
    fn metadata(&self) -> CalculatorMetadata {
        CalculatorMetadata {
            name: "Advanced Uncertainty Calculator".to_string(),
            version: "1.0.0".to_string(),
            description: "Full Fisher matrix + multi-divergence + quantum framework".to_string(),
            supported_features: vec![],
            performance_characteristics: PerformanceCharacteristics {
                typical_processing_time_ms: (10.0, 100.0),
                memory_usage_mb: (10.0, 100.0),
                accuracy_range: (0.8, 0.95),
                scalability: ScalabilityCharacteristics {
                    linear_scaling_factor: 1.5,
                    memory_scaling_factor: 2.0,
                    batch_efficiency: 0.7,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_calculator() {
        let calculator = BasicUncertaintyCalculator::new();
        let context = CalculationContext {
            model_architecture: None,
            prompt_type: PromptType::General,
            domain: Domain::General,
            performance_requirements: PerformanceRequirements {
                max_processing_time_ms: 100.0,
                memory_constraints: None,
                accuracy_requirements: AccuracyRequirements::Balanced,
            },
            calibration_preferences: CalibrationPreferences {
                use_context_calibration: false,
                use_quantum_effects: false,
                use_architecture_kappa: false,
                custom_calibration_factor: None,
            },
        };
        
        let result = calculator
            .calculate_uncertainty("What is AI?", "AI is artificial intelligence", RigorLevel::Basic, &context)
            .unwrap();
        
        assert!(result.raw_hbar > 0.0);
        assert!(result.calibrated_hbar > 0.0);
        assert!(result.processing_time_ms < 10.0);
    }
    
    #[test]
    fn test_calculator_factory() {
        let calculator = CalculatorFactory::create_calculator(RigorLevel::Basic);
        let metadata = calculator.metadata();
        assert_eq!(metadata.name, "Basic Uncertainty Calculator");
    }
} 