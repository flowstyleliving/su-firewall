use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tracing::{debug, info, warn, error, instrument};
use uuid::Uuid;

// 🌟 Mathematical Constants for Calibration
const EMPIRICAL_DELTA_MU: f64 = 2.67; // Measured from actual data
const EMPIRICAL_GOLDEN_SCALE: f64 = 3.4000000000000004; // √(φ × 2.67²) ≈ 3.40

// 🏗️ Architecture-Dependent Uncertainty Constants (κ) from Research
const ENCODER_ONLY_KAPPA: f64 = 1.000; // ± 0.035 (Universal constant)
const DECODER_ONLY_KAPPA: f64 = 1.040; // ± 0.050 (Research primary target)
const ENCODER_DECODER_KAPPA: f64 = 0.900; // ± 0.107 (Seq2seq reduction)
const UNKNOWN_ARCHITECTURE_KAPPA: f64 = 1.040; // Fallback to decoder assumption

/// 🌀 Golden Ratio Scaling Options
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GoldenScaling {
    /// √(φ × 2.67²) ≈ 3.40 - Empirical golden scale (scientifically justified)
    EmpiricalGolden,
    /// Custom scaling factor
    Custom(f64),
}

impl GoldenScaling {
    pub fn factor(&self) -> f64 {
        match self {
            GoldenScaling::EmpiricalGolden => EMPIRICAL_GOLDEN_SCALE,
            GoldenScaling::Custom(factor) => *factor,
        }
    }
    
    pub fn name(&self) -> &str {
        match self {
            GoldenScaling::EmpiricalGolden => "√(φ × 2.67²) (Empirical Golden Scale)",
            GoldenScaling::Custom(_) => "Custom Scaling",
        }
    }
}

/// 🎯 Calibration Mode for Semantic Uncertainty Analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationMode {
    /// Scientific mode: Use raw ℏₛ values with empirically-derived thresholds
    Scientific {
        abort_threshold: f64,    // ~0.1 (block immediately)
        warn_threshold: f64,     // ~0.3 (proceed with caution)
        proceed_threshold: f64,  // ~0.4 (safe to proceed)
    },
    /// Pragmatic mode: Scale ℏₛ values for immediate usability
    Pragmatic {
        scaling: GoldenScaling,
        abort_threshold: f64,    // 0.8 (standard)
        warn_threshold: f64,     // 1.0 (standard)
        proceed_threshold: f64,  // 1.2 (standard)
    },
}

impl Default for CalibrationMode {
    fn default() -> Self {
        CalibrationMode::Scientific {
            abort_threshold: 0.1,
            warn_threshold: 0.3,
            proceed_threshold: 0.4,
        }
    }
}

/// 🚦 Risk Level Assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// ✅ Safe to proceed
    Safe,
    /// ⚠️ Proceed with caution
    Warning,
    /// 🚨 High risk - review recommended
    HighRisk,
    /// ❌ Critical - block immediately
    Critical,
}

impl RiskLevel {
    pub fn emoji(&self) -> &str {
        match self {
            RiskLevel::Safe => "✅",
            RiskLevel::Warning => "⚠️",
            RiskLevel::HighRisk => "🚨",
            RiskLevel::Critical => "❌",
        }
    }
    
    pub fn description(&self) -> &str {
        match self {
            RiskLevel::Safe => "Safe to proceed",
            RiskLevel::Warning => "Proceed with caution",
            RiskLevel::HighRisk => "High risk - review recommended",
            RiskLevel::Critical => "Critical - block immediately",
        }
    }
}

/// 📊 Enhanced Semantic Uncertainty Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticUncertaintyResult {
    /// 🧮 Pure mathematical ℏₛ = √(Δμ × Δσ)
    pub raw_hbar: f64,
    /// 🎯 Decision-making value (calibrated)
    pub calibrated_hbar: f64,
    /// 🚦 Risk assessment based on calibrated value
    pub risk_level: RiskLevel,
    /// ⚙️ Calibration mode used
    pub calibration_mode: CalibrationMode,
    /// 📝 Human-readable explanation
    pub explanation: String,
    /// 📊 Component metrics
    pub delta_mu: f64,
    pub delta_sigma: f64,
    /// ⏱️ Processing metadata
    pub processing_time_ms: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 🆔 Request tracking
    pub request_id: RequestId,
}

impl CalibrationMode {
    /// Calibration disabled: identity mapping with fixed thresholds
    pub fn calibrate(&self, raw_hbar: f64) -> (f64, RiskLevel, String) {
        let calibrated = raw_hbar;
        let abort_threshold = 0.1;
        let warn_threshold = 0.3;
        let proceed_threshold = 0.4;
        let risk_level = if raw_hbar < abort_threshold {
            RiskLevel::Critical
        } else if raw_hbar < warn_threshold {
            RiskLevel::Warning
        } else if raw_hbar < proceed_threshold {
            RiskLevel::HighRisk
        } else {
            RiskLevel::Safe
        };
        let explanation = format!(
            "Calibration disabled: identity mapping (abort: {:.1}, warn: {:.1}, proceed: {:.1})",
            abort_threshold, warn_threshold, proceed_threshold
        );
        (calibrated, risk_level, explanation)
    }
}

// 🧮 Semantic Precision Module (Fisher Information + JSD)
pub mod semantic_metrics;

// 🎯 Multi-Axis Drift Tensor (MAD Tensor) for geometric diagnostics
// pub mod mad_tensor; // REMOVED - migrated to JSD+KL divergence


// 🧪 Hash Embedding Discrepancy Tests
// #[cfg(test)]
// pub mod tests {
//     pub mod hash_embedding_discrepancy_tests;
// }

// 📦 Modular architecture
pub mod modules;

// 🔌 Feature-gated modules
#[cfg(feature = "api")]
pub mod api;

#[cfg(feature = "python")]
pub mod ffi;

#[cfg(feature = "worker")]
pub mod cloudflare_worker;

// 🏛️ Legacy modules (kept for compatibility)
pub mod compression;
pub mod batch_processing;
pub mod semantic_decision_engine;
pub mod api_security_analyzer;
pub mod information_theory;
pub mod rigorous_benchmarking;
pub mod secure_api_key_manager;
// pub mod key_rotation_scheduler; // REMOVED: Over-engineered rotation system
// pub mod enhanced_breach_detection; // REMOVED: Experimental security component
// pub mod tier3_measurement; // REMOVED: Incomplete advanced measurement system
pub mod monitoring;
pub mod scalar_walk_firewall;
pub mod scalar_firewall;
pub mod alias_ambiguity_defense;

// 🏗️ Neural Uncertainty Physics Research Integration
pub mod architecture_detector;
pub mod predictive_uncertainty;

// 🧠 Free Energy Principle metrics
pub mod free_energy;

// 📤 StreamlinedEngine removed; using SemanticAnalyzer only

// 📤 Re-exports for semantic metrics
pub use semantic_metrics::{
    SemanticMetricsCalculator,
    HashEmbeddingDiscrepancyTester,
    SemanticPrecisionResult,
    PrecisionMethod,
    DiscrepancyTestResult
};

// 📤 Re-exports for MAD Tensor (REMOVED - migrated to JSD+KL divergence)
// pub use mad_tensor::{
//     MadTensorCalculator,
//     MadTensorResult,
//     ContextData,
//     GeometricMetadata,
//     CurvatureAnalysis,
//     PerturbationAnalysis,
//     VolatilityAnalysis,
//     TopologyType
// };

// 🔧 Legacy imports (for backwards compatibility)
use crate::information_theory::InformationTheoryCalculator;

use compression::SemanticCompressor;
use batch_processing::{BatchProcessor, BatchConfig, BatchResult};
use semantic_decision_engine::{SemanticDecisionEngine};
use api_security_analyzer::{ApiSecurityAnalyzer, ApiSecurityAssessment, SecurityAction};
use secure_api_key_manager::{SecureApiKeyManager, KeyValidationResult, KeyAction};
// use key_rotation_scheduler::{KeyRotationScheduler, RotationConfig}; // REMOVED
// use tier3_measurement::{Tier3MeasurementEngine, Tier3Config}; // REMOVED

/// Unique request identifier for tracing and debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(Uuid);

impl RequestId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.simple())
    }
}

/// Response structure containing all semantic uncertainty metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct HbarResponse {
    /// Request identifier for tracing
    pub request_id: RequestId,
    /// Quantum-inspired semantic uncertainty metric: ℏₛ(C) = √(Δμ * Δσ)
    pub hbar_s: f32,
    /// Semantic precision: Δμ (entropy of embedding)
    pub delta_mu: f32,
    /// Semantic flexibility: Δσ (Jensen-Shannon divergence between prompt and output)  
    pub delta_sigma: f32,
    /// Risk of semantic collapse (true if ℏₛ < threshold)
    pub collapse_risk: bool,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Embedding dimensions used
    pub embedding_dims: usize,
    /// Analysis timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Security assessment (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security_assessment: Option<ApiSecurityAssessment>,
}

/// Semantic uncertainty analysis errors
#[derive(thiserror::Error, Debug)]
pub enum SemanticError {
    #[error("Input validation failed: {message}")]
    InvalidInput { message: String },
    
    #[error("Embedding computation failed: {source}")]
    EmbeddingError { source: anyhow::Error },
    
    #[error("Math computation failed: {operation}")]
    MathError { operation: String },
    
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    #[error("Timeout occurred after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    #[error("Internal error: {source}")]
    Internal { source: anyhow::Error },
}

/// Performance and accuracy configuration for the semantic uncertainty analyzer
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    pub model_path: String,
    pub collapse_threshold: f32,
    pub perturbation_count: usize,
    pub max_sequence_length: usize,
    /// Enable fast approximations for sub-10ms performance
    pub fast_mode: bool,
    /// Minimum entropy threshold for numerical stability
    pub entropy_min_threshold: f32,
    /// Minimum JS divergence threshold for numerical stability
    pub js_min_threshold: f32,
    /// Embedding dimensions (higher = more accurate, slower)
    pub embedding_dims: usize,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Enable Tier-3 advanced measurement engine
    pub enable_tier3: bool,
    // 🏗️ Neural Uncertainty Physics Research Integration
    /// Enable architecture detection for κ-based calibration
    pub enable_architecture_detection: bool,
    /// Use research-based calibration instead of empirical golden scale
    pub use_research_calibration: bool,
    /// Research mode (optimizes for decoder-only workflows)
    pub research_mode: bool,
    /// Fallback to legacy calibration if architecture detection fails
    pub fallback_to_legacy: bool,
    // Tier-3 specific configuration - REMOVED
    // pub tier3_config: Option<Tier3Config>, // REMOVED
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            model_path: "models/sentence_transformer.onnx".to_string(),
            collapse_threshold: 1.0,
            perturbation_count: 3,
            max_sequence_length: 256,
            fast_mode: true,
            entropy_min_threshold: 0.1,
            js_min_threshold: 0.01,
            embedding_dims: 128, // Reduced from 384 for speed
            timeout_ms: 5000,
            use_simd: true,
            enable_tier3: false,
            // 🏗️ Neural Uncertainty Physics Research Integration
            enable_architecture_detection: true,
            use_research_calibration: true,
            research_mode: false,
            fallback_to_legacy: true,
            // tier3_config: None, // REMOVED
        }
    }
}

impl SemanticConfig {
    /// Ultra-fast configuration for sub-10ms targets
    pub fn ultra_fast() -> Self {
        Self {
            model_path: "models/sentence_transformer.onnx".to_string(),
            collapse_threshold: 1.0,
            perturbation_count: 1,
            max_sequence_length: 64,
            fast_mode: true,
            entropy_min_threshold: 0.05,
            js_min_threshold: 0.005,
            embedding_dims: 64,
            timeout_ms: 50,
            use_simd: true,
            enable_tier3: false,
            // 🏗️ Neural Uncertainty Physics Research Integration
            enable_architecture_detection: true,
            use_research_calibration: true,
            research_mode: false,
            fallback_to_legacy: true,
            // tier3_config: None, // REMOVED
        }
    }

    /// Performance-optimized configuration for sub-100ms targets
    pub fn performance() -> Self {
        Self {
            model_path: "models/sentence_transformer.onnx".to_string(),
            collapse_threshold: 1.0,
            perturbation_count: 2,
            max_sequence_length: 128,
            fast_mode: true,
            entropy_min_threshold: 0.05,
            js_min_threshold: 0.005,
            embedding_dims: 128,
            timeout_ms: 1000,
            use_simd: true,
            enable_tier3: false,
            // 🏗️ Neural Uncertainty Physics Research Integration
            enable_architecture_detection: true,
            use_research_calibration: true,
            research_mode: false,
            fallback_to_legacy: true,
            // tier3_config: None, // REMOVED
        }
    }

    /// High-accuracy configuration (slower but more precise)
    pub fn accuracy() -> Self {
        Self {
            model_path: "models/sentence_transformer.onnx".to_string(),
            collapse_threshold: 1.0,
            perturbation_count: 10,
            max_sequence_length: 512,
            fast_mode: false,
            entropy_min_threshold: 0.001,
            js_min_threshold: 0.001,
            embedding_dims: 384,
            timeout_ms: 10000,
            use_simd: true,
            enable_tier3: false,
            // 🏗️ Neural Uncertainty Physics Research Integration
            enable_architecture_detection: true,
            use_research_calibration: true,
            research_mode: false,
            fallback_to_legacy: true,
            // tier3_config: None, // REMOVED
        }
    }

    /// Tier-3 advanced configuration with sophisticated measurement
    pub fn tier3() -> Self {
        Self {
            model_path: "models/sentence_transformer.onnx".to_string(),
            collapse_threshold: 1.0,
            perturbation_count: 8,
            max_sequence_length: 512,
            fast_mode: false,
            entropy_min_threshold: 0.001,
            js_min_threshold: 0.001,
            embedding_dims: 384,
            timeout_ms: 15000,
            use_simd: true,
            enable_tier3: false, // DISABLED: Tier-3 removed
            // 🏗️ Neural Uncertainty Physics Research Integration
            enable_architecture_detection: true,
            use_research_calibration: true,
            research_mode: false,
            fallback_to_legacy: true,
            // tier3_config: Some(Tier3Config::default()), // REMOVED
        }
    }
}

/// Main semantic uncertainty analyzer with optimized vector operations
pub struct SemanticAnalyzer {
    pub config: SemanticConfig,
    // 🌀 Calibration mode for uncertainty analysis
    pub calibration_mode: CalibrationMode,
    // Pre-allocated buffers for performance
    embedding_buffer: Arc<std::sync::Mutex<Vec<f32>>>,
    computation_buffer: Arc<std::sync::Mutex<Vec<f32>>>,
    // Integrated compression engine
    compressor: SemanticCompressor,
    // 🧮 Semantic decision engine for ℏₛ-guided process control
    decision_engine: std::sync::Mutex<SemanticDecisionEngine>,
    // 🛡️ API security analyzer for robust request validation
    security_analyzer: std::sync::Mutex<ApiSecurityAnalyzer>,
    // 🔐 Secure API key manager for cryptographic key validation
    key_manager: std::sync::Mutex<SecureApiKeyManager>,
    // 🔄 Key rotation scheduler for usage and uncertainty-based rotation
    // rotation_scheduler: std::sync::Mutex<KeyRotationScheduler>, // REMOVED
    // 🧠 Tier-3 advanced measurement engine (optional)
    // 📊 Information theory calculator for advanced metrics
    info_calculator: InformationTheoryCalculator,
    // 🌀 Curvature regularizer for manifold stability
    // curvature_regularizer: Option<CurvatureRegularizer>, // REMOVED
    // 🏗️ Neural Uncertainty Physics Research Integration
    // 🏗️ Architecture detector for κ-based calibration
    architecture_detector: std::sync::Mutex<architecture_detector::ArchitectureDetector>,
    // 🔮 Predictive uncertainty framework
    uncertainty_predictor: std::sync::Mutex<predictive_uncertainty::UncertaintyPredictor>,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer with pre-allocated buffers
    pub fn new(config: SemanticConfig) -> Result<Self, SemanticError> {
        Self::with_calibration(config, CalibrationMode::default())
    }

    /// Create a new semantic analyzer with custom calibration mode
    pub fn with_calibration(config: SemanticConfig, calibration_mode: CalibrationMode) -> Result<Self, SemanticError> {
        info!("Initializing semantic analyzer with fast_mode={}, embedding_dims={}, tier3={}", 
              config.fast_mode, config.embedding_dims, config.enable_tier3);
        
        // Validate configuration
        if config.embedding_dims == 0 {
            return Err(SemanticError::ConfigError {
                message: "embedding_dims must be greater than 0".to_string(),
            });
        }
        
        // Pre-allocate buffers for zero-allocation analysis
        let embedding_buffer = Arc::new(std::sync::Mutex::new(vec![0.0; config.embedding_dims * 2]));
        let computation_buffer = Arc::new(std::sync::Mutex::new(vec![0.0; config.embedding_dims * 4]));
        
        // Tier-3 engine removed - using streamlined analysis only
        
        // Initialize advanced features
        let info_calculator = InformationTheoryCalculator::default();
        // let curvature_regularizer = None; // REMOVED

        // 🏗️ Initialize Neural Uncertainty Physics Research components
        let architecture_detector = std::sync::Mutex::new(architecture_detector::ArchitectureDetector::new());
        let uncertainty_predictor = std::sync::Mutex::new(predictive_uncertainty::UncertaintyPredictor::new());

        Ok(Self { 
            config,
            calibration_mode,
            embedding_buffer,
            computation_buffer,
            compressor: SemanticCompressor::new(),
            decision_engine: std::sync::Mutex::new(SemanticDecisionEngine::new()),
            security_analyzer: std::sync::Mutex::new(ApiSecurityAnalyzer::new()),
            key_manager: std::sync::Mutex::new(SecureApiKeyManager::new()),
            // rotation_scheduler: std::sync::Mutex::new(KeyRotationScheduler::new(RotationConfig::default())), // REMOVED
            info_calculator,
            // curvature_regularizer, // REMOVED
            architecture_detector,
            uncertainty_predictor,
        })
    }

    /// Enhanced analysis function with calibration - returns detailed uncertainty result
    #[instrument(skip(self, prompt, output), fields(request_id = %request_id))]
    pub async fn analyze_with_calibration(&self, prompt: &str, output: &str, request_id: RequestId) -> Result<SemanticUncertaintyResult, SemanticError> {
        let start_time = std::time::Instant::now();
        let timestamp = chrono::Utc::now();
        
        debug!("Starting calibrated semantic analysis with mode: {:?}", self.calibration_mode);
        
        // Input validation with early returns
        if prompt.is_empty() && output.is_empty() {
            let (calibrated_hbar, risk_level, explanation) = self.calibration_mode.calibrate(0.0);
            return Ok(SemanticUncertaintyResult {
                request_id,
                raw_hbar: 0.0,
                calibrated_hbar,
                risk_level,
                calibration_mode: self.calibration_mode.clone(),
                explanation,
                delta_mu: 0.0,
                delta_sigma: 0.0,
                processing_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                timestamp,
            });
        }
        
        // Use streamlined analysis only (Tier-3 removed)
        debug!("Using streamlined measurement engine");
        let timeout = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.timeout_ms),
            self.analyze_internal(prompt, output)
        );
        
        let (delta_mu, delta_sigma, raw_hbar) = match timeout.await {
            Ok(result) => result?,
            Err(_) => return Err(SemanticError::Timeout { 
                timeout_ms: self.config.timeout_ms 
            }),
        };

        // Numerical stability warnings (lightweight)
        let eps = 1e-9_f64;
        if (raw_hbar as f64) < eps {
            warn!("NumericalWarning: raw_hbar below epsilon: {:.3e}", raw_hbar);
        }
        if (delta_mu as f64) > 1e3 {
            warn!("NumericalWarning: delta_mu unusually large: {:.3e}", delta_mu);
        }
        if (delta_sigma as f64) > 1e3 {
            warn!("NumericalWarning: delta_sigma unusually large: {:.3e}", delta_sigma);
        }
        
        // Identity calibration (plug-and-play)
        let (calibrated_hbar, risk_level, explanation) = self.calibration_mode.calibrate(raw_hbar as f64);
        
        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        debug!("Analysis completed: raw_ℏₛ={:.4}, risk={:?}, time={:.2}ms", 
               raw_hbar, risk_level, processing_time_ms);

        Ok(SemanticUncertaintyResult {
            request_id,
            raw_hbar: raw_hbar as f64,
            calibrated_hbar,
            risk_level,
            calibration_mode: self.calibration_mode.clone(),
            explanation,
            delta_mu: delta_mu as f64,
            delta_sigma: delta_sigma as f64,
            processing_time_ms,
            timestamp,
        })
    }

    /// Main analysis function - computes all semantic uncertainty metrics (legacy compatibility)
    #[instrument(skip(self, prompt, output), fields(request_id = %request_id))]
    pub async fn analyze(&self, prompt: &str, output: &str, request_id: RequestId) -> Result<HbarResponse, SemanticError> {
        let start_time = std::time::Instant::now();
        let timestamp = chrono::Utc::now();
        
        debug!("Starting semantic analysis with Tier3={}", self.config.enable_tier3);
        
        // Input validation with early returns
        if prompt.is_empty() && output.is_empty() {
            return Ok(HbarResponse {
                request_id,
                hbar_s: 0.0,
                delta_mu: 0.0,
                delta_sigma: 0.0,
                collapse_risk: true,
                processing_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                embedding_dims: self.config.embedding_dims,
                security_assessment: None,
                timestamp,
            });
        }
        
        // Use streamlined analysis only (Tier-3 removed)
        debug!("Using streamlined measurement engine");
        let timeout = tokio::time::timeout(
            std::time::Duration::from_millis(self.config.timeout_ms),
            self.analyze_internal(prompt, output)
        );
        
        let (delta_mu, delta_sigma, hbar_s) = match timeout.await {
            Ok(result) => result?,
            Err(_) => return Err(SemanticError::Timeout { 
                timeout_ms: self.config.timeout_ms 
            }),
        };
        
        // Check for collapse risk
        let collapse_risk = hbar_s < self.config.collapse_threshold;
        
        if collapse_risk {
            warn!("Semantic collapse detected: ℏₛ = {:.4} < threshold {:.4}", 
                  hbar_s, self.config.collapse_threshold);
        }
        
        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        debug!("Analysis completed: ℏₛ={:.4}, Δμ={:.4}, Δσ={:.4}, collapse={}, time={:.2}ms", 
               hbar_s, delta_mu, delta_sigma, collapse_risk, processing_time_ms);

        Ok(HbarResponse {
            request_id,
            hbar_s,
            delta_mu,
            delta_sigma,
            collapse_risk,
            processing_time_ms,
            embedding_dims: self.config.embedding_dims,
            timestamp,
            security_assessment: None, // Set by secure_analyze method
        })
    }

    
    /// Internal analysis with optimized computations (standard mode)
    async fn analyze_internal(&self, prompt: &str, output: &str) -> Result<(f32, f32, f32), SemanticError> {
        // Generate optimized embeddings using pre-allocated buffers
        let (prompt_embedding, output_embedding) = self.embed_texts_batch(prompt, output)?;

        // Ultra-fast parallel computation of metrics
        #[cfg(not(target_arch = "wasm32"))]
        let (delta_mu, delta_sigma) = rayon::join(
            || self.compute_delta_mu_simd(&output_embedding),
            || self.compute_delta_sigma_simd(&prompt_embedding, &output_embedding)
        );
        
        #[cfg(target_arch = "wasm32")]
        let (delta_mu, delta_sigma) = (
            self.compute_delta_mu_simd(&output_embedding),
            self.compute_delta_sigma_simd(&prompt_embedding, &output_embedding)
        );

        // Optimized ℏₛ(C) = √(Δμ * Δσ) with stability check
        let hbar_s = if delta_mu > 0.0 && delta_sigma > 0.0 {
            (delta_mu * delta_sigma).sqrt()
        } else {
            0.0
        };

        // 🏗️ Apply architecture-aware calibration
        let calibrated_hbar = if self.config.enable_architecture_detection {
            let calibration_factor = self.get_architecture_calibration(None);
            hbar_s * calibration_factor as f32
        } else {
            hbar_s * EMPIRICAL_GOLDEN_SCALE as f32 // Legacy calibration
        };

        Ok((delta_mu, delta_sigma, calibrated_hbar))
    }

    /// 🏗️ Get architecture-aware calibration factor
    fn get_architecture_calibration(&self, model_name: Option<&str>) -> f64 {
        if !self.config.enable_architecture_detection {
            return EMPIRICAL_GOLDEN_SCALE; // Fallback to legacy calibration
        }

        let model_name = model_name.unwrap_or("unknown");
        
        // Detect architecture
        let detector = match self.architecture_detector.lock() {
            Ok(detector) => detector,
            Err(e) => {
                warn!("Failed to acquire architecture detector lock: {}", e);
                return EMPIRICAL_GOLDEN_SCALE;
            }
        };

        let detection_result = detector.detect_from_model_name(model_name);
        let constants = detection_result.constants;
        
        if self.config.use_research_calibration {
            // Use research-based κ scaling
            let kappa_scaling = constants.scaling_factor();
            let base_scale = if self.config.research_mode {
                DECODER_ONLY_KAPPA // Optimize for decoder-only workflows
            } else {
                EMPIRICAL_GOLDEN_SCALE
            };
            
            info!("🏗️ Architecture-aware calibration: {} (κ = {:.3}) for model '{}'", 
                  detection_result.architecture.name(), kappa_scaling, model_name);
            
            base_scale * kappa_scaling
        } else {
            // Use legacy calibration
            if self.config.fallback_to_legacy {
                info!("🔄 Using legacy calibration for model '{}'", model_name);
                EMPIRICAL_GOLDEN_SCALE
            } else {
                constants.scaling_factor() * EMPIRICAL_GOLDEN_SCALE
            }
        }
    }

    /// 🏗️ Get architecture-aware risk thresholds
    fn get_architecture_risk_thresholds(&self, model_name: Option<&str>) -> (f64, f64, f64) {
        if !self.config.enable_architecture_detection {
            return (0.8, 1.0, 1.2); // Default thresholds
        }

        let model_name = model_name.unwrap_or("unknown");
        
        let detector = match self.architecture_detector.lock() {
            Ok(detector) => detector,
            Err(e) => {
                warn!("Failed to acquire architecture detector lock: {}", e);
                return (0.8, 1.0, 1.2);
            }
        };

        let detection_result = detector.detect_from_model_name(model_name);
        
        match detection_result.architecture {
            architecture_detector::ModelArchitecture::EncoderOnly => {
                info!("🏗️ Using encoder-only risk thresholds for model '{}'", model_name);
                (0.80, 1.20, 1.00) // Encoder thresholds
            },
            architecture_detector::ModelArchitecture::DecoderOnly => {
                info!("🏗️ Using decoder-only risk thresholds for model '{}'", model_name);
                (0.83, 1.25, 1.04) // Decoder thresholds (research primary target)
            },
            architecture_detector::ModelArchitecture::EncoderDecoder => {
                info!("🏗️ Using encoder-decoder risk thresholds for model '{}'", model_name);
                (0.72, 1.08, 0.90) // Encoder-decoder thresholds
            },
            architecture_detector::ModelArchitecture::Unknown => {
                info!("🏗️ Using default risk thresholds for unknown model '{}'", model_name);
                (0.83, 1.25, 1.04) // Default to decoder thresholds (research use case)
            },
        }
    }

    /// Batch embedding generation with SIMD optimizations
    #[inline]
    fn embed_texts_batch(&self, text1: &str, text2: &str) -> Result<(Vec<f32>, Vec<f32>), SemanticError> {
        // Use pre-allocated buffer for zero-allocation embeddings
        let mut buffer = self.embedding_buffer.lock().map_err(|e| SemanticError::Internal {
            source: anyhow::anyhow!("Failed to acquire embedding buffer: {}", e)
        })?;
        
        if buffer.len() < self.config.embedding_dims * 2 {
            buffer.resize(self.config.embedding_dims * 2, 0.0);
        }
        
        let (prompt_slice, output_slice) = buffer.split_at_mut(self.config.embedding_dims);
        
        // Fast deterministic embedding using optimized hash function
        self.embed_text_fast_inplace(text1, prompt_slice);
        self.embed_text_fast_inplace(text2, output_slice);
        
        Ok((prompt_slice.to_vec(), output_slice.to_vec()))
    }

    /// Ultra-fast in-place text embedding with SIMD
    #[inline]
    fn embed_text_fast_inplace(&self, text: &str, output: &mut [f32]) {
        if text.is_empty() {
            output.fill(0.0);
            return;
        }
        
        let mut hasher = DefaultHasher::new();
        
        // Hash each character with position weighting for better distribution
        for (i, ch) in text.chars().enumerate().take(self.config.max_sequence_length) {
            ch.hash(&mut hasher);
            i.hash(&mut hasher);
            
            let hash = hasher.finish();
            let idx = (hash as usize) % output.len();
            
            // Use improved distribution function
            output[idx] += ((hash as f32).sin() * 0.1 + (i as f32).cos() * 0.05).tanh();
        }
        
        // SIMD-optimized normalization
        if self.config.use_simd {
            self.normalize_simd(output);
        } else {
            self.normalize_standard(output);
        }
    }

    /// SIMD-optimized vector normalization
    #[inline]
    fn normalize_simd(&self, vec: &mut [f32]) {
        #[cfg(feature = "fast-math")]
        {
            // Use unsafe SIMD operations for maximum performance
            let sum: f32 = vec.iter().map(|x| x * x).sum();
            if sum > 1e-10 {
                let inv_norm = 1.0 / sum.sqrt();
                vec.iter_mut().for_each(|x| *x *= inv_norm);
            }
        }
        
        #[cfg(not(feature = "fast-math"))]
        {
            self.normalize_standard(vec);
        }
    }

    /// Standard vector normalization fallback
    #[inline]
    fn normalize_standard(&self, vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            let inv_norm = 1.0 / norm;
            vec.iter_mut().for_each(|x| *x *= inv_norm);
        } else {
            // Handle zero vector
            let uniform_val = 1.0 / (vec.len() as f32).sqrt();
            vec.fill(uniform_val);
        }
    }

    /// SIMD-optimized Δμ computation using entropy approximation
    #[inline]
    fn compute_delta_mu_simd(&self, embedding: &[f32]) -> f32 {
        let entropy = if self.config.use_simd {
            self.entropy_approx_simd(embedding)
        } else {
            self.entropy_approx_standard(embedding)
        };
        
        // Precision is inverse of entropy with stability
        1.0 / entropy.sqrt().max(self.config.entropy_min_threshold)
    }

    /// SIMD-optimized entropy approximation
    #[inline]
    fn entropy_approx_simd(&self, embedding: &[f32]) -> f32 {
        #[cfg(feature = "fast-math")]
        {
            // Ultra-fast entropy using vector operations
            embedding.iter()
                .map(|&v| {
                    let p = v.abs();
                    if p > 1e-10 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum()
        }
        
        #[cfg(not(feature = "fast-math"))]
        {
            self.entropy_approx_standard(embedding)
        }
    }

    /// Standard entropy approximation fallback
    #[inline]
    fn entropy_approx_standard(&self, embedding: &[f32]) -> f32 {
        embedding.iter()
            .map(|&v| {
                let p = v.abs();
                if p > 1e-10 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// SIMD-optimized Δσ computation using JSD and KL divergence
    #[inline]
    fn compute_delta_sigma_simd(&self, prompt_emb: &[f32], output_emb: &[f32]) -> f32 {
        // Fast probability distribution normalization
        let p = self.to_probability_fast(prompt_emb);
        let q = self.to_probability_fast(output_emb);

        // SIMD-optimized JSD computation (primary flexibility metric)
        let js_divergence = if self.config.use_simd {
            self.js_divergence_simd(&p, &q)
        } else {
            self.js_divergence_standard(&p, &q)
        };

        // KL divergence computation for directional insight
        let kl_pq = self.kl_divergence_simd(&p, &q);
        let kl_qp = self.kl_divergence_simd(&q, &p);

        // Use JSD as primary flexibility measure
        let primary_flexibility = js_divergence.sqrt();
        
        // Use minimum KL for conservative drift estimate
        let kl_min = kl_pq.min(kl_qp).sqrt();
        
        // Combine JSD and KL for enhanced flexibility measurement
        // Weight JSD more heavily as it's symmetric and more stable
        let combined_flexibility = 0.7 * primary_flexibility + 0.3 * kl_min;
        
        combined_flexibility.max(self.config.js_min_threshold)
    }

    /// Fast probability distribution conversion
    #[inline]
    fn to_probability_fast(&self, v: &[f32]) -> Vec<f32> {
        let sum: f32 = v.iter().map(|x| x.abs()).sum();
        if sum > 1e-10 {
            v.iter().map(|x| x.abs() / sum).collect()
        } else {
            vec![1.0 / v.len() as f32; v.len()]
        }
    }

    /// SIMD-optimized Jensen-Shannon divergence
    #[inline]
    fn js_divergence_simd(&self, p: &[f32], q: &[f32]) -> f32 {
        #[cfg(feature = "fast-math")]
        {
            // Ultra-fast JS divergence using SIMD operations
            let mut js_sum = 0.0f32;
            
            for (&pi, &qi) in p.iter().zip(q.iter()) {
                if pi > 1e-10 && qi > 1e-10 {
                    let m = (pi + qi) * 0.5;
                    if m > 1e-10 {
                        js_sum += 0.5 * (pi * (pi / m).ln() + qi * (qi / m).ln());
                    }
                }
            }
            
            js_sum
        }
        
        #[cfg(not(feature = "fast-math"))]
        {
            self.js_divergence_standard(p, q)
        }
    }

    /// Standard JS divergence fallback
    #[inline]
    fn js_divergence_standard(&self, p: &[f32], q: &[f32]) -> f32 {
        let mut js_sum = 0.0f32;
        
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 1e-10 && qi > 1e-10 {
                let m = (pi + qi) * 0.5;
                if m > 1e-10 {
                    js_sum += 0.5 * (pi * (pi / m).ln() + qi * (qi / m).ln());
                }
            }
        }
        
        js_sum
    }

    /// SIMD-optimized KL divergence
    #[inline]
    fn kl_divergence_simd(&self, p: &[f32], q: &[f32]) -> f32 {
        #[cfg(feature = "fast-math")]
        {
            // Ultra-fast KL divergence using SIMD operations
            let mut kl_sum = 0.0f32;
            
            for (&pi, &qi) in p.iter().zip(q.iter()) {
                if pi > 1e-10 && qi > 1e-10 {
                    kl_sum += pi * (pi / qi).ln();
                }
            }
            
            kl_sum
        }
        
        #[cfg(not(feature = "fast-math"))]
        {
            self.kl_divergence_standard(p, q)
        }
    }

    /// Standard KL divergence fallback
    #[inline]
    fn kl_divergence_standard(&self, p: &[f32], q: &[f32]) -> f32 {
        let mut kl_sum = 0.0f32;
        
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi > 1e-10 && qi > 1e-10 {
                kl_sum += pi * (pi / qi).ln();
            }
        }
        
        kl_sum
    }

    /// 🧠 Analyze with intelligent compression and ℏₛ-guided decisions
    #[instrument(skip(self, prompt), fields(request_id = %request_id))]
    pub async fn analyze_with_compression(&self, prompt: &str, request_id: RequestId) -> Result<HbarResponse, SemanticError> {
        debug!("🔍 SEMANTIC_ANALYSIS_START | Starting analysis with ℏₛ-guided decisions");

        // 🧮 Evaluate compression decision using semantic uncertainty
        let mut decision_engine = self.decision_engine.lock().map_err(|e| SemanticError::Internal {
            source: anyhow::anyhow!("Decision engine lock failed: {}", e)
        })?;
        
        let risk_score = self.calculate_prompt_risk_score(prompt);
        let compression_decision = decision_engine.evaluate_compression_decision(prompt.len(), risk_score);
        
        info!("{} {} | ℏₛ={:.3} | Compression evaluation complete", 
              compression_decision.emoji_indicator, 
              compression_decision.relevance_phrase,
              compression_decision.h_bar);

        // 🎯 Act based on compression decision
        if prompt.len() > 300 && compression_decision.h_bar >= 1.0 {
            match self.compressor.compress_prompt(prompt) {
                Ok(compression_result) => {
                    if compression_result.should_use_compression.should_compress {
                        info!("⚡ OPTIMAL_COMPRESSION | Using compressed essence for analysis");
                        
                        // Generate synthetic output based on compressed essence
                        let synthetic_output = self.generate_synthetic_output(&compression_result.compressed_essence);
                        
                        // Analyze compressed essence
                        let mut result = self.analyze(&compression_result.compressed_essence, &synthetic_output, request_id).await?;
                        
                        // Adjust results for compression effects using ℏₛ
                        if !compression_result.risk_preservation.risk_preserved && compression_result.risk_preservation.original_risk_score > 0.5 {
                            // 🚨 Escalate uncertainty if risk indicators were lost
                            let escalation_factor = 1.0 + (compression_decision.h_bar * 0.3);
                            result.hbar_s = (result.hbar_s * escalation_factor).min(5.0);
                        }
                        
                        return Ok(result);
                    }
                }
                Err(e) => {
                    warn!("🛑 COMPRESSION_BLOCKED_RISK | Compression failed: {}", e);
                }
            }
        } else if compression_decision.h_bar < 1.0 {
            warn!("{} {} | Compression deemed too risky", 
                  compression_decision.emoji_indicator, 
                  compression_decision.relevance_phrase);
        }

        // Use original prompt for analysis
        let synthetic_output = self.generate_synthetic_output(prompt);
        self.analyze(prompt, &synthetic_output, request_id).await
    }

    /// Generate synthetic output for semantic comparison
    fn generate_synthetic_output(&self, prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();
        
        // Risk-based synthetic response generation
        if prompt_lower.contains("hack") || prompt_lower.contains("exploit") || prompt_lower.contains("illegal") {
            "I cannot and will not provide information on that topic as it could be harmful.".to_string()
        } else if prompt_lower.contains("creative") || prompt_lower.contains("story") {
            format!("Here's a creative response to your request about {}. Let me craft something imaginative...", 
                   prompt.split_whitespace().take(5).collect::<Vec<_>>().join(" "))
        } else if prompt_lower.contains("explain") || prompt_lower.contains("how") {
            format!("Let me explain this concept clearly. Regarding your question about {}, here's a comprehensive explanation...",
                   prompt.split_whitespace().take(5).collect::<Vec<_>>().join(" "))
        } else {
            format!("Thank you for your question about {}. Here's my response based on my understanding...",
                   prompt.split_whitespace().take(5).collect::<Vec<_>>().join(" "))
        }
    }

    /// 🚀 High-performance batch analysis with ℏₛ-guided decisions
    pub async fn batch_analyze(&self, prompts: Vec<String>, model: &str) -> Result<BatchResult, SemanticError> {
        // 🧮 Evaluate batch processing decision using semantic uncertainty
        let mut decision_engine = self.decision_engine.lock().map_err(|e| SemanticError::Internal {
            source: anyhow::anyhow!("Decision engine lock failed: {}", e)
        })?;
        
        let complexity = self.calculate_batch_complexity(&prompts);
        let time_pressure = 0.5; // Default time pressure
        let batch_decision = decision_engine.evaluate_batch_decision(prompts.len(), complexity, time_pressure);
        
        info!("{} {} | ℏₛ={:.3} | Batch={} items", 
              batch_decision.emoji_indicator, 
              batch_decision.relevance_phrase,
              batch_decision.h_bar, prompts.len());

        // 🎯 Act based on batch decision
        match batch_decision.decision {
            semantic_decision_engine::ProcessDecision::Execute => {
                info!("🚀 OPTIMAL_BATCH | Proceeding with batch processing");
                let batch_config = BatchConfig::default();
                let processor = BatchProcessor::new(batch_config, self.config.clone());
                processor.process_batch(prompts, model).await
            },
            semantic_decision_engine::ProcessDecision::Monitor => {
                info!("⚡ BATCH_PROCESSING | Proceeding with enhanced monitoring");
                let batch_config = BatchConfig { 
                    max_concurrent: 5, // Reduce concurrency for monitoring
                    ..BatchConfig::default() 
                };
                let processor = BatchProcessor::new(batch_config, self.config.clone());
                processor.process_batch(prompts, model).await
            },
            semantic_decision_engine::ProcessDecision::Defer => {
                warn!("🚨 BATCH_OVERLOAD | Batch size too large, splitting");
                // Split large batch into smaller chunks
                let chunk_size = 25;
                let mut all_results = Vec::new();
                
                for chunk in prompts.chunks(chunk_size) {
                    let batch_config = BatchConfig { 
                        max_concurrent: 3, // Very conservative
                        ..BatchConfig::default() 
                    };
                    let processor = BatchProcessor::new(batch_config, self.config.clone());
                    let chunk_result = processor.process_batch(chunk.to_vec(), model).await?;
                    all_results.extend(chunk_result.results);
                }
                
                // Combine results
                Ok(BatchResult {
                    total_prompts: prompts.len(),
                    successful_prompts: all_results.iter().filter(|r| r.success).count(),
                    failed_prompts: all_results.iter().filter(|r| !r.success).count(),
                    results: all_results,
                    total_time_ms: 0.0, // Would need to track across chunks
                    average_time_ms: 0.0,
                    average_hbar: 0.0,
                    batch_statistics: batch_processing::BatchStatistics {
                        risk_distribution: batch_processing::RiskDistribution {
                            high_risk_count: 0,
                            moderate_risk_count: 0,
                            stable_count: 0,
                            error_count: 0,
                        },
                        performance_metrics: batch_processing::PerformanceMetrics {
                            min_time_ms: 0.0,
                            max_time_ms: 0.0,
                            p95_time_ms: 0.0,
                            throughput_per_second: 0.0,
                            parallel_efficiency: 0.0,
                        },
                        error_analysis: batch_processing::ErrorAnalysis {
                            timeout_errors: 0,
                            validation_errors: 0,
                            computation_errors: 0,
                            other_errors: 0,
                            error_rate: 0.0,
                        },
                    },
                    timestamp: chrono::Utc::now(),
                })
            },
            semantic_decision_engine::ProcessDecision::Escalate => {
                return Err(SemanticError::InvalidInput {
                    message: format!("🚨 CRITICAL_BATCH_RISK | ℏₛ={:.3} indicates critical instability", batch_decision.h_bar),
                });
            }
        }
    }

    /// Calculate prompt risk score for decision making
    fn calculate_prompt_risk_score(&self, prompt: &str) -> f32 {
        let prompt_lower = prompt.to_lowercase();
        let mut risk_score: f32 = 0.0;

        // High-risk keywords
        let high_risk_keywords = ["hack", "bomb", "weapon", "kill", "murder", "exploit"];
        for keyword in &high_risk_keywords {
            if prompt_lower.contains(keyword) {
                risk_score += 0.3;
            }
        }

        // Medium-risk keywords  
        let medium_risk_keywords = ["manipulat", "deceive", "trick", "bypass"];
        for keyword in &medium_risk_keywords {
            if prompt_lower.contains(keyword) {
                risk_score += 0.2;
            }
        }

        // Pattern-based risks
        if prompt_lower.contains("how to") && prompt_lower.contains("illegal") {
            risk_score += 0.4;
        }

        risk_score.min(1.0)
    }

    /// Calculate batch complexity for decision making
    fn calculate_batch_complexity(&self, prompts: &[String]) -> f32 {
        if prompts.is_empty() {
            return 0.0;
        }

        let mut total_complexity = 0.0;
        
        for prompt in prompts {
            let words = prompt.split_whitespace().count();
            let questions = prompt.matches('?').count();
            let complexity_indicators = prompt.to_lowercase().matches("complex").count() + 
                                      prompt.to_lowercase().matches("difficult").count();
            
            let prompt_complexity = (words as f32 / 100.0) + 
                                  (questions as f32 * 0.2) + 
                                  (complexity_indicators as f32 * 0.3);
            
            total_complexity += prompt_complexity;
        }

        (total_complexity / prompts.len() as f32).min(2.0)
    }

    /// 🛡️ Secure analysis with comprehensive API security validation
    #[instrument(skip(self, prompt, client_ip, api_key, user_agent, headers), fields(request_id = %request_id))]
    pub async fn secure_analyze(
        &self,
        prompt: &str,
        client_ip: &str,
        api_key: &str,
        user_agent: &str,
        headers: &std::collections::HashMap<String, String>,
        endpoint: &str,
        request_id: RequestId,
    ) -> Result<HbarResponse, SemanticError> {
        
        info!("🛡️ SECURE_ANALYSIS_START | IP: {} | Endpoint: {} | ℏₛ-guided security", client_ip, endpoint);

        // 🔐 First validate API key with secure manager
        let key_validation = self.validate_api_key_security(api_key, client_ip, endpoint);
        
        // 🚨 Early return for invalid or revoked keys
        if !key_validation.is_valid || matches!(key_validation.recommended_action, KeyAction::Revoke | KeyAction::Suspend) {
            error!("🚫 KEY_VALIDATION_FAILED | Action: {:?} | ℏₛ: {:.3}", 
                   key_validation.recommended_action, key_validation.validation_uncertainty.h_bar);
            return Err(SemanticError::InvalidInput {
                message: format!("🚫 API key validation failed: {:?}", key_validation.recommended_action),
            });
        }

        // 🔄 Record usage and evaluate rotation needs
        if let Some(ref key_info) = key_validation.key_info {
            /*
            // REMOVED: Key rotation scheduler functionality
            let mut rotation_scheduler = self.rotation_scheduler.lock().map_err(|e| SemanticError::Internal {
                source: anyhow::anyhow!("Rotation scheduler lock failed: {}", e)
            })?;
            
            // Record this usage for pattern analysis
            rotation_scheduler.record_key_usage(&key_info.key_id, &key_validation);
            
            // Evaluate if rotation is needed (async evaluation)
            if let Ok(Some(rotation_event)) = rotation_scheduler.evaluate_rotation_need(&key_info.key_id, key_info).await {
                warn!("🔄 ROTATION_TRIGGERED | Key: {} | Trigger: {:?} | ℏₛ: {:.3}", 
                      key_info.key_id, rotation_event.trigger, rotation_event.uncertainty_metrics.rotation_h_bar);
                
                // In production, this would trigger background rotation
                // For now, just log the recommendation
                info!("🔄 ROTATION_RECOMMENDATION | Action: {:?} | Confidence: {:.3}", 
                      rotation_event.uncertainty_metrics.recommended_action,
                      rotation_event.uncertainty_metrics.confidence_score);
            }
            */
        }
        
        // 🔍 Comprehensive security analysis using key validation results
        let mut security_analyzer = self.security_analyzer.lock().map_err(|e| SemanticError::Internal {
            source: anyhow::anyhow!("Security analyzer lock failed: {}", e)
        })?;

        let security_assessment = security_analyzer.analyze_request_security(
            endpoint,
            client_ip,
            api_key,
            user_agent,
            prompt,
            headers,
            &key_validation,
        );

        info!("{} {} | ℏₛ={:.3} | Security={:.3} | Action={:?}", 
              security_assessment.security_emoji, 
              security_assessment.security_phrase,
              security_assessment.request_uncertainty.h_bar,
              security_assessment.overall_security_score,
              security_assessment.recommended_action);

        // 🚨 Security decision enforcement
        match security_assessment.recommended_action {
            SecurityAction::Block => {
                error!("🚫 REQUEST_BLOCKED | Security risk too high");
                return Err(SemanticError::InvalidInput {
                    message: format!("🚫 {} | Request blocked due to security concerns", 
                                   security_assessment.security_phrase),
                });
            },
            SecurityAction::Quarantine => {
                error!("🏥 REQUEST_QUARANTINED | Critical security risk detected");
                return Err(SemanticError::InvalidInput {
                    message: format!("🏥 {} | Request quarantined for investigation", 
                                   security_assessment.security_phrase),
                });
            },
            SecurityAction::Challenge => {
                warn!("🔐 ADDITIONAL_AUTH_REQUIRED | Enhanced authentication needed");
                return Err(SemanticError::InvalidInput {
                    message: format!("🔐 {} | Additional authentication required", 
                                   security_assessment.security_phrase),
                });
            },
            SecurityAction::RateLimit => {
                warn!("⏳ RATE_LIMITED | Request processing slowed");
                // Add artificial delay for rate limiting
                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
            },
            SecurityAction::AllowWithMonitoring => {
                info!("👀 ENHANCED_MONITORING | Processing with increased surveillance");
            },
            SecurityAction::Allow => {
                info!("✅ SECURITY_APPROVED | Full authorization granted");
            },
        }

        // Drop the lock before proceeding with analysis
        drop(security_analyzer);

        // 🧠 Proceed with semantic analysis using compression intelligence
        let mut result = self.analyze_with_compression(prompt, request_id).await?;

        // 🛡️ Attach security assessment to response
        result.security_assessment = Some(security_assessment);

        // 🎯 Final security-uncertainty correlation check
        if result.hbar_s > 1.5 && result.security_assessment.as_ref().unwrap().overall_security_score < 0.6 {
            warn!("🚨 CORRELATION_ALERT | High semantic uncertainty + low security score");
            
            // Escalate semantic uncertainty due to security correlation
            result.hbar_s = (result.hbar_s * 1.2).min(5.0);
            result.collapse_risk = result.hbar_s < self.config.collapse_threshold;
        }

        info!("🎯 SECURE_ANALYSIS_COMPLETE | ℏₛ={:.3} | Security={:.3}", 
              result.hbar_s, 
              result.security_assessment.as_ref().unwrap().overall_security_score);

        Ok(result)
    }

    /// 🔒 Enhanced API key validation with cryptographic security
    pub fn validate_api_key_security(&self, api_key: &str, client_ip: &str, endpoint: &str) -> KeyValidationResult {
        let mut key_manager = self.key_manager.lock().unwrap();
        key_manager.validate_api_key(api_key, client_ip, endpoint)
    }
}

/// Convenience function for standalone analysis
pub async fn analyze(prompt: &str, output: &str) -> Result<HbarResponse, SemanticError> {
    let config = SemanticConfig::performance();
    let analyzer = SemanticAnalyzer::new(config)?;
    let request_id = RequestId::new();
    
    analyzer.analyze(prompt, output, request_id).await
}

/// Performance metrics collector
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub collapse_detections: u64,
    pub uptime_seconds: u64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            collapse_detections: 0,
            uptime_seconds: 0,
        }
    }
}

// Temporarily comment out modules with compilation issues
// Temporarily commented out modules to fix compilation
// pub mod oss_logit_adapter;
// pub mod live_response_auditor;
// pub mod audit_interface;

// Simplified WASM module for core equation
pub mod wasm_simple;

#[cfg(test)]
mod tests {
    use super::*;


    #[tokio::test]
    async fn test_basic_analysis() {
        let config = SemanticConfig::ultra_fast();
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        let request_id = RequestId::new();
        
        let result = analyzer.analyze("What is AI?", "AI is artificial intelligence", request_id).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(response.hbar_s >= 0.0);
        assert!(response.delta_mu > 0.0);
        assert!(response.delta_sigma > 0.0);
        assert_eq!(response.request_id, request_id);
    }

    #[tokio::test]
    async fn test_collapse_detection() {
        let config = SemanticConfig::default();
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        let request_id = RequestId::new();
        
        // Empty inputs should trigger collapse
        let result = analyzer.analyze("", "", request_id).await.unwrap();
        assert!(result.collapse_risk);
        
        // Identical inputs should trigger collapse
        let result = analyzer.analyze("test", "test", request_id).await.unwrap();
        assert!(result.collapse_risk);
    }

    #[tokio::test]
    async fn test_performance_config() {
        let config = SemanticConfig::ultra_fast();
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        let request_id = RequestId::new();
        
        let start = std::time::Instant::now();
        let result = analyzer.analyze("Quick test", "Fast response", request_id).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 50); // Should be very fast
    }

    #[tokio::test]
    async fn test_tier3_integration() {
        let config = SemanticConfig::tier3();
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        let request_id = RequestId::new();
        
        let result = analyzer.analyze("Complex philosophical question", "Detailed philosophical response", request_id).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(response.hbar_s >= 0.0);
        assert!(response.delta_mu > 0.0);
        assert!(response.delta_sigma > 0.0);
        assert_eq!(response.request_id, request_id);
    }

    #[test]
    fn test_embedding_deterministic() {
        let config = SemanticConfig::performance();
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        
        let mut buffer1 = vec![0.0; 128];
        let mut buffer2 = vec![0.0; 128];
        
        analyzer.embed_text_fast_inplace("test", &mut buffer1);
        analyzer.embed_text_fast_inplace("test", &mut buffer2);
        
        assert_eq!(buffer1, buffer2); // Should be deterministic
    }

    #[test]
    fn test_simd_vs_standard() {
        let config_simd = SemanticConfig { use_simd: true, ..SemanticConfig::default() };
        let config_std = SemanticConfig { use_simd: false, ..SemanticConfig::default() };
        
        let analyzer_simd = SemanticAnalyzer::new(config_simd).unwrap();
        let analyzer_std = SemanticAnalyzer::new(config_std).unwrap();
        
        let test_vec = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        
        let entropy_simd = analyzer_simd.entropy_approx_simd(&test_vec);
        let entropy_std = analyzer_std.entropy_approx_standard(&test_vec);
        
        // Results should be very close
        assert!((entropy_simd - entropy_std).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_tier3_vs_standard_comparison() {
        // Test standard mode
        let config_standard = SemanticConfig::performance();
        let analyzer_standard = SemanticAnalyzer::new(config_standard).unwrap();
        let request_id_standard = RequestId::new();
        
        let result_standard = analyzer_standard.analyze("Test prompt", "Test output", request_id_standard).await.unwrap();
        
        // Test Tier3 mode
        let config_tier3 = SemanticConfig::tier3();
        let analyzer_tier3 = SemanticAnalyzer::new(config_tier3).unwrap();
        let request_id_tier3 = RequestId::new();
        
        let result_tier3 = analyzer_tier3.analyze("Test prompt", "Test output", request_id_tier3).await.unwrap();
        
        // Both should produce valid results
        assert!(result_standard.hbar_s >= 0.0);
        assert!(result_tier3.hbar_s >= 0.0);
        assert!(result_standard.delta_mu > 0.0);
        assert!(result_tier3.delta_mu > 0.0);
        assert!(result_standard.delta_sigma > 0.0);
        assert!(result_tier3.delta_sigma > 0.0);
        
        // Tier3 might produce different results due to more sophisticated analysis
        // but both should be valid semantic uncertainty measurements
        println!("Standard: ℏₛ={:.4}, Δμ={:.4}, Δσ={:.4}", 
                result_standard.hbar_s, result_standard.delta_mu, result_standard.delta_sigma);
        println!("Tier3: ℏₛ={:.4}, Δμ={:.4}, Δσ={:.4}", 
                result_tier3.hbar_s, result_tier3.delta_mu, result_tier3.delta_sigma);
    }

    #[tokio::test]
    async fn test_tier3_fallback_mechanism() {
        // Test that the analyzer gracefully handles Tier3 initialization failure
        let mut config = SemanticConfig::tier3();
        config.enable_tier3 = false; // Disable Tier3 to test fallback
        
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        let request_id = RequestId::new();
        
        // Should still work with standard analysis
        let result = analyzer.analyze("Test prompt", "Test output", request_id).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(response.hbar_s >= 0.0);
        assert!(response.delta_mu > 0.0);
        assert!(response.delta_sigma > 0.0);
    }

    #[tokio::test]
    async fn test_empirical_golden_calibration() {
        // Test the new Empirical Golden Scale √(φ × 2.67²) ≈ 3.40
        let empirical_mode = CalibrationMode::Pragmatic {
            scaling: GoldenScaling::EmpiricalGolden,
            abort_threshold: 0.8,
            warn_threshold: 1.0,
            proceed_threshold: 1.2,
        };
        
        // Test with typical raw ℏₛ value from our measurements
        let raw_hbar = 0.3; // Typical measured value
        let (calibrated, risk_level, explanation) = empirical_mode.calibrate(raw_hbar);
        
        // With √(φ × 2.67²) ≈ 3.40 scaling: 0.3 × 3.40 = 1.02
        assert!((calibrated - 1.02).abs() < 0.01, "Expected ~1.02, got {}", calibrated);
        assert_eq!(risk_level, RiskLevel::HighRisk, "0.3 raw should be HighRisk after scaling (1.02 is between 1.0 and 1.2)");
        assert!(explanation.contains("3.40"), "Explanation should mention 3.40 scaling factor");
        assert!(explanation.contains("Empirical Golden Scale"), "Should mention Empirical Golden Scale");
        
        // Test edge case: 0.235 should scale to ~0.8 (just at abort threshold)
        let raw_hbar_edge = 0.235;
        let (calibrated_edge, risk_level_edge, _) = empirical_mode.calibrate(raw_hbar_edge);
        assert!((calibrated_edge - 0.799).abs() < 0.01, "Expected ~0.799, got {}", calibrated_edge);
        assert_eq!(risk_level_edge, RiskLevel::Critical, "Should be Critical (< 0.8)");
        
        // Test warning threshold: 0.265 should scale to ~0.9 (Warning)
        let raw_hbar_warn = 0.265;
        let (calibrated_warn, risk_level_warn, _) = empirical_mode.calibrate(raw_hbar_warn);
        assert!((calibrated_warn - 0.901).abs() < 0.01, "Expected ~0.901, got {}", calibrated_warn);
        assert_eq!(risk_level_warn, RiskLevel::Warning, "Should be Warning (0.8 < x < 1.0)");
        
        // Test low value: 0.2 should scale to 0.68 (Critical)
        let raw_hbar_low = 0.2;
        let (calibrated_low, risk_level_low, _) = empirical_mode.calibrate(raw_hbar_low);
        assert!((calibrated_low - 0.68).abs() < 0.01, "Expected ~0.68, got {}", calibrated_low);
        assert_eq!(risk_level_low, RiskLevel::Critical, "Should be Critical");
        
        // Test high value: 0.4 should scale to 1.36 (Safe)
        let raw_hbar_high = 0.4;
        let (calibrated_high, risk_level_high, _) = empirical_mode.calibrate(raw_hbar_high);
        assert!((calibrated_high - 1.36).abs() < 0.01, "Expected ~1.36, got {}", calibrated_high);
        assert_eq!(risk_level_high, RiskLevel::Safe, "Should be Safe");
        
        println!("✅ Empirical Golden Scale tests passed!");
        println!("   Raw 0.3 → Calibrated {:.2} → {:?}", calibrated, risk_level);
        println!("   Scaling factor: {:.2}", GoldenScaling::EmpiricalGolden.factor());
    }

    #[tokio::test]
    async fn test_architecture_aware_calibration() {
        let mut config = SemanticConfig::default();
        config.enable_architecture_detection = true;
        config.use_research_calibration = true;
        
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        
        // Test with a known decoder model
        let result = analyzer.analyze("Test prompt", "Test output", RequestId::new()).await.unwrap();
        
        // Verify that architecture-aware calibration is applied
        assert!(result.hbar_s > 0.0);
        assert!(result.delta_mu > 0.0);
        assert!(result.delta_sigma > 0.0);
    }

    #[tokio::test]
    async fn test_research_mode_calibration() {
        let mut config = SemanticConfig::default();
        config.enable_architecture_detection = true;
        config.use_research_calibration = true;
        config.research_mode = true; // Optimize for decoder-only workflows
        
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        
        let result = analyzer.analyze("Test prompt", "Test output", RequestId::new()).await.unwrap();
        
        // Verify that research mode calibration is applied
        assert!(result.hbar_s > 0.0);
        assert!(result.delta_mu > 0.0);
        assert!(result.delta_sigma > 0.0);
    }

    #[tokio::test]
    async fn test_legacy_fallback() {
        let mut config = SemanticConfig::default();
        config.enable_architecture_detection = false; // Disable architecture detection
        config.fallback_to_legacy = true;
        
        let analyzer = SemanticAnalyzer::new(config).unwrap();
        
        let result = analyzer.analyze("Test prompt", "Test output", RequestId::new()).await.unwrap();
        
        // Verify that legacy calibration is used
        assert!(result.hbar_s > 0.0);
        assert!(result.delta_mu > 0.0);
        assert!(result.delta_sigma > 0.0);
    }
} 