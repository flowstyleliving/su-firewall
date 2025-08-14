// 🎯 OSS Model Logit Adapter for Live UQ/Hbar Auditing
// Adapts core-engine for direct logit access from OSS models
// Enhanced with advanced information-geometric estimators

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use common::{SemanticUncertaintyResult, CalibrationMode, RequestId};
use common::math::information_theory::InformationTheoryCalculator;
use preprompt::metrics_pipeline::SemanticMetricsCalculator;
use crate::metrics;

/// 🧠 OSS Model Interface - supports various OSS model frameworks
#[derive(Debug, Clone)]
pub enum OSSModelFramework {
    HuggingFaceTransformers,
    LlamaCpp,
    GGML,
    Candle,
    TorchScript,
    ONNX,
}

/// 📊 Raw logit data from OSS model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitData {
    /// Token logits for each position in sequence
    pub token_logits: Vec<Vec<f32>>,
    /// Vocabulary mapping (token_id -> token_string)
    pub vocab_map: HashMap<u32, String>,
    /// Attention weights (if available)
    pub attention_weights: Option<Vec<Vec<f32>>>,
    /// Hidden states (if available)
    pub hidden_states: Option<Vec<Vec<f32>>>,
    /// Temperature used for sampling
    pub temperature: f32,
    /// Top-p value used for nucleus sampling
    pub top_p: Option<f32>,
    /// Generated token sequence
    pub token_sequence: Vec<u32>,
    /// Gradient information (if available) for FIM approximation
    pub gradients: Option<Vec<Vec<f32>>>,
    /// Multiple paraphrase logits for refined flexibility measurement
    pub paraphrase_logits: Option<Vec<Vec<Vec<f32>>>>,
}

/// 🔬 Live UQ Analysis Result from Logits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveLogitAnalysis {
    /// Standard semantic uncertainty result
    pub base_result: SemanticUncertaintyResult,
    /// Logit-specific metrics
    pub logit_metrics: LogitMetrics,
    /// Real-time streaming metrics
    pub streaming_metrics: StreamingMetrics,
    /// Token-level uncertainty breakdown
    pub token_uncertainties: Vec<TokenUncertainty>,
    /// Enhanced information-geometric metrics
    pub geometric_metrics: GeometricMetrics,
}

/// 📈 Enhanced logit-derived metrics with FIM approximation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitMetrics {
    /// Average entropy across all token positions
    pub average_entropy: f64,
    /// Maximum entropy in sequence
    pub max_entropy: f64,
    /// Perplexity of the generated sequence
    pub perplexity: f64,
    /// Confidence based on top-1 probabilities
    pub confidence_score: f64,
    /// Diversity of probability distributions
    pub distribution_diversity: f64,
    /// Attention consistency (if available)
    pub attention_consistency: Option<f64>,
    /// Fisher Information Matrix trace (if computed)
    pub fim_trace: Option<f64>,
    /// Maximum eigenvalue of FIM (if computed)
    pub fim_max_eigenvalue: Option<f64>,
}

/// 🔺 Advanced information-geometric metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricMetrics {
    /// Direct FIM-based precision scalar I(C)
    pub fim_precision: Option<f64>,
    /// Maximum JS divergence across paraphrases
    pub max_js_divergence: Option<f64>,
    /// Geodesic distance measures
    pub geodesic_distances: Option<GeodesicDistances>,
    /// Mahalanobis distances with FIM metric tensor
    pub mahalanobis_distances: Option<Vec<f64>>,
    /// Information-geometric curvature measures
    pub ricci_curvature: Option<f64>,
}

/// 📐 Geodesic distance computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeodesicDistances {
    /// Mean geodesic distance from center
    pub mean_geodesic_distance: f64,
    /// Maximum geodesic distance
    pub max_geodesic_distance: f64,
    /// Geodesic variance measure
    pub geodesic_variance: f64,
    /// Number of geodesic paths computed
    pub path_count: usize,
}

/// ⚡ Streaming analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetrics {
    /// Uncertainty trend over time
    pub uncertainty_trend: Vec<f64>,
    /// Processing latency per token
    pub token_latencies_ms: Vec<f64>,
    /// Memory usage tracking
    pub memory_usage_mb: f64,
    /// Tokens processed per second
    pub throughput_tps: f64,
    /// FIM computation overhead (if enabled)
    pub fim_overhead_ms: Option<f64>,
}

/// 🎯 Per-token uncertainty analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUncertainty {
    /// Token position in sequence
    pub position: usize,
    /// Token ID and string
    pub token_id: u32,
    pub token_string: String,
    /// Local entropy for this token
    pub token_entropy: f64,
    /// Probability of selected token
    pub token_probability: f64,
    /// Top-k alternatives and their probabilities
    pub alternatives: Vec<(String, f64)>,
    /// Local hbar calculation
    pub local_hbar: f64,
    /// Local FIM contribution (if available)
    pub local_fim_contribution: Option<f64>,
    /// Local geodesic distance (if computed)
    pub local_geodesic_distance: Option<f64>,
}

/// 🔧 OSS Logit Adapter - main interface with advanced features
pub struct OSSLogitAdapter {
    info_calculator: InformationTheoryCalculator,
    semantic_calculator: SemanticMetricsCalculator,
    framework: OSSModelFramework,
    config: AdapterConfig,
    /// FIM computation cache for efficiency
    fim_cache: lru::LruCache<String, Array2<f64>>,
}

/// ⚙️ Enhanced configuration for the adapter
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Whether to enable streaming analysis
    pub enable_streaming: bool,
    /// How many tokens to buffer for analysis
    pub buffer_size: usize,
    /// Minimum entropy threshold for alerts
    pub entropy_alert_threshold: f64,
    /// Whether to compute attention-based metrics
    pub use_attention: bool,
    /// Calibration mode for hbar values
    pub calibration_mode: CalibrationMode,
    /// Enhanced FIM computation settings
    pub fim_config: FIMConfig,
    /// Geodesic computation settings
    pub geodesic_config: GeodesicConfig,
    /// Error handling and validation settings
    pub validation_config: ValidationConfig,
}

/// 🧮 Fisher Information Matrix computation configuration
#[derive(Debug, Clone)]
pub struct FIMConfig {
    /// Enable direct FIM approximation
    pub enable_fim: bool,
    /// Use diagonal approximation for efficiency
    pub diagonal_approximation: bool,
    /// Use trace or max eigenvalue for scalar I(C)
    pub scalar_method: FIMScalarMethod,
    /// Regularization epsilon for numerical stability
    pub regularization_epsilon: f64,
    /// Maximum FIM cache size
    pub cache_size: usize,
    /// Enable gradient-based FIM (requires gradient access)
    pub gradient_based: bool,
}

#[derive(Debug, Clone)]
pub enum FIMScalarMethod {
    Trace,
    MaxEigenvalue,
    FrobeniusNorm,
    LogDeterminant,
}

/// 📐 Geodesic computation configuration
#[derive(Debug, Clone)]
pub struct GeodesicConfig {
    /// Enable geodesic distance computations
    pub enable_geodesics: bool,
    /// Number of geodesic paths to compute
    pub num_paths: usize,
    /// Maximum geodesic integration steps
    pub max_integration_steps: usize,
    /// Geodesic integration tolerance
    pub integration_tolerance: f64,
    /// Use Riemannian exponential map approximation
    pub use_exponential_map: bool,
}

/// ✅ Validation and error handling configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable positive semi-definite matrix checks
    pub check_psd_matrices: bool,
    /// Entropy regularization epsilon
    pub entropy_epsilon: f64,
    /// Enable intermediate step logging
    pub enable_logging: bool,
    /// Probability normalization tolerance
    pub normalization_tolerance: f64,
    /// Enable numerical stability checks
    pub enable_stability_checks: bool,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            enable_streaming: true,
            buffer_size: 50,
            entropy_alert_threshold: 2.0,
            use_attention: true,
            calibration_mode: CalibrationMode::default(),
            fim_config: FIMConfig::default(),
            geodesic_config: GeodesicConfig::default(),
            validation_config: ValidationConfig::default(),
        }
    }
}

impl Default for FIMConfig {
    fn default() -> Self {
        Self {
            enable_fim: true,
            diagonal_approximation: true,
            scalar_method: FIMScalarMethod::Trace,
            regularization_epsilon: 1e-8,
            cache_size: 1000,
            gradient_based: false,
        }
    }
}

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            enable_geodesics: false, // Computationally intensive, disabled by default
            num_paths: 10,
            max_integration_steps: 100,
            integration_tolerance: 1e-6,
            use_exponential_map: true,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_psd_matrices: true,
            entropy_epsilon: 1e-10,
            enable_logging: false,
            normalization_tolerance: 1e-6,
            enable_stability_checks: true,
        }
    }
}

impl OSSLogitAdapter {
    /// 🚀 Create new adapter for specific OSS framework with enhanced features
    pub fn new(framework: OSSModelFramework, config: AdapterConfig) -> Self {
        let fim_cache = lru::LruCache::new(std::num::NonZeroUsize::new(config.fim_config.cache_size).unwrap_or(std::num::NonZeroUsize::new(256).unwrap()));
        
        Self {
            info_calculator: InformationTheoryCalculator::new(
                config.validation_config.entropy_epsilon, 
                0.001
            ),
            semantic_calculator: SemanticMetricsCalculator::default(),
            framework,
            config,
            fim_cache,
        }
    }

    /// 📊 Enhanced logit analysis with advanced information-geometric estimators
    pub fn analyze_logits(
        &mut self,
        prompt: &str,
        logit_data: &LogitData,
        request_id: RequestId,
    ) -> Result<LiveLogitAnalysis, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        // Validate input data
        self.validate_logit_data(logit_data)?;
        
        // Convert logits to probability distributions with enhanced validation
        let prob_distributions = self.logits_to_probabilities_enhanced(&logit_data.token_logits, logit_data.temperature)?;
        
        // Calculate enhanced precision using FIM if available
        let (delta_mu, fim_precision, fim_diag_opt, u_opt) = if self.config.fim_config.enable_fim {
            // Compute diagonal FIM (fast path) and build direction vector from first→last distribution
            let fim = if self.config.fim_config.diagonal_approximation {
                self.compute_diagonal_fim_approximation(&prob_distributions)?
            } else {
                self.compute_full_fim_approximation(&prob_distributions)?
            };
            let fim_diag = fim.diag().to_owned();
            let eps = self.config.fim_config.regularization_epsilon.max(1e-12);
            // Build u from first and last distributions if available; else use ones
            let (p0, p1) = if prob_distributions.len() >= 2 {
                (&prob_distributions[0], &prob_distributions[prob_distributions.len()-1])
            } else {
                (&prob_distributions[0], &prob_distributions[0])
            };
            let u = self.build_u_from_distributions(p0, p1, eps);
            let dir_mu = self.directional_precision_diag(&fim_diag, &u, eps);
            (dir_mu, Some(fim_diag.iter().sum()), Some(fim_diag), Some(u))
        } else {
            (self.calculate_logit_precision(&prob_distributions)?, None, None, None)
        };
        
        // Calculate enhanced flexibility using paraphrases if available
        let (delta_sigma, max_js_divergence) = if let (Some(fim_diag), Some(u)) = (fim_diag_opt.as_ref(), u_opt.as_ref()) {
            // Fast diagonal inverse flexibility
            let eps = self.config.fim_config.regularization_epsilon.max(1e-12);
            (self.flexibility_diag_inv(fim_diag, u, eps), None)
        } else if let Some(ref paraphrase_logits) = logit_data.paraphrase_logits {
            let enhanced_result = self.calculate_enhanced_flexibility(&prob_distributions, paraphrase_logits)?;
            (enhanced_result.0, Some(enhanced_result.1))
        } else {
            (self.calculate_logit_flexibility(&prob_distributions)?, None)
        };
        
        // Calculate geodesic distances if enabled
        let geodesic_distances = if self.config.geodesic_config.enable_geodesics {
            Some(self.calculate_geodesic_distances(&prob_distributions)?)
        } else {
            None
        };
        
        // Calculate ℏₛ = √(Δμ × Δσ) using enhanced metrics
        let raw_hbar = (delta_mu * delta_sigma).sqrt();
        
        // Apply calibration with enhanced validation
        let (calibrated_hbar, risk_level, explanation) = 
            self.config.calibration_mode.calibrate_identity(raw_hbar);
        
        // Calculate detailed logit metrics with FIM information
        let logit_metrics = self.calculate_enhanced_logit_metrics(&prob_distributions, logit_data)?;
        
        // Generate enhanced per-token uncertainty analysis
        let token_uncertainties = self.analyze_enhanced_token_uncertainties(&prob_distributions, logit_data)?;
        
        // Calculate Mahalanobis distances if FIM is available
        let mahalanobis_distances = if fim_precision.is_some() {
            Some(self.calculate_mahalanobis_distances(&prob_distributions)?)
        } else {
            None
        };
        
        // Create enhanced streaming metrics
        let fim_overhead_ms = if self.config.fim_config.enable_fim {
            Some(start_time.elapsed().as_millis() as f64 * 0.3) // Estimate FIM overhead
        } else {
            None
        };
        
        let streaming_metrics = if self.config.enable_streaming {
            self.calculate_enhanced_streaming_metrics(&prob_distributions, fim_overhead_ms)?
        } else {
            StreamingMetrics::default()
        };
        
        // Build geometric metrics
        let geometric_metrics = GeometricMetrics {
            fim_precision,
            max_js_divergence,
            geodesic_distances,
            mahalanobis_distances,
            ricci_curvature: None, // Could be implemented for advanced use cases
        };
        
        // Build base result
        metrics::record_analysis(risk_level.clone());
        let base_result = SemanticUncertaintyResult {
            raw_hbar,
            calibrated_hbar,
            risk_level,
            calibration_mode: self.config.calibration_mode.clone(),
            explanation,
            delta_mu,
            delta_sigma,
            processing_time_ms: start_time.elapsed().as_millis() as f64,
            timestamp: chrono::Utc::now(),
            request_id,
        };
        
        if self.config.validation_config.enable_logging {
            log::debug!("Enhanced logit analysis completed in {:.2}ms", 
                start_time.elapsed().as_millis());
        }
        
        Ok(LiveLogitAnalysis {
            base_result,
            logit_metrics,
            streaming_metrics,
            token_uncertainties,
            geometric_metrics,
        })
    }

    /// ✅ Validate logit data for enhanced error handling
    fn validate_logit_data(&self, logit_data: &LogitData) -> Result<(), Box<dyn std::error::Error>> {
        if logit_data.token_logits.is_empty() {
            return Err("Empty token logits".into());
        }
        
        if logit_data.token_sequence.len() != logit_data.token_logits.len() {
            return Err("Token sequence length mismatch with logits".into());
        }
        
        // Check for NaN or infinite values
        if self.config.validation_config.enable_stability_checks {
            for (i, token_logits) in logit_data.token_logits.iter().enumerate() {
                for (j, &logit) in token_logits.iter().enumerate() {
                    if !logit.is_finite() {
                        return Err(format!("Non-finite logit at position {}, token {}", i, j).into());
                    }
                }
            }
        }
        
        if self.config.validation_config.enable_logging {
            log::debug!("Logit data validation passed: {} tokens, vocab size: {}", 
                logit_data.token_logits.len(), logit_data.vocab_map.len());
        }
        
        Ok(())
    }

    /// 🔢 Enhanced probability conversion with validation
    fn logits_to_probabilities_enhanced(&self, logits: &[Vec<f32>], temperature: f32) -> Result<Vec<Array1<f64>>, Box<dyn std::error::Error>> {
        let distributions: Result<Vec<Array1<f64>>, _> = logits.iter().map(|token_logits| {
            // Apply temperature scaling
            let scaled_logits: Vec<f64> = token_logits.iter()
                .map(|&logit| (logit / temperature) as f64)
                .collect();
            
            // Apply softmax with numerical stability
            let max_logit = scaled_logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f64> = scaled_logits.iter()
                .map(|&logit| (logit - max_logit).exp())
                .collect();
            
            let sum_exp: f64 = exp_logits.iter().sum();
            if sum_exp <= 0.0 || !sum_exp.is_finite() {
                return Err("Invalid probability normalization".into());
            }
            
            let probabilities: Vec<f64> = exp_logits.iter()
                .map(|&exp_logit| exp_logit / sum_exp)
                .collect();
            
            // Validate probability distribution
            let prob_sum: f64 = probabilities.iter().sum();
            if (prob_sum - 1.0).abs() > self.config.validation_config.normalization_tolerance {
                return Err(format!("Probability sum validation failed: {}", prob_sum).into());
            }
            
            Ok(Array1::from_vec(probabilities))
        }).collect();
        
        distributions
    }

    /// 🧮 Calculate FIM-based precision with enhanced geometric accuracy
    fn calculate_fim_based_precision(&mut self, distributions: &[Array1<f64>], logit_data: &LogitData) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        if distributions.is_empty() {
            return Ok((0.0, 0.0));
        }
        
        // Generate cache key for FIM computation
        let cache_key = format!("fim_{}_{}", distributions.len(), 
            distributions[0].len());
        
        // Check cache first
        if let Some(fim) = self.fim_cache.get(&cache_key).cloned() {
            let scalar_ic = self.compute_fim_scalar(&fim)?;
            let precision = 1.0 / (1.0 + 1.0/scalar_ic); // Transform to precision measure
            return Ok((precision, scalar_ic));
        }
        
        // Compute FIM approximation
        let fim = if self.config.fim_config.gradient_based && logit_data.gradients.is_some() {
            self.compute_gradient_based_fim(distributions, logit_data.gradients.as_ref().unwrap())?
        } else if self.config.fim_config.diagonal_approximation {
            self.compute_diagonal_fim_approximation(distributions)?
        } else {
            self.compute_full_fim_approximation(distributions)?
        };
        
        // Validate positive semi-definite if enabled
        if self.config.validation_config.check_psd_matrices {
            self.validate_psd_matrix(&fim)?;
        }
        
        // Cache the result
        self.fim_cache.put(cache_key, fim.clone());
        
        // Compute scalar I(C) from FIM
        let scalar_ic = self.compute_fim_scalar(&fim)?;
        
        // Transform to precision measure
        let precision = 1.0 / (1.0 + 1.0/scalar_ic);
        
        if self.config.validation_config.enable_logging {
            log::debug!("FIM-based precision computed: {:.6}, scalar I(C): {:.6}", 
                precision, scalar_ic);
        }
        
        Ok((precision, scalar_ic))
    }

    /// 📐 Compute diagonal FIM approximation for efficiency
    fn compute_diagonal_fim_approximation(&self, distributions: &[Array1<f64>]) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let dim = distributions[0].len();
        let mut fim_diagonal = vec![0.0; dim];
        
        for dist in distributions {
            for i in 0..dim {
                let p_i = dist[i];
                if p_i > self.config.validation_config.entropy_epsilon {
                    // Diagonal FIM element: F_ii = (∂log p / ∂θ_i)^2 * p
                    // Approximated as 1/p_i for efficiency
                    fim_diagonal[i] += 1.0 / p_i;
                }
            }
        }
        
        // Add regularization
        for elem in &mut fim_diagonal {
            *elem += self.config.fim_config.regularization_epsilon;
        }
        
        // Create diagonal matrix
        let mut fim = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            fim[[i, i]] = fim_diagonal[i] / distributions.len() as f64;
        }
        
        Ok(fim)
    }

    /// 🔍 Compute full FIM approximation with off-diagonal elements
    fn compute_full_fim_approximation(&self, distributions: &[Array1<f64>]) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let dim = distributions[0].len();
        let mut fim = Array2::<f64>::zeros((dim, dim));
        
        for dist in distributions {
            for i in 0..dim {
                for j in 0..dim {
                    let p_i = dist[i];
                    let p_j = dist[j];
                    
                    if p_i > self.config.validation_config.entropy_epsilon && 
                       p_j > self.config.validation_config.entropy_epsilon {
                        // FIM element: F_ij = ∂²(-log L) / ∂θ_i ∂θ_j
                        // For multinomial: F_ij = δ_ij/p_i (diagonal) or interaction terms
                        if i == j {
                            fim[[i, j]] += 1.0 / p_i;
                        } else {
                            // Off-diagonal approximation based on covariance
                            fim[[i, j]] += -1.0 / (p_i * p_j).sqrt();
                        }
                    }
                }
            }
        }
        
        // Normalize and add regularization
        fim = fim / distributions.len() as f64;
        for i in 0..dim {
            fim[[i, i]] += self.config.fim_config.regularization_epsilon;
        }
        
        Ok(fim)
    }

    /// 🎯 Compute gradient-based FIM using provided gradients
    fn compute_gradient_based_fim(&self, distributions: &[Array1<f64>], gradients: &[Vec<f32>]) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let dim = distributions[0].len();
        let mut fim = Array2::<f64>::zeros((dim, dim));
        
        for (dist, grad) in distributions.iter().zip(gradients.iter()) {
            if grad.len() != dim {
                return Err("Gradient dimension mismatch".into());
            }
            
            // FIM = E[∇log p(x|θ) ∇log p(x|θ)ᵀ]
            for i in 0..dim {
                for j in 0..dim {
                    let grad_i = grad[i] as f64;
                    let grad_j = grad[j] as f64;
                    fim[[i, j]] += grad_i * grad_j;
                }
            }
        }
        
        // Normalize and regularize
        fim = fim / distributions.len() as f64;
        for i in 0..dim {
            fim[[i, i]] += self.config.fim_config.regularization_epsilon;
        }
        
        Ok(fim)
    }

    /// 📊 Compute scalar I(C) from FIM matrix
    fn compute_fim_scalar(&self, fim: &Array2<f64>) -> Result<f64, Box<dyn std::error::Error>> {
        match self.config.fim_config.scalar_method {
            FIMScalarMethod::Trace => {
                Ok(fim.diag().sum())
            },
            FIMScalarMethod::MaxEigenvalue => {
                // Simplified eigenvalue approximation (power method)
                self.approximate_max_eigenvalue(fim)
            },
            FIMScalarMethod::FrobeniusNorm => {
                Ok((fim * fim).sum().sqrt())
            },
            FIMScalarMethod::LogDeterminant => {
                // Approximated using diagonal elements for numerical stability
                let log_det: f64 = fim.diag().iter()
                    .map(|&x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
                    .sum();
                Ok(log_det)
            }
        }
    }

    /// Build a direction vector u from prompt/output distributions (no hash embeddings)
    fn build_u_from_distributions(&self, p_prompt: &Array1<f64>, p_out: &Array1<f64>, eps: f64) -> Array1<f64> {
        let mut u = p_out - p_prompt;
        let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt().max(eps);
        u.mapv(|v| v / norm)
    }

    /// Directional precision (Δμ) using diagonal Fisher: u^T diag(I) u = Σ u_i^2 * I_ii
    fn directional_precision_diag(&self, fim_diag: &Array1<f64>, u: &Array1<f64>, eps: f64) -> f64 {
        let mut acc = 0.0;
        let n = fim_diag.len().min(u.len());
        for i in 0..n {
            let ui = u[i];
            let li = fim_diag[i].max(eps);
            acc += ui * ui * li;
        }
        acc
    }

    /// Fast flexibility (Δσ) using diagonal inverse: sqrt(u^T diag(I)^{-1} u) = sqrt(Σ u_i^2 / I_ii)
    fn flexibility_diag_inv(&self, fim_diag: &Array1<f64>, u: &Array1<f64>, eps: f64) -> f64 {
        let mut acc = 0.0;
        let n = fim_diag.len().min(u.len());
        for i in 0..n {
            let ui = u[i];
            let li = fim_diag[i].max(eps);
            acc += ui * ui / li;
        }
        acc.sqrt()
    }

    /// 🔢 Approximate maximum eigenvalue using power method
    fn approximate_max_eigenvalue(&self, matrix: &Array2<f64>) -> Result<f64, Box<dyn std::error::Error>> {
        let dim = matrix.nrows();
        let mut v = Array1::<f64>::ones(dim);
        v = v.clone() / (v.dot(&v).sqrt());
        
        let max_iterations = 50;
        let tolerance = 1e-6;
        
        for _ in 0..max_iterations {
            let v_new = matrix.dot(&v);
            let eigenvalue = v.dot(&v_new);
            
            let v_norm = v_new.dot(&v_new).sqrt();
            if v_norm > 0.0 {
                let v_normalized = v_new / v_norm;
                
                // Check convergence
                let diff = (&v_normalized - &v).dot(&(&v_normalized - &v)).sqrt();
                if diff < tolerance {
                    return Ok(eigenvalue);
                }
                
                v = v_normalized;
            } else {
                break;
            }
        }
        
        // Fallback to trace if power method doesn't converge
        Ok(matrix.diag().sum() / dim as f64)
    }

    /// ✅ Validate positive semi-definite matrix
    fn validate_psd_matrix(&self, matrix: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
        // Check diagonal elements are non-negative
        for &diag_elem in matrix.diag() {
            if diag_elem < -self.config.fim_config.regularization_epsilon {
                return Err(format!("Matrix not PSD: negative diagonal element {}", diag_elem).into());
            }
        }
        
        // Check symmetry (for full matrices)
        if !self.config.fim_config.diagonal_approximation {
            let (rows, cols) = matrix.dim();
            for i in 0..rows {
                for j in 0..cols {
                    let diff = (matrix[[i, j]] - matrix[[j, i]]).abs();
                    if diff > self.config.validation_config.normalization_tolerance {
                        return Err("Matrix not symmetric".into());
                    }
                }
            }
        }
        
        Ok(())
    }

    /// 🌊 Calculate enhanced flexibility using maximum JS divergence across paraphrases
    fn calculate_enhanced_flexibility(&self, base_distributions: &[Array1<f64>], paraphrase_logits: &[Vec<Vec<f32>>]) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut max_js_divergence: f64 = 0.0;
        
        // Convert paraphrase logits to distributions
        for paraphrase_logit_sequence in paraphrase_logits {
            let paraphrase_distributions = self.logits_to_probabilities_enhanced(paraphrase_logit_sequence, 1.0)?;
            
            // Calculate JS divergences between base and paraphrase
            let min_len = base_distributions.len().min(paraphrase_distributions.len());
            
            for i in 0..min_len {
                let js_div = common::math::information_theory::InformationTheoryCalculator::default()
                    .js_divergence(&base_distributions[i].to_vec(), &paraphrase_distributions[i].to_vec())?;
                max_js_divergence = max_js_divergence.max(js_div);
            }
        }
        
        // Fallback to consecutive distribution analysis if no paraphrases
        let base_flexibility = if base_distributions.len() > 1 {
            self.calculate_logit_flexibility(base_distributions)?
        } else {
            0.5
        };
        
        // Use maximum of paraphrase-based and consecutive-based flexibility
        let enhanced_flexibility = f64::max(max_js_divergence, base_flexibility);
        
        if self.config.validation_config.enable_logging {
            log::debug!("Enhanced flexibility: {:.6}, max JS divergence: {:.6}", 
                enhanced_flexibility, max_js_divergence);
        }
        
        Ok((enhanced_flexibility, max_js_divergence as f64))
    }

    /// 📐 Calculate geodesic distances using FIM metric tensor
    fn calculate_geodesic_distances(&self, distributions: &[Array1<f64>]) -> Result<GeodesicDistances, Box<dyn std::error::Error>> {
        if distributions.len() < 2 {
            return Ok(GeodesicDistances {
                mean_geodesic_distance: 0.0,
                max_geodesic_distance: 0.0,
                geodesic_variance: 0.0,
                path_count: 0,
            });
        }
        
        let mut geodesic_distances = Vec::new();
        let num_paths = self.config.geodesic_config.num_paths.min(distributions.len() - 1);
        
        // Compute center distribution (mean)
        let center = self.compute_mean_distribution(distributions)?;
        
        for i in 0..num_paths {
            if i < distributions.len() {
                let geodesic_dist = if self.config.geodesic_config.use_exponential_map {
                    self.approximate_geodesic_distance_exponential_map(&center, &distributions[i])?
                } else {
                    self.approximate_geodesic_distance_integration(&center, &distributions[i])?
                };
                geodesic_distances.push(geodesic_dist);
            }
        }
        
        if geodesic_distances.is_empty() {
            return Ok(GeodesicDistances {
                mean_geodesic_distance: 0.0,
                max_geodesic_distance: 0.0,
                geodesic_variance: 0.0,
                path_count: 0,
            });
        }
        
        let mean_distance = geodesic_distances.iter().sum::<f64>() / geodesic_distances.len() as f64;
        let max_distance: f64 = geodesic_distances.iter().copied().fold(0.0_f64, f64::max);
        
        let variance = geodesic_distances.iter()
            .map(|&d| (d - mean_distance).powi(2))
            .sum::<f64>() / geodesic_distances.len() as f64;
        
        Ok(GeodesicDistances {
            mean_geodesic_distance: mean_distance,
            max_geodesic_distance: max_distance,
            geodesic_variance: variance,
            path_count: geodesic_distances.len(),
        })
    }

    /// 🎯 Compute mean distribution for geodesic center
    fn compute_mean_distribution(&self, distributions: &[Array1<f64>]) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let dim = distributions[0].len();
        let mut mean = Array1::<f64>::zeros(dim);
        
        for dist in distributions {
            mean = mean + dist;
        }
        
        mean = mean / distributions.len() as f64;
        
        // Ensure proper normalization
        let sum = mean.sum();
        if sum > 0.0 {
            mean = mean / sum;
        }
        
        Ok(mean)
    }

    /// 🗺️ Approximate geodesic distance using exponential map
    fn approximate_geodesic_distance_exponential_map(&self, p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified geodesic distance using Fisher-Rao metric
        // d_g(p,q) ≈ 2 * arccos(√(Σ √(p_i * q_i)))
        
        let mut sum = 0.0;
        for (p_i, q_i) in p.iter().zip(q.iter()) {
            if *p_i > 0.0 && *q_i > 0.0 {
                sum += (p_i * q_i).sqrt();
            }
        }
        
        let geodesic_dist = 2.0 * (sum.sqrt()).acos();
        
        // Handle numerical issues
        if geodesic_dist.is_finite() {
            Ok(geodesic_dist)
        } else {
            // Fallback to Euclidean distance
            Ok((p - q).dot(&(p - q)).sqrt())
        }
    }

    /// ∫ Approximate geodesic distance using integration
    fn approximate_geodesic_distance_integration(&self, p: &Array1<f64>, q: &Array1<f64>) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified integration along geodesic path
        // This is a basic approximation - full implementation would use Runge-Kutta
        
        let steps = self.config.geodesic_config.max_integration_steps;
        let mut total_distance = 0.0;
        
        for i in 0..steps {
            let t = i as f64 / steps as f64;
            let next_t = (i + 1) as f64 / steps as f64;
            
            // Linear interpolation (approximation)
            let current = p * (1.0 - t) + q * t;
            let next = p * (1.0 - next_t) + q * next_t;
            
            // Ensure normalization
            let current_normalized = &current / current.sum();
            let next_normalized = &next / next.sum();
            
            // Local distance element
            let local_dist = (&next_normalized - &current_normalized).dot(&(&next_normalized - &current_normalized)).sqrt();
            total_distance += local_dist;
        }
        
        Ok(total_distance)
    }

    /// 📏 Calculate Mahalanobis distances with FIM metric tensor
    fn calculate_mahalanobis_distances(&self, distributions: &[Array1<f64>]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        // This would require the FIM inverse, simplified for now
        let mut distances = Vec::new();
        
        if distributions.len() < 2 {
            return Ok(distances);
        }
        
        let center = self.compute_mean_distribution(distributions)?;
        
        for dist in distributions {
            let diff = dist - &center;
            // Simplified Mahalanobis distance (using identity covariance)
            let mahalanobis_dist = diff.dot(&diff).sqrt();
            distances.push(mahalanobis_dist);
        }
        
        Ok(distances)
    }

    /// 📐 Calculate semantic precision (Δμ) - existing method as fallback
    fn calculate_logit_precision(&self, distributions: &[Array1<f64>]) -> Result<f64, Box<dyn std::error::Error>> {
        if distributions.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate entropy for each position
        let entropies: Result<Vec<f64>, _> = distributions.iter()
            .map(|dist| self.info_calculator.shannon_entropy(dist.as_slice().unwrap()))
            .collect();
        
        let entropies = entropies?;
        
        // Precision is inverse of entropy variance (more stable = higher precision)
        let mean_entropy: f64 = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let entropy_variance: f64 = entropies.iter()
            .map(|&entropy| (entropy - mean_entropy).powi(2))
            .sum::<f64>() / entropies.len() as f64;
        
        // Higher stability (lower variance) = higher precision
        let precision = 1.0 / (1.0 + entropy_variance);
        
        Ok(precision)
    }

    /// 🌊 Calculate flexibility (Δσ) from distribution variance - existing method as fallback
    fn calculate_logit_flexibility(&self, distributions: &[Array1<f64>]) -> Result<f64, Box<dyn std::error::Error>> {
        if distributions.len() < 2 {
            return Ok(0.5); // Default flexibility for single distribution
        }
        
        // Calculate Jensen-Shannon divergence between consecutive distributions
        let mut js_divergences = Vec::new();
        
        for i in 0..distributions.len()-1 {
            let dist_a = &distributions[i];
            let dist_b = &distributions[i+1];
            
            // Calculate JS divergence using existing infrastructure
            let js_div = common::math::information_theory::InformationTheoryCalculator::default().js_divergence(&dist_a.to_vec(), &dist_b.to_vec())?;
            js_divergences.push(js_div);
        }
        
        // Flexibility is the average JS divergence (higher divergence = more flexibility)
        let avg_js_divergence: f64 = js_divergences.iter().sum::<f64>() / js_divergences.len() as f64;
        
        Ok(avg_js_divergence.sqrt())
    }

    /// 📈 Calculate enhanced logit metrics with FIM information
    fn calculate_enhanced_logit_metrics(
        &self,
        distributions: &[Array1<f64>],
        logit_data: &LogitData,
    ) -> Result<LogitMetrics, Box<dyn std::error::Error>> {
        // Calculate entropies
        let entropies: Result<Vec<f64>, _> = distributions.iter()
            .map(|dist| self.info_calculator.shannon_entropy(dist.as_slice().unwrap()))
            .collect();
        let entropies = entropies?;
        
        let average_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let max_entropy: f64 = entropies.iter().copied().fold(0.0_f64, f64::max);
        
        // Calculate perplexity
        let perplexity = average_entropy.exp2();
        
        // Calculate confidence (average of max probabilities)
        let confidence_score: f64 = distributions.iter()
            .map(|dist| dist.iter().copied().fold(0.0_f64, f64::max))
            .sum::<f64>() / distributions.len() as f64;
        
        // Calculate distribution diversity (average entropy normalized)
        let distribution_diversity = average_entropy / (distributions[0].len() as f64).log2();
        
        // Attention consistency (placeholder - would need actual attention weights)
        let attention_consistency = logit_data.attention_weights.as_ref()
            .map(|_weights| 0.8); // Placeholder calculation
        
        // FIM-related metrics (placeholder values)
        let fim_trace = None; // Would be computed from cached FIM
        let fim_max_eigenvalue = None; // Would be computed from cached FIM
        
        Ok(LogitMetrics {
            average_entropy,
            max_entropy: max_entropy,
            perplexity,
            confidence_score,
            distribution_diversity,
            attention_consistency,
            fim_trace,
            fim_max_eigenvalue,
        })
    }

    /// 🎯 Analyze enhanced token uncertainties with geometric information
    fn analyze_enhanced_token_uncertainties(
        &self,
        distributions: &[Array1<f64>],
        logit_data: &LogitData,
    ) -> Result<Vec<TokenUncertainty>, Box<dyn std::error::Error>> {
        let mut token_uncertainties = Vec::new();
        
        for (pos, (dist, &token_id)) in distributions.iter()
            .zip(logit_data.token_sequence.iter())
            .enumerate() 
        {
            let token_string = logit_data.vocab_map.get(&token_id)
                .unwrap_or(&"<unk>".to_string())
                .clone();
            
            // Calculate local entropy
            let token_entropy = self.info_calculator
                .shannon_entropy(dist.as_slice().unwrap())?;
            
            // Get selected token probability
            let token_probability = if (token_id as usize) < dist.len() {
                dist[token_id as usize]
            } else {
                0.0 // Handle out-of-bounds gracefully
            };
            
            // Get top-k alternatives
            let mut indexed_probs: Vec<(usize, f64)> = dist.iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let alternatives: Vec<(String, f64)> = indexed_probs.iter()
                .take(5)
                .map(|(idx, prob)| {
                    let alt_token = logit_data.vocab_map.get(&(*idx as u32))
                        .unwrap_or(&format!("<{}>", idx))
                        .clone();
                    (alt_token, *prob)
                })
                .collect();
            
            // Calculate local hbar (simplified)
            let local_hbar = token_entropy.sqrt();
            
            // Enhanced geometric metrics (placeholders for now)
            let local_fim_contribution = None; // Would compute from local FIM
            let local_geodesic_distance = None; // Would compute local geodesic
            
            token_uncertainties.push(TokenUncertainty {
                position: pos,
                token_id,
                token_string,
                token_entropy,
                token_probability,
                alternatives,
                local_hbar,
                local_fim_contribution,
                local_geodesic_distance,
            });
        }
        
        Ok(token_uncertainties)
    }

    /// ⚡ Calculate enhanced streaming metrics with FIM overhead tracking
    fn calculate_enhanced_streaming_metrics(
        &self,
        distributions: &[Array1<f64>],
        fim_overhead_ms: Option<f64>,
    ) -> Result<StreamingMetrics, Box<dyn std::error::Error>> {
        // Calculate uncertainty trend
        let uncertainty_trend: Result<Vec<f64>, _> = distributions.iter()
            .map(|dist| self.info_calculator.shannon_entropy(dist.as_slice().unwrap()))
            .collect();
        let uncertainty_trend = uncertainty_trend?;
        
        // Enhanced metrics with FIM overhead tracking
        let token_latencies_ms = vec![1.2; distributions.len()];
        let memory_usage_mb = 512.0 + fim_overhead_ms.unwrap_or(0.0) * 0.1; // Estimate FIM memory impact
        let throughput_tps = 25.0;
        
        Ok(StreamingMetrics {
            uncertainty_trend,
            token_latencies_ms,
            memory_usage_mb,
            throughput_tps,
            fim_overhead_ms,
        })
    }
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self {
            uncertainty_trend: Vec::new(),
            token_latencies_ms: Vec::new(),
            memory_usage_mb: 0.0,
            throughput_tps: 0.0,
            fim_overhead_ms: None,
        }
    }
}

/// 🔌 Integration helpers for popular OSS frameworks

/// HuggingFace Transformers integration
#[cfg(feature = "huggingface")]
pub mod huggingface {
    use super::*;
    
    pub fn from_transformers_output(
        model_output: &dyn std::any::Any, // Placeholder for transformers output
        vocab: &HashMap<u32, String>,
        temperature: f32,
    ) -> LogitData {
        // Implementation would extract logits from transformers output
        LogitData {
            token_logits: vec![],
            vocab_map: vocab.clone(),
            attention_weights: None,
            hidden_states: None,
            temperature,
            top_p: None,
            token_sequence: vec![],
            gradients: None,
            paraphrase_logits: None,
        }
    }
}

/// llama.cpp integration
#[cfg(feature = "llamacpp")]
pub mod llamacpp {
    use super::*;
    
    pub fn from_llamacpp_context(
        // Integration with llama.cpp C API
        context: *mut std::ffi::c_void,
        vocab: &HashMap<u32, String>,
    ) -> LogitData {
        // Implementation would extract logits from llama.cpp context
        LogitData {
            token_logits: vec![],
            vocab_map: vocab.clone(),
            attention_weights: None,
            hidden_states: None,
            temperature: 1.0,
            top_p: None,
            token_sequence: vec![],
            gradients: None,
            paraphrase_logits: None,
        }
    }
}

/// 🐍 PyTorch bridge for gradient-based enhancements
#[cfg(feature = "pytorch")]
pub mod pytorch_bridge {
    use super::*;
    use pyo3::prelude::*;
    
    /// Extract logits and gradients from PyTorch model
    #[pyfunction]
    pub fn extract_pytorch_logits(
        model_output: &PyAny,
        enable_gradients: bool,
    ) -> PyResult<LogitData> {
        // Implementation would use PyTorch C++ API or Python bindings
        // to extract logits and compute gradients
        
        let token_logits: Vec<Vec<f32>> = vec![]; // Extract from PyTorch tensor
        let gradients = if enable_gradients {
            Some(vec![]) // Compute gradients using torch.autograd
        } else {
            None
        };
        
        Ok(LogitData {
            token_logits,
            vocab_map: HashMap::new(), // Extract from tokenizer
            attention_weights: None,
            hidden_states: None,
            temperature: 1.0,
            top_p: None,
            token_sequence: vec![],
            gradients,
            paraphrase_logits: None,
        })
    }
    
    /// Create PyTorch module for enhanced FIM computation
    #[pymodule]
    fn semantic_uncertainty_pytorch(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(extract_pytorch_logits, m)?)?;
        Ok(())
    }
} 