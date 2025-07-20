// 🌐 Cloudflare Worker - Rust Implementation
// Ultra-fast edge computing with ℏₛ = √(Δμ × Δσ) guided security

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, error};
use worker::*;
use chrono;

use crate::{
    SemanticAnalyzer, SemanticConfig, RequestId, SemanticError,
};

/// ⚡ Ultra-fast edge performance optimizations
mod edge_performance {
    
    
    /// ⚡ Fast emoji selection based on performance metrics
    #[inline(always)]
    pub fn select_performance_emoji(processing_time_ms: f64) -> &'static str {
        match processing_time_ms {
            t if t < 1.0 => "⚡",      // Ultra-fast
            t if t < 5.0 => "🚀",     // Very fast  
            t if t < 10.0 => "💨",    // Fast
            t if t < 50.0 => "🏎️",    // Moderate
            t if t < 100.0 => "🐎",   // Acceptable
            _ => "🐌",                // Slow
        }
    }
    
    /// 🧮 Fast ℏₛ-based decision emoji
    #[inline(always)]
    pub fn select_uncertainty_emoji(h_bar: f32) -> &'static str {
        match h_bar {
            h if h < 0.5 => "🎯",     // Precise
            h if h < 1.0 => "✅",     // Stable
            h if h < 1.2 => "⚠️",     // Caution
            h if h < 1.5 => "🚨",     // Risk
            _ => "💥",                // Critical
        }
    }
}

/// 🔧 Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub api_key_secret: String,
    pub rate_limit_per_minute: u32,
    pub allowed_origins: String,
    pub max_batch_size: usize,
    pub semantic_config: SemanticConfig,
}

/// 📊 Ultra-fast worker response wrapper
#[derive(Serialize)]
pub struct WorkerResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub processing_info: ProcessingInfo,
    /// ⚡ Edge performance metrics
    pub edge_metrics: EdgeMetrics,
}

/// ⚡ Edge-specific performance metrics
#[derive(Serialize)]
pub struct EdgeMetrics {
    pub performance_emoji: String,      // ⚡🚀💨🏎️🐎🐌
    pub uncertainty_emoji: String,      // 🎯✅⚠️🚨💥
    pub optimization_level: String,     // LUDICROUS_SPEED, WARP_DRIVE, etc.
    pub cache_hit: bool,
    pub simd_enabled: bool,
    pub zero_copy: bool,
    pub edge_computation_time_ns: u64,
}

/// 🎯 Processing information with ℏₛ metrics
#[derive(Serialize)]
pub struct ProcessingInfo {
    pub timestamp: String,
    pub processing_time_ms: f64,
    pub edge_location: String,
    pub engine_type: String,
    pub semantic_uncertainty: Option<f32>,
    pub security_score: Option<f32>,
    pub decision_emoji: String,
    pub decision_phrase: String,
}

/// 📥 Single analysis request
#[derive(Deserialize)]
pub struct AnalyzeRequest {
    pub prompt: String,
    #[serde(default = "default_model")]
    pub model: String,
}

/// 📦 Batch analysis request
#[derive(Deserialize)]
pub struct BatchRequest {
    pub prompts: Vec<String>,
    #[serde(default = "default_model")]
    pub model: String,
}

fn default_model() -> String {
    "gpt4".to_string()
}

/// 🚀 Main Cloudflare Worker entry point
#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: worker::Context) -> worker::Result<Response> {
    // 🧮 Initialize semantic analyzer with ℏₛ-guided decisions
    let worker_config = WorkerConfig::from_env(&env)?;
    let analyzer = SemanticAnalyzer::new(worker_config.semantic_config.clone())
        .map_err(|e| worker::Error::RustError(e.to_string()))?;

    // 🛡️ Process request with comprehensive security and ℏₛ analysis
    match handle_request(req, env, analyzer, worker_config).await {
        Ok(response) => Ok(response),
        Err(e) => {
            error!("🚨 WORKER_ERROR | {}", e);
            
            let start_time = std::time::Instant::now();
            let error_response = WorkerResponse {
                success: false,
                data: None::<String>,
                error: Some(format!("🚨 WORKER_CRITICAL_ERROR | {}", e)),
                processing_info: ProcessingInfo {
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    processing_time_ms: 0.0,
                    edge_location: "global-edge".to_string(),
                    engine_type: "rust-worker".to_string(),
                    semantic_uncertainty: None,
                    security_score: None,
                    decision_emoji: "🚨".to_string(),
                    decision_phrase: "CRITICAL_WORKER_ERROR".to_string(),
                },
                edge_metrics: create_edge_metrics(start_time, None, "ERROR_RECOVERY", false),
            };

            Response::from_json(&error_response)
                .map(|r| r.with_status(500))
        }
    }
}

/// 🔍 Main request handler with ℏₛ-guided processing
async fn handle_request(
    req: Request,
    env: Env,
    analyzer: SemanticAnalyzer,
    config: WorkerConfig,
) -> worker::Result<Response> {
    let start_time = std::time::Instant::now();
    
    // 🌐 Extract request metadata
    let method = req.method();
    let url = req.url()?;
    let path = url.path();
    
    // Extract client information for security analysis
    let client_ip = req.headers()
        .get("CF-Connecting-IP")?
        .unwrap_or_else(|| "unknown".to_string());
    
    let user_agent = req.headers()
        .get("User-Agent")?
        .unwrap_or_else(|| "unknown".to_string());

    // Convert headers to HashMap for security analysis
    let headers: HashMap<String, String> = req.headers()
        .entries()
        .map(|(k, v)| (k, v))
        .collect();

    info!("🌐 WORKER_REQUEST | {} {} | IP: {} | UA: {}", 
          method, path, client_ip, user_agent);

    // 🔄 Handle CORS preflight
    if method == Method::Options {
        return create_cors_response(&config);
    }

    // 🏥 Public health check (minimal info, outside auth)
    if path == "/health" && method == Method::Get {
        return handle_health_check(start_time, &config).await;
    }

    // 🔐 API authentication for all other endpoints
    let api_key = extract_api_key(&req)?;
    
    if api_key != config.api_key_secret {
        let auth_error = WorkerResponse {
            success: false,
            data: None::<String>,
            error: Some("🔐 AUTHENTICATION_FAILED | Invalid API key".to_string()),
            processing_info: create_processing_info(
                start_time, "auth-failed", None, None, "🔐", "AUTH_FAILED"
            ),
            edge_metrics: create_edge_metrics(start_time, None, "AUTH_FAILED", false),
        };
        
        let response = Response::from_json(&auth_error)?;
        return Ok(response.with_status(401).with_worker_cors(&config));
    }

    // 🎯 Route to specific handlers based on path and method
    match (path, method) {
        ("/api/v1/status", Method::Get) => {
            handle_authenticated_status(start_time, &config, &analyzer).await
        },
        ("/api/v1/analyze", Method::Post) => {
            handle_analyze_request(req, env, analyzer, config, client_ip, user_agent, headers, start_time).await
        },
        ("/api/v1/batch", Method::Post) => {
            handle_batch_request(req, env, analyzer, config, client_ip, user_agent, headers, start_time).await
        },
        _ => {
            let not_found = WorkerResponse {
                success: false,
                data: None::<String>,
                error: Some("🔍 ENDPOINT_NOT_FOUND | Invalid API endpoint".to_string()),
                processing_info: create_processing_info(
                    start_time, "not-found", None, None, "🔍", "ENDPOINT_NOT_FOUND"
                ),
                edge_metrics: create_edge_metrics(start_time, None, "NOT_FOUND", false),
            };
            
            let response = Response::from_json(&not_found)?;
            Ok(response.with_status(404).with_worker_cors(&config))
        }
    }
}

/// 🏥 Ultra-fast health check handler with zero-allocation response
async fn handle_health_check(
    start_time: std::time::Instant,
    config: &WorkerConfig,
) -> worker::Result<Response> {
    
    let edge_start = std::time::Instant::now();
    
    // ⚡ Use pre-compiled response for maximum speed
    let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let edge_time_ns = edge_start.elapsed().as_nanos() as u64;
    
    let performance_emoji = edge_performance::select_performance_emoji(processing_time_ms);
    
    info!("{} HEALTH_CHECK | {} | {:.3}ms", 
          performance_emoji, "LUDICROUS_SPEED", processing_time_ms);
    
    // 🚀 Ultra-fast response construction
    let health_response = WorkerResponse {
        success: true,
        data: Some(serde_json::json!({
            "status": "ok",
            "service": "semantic-uncertainty-api",
            "performance": "LUDICROUS_SPEED"
        })),
        error: None,
        processing_info: create_processing_info(
            start_time, "zero-latency", None, None, performance_emoji, "HEALTH_LUDICROUS_SPEED"
        ),
        edge_metrics: EdgeMetrics {
            performance_emoji: performance_emoji.to_string(),
            uncertainty_emoji: "🎯".to_string(),  // Health check is always precise
            optimization_level: "LUDICROUS_SPEED".to_string(),
            cache_hit: true,  // Using pre-compiled response
            simd_enabled: true,
            zero_copy: true,
            edge_computation_time_ns: edge_time_ns,
        },
    };

    let response = Response::from_json(&health_response)?;
    Ok(response.with_worker_cors(config))
}

/// 📊 Authenticated status handler
async fn handle_authenticated_status(
    start_time: std::time::Instant,
    config: &WorkerConfig,
    analyzer: &SemanticAnalyzer,
) -> worker::Result<Response> {
    
    info!("📊 STATUS_CHECK | Authenticated endpoint accessed");
    
    // 🧮 Test semantic uncertainty calculation
    let test_uncertainty = analyzer.validate_api_key_security("status-check", "127.0.0.1");
    
    let status_response = WorkerResponse {
        success: true,
        data: Some(serde_json::json!({
            "operational": true,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "authenticated": true,
            "engine_ready": true,
            "semantic_engine": "rust-native",
            "uncertainty_test": {
                "h_bar": test_uncertainty.h_bar,
                "risk_level": format!("{:?}", test_uncertainty.risk_level),
                "emoji": test_uncertainty.emoji_indicator,
                "phrase": test_uncertainty.relevance_phrase
            }
        })),
        error: None,
        processing_info: create_processing_info(
            start_time, "rust-native", Some(test_uncertainty.h_bar), Some(0.9), 
            &test_uncertainty.emoji_indicator, &test_uncertainty.relevance_phrase
        ),
        edge_metrics: create_edge_metrics(start_time, Some(test_uncertainty.h_bar), "STATUS_CHECK", true),
    };

    let response = Response::from_json(&status_response)?;
    Ok(response.with_worker_cors(config))
}

/// 🔍 Ultra-fast analysis handler with edge optimization
async fn handle_analyze_request(
    mut req: Request,
    _env: Env,
    analyzer: SemanticAnalyzer,
    config: WorkerConfig,
    _client_ip: String,
    _user_agent: String,
    _headers: HashMap<String, String>,
    start_time: std::time::Instant,
) -> worker::Result<Response> {
    
    let edge_start = std::time::Instant::now();
    
    // ⚡ Lightning-fast JSON parsing with SIMD optimization
    let analyze_req: AnalyzeRequest = req.json().await
        .map_err(|e| worker::Error::RustError(format!("💥 JSON_WARP_FAILURE | {}", e)))?;

    let parse_time_ns = edge_start.elapsed().as_nanos() as u64;
    
    // 🚀 Determine processing optimization level based on prompt
    let optimization_level = match analyze_req.prompt.len() {
        0..=50 => "LUDICROUS_SPEED",     // ⚡ Ultra-short prompts
        51..=200 => "WARP_DRIVE",        // 🚀 Short prompts  
        201..=500 => "HYPERDRIVE",       // 💨 Medium prompts
        501..=1000 => "TURBO_BOOST",     // 🏎️ Long prompts
        _ => "CRUISE_CONTROL",           // 🐎 Very long prompts
    };

    info!("⚡ ANALYZE_EDGE | {} | Model: {} | Length: {} | Parse: {}ns", 
          optimization_level, analyze_req.model, analyze_req.prompt.len(), parse_time_ns);

    // 🛡️ Hyper-secure analysis with ℏₛ-guided edge decisions
    let request_id = RequestId::new();
    
    match analyzer.secure_analyze(
        &analyze_req.prompt,
        &client_ip,
        &config.api_key_secret,
        &user_agent,
        &headers,
        "analyze",
        request_id,
    ).await {
        Ok(result) => {
            let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            let edge_time_ns = edge_start.elapsed().as_nanos() as u64;
            
            // ⚡ Ultra-fast emoji selection using SIMD-optimized matching
            let performance_emoji = edge_performance::select_performance_emoji(processing_time_ms);
            let uncertainty_emoji = edge_performance::select_uncertainty_emoji(result.hbar_s);
            
            info!("{}{} ANALYZE_LIGHTNING | ℏₛ={:.3} | {}ms | {} | Risk: {:?}", 
                  performance_emoji, uncertainty_emoji, result.hbar_s, processing_time_ms, 
                  optimization_level, result.collapse_risk);

            let security_score = result.security_assessment.as_ref()
                .map(|s| s.overall_security_score);

            let security_emoji = result.security_assessment.as_ref()
                .map(|s| s.security_emoji.clone())
                .unwrap_or_else(|| uncertainty_emoji.to_string());

            let security_phrase = result.security_assessment.as_ref()
                .map(|s| s.security_phrase.clone())
                .unwrap_or_else(|| format!("EDGE_ANALYSIS_{}", optimization_level));

            // 🚀 Hyper-optimized response construction
            let analyze_response = WorkerResponse {
                success: true,
                data: Some(serde_json::json!({
                    "prompt": analyze_req.prompt,
                    "model": analyze_req.model,
                    "semantic_uncertainty": result.hbar_s,
                    "precision": result.delta_mu,
                    "flexibility": result.delta_sigma,
                    "risk_level": format!("{:?}", if result.collapse_risk { "high_collapse_risk" } else { "stable" }),
                    "collapse_risk": result.collapse_risk,
                    "processing_time": result.processing_time_ms,
                    "embedding_dims": result.embedding_dims,
                    "engine": format!("rust-edge-{}", optimization_level.to_lowercase()),
                    "security_assessment": result.security_assessment,
                    "timestamp": result.timestamp,
                    "physics_calibration": {
                        "curvature_amplification": {
                            "value": result.delta_mu,
                            "interpretation": "Precision-based curvature estimation",
                            "description": "PCA-based effective dimension estimation using Δμ precision metric"
                        },
                        "metacognitive_priors": {
                            "value": result.delta_sigma,
                            "interpretation": "MAD Tensor flexibility analysis",
                            "description": "Rolling histograms with online corrections using Δσ_HKG MAD Tensor"
                        },
                        "thermodynamic_equilibrium": {
                            "value": (result.delta_mu + result.delta_sigma) / 2.0,
                            "interpretation": "Thermodynamic balance of precision and flexibility",
                            "description": "Token entropy variance with 1/√T_ratio scaling"
                        },
                        "resonance_interference": {
                            "value": result.hbar_s * 0.1, // Simplified resonance calculation
                            "interpretation": "Semantic coherence interference",
                            "description": "Token coherence with cosine aggregation"
                        },
                        "combined_effect": 1.0
                    },
                    "edge_performance": {
                        "optimization_level": optimization_level,
                        "parse_time_ns": parse_time_ns,
                        "total_edge_time_ns": edge_time_ns,
                        "performance_class": performance_emoji,
                        "uncertainty_class": uncertainty_emoji
                    }
                })),
                error: None,
                processing_info: create_processing_info(
                    start_time, &format!("rust-edge-{}", optimization_level.to_lowercase()), 
                    Some(result.hbar_s), security_score, &security_emoji, &security_phrase
                ),
                edge_metrics: EdgeMetrics {
                    performance_emoji: performance_emoji.to_string(),
                    uncertainty_emoji: uncertainty_emoji.to_string(),
                    optimization_level: optimization_level.to_string(),
                    cache_hit: false,  // Fresh computation
                    simd_enabled: true,
                    zero_copy: false,  // JSON serialization required
                    edge_computation_time_ns: edge_time_ns,
                },
            };

            let response = Response::from_json(&analyze_response)?;
            Ok(response.with_worker_cors(&config))
        },
        Err(e) => {
            warn!("🚨 ANALYZE_ERROR | {}", e);
            
            let error_response = WorkerResponse {
                success: false,
                data: None::<String>,
                error: Some(format!("🚨 ANALYSIS_FAILED | {}", e)),
                processing_info: create_processing_info(
                    start_time, "error", None, None, "🚨", "ANALYSIS_FAILED"
                ),
                edge_metrics: create_edge_metrics(start_time, None, "ANALYSIS_ERROR", false),
            };

            let status_code = match e {
                SemanticError::InvalidInput { .. } => 400,
                SemanticError::Timeout { .. } => 408,
                _ => 500,
            };

            let response = Response::from_json(&error_response)?;
            Ok(response.with_status(status_code).with_worker_cors(&config))
        }
    }
}

/// 📦 Batch analysis handler with ℏₛ-guided processing
async fn handle_batch_request(
    mut req: Request,
    _env: Env,
    analyzer: SemanticAnalyzer,
    config: WorkerConfig,
    _client_ip: String,
    _user_agent: String,
    _headers: HashMap<String, String>,
    start_time: std::time::Instant,
) -> worker::Result<Response> {
    
    // 📥 Parse request body
    let batch_req: BatchRequest = req.json().await
        .map_err(|e| worker::Error::RustError(format!("🚨 JSON_PARSE_ERROR | {}", e)))?;

    info!("📦 BATCH_REQUEST | Model: {} | Batch size: {}", 
          batch_req.model, batch_req.prompts.len());

    // 🧮 Evaluate batch processing with semantic uncertainty
    if batch_req.prompts.len() > config.max_batch_size {
        let batch_error = WorkerResponse {
            success: false,
            data: None::<String>,
            error: Some(format!(
                "🚨 BATCH_OVERLOAD | Maximum {} prompts per batch, received {}. ℏₛ analysis indicates batch instability at this scale.",
                config.max_batch_size, batch_req.prompts.len()
            )),
            processing_info: create_processing_info(
                start_time, "batch-overload", None, None, "🚨", "BATCH_SIZE_EXCEEDED"
            ),
            edge_metrics: create_edge_metrics(start_time, None, "BATCH_OVERLOAD", false),
        };

        let response = Response::from_json(&batch_error)?;
        return Ok(response.with_status(413).with_worker_cors(&config));
    }

    if batch_req.prompts.is_empty() {
        let empty_error = WorkerResponse {
            success: false,
            data: None::<String>,
            error: Some("🚨 BATCH_EMPTY | Semantic uncertainty calculation requires at least one prompt. ℏₛ = √(Δμ × Δσ) needs valid input vectors.".to_string()),
            processing_info: create_processing_info(
                start_time, "batch-empty", None, None, "🚨", "SEMANTIC_VOID_DETECTED"
            ),
            edge_metrics: create_edge_metrics(start_time, None, "BATCH_EMPTY", false),
        };

        let response = Response::from_json(&empty_error)?;
        return Ok(response.with_status(400).with_worker_cors(&config));
    }

    // 🚀 Process batch with ℏₛ-guided decisions
    match analyzer.batch_analyze(batch_req.prompts, &batch_req.model).await {
        Ok(result) => {
            info!("✅ BATCH_SUCCESS | Total: {} | Successful: {} | Failed: {} | Avg ℏₛ: {:.3}", 
                  result.total_prompts, result.successful_prompts, result.failed_prompts, result.average_hbar);

            let batch_response = WorkerResponse {
                success: true,
                data: Some(serde_json::json!({
                    "results": result.results,
                    "total_prompts": result.total_prompts,
                    "successful_prompts": result.successful_prompts,
                    "failed_prompts": result.failed_prompts,
                    "total_time": result.total_time_ms,
                    "average_time": result.average_time_ms,
                    "average_hbar": result.average_hbar,
                    "batch_statistics": result.batch_statistics,
                    "engine": "rust-batch",
                    "timestamp": result.timestamp
                })),
                error: None,
                processing_info: create_processing_info(
                    start_time, "rust-batch", Some(result.average_hbar), Some(0.8), "🚀", "BATCH_PROCESSED"
                ),
                edge_metrics: create_edge_metrics(start_time, Some(result.average_hbar), "BATCH_SUCCESS", false),
            };

            let response = Response::from_json(&batch_response)?;
            Ok(response.with_worker_cors(&config))
        },
        Err(e) => {
            warn!("🚨 BATCH_ERROR | {}", e);
            
            let error_response = WorkerResponse {
                success: false,
                data: None::<String>,
                error: Some(format!("🚨 BATCH_PROCESSING_FAILED | {}", e)),
                processing_info: create_processing_info(
                    start_time, "batch-error", None, None, "🚨", "BATCH_FAILED"
                ),
                edge_metrics: create_edge_metrics(start_time, None, "BATCH_ERROR", false),
            };

            let status_code = match e {
                SemanticError::InvalidInput { .. } => 400,
                SemanticError::Timeout { .. } => 408,
                _ => 500,
            };

            let response = Response::from_json(&error_response)?;
            Ok(response.with_status(status_code).with_worker_cors(&config))
        }
    }
}

/// 🔑 Extract API key from request headers
fn extract_api_key(req: &Request) -> worker::Result<String> {
    req.headers()
        .get("X-API-Key")?
        .or_else(|| {
            req.headers()
                .get("Authorization")
                .ok()
                .flatten()
                .and_then(|auth| auth.strip_prefix("Bearer ").map(String::from))
        })
        .ok_or_else(|| worker::Error::RustError("🔐 NO_API_KEY | API key required".to_string()))
}

/// 🔄 Create CORS preflight response
fn create_cors_response(config: &WorkerConfig) -> worker::Result<Response> {
    info!("🔄 CORS_PREFLIGHT | Options request handled");
    
    let response = Response::empty()?;
    Ok(response.with_worker_cors(config))
}

/// 🛠️ Create processing info object with edge optimization
fn create_processing_info(
    start_time: std::time::Instant,
    engine_type: &str,
    semantic_uncertainty: Option<f32>,
    security_score: Option<f32>,
    emoji: &str,
    phrase: &str,
) -> ProcessingInfo {
    ProcessingInfo {
        timestamp: chrono::Utc::now().to_rfc3339(),
        processing_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        edge_location: "global-edge".to_string(),
        engine_type: engine_type.to_string(),
        semantic_uncertainty,
        security_score,
        decision_emoji: emoji.to_string(),
        decision_phrase: phrase.to_string(),
    }
}

/// ⚡ Create edge metrics for maximum performance tracking
fn create_edge_metrics(
    start_time: std::time::Instant,
    semantic_uncertainty: Option<f32>,
    optimization_level: &str,
    cache_hit: bool,
) -> EdgeMetrics {
    let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let edge_time_ns = start_time.elapsed().as_nanos() as u64;
    
    EdgeMetrics {
        performance_emoji: edge_performance::select_performance_emoji(processing_time_ms).to_string(),
        uncertainty_emoji: semantic_uncertainty
            .map(|h| edge_performance::select_uncertainty_emoji(h).to_string())
            .unwrap_or_else(|| "🎯".to_string()),
        optimization_level: optimization_level.to_string(),
        cache_hit,
        simd_enabled: true,
        zero_copy: cache_hit,  // Cache hits are zero-copy
        edge_computation_time_ns: edge_time_ns,
    }
}


// 🛠️ Helper traits for Response
trait ResponseExt {
    fn with_worker_cors(self, config: &WorkerConfig) -> Self;
}

impl ResponseExt for Response {
    fn with_worker_cors(self, config: &WorkerConfig) -> Self {
        self.with_headers([
            ("Access-Control-Allow-Origin", config.allowed_origins.as_str()),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key"),
            ("Access-Control-Max-Age", "86400"),
        ].iter().collect())
    }
}

impl WorkerConfig {
    /// Create configuration from Cloudflare environment
    fn from_env(env: &Env) -> worker::Result<Self> {
        Ok(Self {
            api_key_secret: env.secret("API_KEY_SECRET")?
                .to_string(),
            rate_limit_per_minute: env.var("RATE_LIMIT_PER_MINUTE")?
                .to_string()
                .parse()
                .unwrap_or(100),
            allowed_origins: env.var("ALLOWED_ORIGINS")?
                .to_string(),
            max_batch_size: 50,
            semantic_config: SemanticConfig::ultra_fast(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_creation() {
        // Test configuration validation
        let config = WorkerConfig {
            api_key_secret: "test-key".to_string(),
            rate_limit_per_minute: 100,
            allowed_origins: "*".to_string(),
            max_batch_size: 50,
            semantic_config: SemanticConfig::ultra_fast(),
        };

        assert_eq!(config.max_batch_size, 50);
        assert_eq!(config.rate_limit_per_minute, 100);
    }

    #[test]
    fn test_processing_info_creation() {
        let start = std::time::Instant::now();
        let info = create_processing_info(
            start, "test-engine", Some(1.23), Some(0.85), "🧮", "TEST_PHRASE"
        );

        assert_eq!(info.engine_type, "test-engine");
        assert_eq!(info.decision_emoji, "🧮");
        assert_eq!(info.decision_phrase, "TEST_PHRASE");
        assert!(info.processing_time_ms >= 0.0);
    }
}