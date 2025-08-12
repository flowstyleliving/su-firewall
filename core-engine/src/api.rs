// Production-ready Semantic Uncertainty API
// Optimized for sub-10ms latency with comprehensive error handling

use crate::{SemanticAnalyzer, SemanticConfig, HbarResponse, SemanticError, RequestId, PerformanceMetrics};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{Json as ResponseJson, IntoResponse},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
    limit::RequestBodyLimitLayer,
};
use tracing::{info, error, debug, instrument};

#[cfg(feature = "api")]
use utoipa::{OpenApi, ToSchema};
#[cfg(feature = "api")]
use utoipa_swagger_ui::SwaggerUi;

// --- NEW: streaming/session imports ---
use std::collections::HashMap;
use std::sync::RwLock;
use axum::extract::Path;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use tokio::sync::broadcast;
use chrono::Utc;
use uuid::Uuid;

/// Streamlined API request for maximum performance
#[derive(Debug, Deserialize, ToSchema)]
pub struct AnalysisRequest {
    /// Input prompt (max 10,000 characters)
    pub prompt: String,
    /// Output response (max 10,000 characters)  
    pub output: String,
    /// Optional method selector (e.g., "jsd-kl", "fisher", "both")
    #[serde(default)]
    pub method: Option<String>,
    /// Optional advanced configuration
    #[serde(default)]
    pub config: Option<AnalysisConfig>,
}

/// Optional advanced analysis controls
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnalysisConfig {
    pub precision_method: Option<String>,
    pub flexibility_method: Option<String>,
    pub hash_embeddings_precision: Option<bool>,
    pub hash_embeddings_flexibility: Option<bool>,
    pub domain_tuning: Option<String>,
    pub performance_mode: Option<String>,
}

/// Production API response with comprehensive metadata
#[derive(Debug, Serialize, ToSchema)]
pub struct ApiResponse {
    /// Success indicator
    pub success: bool,
    /// Analysis results
    pub data: Option<HbarResponse>,
    /// Error information
    pub error: Option<String>,
    /// Processing metadata
    pub metadata: ResponseMetadata,
}

/// Response metadata for monitoring and debugging
#[derive(Debug, Serialize, ToSchema, Clone)]
pub struct ResponseMetadata {
    /// Request ID for tracing
    pub request_id: String,
    /// Server processing time in milliseconds
    pub processing_time_ms: f64,
    /// API version
    pub version: String,
    /// Response timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Health check response
#[derive(Debug, Serialize, ToSchema, Clone)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub metrics: PerformanceMetrics,
}

/// Thread-safe API state
pub struct ApiState {
    pub analyzer: Arc<SemanticAnalyzer>,
    pub metrics: Arc<Mutex<PerformanceMetrics>>,
    pub start_time: Instant,
    // --- NEW: streaming/session state ---
    pub broadcaster: broadcast::Sender<ServerEvent>,
    pub sessions: Arc<RwLock<HashMap<String, SessionState>>>,
    pub failure_law: FailureLawConfig,
}

/// WebSocket server events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerEvent {
    Welcome { session_id: String },
    TokenUpdate { session_id: String, payload: serde_json::Value },
}

/// Failure law configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureLawConfig {
    pub lambda: f64,
    pub tau: f64,
    pub risk_pfail_thresholds: RiskPfailThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskPfailThresholds {
    pub critical: f64,
    pub high_risk: f64,
    pub warning: f64,
}

/// Derived ℏ* thresholds from Pfail bands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HbarThresholds {
    pub supercritical: f64,
    pub critical: f64,
}

/// Per-session configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionConfig {
    pub precision_method: Option<String>,
    pub flexibility_method: Option<String>,
    pub hash_embeddings_precision: Option<bool>,
    pub hash_embeddings_flexibility: Option<bool>,
    pub domain_tuning: Option<String>,
    pub performance_mode: Option<String>,
}

/// Per-session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub id: String,
    pub created_at: i64,
    pub token_count: usize,
    pub model_id: String,
    pub lambda: f64,
    pub tau: f64,
    pub thresholds: HbarThresholds,
    pub prev_probs: Option<Vec<f64>>, // for JSD
    pub ema_hbar: Option<f64>,        // for rolling hbar
    pub config: SessionConfig,
}

/// Token metrics update from bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetricsUpdate {
    pub token_index: usize,
    pub token_text: String,
    pub probability: f64,
    pub entropy: f64,
    pub jsd: f64,
}

/// OpenAPI documentation
#[derive(OpenApi)]
#[openapi(
    paths(analyze_handler, health_handler),
    components(schemas(AnalysisRequest, ApiResponse, HealthResponse, HbarResponse)),
    info(
        title = "Semantic Uncertainty Runtime API",
        description = "High-performance semantic uncertainty analysis for AI collapse prevention",
        version = "1.0.0"
    )
)]
struct ApiDoc;

/// Create optimized production application
pub async fn create_app() -> Result<Router, Box<dyn std::error::Error>> {
    info!("🚀 Creating production API");
    
    // Ultra-fast configuration for sub-10ms performance
    let config = SemanticConfig::performance();
    let analyzer = Arc::new(SemanticAnalyzer::new(config)?);

    // NEW: initialize broadcaster and session map
    let (tx, _rx) = broadcast::channel::<ServerEvent>(512);
    let sessions: Arc<RwLock<HashMap<String, SessionState>>> = Arc::new(RwLock::new(HashMap::new()));

    let failure_law = load_failure_law_config().unwrap_or(FailureLawConfig{
        lambda: 5.0,
        tau: 1.0,
        risk_pfail_thresholds: RiskPfailThresholds{ critical: 0.8, high_risk: 0.5, warning: 0.2 }
    });

    let state = Arc::new(ApiState {
        analyzer,
        metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
        start_time: Instant::now(),
        broadcaster: tx,
        sessions,
        failure_law,
    });

    // High-performance CORS
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    // Create optimized routes
    let api_routes = Router::new()
        .route("/analyze", post(analyze_handler))
        .route("/health", get(health_handler))
        // NEW: streaming/session routes
        .route("/models", get(models_handler))
        .route("/session/new", post(session_new))
        .route("/session/:id/failure_law", get(get_session_failure_law).post(update_session_failure_law))
        .route("/session/:id/config", post(update_session_config))
        .route("/session/:id/token_metrics", post(token_metrics_update))
        .route("/ws", get(ws_handler))
        .with_state(state);

    // Build with minimal middleware for maximum speed
    let app = Router::new()
        .nest("/api/v1", api_routes)
        .merge(SwaggerUi::new("/docs").url("/api/v1/openapi.json", ApiDoc::openapi()))
        .layer(TimeoutLayer::new(Duration::from_millis(5000)))
        .layer(RequestBodyLimitLayer::new(1024 * 100))
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    info!("✅ API ready");
    Ok(app)
}

/// Ultra-fast analysis endpoint
#[utoipa::path(
    post,
    path = "/analyze",
    request_body = AnalysisRequest,
    responses(
        (status = 200, description = "Analysis completed", body = ApiResponse),
        (status = 400, description = "Invalid input", body = ApiResponse)
    )
)]
#[instrument(skip(state), fields(request_id))]
async fn analyze_handler(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<AnalysisRequest>,
) -> impl IntoResponse {
    let start_time = Instant::now();
    let request_id = RequestId::new();
    
    tracing::Span::current().record("request_id", &tracing::field::display(&request_id));
    
    // Fast input validation
    if request.prompt.len() > 10_000 || request.output.len() > 10_000 {
        return create_error_response("Input too long", request_id, start_time);
    }

    // Perform ultra-fast analysis
    match state.analyzer.analyze(&request.prompt, &request.output, request_id).await {
        Ok(response) => {
            let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            
            // Update metrics
            if let Ok(mut metrics) = state.metrics.lock() {
                metrics.total_requests += 1;
                metrics.successful_requests += 1;
                if response.collapse_risk {
                    metrics.collapse_detections += 1;
                }
                metrics.average_latency_ms = 
                    (metrics.average_latency_ms * (metrics.total_requests - 1) as f64 + processing_time_ms) 
                    / metrics.total_requests as f64;
            }
            
            debug!("✅ Analysis: ℏₛ={:.4}, {:.2}ms", response.hbar_s, processing_time_ms);
            
            let api_response = ApiResponse {
                success: true,
                data: Some(response),
                error: None,
                metadata: ResponseMetadata {
                    request_id: request_id.to_string(),
                    processing_time_ms,
                    version: "1.0.0".to_string(),
                    timestamp: chrono::Utc::now(),
                },
            };
            
            (StatusCode::OK, ResponseJson(api_response))
        }
        Err(e) => {
            if let Ok(mut metrics) = state.metrics.lock() {
                metrics.total_requests += 1;
                metrics.failed_requests += 1;
            }
            
            error!("❌ Analysis failed: {}", e);
            create_error_response(&e.to_string(), request_id, start_time)
        }
    }
}

/// Health check endpoint
#[utoipa::path(
    get,
    path = "/health",
    responses((status = 200, description = "Service health", body = HealthResponse))
)]
async fn health_handler(State(state): State<Arc<ApiState>>) -> impl IntoResponse {
    let uptime_seconds = state.start_time.elapsed().as_secs();
    let metrics = state.metrics.lock().unwrap().clone();
    
    let health = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds,
        metrics,
    };
    
    (StatusCode::OK, ResponseJson(health))
}

/// Models endpoint (reads config/models.json)
async fn models_handler() -> impl IntoResponse {
    let candidates = ["config/models.json", "../config/models.json"]; 
    for p in candidates.iter() {
        if let Ok(data) = std::fs::read_to_string(p) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                return (StatusCode::OK, ResponseJson(json));
            }
        }
    }
    (StatusCode::OK, ResponseJson(serde_json::json!({"models": [], "default_model_id": null})))
}

/// Create a new streaming session
#[derive(Debug, Deserialize)]
struct NewSessionReq { model_id: Option<String> }
#[derive(Debug, Serialize)]
struct NewSessionResp { session_id: String }

async fn session_new(State(state): State<Arc<ApiState>>, Json(req): Json<NewSessionReq>) -> impl IntoResponse {
    let id = Uuid::new_v4().to_string();
    let model_id = req.model_id.unwrap_or_else(|| "unknown".to_string());
    let thr = derive_hbar_thresholds(&state.failure_law);
    let sess = SessionState{
        id: id.clone(),
        created_at: Utc::now().timestamp_millis(),
        token_count: 0,
        model_id,
        lambda: state.failure_law.lambda,
        tau: state.failure_law.tau,
        thresholds: thr,
        prev_probs: None,
        ema_hbar: None,
        config: SessionConfig::default(),
    };
    state.sessions.write().unwrap().insert(id.clone(), sess);
    let _ = state.broadcaster.send(ServerEvent::Welcome{ session_id: id.clone() });
    (StatusCode::OK, ResponseJson(NewSessionResp{ session_id: id }))
}

/// Get/Update failure law per-session
async fn get_session_failure_law(State(state): State<Arc<ApiState>>, Path(id): Path<String>) -> impl IntoResponse {
    if let Some(sess) = state.sessions.read().unwrap().get(&id) {
        return (StatusCode::OK, ResponseJson(serde_json::json!({
            "lambda": sess.lambda,
            "tau": sess.tau,
            "hbar_thresholds": {
                "supercritical": sess.thresholds.supercritical,
                "critical": sess.thresholds.critical
            },
            "model_id": sess.model_id,
        })));
    }
    (StatusCode::OK, ResponseJson(serde_json::json!({ "error": "unknown session" })))
}

#[derive(Debug, Deserialize)]
struct FailureLawUpdate { lambda: Option<f64>, tau: Option<f64> }
async fn update_session_failure_law(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
    Json(body): Json<FailureLawUpdate>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().unwrap();
    if let Some(sess) = sessions.get_mut(&id) {
        if let Some(l) = body.lambda { sess.lambda = l; }
        if let Some(t) = body.tau { sess.tau = t; }
        sess.thresholds = derive_hbar_thresholds(&state.failure_law);
        return (StatusCode::OK, ResponseJson(serde_json::json!({
            "lambda": sess.lambda,
            "tau": sess.tau,
            "hbar_thresholds": {
                "supercritical": sess.thresholds.supercritical,
                "critical": sess.thresholds.critical
            }
        })));
    }
    (StatusCode::OK, ResponseJson(serde_json::json!({ "error": "unknown session" })))
}

/// Update session analysis configuration
async fn update_session_config(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
    Json(cfg): Json<SessionConfig>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().unwrap();
    if let Some(sess) = sessions.get_mut(&id) {
        // Enforce rule: Fisher Full Matrix disables hash embeddings
        let mut cfg_mut = cfg.clone();
        if let Some(pm) = &cfg_mut.precision_method {
            if pm.to_lowercase().contains("fisher") && pm.to_lowercase().contains("full") {
                cfg_mut.hash_embeddings_precision = Some(false);
                cfg_mut.hash_embeddings_flexibility = Some(false);
            }
        }
        sess.config = cfg_mut;
        return (StatusCode::OK, ResponseJson(serde_json::json!({"ok": true})));
    }
    (StatusCode::OK, ResponseJson(serde_json::json!({ "error": "unknown session" })))
}

/// Handle token metrics updates (streaming)
async fn token_metrics_update(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
    Json(body): Json<TokenMetricsUpdate>,
) -> impl IntoResponse {
    let handler_start = Instant::now();

    let mut sessions = state.sessions.write().unwrap();
    let sess = match sessions.get_mut(&id) {
        Some(s) => s,
        None => {
            return (StatusCode::OK, ResponseJson(serde_json::json!({ "error": "unknown session" })));
        }
    };

    sess.token_count += 1;

    let entropy = body.entropy.max(0.0);
    let jsd = body.jsd.max(0.0);
    let hbar = (entropy * jsd).sqrt();

    let ema = match sess.ema_hbar {
        Some(prev) => 0.8 * prev + 0.2 * hbar,
        None => hbar,
    };
    sess.ema_hbar = Some(ema);

    let failure_probability = failure_prob_from_hbar(hbar, sess.lambda, sess.tau);
    let risk_level = risk_from_pfail(failure_probability, &state.failure_law.risk_pfail_thresholds);
    let regime = regime_from_hbar(hbar, &sess.thresholds);

    let processing_time_ms = handler_start.elapsed().as_secs_f64() * 1000.0;
    let ts_ms = Utc::now().timestamp_millis();

    let payload = serde_json::json!({
        "type": "generation_update",
        "session_id": id,
        "model_id": sess.model_id,
        "token_index": body.token_index,
        "token": body.token_text,
        "probability": body.probability.max(1e-12),
        "hbar_s": hbar,
        "rolling_hbar_s": ema,
        "failure_probability": failure_probability,
        "risk_level": risk_level,
        "regime": regime,
        "processing_time_ms": processing_time_ms,
        "timestamp_ms": ts_ms
    });

    let _ = state.broadcaster.send(ServerEvent::TokenUpdate{ session_id: id.clone(), payload: payload.clone() });

    (StatusCode::OK, ResponseJson(payload))
}

/// WebSocket streaming endpoint
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<ApiState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<ApiState>) {
    let mut rx = state.broadcaster.subscribe();
    let _ = socket
        .send(Message::Text(serde_json::json!({"type":"hello","timestamp_ms": Utc::now().timestamp_millis()}).to_string()))
        .await;
    loop {
        match rx.recv().await {
            Ok(event) => {
                if let Ok(text) = serde_json::to_string(&event_to_json(&event)) {
                    if socket.send(Message::Text(text)).await.is_err() { break; }
                }
            }
            Err(_) => break,
        }
    }
}

fn event_to_json(ev: &ServerEvent) -> serde_json::Value {
    match ev {
        ServerEvent::Welcome { session_id } => serde_json::json!({
            "type": "welcome",
            "session_id": session_id,
            "timestamp_ms": Utc::now().timestamp_millis(),
        }),
        ServerEvent::TokenUpdate { session_id, payload } => {
            let mut obj = payload.clone();
            if let Some(map) = obj.as_object_mut() {
                map.insert("session_id".to_string(), serde_json::json!(session_id));
            }
            obj
        }
    }
}

/// Helper: load failure law config
fn load_failure_law_config() -> anyhow::Result<FailureLawConfig> {
    let candidates = [
        "config/failure_law.json",
        "../config/failure_law.json",
    ];
    for path in candidates.iter() {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(cfg) = serde_json::from_str::<FailureLawConfig>(&text) {
                return Ok(cfg);
            }
        }
    }
    let default_text = r#"{
      "lambda": 5.0,
      "tau": 1.0,
      "risk_pfail_thresholds": { "critical": 0.8, "high_risk": 0.5, "warning": 0.2 }
    }"#;
    let cfg: FailureLawConfig = serde_json::from_str(default_text)?;
    Ok(cfg)
}

/// Helper: Pfail from hbar
fn failure_prob_from_hbar(hbar: f64, lambda: f64, tau: f64) -> f64 {
    1.0 / (1.0 + (lambda * (hbar - tau)).exp())
}

/// Helper: invert failure law to get hbar thresholds
fn invert_failure_law_for_hbar(pfail: f64, lambda: f64, tau: f64) -> f64 {
    let odds_inv = (1.0 / pfail) - 1.0;
    tau + odds_inv.ln() / lambda
}

fn derive_hbar_thresholds(cfg: &FailureLawConfig) -> HbarThresholds {
    let supercritical_hbar = invert_failure_law_for_hbar(cfg.risk_pfail_thresholds.critical, cfg.lambda, cfg.tau);
    let critical_hbar = invert_failure_law_for_hbar(cfg.risk_pfail_thresholds.warning, cfg.lambda, cfg.tau);
    HbarThresholds { supercritical: supercritical_hbar, critical: critical_hbar }
}

fn risk_from_pfail(p: f64, thr: &RiskPfailThresholds) -> &'static str {
    if p >= thr.critical { "critical" }
    else if p >= thr.high_risk { "high_risk" }
    else if p >= thr.warning { "warning" }
    else { "safe" }
}

fn regime_from_hbar(hbar: f64, thr: &HbarThresholds) -> &'static str {
    if hbar < thr.supercritical { "stable" }
    else if hbar < thr.critical { "transitional" }
    else { "unstable" }
}

/// Helper function to create error responses
fn create_error_response(
    message: &str,
    request_id: RequestId,
    start_time: Instant,
) -> (StatusCode, ResponseJson<ApiResponse>) {
    let response = ApiResponse {
        success: false,
        data: None,
        error: Some(message.to_string()),
        metadata: ResponseMetadata {
            request_id: request_id.to_string(),
            processing_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now(),
        },
    };
    
    (StatusCode::BAD_REQUEST, ResponseJson(response))
} 