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

/// Streamlined API request for maximum performance
#[derive(Debug, Deserialize, ToSchema)]
pub struct AnalysisRequest {
    /// Input prompt (max 10,000 characters)
    pub prompt: String,
    /// Output response (max 10,000 characters)  
    pub output: String,
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
#[derive(Debug, Serialize, ToSchema)]
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
#[derive(Debug, Serialize, ToSchema)]
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
    
    let state = Arc::new(ApiState {
        analyzer,
        metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
        start_time: Instant::now(),
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
        .with_state(state);

    // Build with minimal middleware for maximum speed
    let app = Router::new()
        .nest("/api/v1", api_routes)
        .merge(SwaggerUi::new("/docs").url("/api/v1/openapi.json", ApiDoc::openapi()))
        .layer(
            ServiceBuilder::new()
                .layer(TimeoutLayer::new(Duration::from_millis(5000)))
                .layer(RequestBodyLimitLayer::new(1024 * 100)) // 100KB limit for speed
                .layer(TraceLayer::new_for_http())
                .layer(cors)
        );

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
            let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            
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