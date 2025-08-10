use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use axum::{routing::{get, post}, Router, extract::{Path, State}, Json};
use axum::response::IntoResponse;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{info, error};
use uuid::Uuid;
use chrono::Utc;

use semantic_uncertainty_runtime::free_energy::{compute_free_energy_for_token};
use semantic_uncertainty_runtime::information_theory::InformationTheoryCalculator;

mod models;
use models::{ModelsRegistry, ModelSpec};

#[derive(Clone)]
struct AppState {
    broadcaster: broadcast::Sender<ServerEvent>,
    sessions: Arc<RwLock<HashMap<String, SessionState>>>,
    registry: Arc<ModelsRegistry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionState {
    id: String,
    created_at: i64,
    token_count: usize,
    model_id: String,
    // rolling stats
    prev_probs: Option<Vec<f64>>,          // for JSD
    ema_hbar: Option<f64>,                 // exponential moving average for rolling_hbar_s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerEvent {
    Welcome { session_id: String },
    TokenUpdate { session_id: String, payload: serde_json::Value },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenInfo {
    index: usize,
    text: String,
    prob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenProbUpdate {
    token_index: usize,
    token_text: String,
    probabilities: Vec<f64>,
    observed_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NewSessionRequest {
    #[serde(default = "default_model_id")] 
    model_id: String,
}

fn default_model_id() -> String { "oss-model".to_string() }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let registry = Arc::new(ModelsRegistry::load_from_file("config/models.json")?);

    let (tx, _rx) = broadcast::channel(1024);
    let app_state = AppState {
        broadcaster: tx.clone(),
        sessions: Arc::new(RwLock::new(HashMap::new())),
        registry,
    };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/ws", get(ws_handler))
        .route("/models", get(list_models))
        .route("/session/new", post(new_session))
        .route("/session/:id/token", post(token_update))
        .with_state(app_state.clone());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8767));
    info!("Starting real-time-engine on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let list: Vec<&ModelSpec> = state.registry.list();
    Json(serde_json::json!({
        "default_model_id": state.registry.default_id,
        "models": list
    }))
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: AppState) {
    let mut rx = state.broadcaster.subscribe();

    let _ = socket
        .send(Message::Text(serde_json::json!({"type":"hello","timestamp_ms": Utc::now().timestamp_millis()}).to_string()))
        .await;

    loop {
        match rx.recv().await {
            Ok(event) => {
                if let Ok(text) = serde_json::to_string(&event_to_json(&event)) {
                    if socket.send(Message::Text(text)).await.is_err() {
                        break;
                    }
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

async fn new_session(State(state): State<AppState>, Json(req): Json<NewSessionRequest>) -> impl IntoResponse {
    // validate/normalize model id
    let chosen = if !req.model_id.is_empty() && state.registry.get(&req.model_id).is_some() {
        req.model_id
    } else {
        state.registry.default_id.clone()
    };

    let id = Uuid::new_v4().to_string();
    let session = SessionState { 
        id: id.clone(), 
        created_at: Utc::now().timestamp(), 
        token_count: 0,
        model_id: chosen,
        prev_probs: None,
        ema_hbar: None,
    };
    state.sessions.write().unwrap().insert(id.clone(), session);
    let _ = state.broadcaster.send(ServerEvent::Welcome { session_id: id.clone() });
    Json(serde_json::json!({ "session_id": id }))
}

async fn token_update(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<TokenProbUpdate>,
) -> impl IntoResponse {
    let probs = body.probabilities;
    let observed_index = body.observed_index.or(Some(argmax_index(&probs)));
    let ts_ms = Utc::now().timestamp_millis();

    let fep = match compute_free_energy_for_token(&probs, observed_index, None, None) {
        Ok(m) => m,
        Err(e) => {
            error!("FEP compute error: {}", e);
            return Json(serde_json::json!({ "error": e.to_string() }));
        }
    };

    let mut sessions = state.sessions.write().unwrap();
    let sess = match sessions.get_mut(&id) {
        Some(s) => s,
        None => {
            return Json(serde_json::json!({ "error": "unknown session" }));
        }
    };

    sess.token_count += 1;
    let model_id = sess.model_id.clone();

    let info = InformationTheoryCalculator::default();
    let entropy = info.shannon_entropy(&probs.iter().map(|p| *p).collect::<Vec<_>>()).unwrap_or(0.0);

    let jsd = if let Some(prev) = &sess.prev_probs {
        if prev.len() == probs.len() {
            info.js_divergence(prev, &probs).unwrap_or(0.0)
        } else { 0.0 }
    } else { 0.0 };

    let hbar = (entropy * jsd).sqrt();

    let ema = match sess.ema_hbar {
        Some(prev) => 0.8 * prev + 0.2 * hbar,
        None => hbar,
    };
    sess.ema_hbar = Some(ema);
    sess.prev_probs = Some(probs.clone());

    let failure_probability = failure_prob_from_hbar(hbar);

    let risk_level = risk_from_hbar(hbar);
    let regime = regime_from_hbar(hbar);

    let anomaly = hbar < 1e-9 || entropy > 1e3 || jsd > 1e3;

    let payload = serde_json::json!({
        "type": "generation_update",
        "session_id": id,
        "model_id": model_id,
        "token_index": body.token_index,
        "token": body.token_text,
        "probability": probs[observed_index.unwrap_or(0)].max(1e-12),
        "hbar_s": hbar,
        "rolling_hbar_s": ema,
        "failure_probability": failure_probability,
        "risk_level": risk_level,
        "regime": regime,
        "fep_surprise": fep.surprise,
        "fep_ambiguity": fep.ambiguity,
        "fep_free_energy": fep.free_energy,
        "anomaly": anomaly,
        "timestamp_ms": ts_ms
    });

    let _ = state.broadcaster.send(ServerEvent::TokenUpdate { session_id: id.clone(), payload: payload.clone() });

    Json(payload)
}

fn failure_prob_from_hbar(hbar: f64) -> f64 {
    let alpha = 5.0; // steepness
    let beta = 1.0;  // threshold
    1.0 / (1.0 + (alpha * (hbar - beta)).exp())
}

fn risk_from_hbar(hbar: f64) -> &'static str {
    if hbar < 0.1 { "critical" }
    else if hbar < 0.3 { "warning" }
    else if hbar < 0.4 { "high_risk" }
    else { "safe" }
}

fn regime_from_hbar(hbar: f64) -> &'static str {
    if hbar < 0.1 { "supercritical" }
    else if hbar < 0.4 { "critical" }
    else { "subcritical" }
}

fn argmax_index(values: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val { best_val = v; best_idx = i; }
    }
    best_idx
} 