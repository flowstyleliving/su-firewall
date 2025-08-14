use axum::{routing::{get, post}, Router};
use axum::response::IntoResponse;
use axum::http::StatusCode;
use axum::Json;
use axum::extract::{Path};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Instant;
use chrono::Utc;
use uuid::Uuid;
use crate::metrics;
use crate::{RequestId};
use crate::oss_logit_adapter::{OSSLogitAdapter, LogitData, AdapterConfig, OSSModelFramework};
use thiserror::Error;
use tracing::{info, warn, error, instrument};

static START: OnceLock<Instant> = OnceLock::new();
static MODELS_JSON: OnceLock<serde_json::Value> = OnceLock::new();
static FAILURE_LAW: OnceLock<FailureLaw> = OnceLock::new();

#[derive(Serialize)]
struct Counters {
	requests: u64,
	ws_connections: u64,
	sessions_active: u64,
	analyses: u64,
	firewall_allowed: u64,
	firewall_blocked: u64,
	risk_safe: u64,
	risk_warning: u64,
	risk_high: u64,
	risk_critical: u64,
}

#[derive(Serialize)]
struct Health {
	status: &'static str,
	uptime_ms: u128,
	counters: Counters,
}

#[derive(Serialize)]
struct SessionStartResponse {
	session_id: String,
	started_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FailureLaw {
	lambda: f64,
	tau: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct AnalyzeRequest {
	prompt: String,
	output: String,
	#[serde(default)]
	model_id: Option<String>,
	#[serde(default)]
	method: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AnalyzeResponse {
	request_id: String,
	hbar_s: f64,
	delta_mu: f64,
	delta_sigma: f64,
	p_fail: f64,
	processing_time_ms: f64,
	timestamp: String,
	method: String,
	model_id: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	free_energy: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnalyzeLogitsRequest {
	#[serde(default)]
	model_id: Option<String>,
	/// Token logits for each position (time-major)
	token_logits: Vec<Vec<f32>>,
	/// Optional vocabulary map list (index -> token)
	#[serde(default)]
	vocab: Option<Vec<String>>,
	#[serde(default)]
	temperature: Option<f32>,
	#[serde(default)]
	paraphrase_logits: Option<Vec<Vec<Vec<f32>>>>,
	/// Optional prompt-only next-token logits to build a direction u
	#[serde(default)]
	prompt_next_logits: Option<Vec<f32>>,
	/// Optional method override (e.g., "full_fim_dir")
	#[serde(default)]
	method: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AnalyzeLogitsResponse {
	request_id: String,
	hbar_s: f64,
	delta_mu: f64,
	delta_sigma: f64,
	p_fail: f64,
	processing_time_ms: f64,
	timestamp: String,
	method: String,
	model_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnalyzeTopkRequest {
	#[serde(default)]
	model_id: Option<String>,
	/// For each step, top-k indices and probs; rest_mass for the remainder
	topk_indices: Vec<Vec<usize>>,
	topk_probs: Vec<Vec<f64>>,
	rest_mass: Vec<f64>,
	/// Prompt-side top-k for next-token baseline
	#[serde(default)]
	prompt_next_topk_indices: Option<Vec<usize>>,
	#[serde(default)]
	prompt_next_topk_probs: Option<Vec<f64>>,
	#[serde(default)]
	prompt_next_rest_mass: Option<f64>,
	#[serde(default)]
	vocab_size: Option<usize>,
	#[serde(default)]
	method: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AnalyzeTopkResponse {
	request_id: String,
	hbar_s: f64,
	delta_mu: f64,
	delta_sigma: f64,
	p_fail: f64,
	#[serde(skip_serializing_if = "Option::is_none")]
	free_energy: Option<f64>,
	processing_time_ms: f64,
	timestamp: String,
	method: String,
	model_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnalyzeTopkCompactRequest {
	#[serde(default)]
	model_id: Option<String>,
	// Prompt baseline (p)
	#[serde(default)]
	prompt_next_topk_indices: Option<Vec<usize>>,
	#[serde(default)]
	prompt_next_topk_probs: Option<Vec<f64>>,
	#[serde(default)]
	prompt_next_rest_mass: Option<f64>,
	// Output step (q)
	topk_indices: Vec<usize>,
	topk_probs: Vec<f64>,
	rest_mass: f64,
	#[serde(default)]
	method: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AnalyzeTopkCompactResponse {
	request_id: String,
	hbar_s: f64,
	delta_mu: f64,
	delta_sigma: f64,
	p_fail: f64,
	#[serde(skip_serializing_if = "Option::is_none")]
	free_energy: Option<f64>,
	processing_time_ms: f64,
	timestamp: String,
	method: String,
	model_id: Option<String>,
}

#[derive(Debug, Error)]
enum ApiError {
	#[error("bad request: {0}")]
	BadRequest(String),
	#[error("internal error: {0}")]
	Internal(String),
}

impl IntoResponse for ApiError {
	fn into_response(self) -> axum::response::Response {
		let (code, msg) = match &self {
			ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, m.clone()),
			ApiError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m.clone()),
		};
		(code, msg).into_response()
	}
}

pub fn router() -> Router {
	START.get_or_init(Instant::now);
	// Lazy init configs
	let _ = MODELS_JSON.get_or_init(|| {
		let path = std::path::Path::new("config/models.json");
		match std::fs::read_to_string(path) {
			Ok(s) => serde_json::from_str(&s).unwrap_or(serde_json::json!({"default_model_id": null, "models": []})),
			Err(_) => serde_json::json!({"default_model_id": null, "models": []}),
		}
	});
	let _ = FAILURE_LAW.get_or_init(|| {
		let path = std::path::Path::new("config/failure_law.json");
		match std::fs::read_to_string(path) {
			Ok(s) => {
				let v: serde_json::Value = serde_json::from_str(&s).unwrap_or(serde_json::json!({"lambda":5.0,"tau":1.0}));
				FailureLaw { lambda: v.get("lambda").and_then(|x| x.as_f64()).unwrap_or(5.0), tau: v.get("tau").and_then(|x| x.as_f64()).unwrap_or(1.0) }
			}
			Err(_) => FailureLaw { lambda: 5.0, tau: 1.0 },
		}
	});

	Router::new()
		.route("/health", get(health))
		// Backward-compat simple routes
		.route("/ws", get(ws_upgrade))
		.route("/session/start", post(session_start))
		.route("/session/:id/close", post(session_close))
		// API v1
		.route("/api/v1/health", get(health))
		.route("/api/v1/models", get(models_list))
		.route("/api/v1/analyze", post(analyze))
		.route("/api/v1/analyze_logits", post(analyze_logits))
		.route("/api/v1/analyze_topk", post(analyze_topk))
		.route("/api/v1/analyze_topk_compact", post(analyze_topk_compact))
}

#[instrument(skip_all)]
async fn health() -> impl IntoResponse {
	metrics::record_request();
	let start = START.get().cloned().unwrap_or_else(Instant::now);
	let uptime_ms = start.elapsed().as_millis();
	let m = metrics::metrics();
	let counters = Counters {
		requests: m.requests_total.load(std::sync::atomic::Ordering::Relaxed),
		ws_connections: m.ws_connections_total.load(std::sync::atomic::Ordering::Relaxed),
		sessions_active: m.sessions_active.load(std::sync::atomic::Ordering::Relaxed),
		analyses: m.analyses_total.load(std::sync::atomic::Ordering::Relaxed),
		firewall_allowed: m.firewall_allowed_total.load(std::sync::atomic::Ordering::Relaxed),
		firewall_blocked: m.firewall_blocked_total.load(std::sync::atomic::Ordering::Relaxed),
		risk_safe: m.risk_safe_total.load(std::sync::atomic::Ordering::Relaxed),
		risk_warning: m.risk_warning_total.load(std::sync::atomic::Ordering::Relaxed),
		risk_high: m.risk_high_total.load(std::sync::atomic::Ordering::Relaxed),
		risk_critical: m.risk_critical_total.load(std::sync::atomic::Ordering::Relaxed),
	};
	Json(Health { status: "ok", uptime_ms, counters })
}

use axum::extract::ws::{WebSocketUpgrade, WebSocket, Message};
#[derive(Debug, Deserialize)]
struct WsAnalyzeTopkMessage {
	model_id: Option<String>,
	topk_indices: Vec<Vec<usize>>,
	topk_probs: Vec<Vec<f64>>,
	rest_mass: Vec<f64>,
	vocab_size: usize,
	alpha: Option<f64>, // EMA alpha
}

#[instrument(skip_all)]
async fn ws_upgrade(ws: WebSocketUpgrade) -> impl IntoResponse {
	metrics::record_request();
	ws.on_upgrade(handle_ws)
}

#[instrument(skip_all)]
async fn handle_ws(mut socket: WebSocket) {
	metrics::record_ws_connection();
	while let Some(msg) = socket.recv().await {
		match msg {
			Ok(Message::Text(t)) => {
				// Try to interpret as analyze_topk streaming request
				match serde_json::from_str::<WsAnalyzeTopkMessage>(&t) {
					Ok(req) => {
						let vocab_size = req.vocab_size;
						let alpha = req.alpha.unwrap_or(0.2);
						let mut ema_h: f64 = 0.0;
						for step in 0..req.topk_indices.len().min(req.topk_probs.len()) {
							let q = densify_from_topk(vocab_size, &req.topk_indices[step], &req.topk_probs[step], *req.rest_mass.get(step).unwrap_or(&0.0));
							let p = if step==0 { q.clone() } else { densify_from_topk(vocab_size, &req.topk_indices[step-1], &req.topk_probs[step-1], *req.rest_mass.get(step-1).unwrap_or(&0.0)) };
							let fim = diag_fim_from_dist(&p,&q);
							let u = build_u(&p,&q);
							let dm = directional_precision_diag(&fim,&u);
							let ds = flexibility_diag_inv(&fim,&u);
							let h = (dm * ds).sqrt();
							ema_h = alpha*h + (1.0-alpha)*ema_h;
							let law = FAILURE_LAW.get().cloned().unwrap_or(FailureLaw{lambda:5.0,tau:1.0});
							let pf = pfail_from_hbar(ema_h, &law);
							let risk = if ema_h <= law.tau { "safe" } else if ema_h <= law.tau+0.5 { "warning" } else if ema_h <= law.tau+1.0 { "high" } else { "critical" };
							let _ = socket.send(Message::Text(serde_json::json!({
								"step": step,
								"ema_hbar": ema_h,
								"p_fail": pf,
								"risk": risk,
							}).to_string())).await;
						}
					}
					Err(_) => {
						let _ = socket.send(Message::Text("unsupported message".to_string())).await;
					}
				}
			}
			Ok(Message::Binary(b)) => { let _ = socket.send(Message::Binary(b)).await; }
			Ok(_) => {}
			Err(e) => { warn!(error=?e, "websocket closed with error"); break },
		}
	}
}

#[instrument(skip_all)]
async fn session_start() -> impl IntoResponse {
	metrics::record_request();
	metrics::inc_session();
	let session_id = Uuid::new_v4().to_string();
	let started_at = Utc::now().to_rfc3339();
	Json(SessionStartResponse { session_id, started_at })
}

#[instrument(skip_all)]
async fn session_close(Path(id): Path<String>) -> impl IntoResponse {
	let _ = id; // placeholder until session store is implemented
	metrics::record_request();
	metrics::dec_session();
	(StatusCode::OK, "closed")
}

#[instrument(skip_all)]
async fn models_list() -> impl IntoResponse {
	metrics::record_request();
	let v = MODELS_JSON.get().cloned().unwrap_or(serde_json::json!({"default_model_id": null, "models": []}));
	Json(v)
}

fn tokenize(text: &str) -> Vec<String> {
	text.to_lowercase()
		.chars()
		.map(|c| if c.is_alphanumeric() { c } else { ' ' })
		.collect::<String>()
		.split_whitespace()
		.map(|s| s.to_string())
		.collect()
}

fn build_distributions(prompt: &str, output: &str) -> (Vec<f64>, Vec<f64>) {
	let p_tokens = tokenize(prompt);
	let o_tokens = tokenize(output);
	let mut vocab: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
	for t in &p_tokens { let _ = vocab.entry(t.clone()).or_insert(0usize); }
	for t in &o_tokens { let _ = vocab.entry(t.clone()).or_insert(0usize); }
	let dim = vocab.len().max(1);
	let mut p = vec![0.0f64; dim];
	let mut q = vec![0.0f64; dim];
	for (idx, (tok, _)) in vocab.into_iter().enumerate() {
		let pc = p_tokens.iter().filter(|x| **x == tok).count() as f64;
		let qc = o_tokens.iter().filter(|x| **x == tok).count() as f64;
		p[idx] = pc;
		q[idx] = qc;
	}
	let sp: f64 = p.iter().sum();
	let sq: f64 = q.iter().sum();
	if sp > 0.0 { for v in &mut p { *v /= sp; } }
	if sq > 0.0 { for v in &mut q { *v /= sq; } }
	(p, q)
}

fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
	let m: Vec<f64> = p.iter().zip(q.iter()).map(|(a,b)| 0.5*(a+b)).collect();
	fn kl(a:&[f64], b:&[f64]) -> f64 {
		let mut s=0.0; let eps=1e-12;
		let sa: f64 = a.iter().sum(); let sb: f64 = b.iter().sum();
		for i in 0..a.len() {
			let ai=(a[i]/sa).max(eps); let bi=(b[i]/sb).max(eps);
			s += ai * (ai/bi).ln();
		}
		s
	}
	0.5*(kl(p,&m)+kl(q,&m))
}

fn diag_fim_from_dist(p: &[f64], q: &[f64]) -> Vec<f64> {
	let eps=1e-12;
	let mut d = vec![0.0; p.len()];
	for i in 0..p.len() {
		let pi = (p[i]+q[i]).max(eps) * 0.5; // combine for stability
		d[i] = 1.0 / pi + 1e-8;
	}
	d
}

fn build_u(p: &[f64], q: &[f64]) -> Vec<f64> {
	let mut u: Vec<f64> = p.iter().zip(q.iter()).map(|(a,b)| b-a).collect();
	let norm = u.iter().map(|v| v*v).sum::<f64>().sqrt().max(1e-12);
	for v in &mut u { *v /= norm; }
	u
}

fn directional_precision_diag(fim_diag: &[f64], u: &[f64]) -> f64 {
	let mut acc=0.0; let n=fim_diag.len().min(u.len());
	for i in 0..n { acc += u[i]*u[i]*fim_diag[i]; }
	acc
}

fn flexibility_diag_inv(fim_diag: &[f64], u: &[f64]) -> f64 {
	let mut acc=0.0; let n=fim_diag.len().min(u.len());
	for i in 0..n { acc += u[i]*u[i]/fim_diag[i].max(1e-12); }
	acc.sqrt()
}

fn pfail_from_hbar(h: f64, law: &FailureLaw) -> f64 {
	1.0 / (1.0 + (-law.lambda * (h - law.tau)).exp())
}

fn fep_summary(p: &[f64], q: &[f64]) -> f64 {
	// Use output argmax as observed; prior=prompt, q_post=output
	let idx = q.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0);
	match common::math::free_energy::compute_free_energy_for_token(q, Some(idx), Some(p), Some(q)) {
		Ok(m) => m.free_energy,
		Err(_) => 0.0,
	}
}

fn densify_from_topk(vocab_size: usize, idx: &[usize], pv: &[f64], rest: f64) -> Vec<f64> {
	let mut p = vec![0.0; vocab_size];
	let mut sum=0.0; for (i,pr) in idx.iter().zip(pv.iter()) { p[*i]=*pr; sum+=*pr; }
	let rem = ((1.0 - sum).max(0.0)).min(rest);
	if vocab_size>0 { let add = rem / (vocab_size as f64); for x in &mut p { *x += add; } }
	// normalize
	let s: f64 = p.iter().sum(); if s>0.0 { for x in &mut p { *x /= s; } }
	p
}

fn build_compact_union_distributions(
	p_idx: Option<&[usize]>, p_prob: Option<&[f64]>, p_rest: f64,
	q_idx: &[usize], q_prob: &[f64], q_rest: f64,
) -> (Vec<f64>, Vec<f64>) {
	use std::collections::BTreeSet;
	let mut union: BTreeSet<usize> = BTreeSet::new();
	if let (Some(pi), _) = (p_idx, p_prob) { for i in pi { union.insert(*i); } }
	for i in q_idx { union.insert(*i); }
	let mut p_map: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
	let mut q_map: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
	if let (Some(pi), Some(pp)) = (p_idx, p_prob) {
		for (i, pr) in pi.iter().zip(pp.iter()) { p_map.insert(*i, *pr); }
	}
	for (i, pr) in q_idx.iter().zip(q_prob.iter()) { q_map.insert(*i, *pr); }
	let mut p_vec: Vec<f64> = Vec::with_capacity(union.len()+1);
	let mut q_vec: Vec<f64> = Vec::with_capacity(union.len()+1);
	let mut psum=0.0f64; let mut qsum=0.0f64;
	for i in union.iter() {
		let pv = *p_map.get(i).unwrap_or(&0.0);
		let qv = *q_map.get(i).unwrap_or(&0.0);
		p_vec.push(pv); q_vec.push(qv);
		psum += pv; qsum += qv;
	}
	let p_other = (1.0 - psum).max(0.0).min(p_rest.max(0.0));
	let q_other = (1.0 - qsum).max(0.0).min(q_rest.max(0.0));
	p_vec.push(p_other); q_vec.push(q_other);
	// normalize defensively
	let sp: f64 = p_vec.iter().sum(); if sp>0.0 { for v in &mut p_vec { *v /= sp; } }
	let sq: f64 = q_vec.iter().sum(); if sq>0.0 { for v in &mut q_vec { *v /= sq; } }
	(p_vec, q_vec)
}

#[instrument(skip_all, fields(method=?req.method, model=?req.model_id))]
async fn analyze(Json(req): Json<AnalyzeRequest>) -> impl IntoResponse {
	metrics::record_request();
	let t0 = std::time::Instant::now();
	let method = req.method.clone().unwrap_or_else(|| "diag_fim_dir".to_string());
	let (p, q) = build_distributions(&req.prompt, &req.output);
	let (delta_mu, delta_sigma) = match method.as_str() {
		"diag_fim_dir" => {
			let fim = diag_fim_from_dist(&p,&q);
			let u = build_u(&p,&q);
			(directional_precision_diag(&fim,&u), flexibility_diag_inv(&fim,&u))
		}
		"scalar_js_kl" => {
			(js_divergence(&p,&q).max(1e-9), js_divergence(&q,&p).max(1e-9))
		}
		"scalar_trace" => {
			let d = diag_fim_from_dist(&p,&q); let tr: f64 = d.iter().sum(); (tr, 1.0/tr.max(1e-9))
		}
		"scalar_fro" => {
			let d = diag_fim_from_dist(&p,&q); let fro: f64 = d.iter().map(|x| x*x).sum::<f64>().sqrt(); (fro, 1.0/fro.max(1e-9))
		}
		_ => {
			let fim = diag_fim_from_dist(&p,&q); let u = build_u(&p,&q); (directional_precision_diag(&fim,&u), flexibility_diag_inv(&fim,&u))
		}
	};
	let hbar_s = (delta_mu * delta_sigma).sqrt();
	let law = FAILURE_LAW.get().cloned().unwrap_or(FailureLaw{lambda:5.0,tau:1.0});
	let p_fail = pfail_from_hbar(hbar_s, &law);
	let fe = fep_summary(&p,&q);
	let resp = AnalyzeResponse {
		request_id: Uuid::new_v4().to_string(),
		hbar_s,
		delta_mu,
		delta_sigma,
		p_fail,
		processing_time_ms: t0.elapsed().as_secs_f64()*1000.0,
		timestamp: Utc::now().to_rfc3339(),
		method,
		model_id: req.model_id.clone(),
		free_energy: Some(fe),
	};
	Json(resp)
}

#[instrument(skip_all, fields(method=?req.method, model=?req.model_id))]
async fn analyze_logits(Json(req): Json<AnalyzeLogitsRequest>) -> impl IntoResponse {
	metrics::record_request();
	let t0 = std::time::Instant::now();
	let vocab_map: std::collections::HashMap<u32, String> = match &req.vocab {
		Some(v) => v.iter().enumerate().map(|(i,s)| (i as u32, s.clone())).collect(),
		None => std::collections::HashMap::new(),
	};
	let logit_data = LogitData {
		token_logits: req.token_logits.clone(),
		vocab_map,
		attention_weights: None,
		hidden_states: None,
		temperature: req.temperature.unwrap_or(1.0),
		top_p: None,
		token_sequence: vec![0u32; req.token_logits.len()],
		gradients: None,
		paraphrase_logits: req.paraphrase_logits.clone(),
	};
	let adapter_cfg = AdapterConfig { ..Default::default() };
	let mut adapter = OSSLogitAdapter::new(OSSModelFramework::HuggingFaceTransformers, adapter_cfg);
	let rid = RequestId::new();
	let result = match adapter.analyze_logits("", &logit_data, rid) {
		Ok(res) => res,
		Err(_) => {
			return (StatusCode::BAD_REQUEST, "invalid logits").into_response();
		}
	};
	let mut base = result.base_result;
	// Optional override for method using prompt_next_logits to build u with full Fisher from logits
	if let Some(m) = &req.method {
		if m == "full_fim_dir" {
			if let Some(prompt_next) = &req.prompt_next_logits {
				// Build distributions p (prompt next) and q (output step 1) from logits via softmax
				fn softmax(v: &Vec<f32>) -> Vec<f64> {
					let maxv = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
					let exps: Vec<f64> = v.iter().map(|x| ((*x - maxv) as f64).exp()).collect();
					let sum: f64 = exps.iter().sum();
					exps.into_iter().map(|e| e / sum.max(1e-300)).collect()
				}
				let p = softmax(prompt_next);
				let q = softmax(&req.token_logits.get(0).cloned().unwrap_or_default());
				// u = q - p
				let u: Vec<f64> = q.iter().zip(p.iter()).map(|(a,b)| a - b).collect();
				// Fisher in prob space: F = diag(p) - p p^T. Directional precision: u^T F u
				let mut quad: f64 = 0.0;
				let mut dot_up: f64 = 0.0;
				for (ui, pi) in u.iter().zip(p.iter()) { dot_up += ui * pi; }
				for (ui, pi) in u.iter().zip(p.iter()) { quad += ui*ui*pi; }
				let dir_prec = (quad - dot_up*dot_up).max(0.0);
				// Flexibility fast diag-inv approximation using p as diag entries
				let mut inv_quad: f64 = 0.0;
				for (ui, pi) in u.iter().zip(p.iter()) { inv_quad += ui*ui / pi.max(1e-12); }
				let dir_flex = inv_quad.sqrt();
				base.delta_mu = dir_prec;
				base.delta_sigma = dir_flex;
				base.raw_hbar = (base.delta_mu * base.delta_sigma).sqrt();
				// mark via response method field below
			}
		}
	}
	let law = FAILURE_LAW.get().cloned().unwrap_or(FailureLaw{lambda:5.0,tau:1.0});
	let p_fail = pfail_from_hbar(base.raw_hbar, &law);
	let resp = AnalyzeLogitsResponse {
		request_id: base.request_id.to_string(),
		hbar_s: base.raw_hbar,
		delta_mu: base.delta_mu,
		delta_sigma: base.delta_sigma,
		p_fail,
		processing_time_ms: base.processing_time_ms,
		timestamp: base.timestamp.to_rfc3339(),
		method: req.method.clone().unwrap_or_else(|| "logits_adapter".to_string()),
		model_id: req.model_id.clone(),
	};
	Json(resp).into_response()
}

#[instrument(skip_all, fields(method=?req.method, model=?req.model_id))]
async fn analyze_topk(Json(req): Json<AnalyzeTopkRequest>) -> impl IntoResponse {
	metrics::record_request();
	let t0 = std::time::Instant::now();
	let vocab_size = req.vocab_size.unwrap_or(0);
	if req.topk_indices.is_empty() || req.topk_probs.is_empty() || vocab_size==0 {
		return (StatusCode::BAD_REQUEST, "missing topk or vocab_size").into_response();
	}
	// Use first step as output q
	let q = densify_from_topk(vocab_size, &req.topk_indices[0], &req.topk_probs[0], *req.rest_mass.get(0).unwrap_or(&0.0));
	// Prompt baseline p
	let p = if let (Some(pi), Some(pv)) = (req.prompt_next_topk_indices.as_ref(), req.prompt_next_topk_probs.as_ref()) {
		densify_from_topk(vocab_size, pi, pv, req.prompt_next_rest_mass.unwrap_or(0.0))
	} else { q.clone() };
	let method = req.method.clone().unwrap_or_else(|| "full_fim_dir".to_string());
	let (delta_mu, delta_sigma) = match method.as_str() {
		"full_fim_dir" => {
			let u: Vec<f64> = q.iter().zip(p.iter()).map(|(a,b)| a-b).collect();
			let dot_up: f64 = u.iter().zip(p.iter()).map(|(ui,pi)| ui*pi).sum();
			let quad: f64 = u.iter().zip(p.iter()).map(|(ui,pi)| ui*ui*pi).sum();
			let dir_prec = (quad - dot_up*dot_up).max(0.0);
			let inv_quad: f64 = u.iter().zip(p.iter()).map(|(ui,pi)| ui*ui / pi.max(1e-12)).sum();
			(dir_prec, inv_quad.sqrt())
		}
		_ => {
			let fim = diag_fim_from_dist(&p,&q);
			let u = build_u(&p,&q);
			(directional_precision_diag(&fim,&u), flexibility_diag_inv(&fim,&u))
		}
	};
	let hbar_s = (delta_mu * delta_sigma).sqrt();
	let law = FAILURE_LAW.get().cloned().unwrap_or(FailureLaw{lambda:5.0,tau:1.0});
	let p_fail = pfail_from_hbar(hbar_s, &law);
	let fe = fep_summary(&p,&q);
	let resp = AnalyzeTopkResponse{
		request_id: Uuid::new_v4().to_string(),
		hbar_s,
		delta_mu,
		delta_sigma,
		p_fail,
		free_energy: Some(fe),
		processing_time_ms: t0.elapsed().as_secs_f64()*1000.0,
		timestamp: Utc::now().to_rfc3339(),
		method,
		model_id: req.model_id.clone(),
	};
	Json(resp).into_response()
}

#[instrument(skip_all, fields(method=?req.method, model=?req.model_id))]
async fn analyze_topk_compact(Json(req): Json<AnalyzeTopkCompactRequest>) -> impl IntoResponse {
	metrics::record_request();
	let t0 = std::time::Instant::now();
	if req.topk_indices.len() != req.topk_probs.len() {
		return (StatusCode::BAD_REQUEST, "length mismatch in topk").into_response();
	}
	let (p, q) = build_compact_union_distributions(
		req.prompt_next_topk_indices.as_deref(),
		req.prompt_next_topk_probs.as_deref(),
		req.prompt_next_rest_mass.unwrap_or(0.0),
		&req.topk_indices,
		&req.topk_probs,
		req.rest_mass,
	);
	let method = req.method.clone().unwrap_or_else(|| "full_fim_dir".to_string());
	let (delta_mu, delta_sigma) = match method.as_str() {
		"full_fim_dir" => {
			let u: Vec<f64> = q.iter().zip(p.iter()).map(|(a,b)| a-b).collect();
			let dot_up: f64 = u.iter().zip(p.iter()).map(|(ui,pi)| ui*pi).sum();
			let quad: f64 = u.iter().zip(p.iter()).map(|(ui,pi)| ui*ui*pi).sum();
			let dir_prec = (quad - dot_up*dot_up).max(0.0);
			let inv_quad: f64 = u.iter().zip(p.iter()).map(|(ui,pi)| ui*ui / pi.max(1e-12)).sum();
			(dir_prec, inv_quad.sqrt())
		}
		_ => {
			let fim = diag_fim_from_dist(&p,&q);
			let u = build_u(&p,&q);
			(directional_precision_diag(&fim,&u), flexibility_diag_inv(&fim,&u))
		}
	};
	let hbar_s = (delta_mu * delta_sigma).sqrt();
	let law = FAILURE_LAW.get().cloned().unwrap_or(FailureLaw{lambda:5.0,tau:1.0});
	let p_fail = pfail_from_hbar(hbar_s, &law);
	let fe = fep_summary(&p,&q);
	let resp = AnalyzeTopkCompactResponse{
		request_id: Uuid::new_v4().to_string(),
		hbar_s,
		delta_mu,
		delta_sigma,
		p_fail,
		free_energy: Some(fe),
		processing_time_ms: t0.elapsed().as_secs_f64()*1000.0,
		timestamp: Utc::now().to_rfc3339(),
		method,
		model_id: req.model_id.clone(),
	};
	Json(resp).into_response()
} 