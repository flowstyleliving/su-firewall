use serde::{Deserialize, Serialize};
use chrono::Utc;
use common::{SemanticError, RequestId};

#[derive(Debug, Clone)]
pub struct SemanticConfig {
	pub collapse_threshold: f32,
	pub embedding_dims: usize,
	pub timeout_ms: u64,
	pub fast_mode: bool,
}

impl Default for SemanticConfig {
	fn default() -> Self {
		Self {
			collapse_threshold: 1.0,
			embedding_dims: 384,
			timeout_ms: 10_000,
			fast_mode: true,
		}
	}
}

impl SemanticConfig {
	pub fn performance() -> Self { Self::default() }
}

#[derive(Debug, Clone)]
pub struct SemanticAnalyzer {
	pub config: SemanticConfig,
}

impl SemanticAnalyzer {
	pub fn new(config: SemanticConfig) -> Result<Self, SemanticError> { Ok(Self { config }) }

	pub async fn analyze(&self, _prompt: &str, _output: &str, request_id: RequestId) -> Result<HbarResponse, SemanticError> {
		let start = std::time::Instant::now();
		let hbar_s: f32 = 0.0;
		let collapse_risk = hbar_s < self.config.collapse_threshold;
		Ok(HbarResponse {
			request_id,
			hbar_s,
			delta_mu: 0.0,
			delta_sigma: 0.0,
			p_fail: None,
			collapse_risk,
			processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
			embedding_dims: self.config.embedding_dims,
			timestamp: Utc::now(),
			security_assessment: None,
		})
	}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HbarResponse {
	pub request_id: RequestId,
	pub hbar_s: f32,
	pub delta_mu: f32,
	pub delta_sigma: f32,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub p_fail: Option<f64>,
	pub collapse_risk: bool,
	pub processing_time_ms: f64,
	pub embedding_dims: usize,
	pub timestamp: chrono::DateTime<chrono::Utc>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub security_assessment: Option<crate::api_security_analyzer::ApiSecurityAssessment>,
} 