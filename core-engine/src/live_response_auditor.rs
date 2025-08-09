// 🔍 Live Response Auditor
// Model-agnostic real-time uncertainty quantification for any LLM response
// Completely separated from specific model implementations

// Note: oss_logit_adapter will be implemented separately for this demo
// use crate::oss_logit_adapter::{LiveLogitAnalysis, TokenUncertainty};
use crate::RiskLevel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use tokio::sync::{mpsc, RwLock, broadcast};
use std::sync::Arc;
use uuid::Uuid;

/// 📊 Generic token data that any model can provide
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    /// Token text content
    pub text: String,
    /// Token ID (if available)
    pub token_id: Option<u32>,
    /// Token probability (if available)
    pub probability: Option<f64>,
    /// Token logits (if available)
    pub logits: Option<Vec<f32>>,
    /// Token position in sequence
    pub position: usize,
    /// Timestamp when token was generated
    pub timestamp_ms: u64,
    /// Alternative tokens considered (if available)
    pub alternatives: Option<Vec<(String, f64)>>,
}

/// 🎯 Live audit session for tracking a single response generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSession {
    /// Unique session identifier
    pub session_id: Uuid,
    /// Original prompt
    pub prompt: String,
    /// Model information
    pub model_info: ModelInfo,
    /// Session start time
    pub start_time: u64,
    /// Current status
    pub status: SessionStatus,
    /// Accumulated tokens
    pub tokens: Vec<TokenData>,
    /// Real-time uncertainty metrics
    pub live_metrics: LiveAuditMetrics,
    /// Alert history for this session
    pub alerts: Vec<AuditAlert>,
}

/// 🤖 Model information (agnostic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: Option<String>,
    pub framework: String,
    pub capabilities: ModelCapabilities,
}

/// 🔧 What capabilities the model provides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub provides_logits: bool,
    pub provides_probabilities: bool,
    pub provides_alternatives: bool,
    pub provides_attention: bool,
    pub supports_streaming: bool,
}

/// 📈 Real-time audit metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveAuditMetrics {
    /// Current uncertainty score (ℏₛ)
    pub current_uncertainty: f64,
    /// Running average uncertainty
    pub average_uncertainty: f64,
    /// Maximum uncertainty seen
    pub max_uncertainty: f64,
    /// Current risk level
    pub risk_level: RiskLevel,
    /// Tokens processed so far
    pub tokens_processed: u32,
    /// Current generation speed (tokens/sec)
    pub tokens_per_second: f64,
    /// Most uncertain token so far
    pub most_uncertain_token: Option<String>,
    /// Uncertainty trend (last 10 tokens)
    pub uncertainty_trend: VecDeque<f64>,
}

/// 🚨 Audit alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAlert {
    pub alert_id: Uuid,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub token_position: Option<usize>,
    pub uncertainty_value: f64,
    pub timestamp_ms: u64,
    pub auto_resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighUncertainty,
    UncertaintySpike,
    LowConfidence,
    AnomalousPattern,
    ModelDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Starting,
    InProgress,
    Completed,
    Aborted,
    Error(String),
}

/// ⚙️ Live auditor configuration
#[derive(Debug, Clone)]
pub struct LiveAuditorConfig {
    /// Uncertainty threshold for alerts
    pub uncertainty_threshold: f64,
    /// Spike detection sensitivity
    pub spike_threshold: f64,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Buffer size for trend analysis
    pub trend_buffer_size: usize,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Session timeout (milliseconds)
    pub session_timeout_ms: u64,
}

impl Default for LiveAuditorConfig {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 2.5,
            spike_threshold: 1.5,
            confidence_threshold: 0.3,
            trend_buffer_size: 10,
            enable_alerts: true,
            max_concurrent_sessions: 100,
            session_timeout_ms: 300_000, // 5 minutes
        }
    }
}

/// 🔍 Main Live Response Auditor
pub struct LiveResponseAuditor {
    config: LiveAuditorConfig,
    active_sessions: Arc<RwLock<HashMap<Uuid, AuditSession>>>,
    alert_sender: broadcast::Sender<AuditAlert>,
    metrics_sender: broadcast::Sender<LiveAuditMetrics>,
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl LiveResponseAuditor {
    /// 🚀 Create new live response auditor
    pub fn new(config: LiveAuditorConfig) -> Self {
        let (alert_sender, _) = broadcast::channel(1000);
        let (metrics_sender, _) = broadcast::channel(1000);
        
        let active_sessions = Arc::new(RwLock::new(HashMap::new()));
        
        // Start cleanup task for expired sessions
        let cleanup_sessions = active_sessions.clone();
        let cleanup_timeout = config.session_timeout_ms;
        let cleanup_handle = Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                Self::cleanup_expired_sessions(&cleanup_sessions, cleanup_timeout).await;
            }
        }));
        
        Self {
            config,
            active_sessions,
            alert_sender,
            metrics_sender,
            cleanup_handle,
        }
    }

    /// 📝 Start a new audit session
    pub async fn start_session(
        &self,
        prompt: String,
        model_info: ModelInfo,
    ) -> Result<Uuid, Box<dyn std::error::Error>> {
        let session_id = Uuid::new_v4();
        
        // Check if we're at capacity
        {
            let sessions = self.active_sessions.read().await;
            if sessions.len() >= self.config.max_concurrent_sessions {
                return Err("Maximum concurrent sessions reached".into());
            }
        }
        
        let session = AuditSession {
            session_id,
            prompt,
            model_info,
            start_time: current_timestamp(),
            status: SessionStatus::Starting,
            tokens: Vec::new(),
            live_metrics: LiveAuditMetrics::new(),
            alerts: Vec::new(),
        };
        
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id, session);
        }
        
        log::info!("Started audit session: {}", session_id);
        Ok(session_id)
    }

    /// 🎯 Process a new token in real-time
    pub async fn process_token(
        &self,
        session_id: Uuid,
        token_data: TokenData,
    ) -> Result<LiveAuditMetrics, Box<dyn std::error::Error>> {
        let mut sessions = self.active_sessions.write().await;
        
        let session = sessions.get_mut(&session_id)
            .ok_or("Session not found")?;
        
        // Update session status
        session.status = SessionStatus::InProgress;
        
        // Add token to session
        session.tokens.push(token_data.clone());
        
        // Calculate uncertainty for this token
        let token_uncertainty = self.calculate_token_uncertainty(&token_data);
        
        // Update live metrics
        self.update_live_metrics(session, token_uncertainty).await;
        
        // Check for alerts
        if self.config.enable_alerts {
            self.check_alerts(session, &token_data, token_uncertainty).await;
        }
        
        // Broadcast updated metrics
        let _ = self.metrics_sender.send(session.live_metrics.clone());
        
        Ok(session.live_metrics.clone())
    }

    /// ✅ Complete an audit session
    pub async fn complete_session(
        &self,
        session_id: Uuid,
    ) -> Result<AuditSession, Box<dyn std::error::Error>> {
        let mut sessions = self.active_sessions.write().await;
        
        let mut session = sessions.remove(&session_id)
            .ok_or("Session not found")?;
        
        session.status = SessionStatus::Completed;
        
        log::info!("Completed audit session: {} with {} tokens", 
            session_id, session.tokens.len());
        
        Ok(session)
    }

    /// 🎭 Abort an audit session
    pub async fn abort_session(
        &self,
        session_id: Uuid,
        reason: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            session.status = SessionStatus::Aborted;
            log::warn!("Aborted audit session {}: {}", session_id, reason);
        }
        
        sessions.remove(&session_id);
        Ok(())
    }

    /// 📊 Get current session status
    pub async fn get_session(
        &self,
        session_id: Uuid,
    ) -> Option<AuditSession> {
        let sessions = self.active_sessions.read().await;
        sessions.get(&session_id).cloned()
    }

    /// 📈 List all active sessions
    pub async fn list_active_sessions(&self) -> Vec<Uuid> {
        let sessions = self.active_sessions.read().await;
        sessions.keys().cloned().collect()
    }

    /// 📡 Subscribe to real-time alerts
    pub fn subscribe_alerts(&self) -> broadcast::Receiver<AuditAlert> {
        self.alert_sender.subscribe()
    }

    /// 📊 Subscribe to real-time metrics
    pub fn subscribe_metrics(&self) -> broadcast::Receiver<LiveAuditMetrics> {
        self.metrics_sender.subscribe()
    }

    /// 🧮 Calculate uncertainty for a single token
    fn calculate_token_uncertainty(&self, token_data: &TokenData) -> f64 {
        // Use different methods based on available data
        if let Some(logits) = &token_data.logits {
            // Calculate entropy from logits
            self.calculate_entropy_from_logits(logits)
        } else if let Some(probability) = token_data.probability {
            // Estimate uncertainty from probability
            -probability * probability.log2()
        } else {
            // Fallback: estimate from text characteristics
            self.estimate_uncertainty_from_text(&token_data.text)
        }
    }

    /// 🎲 Calculate entropy from logits
    fn calculate_entropy_from_logits(&self, logits: &[f32]) -> f64 {
        // Apply softmax to get probabilities
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f64> = logits.iter()
            .map(|&logit| ((logit - max_logit) as f64).exp())
            .collect();
        
        let sum_exp: f64 = exp_logits.iter().sum();
        let probabilities: Vec<f64> = exp_logits.iter()
            .map(|&exp_logit| exp_logit / sum_exp)
            .collect();
        
        // Calculate Shannon entropy
        probabilities.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.log2())
            .sum()
    }

    /// 📝 Estimate uncertainty from text characteristics (fallback)
    fn estimate_uncertainty_from_text(&self, text: &str) -> f64 {
        // Simple heuristic based on text characteristics
        let base_uncertainty = 1.0;
        
        // Higher uncertainty for:
        // - Longer words (potentially more complex)
        // - Special characters
        // - Numbers
        let length_factor = (text.len() as f64 / 10.0).min(2.0);
        let special_char_factor = if text.chars().any(|c| !c.is_alphanumeric()) { 1.2 } else { 1.0 };
        let number_factor = if text.chars().any(|c| c.is_numeric()) { 1.1 } else { 1.0 };
        
        base_uncertainty * length_factor * special_char_factor * number_factor
    }

    /// 📊 Update live metrics for a session
    async fn update_live_metrics(&self, session: &mut AuditSession, token_uncertainty: f64) {
        let metrics = &mut session.live_metrics;
        
        // Update current uncertainty
        metrics.current_uncertainty = token_uncertainty;
        
        // Update running average
        let token_count = session.tokens.len() as f64;
        metrics.average_uncertainty = 
            (metrics.average_uncertainty * (token_count - 1.0) + token_uncertainty) / token_count;
        
        // Update maximum
        metrics.max_uncertainty = metrics.max_uncertainty.max(token_uncertainty);
        
        // Update token count
        metrics.tokens_processed = session.tokens.len() as u32;
        
        // Calculate tokens per second
        let elapsed_secs = (current_timestamp() - session.start_time) as f64 / 1000.0;
        if elapsed_secs > 0.0 {
            metrics.tokens_per_second = token_count / elapsed_secs;
        }
        
        // Update most uncertain token
        if token_uncertainty == metrics.max_uncertainty {
            metrics.most_uncertain_token = session.tokens.last().map(|t| t.text.clone());
        }
        
        // Update trend
        metrics.uncertainty_trend.push_back(token_uncertainty);
        if metrics.uncertainty_trend.len() > self.config.trend_buffer_size {
            metrics.uncertainty_trend.pop_front();
        }
        
        // Update risk level
        metrics.risk_level = if token_uncertainty > self.config.uncertainty_threshold {
            RiskLevel::Critical
        } else if token_uncertainty > self.config.uncertainty_threshold * 0.7 {
            RiskLevel::HighRisk
        } else if token_uncertainty > self.config.uncertainty_threshold * 0.4 {
            RiskLevel::Warning
        } else {
            RiskLevel::Safe
        };
    }

    /// 🚨 Check for alerts and trigger if necessary
    async fn check_alerts(
        &self,
        session: &mut AuditSession,
        token_data: &TokenData,
        token_uncertainty: f64,
    ) {
        let mut alerts = Vec::new();
        
        // High uncertainty alert
        if token_uncertainty > self.config.uncertainty_threshold {
            alerts.push(AuditAlert {
                alert_id: Uuid::new_v4(),
                alert_type: AlertType::HighUncertainty,
                severity: AlertSeverity::Warning,
                message: format!("High uncertainty detected: {:.3}", token_uncertainty),
                token_position: Some(token_data.position),
                uncertainty_value: token_uncertainty,
                timestamp_ms: current_timestamp(),
                auto_resolved: false,
            });
        }
        
        // Uncertainty spike detection
        if session.live_metrics.uncertainty_trend.len() >= 2 {
            let prev_uncertainty = session.live_metrics.uncertainty_trend
                .get(session.live_metrics.uncertainty_trend.len() - 2)
                .unwrap_or(&0.0);
            
            if token_uncertainty > prev_uncertainty * self.config.spike_threshold {
                alerts.push(AuditAlert {
                    alert_id: Uuid::new_v4(),
                    alert_type: AlertType::UncertaintySpike,
                    severity: AlertSeverity::Critical,
                    message: format!("Uncertainty spike: {:.3} -> {:.3}", 
                        prev_uncertainty, token_uncertainty),
                    token_position: Some(token_data.position),
                    uncertainty_value: token_uncertainty,
                    timestamp_ms: current_timestamp(),
                    auto_resolved: false,
                });
            }
        }
        
        // Low confidence alert
        if let Some(prob) = token_data.probability {
            if prob < self.config.confidence_threshold {
                alerts.push(AuditAlert {
                    alert_id: Uuid::new_v4(),
                    alert_type: AlertType::LowConfidence,
                    severity: AlertSeverity::Warning,
                    message: format!("Low confidence token: '{}' (p={:.3})", 
                        token_data.text, prob),
                    token_position: Some(token_data.position),
                    uncertainty_value: token_uncertainty,
                    timestamp_ms: current_timestamp(),
                    auto_resolved: false,
                });
            }
        }
        
        // Add alerts to session and broadcast
        for alert in alerts {
            session.alerts.push(alert.clone());
            let _ = self.alert_sender.send(alert);
        }
    }

    /// 🧹 Clean up expired sessions
    async fn cleanup_expired_sessions(
        sessions: &Arc<RwLock<HashMap<Uuid, AuditSession>>>,
        timeout_ms: u64,
    ) {
        let current_time = current_timestamp();
        let mut sessions_guard = sessions.write().await;
        
        let expired_sessions: Vec<Uuid> = sessions_guard
            .iter()
            .filter(|(_, session)| {
                current_time - session.start_time > timeout_ms
            })
            .map(|(id, _)| *id)
            .collect();
        
        for session_id in expired_sessions {
            sessions_guard.remove(&session_id);
            log::info!("Cleaned up expired session: {}", session_id);
        }
    }
}

impl LiveAuditMetrics {
    fn new() -> Self {
        Self {
            current_uncertainty: 0.0,
            average_uncertainty: 0.0,
            max_uncertainty: 0.0,
            risk_level: RiskLevel::Safe,
            tokens_processed: 0,
            tokens_per_second: 0.0,
            most_uncertain_token: None,
            uncertainty_trend: VecDeque::new(),
        }
    }
}

impl Drop for LiveResponseAuditor {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }
    }
}

/// 🕒 Helper function to get current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// 🧪 Convenience functions for different model integrations

/// 📤 For models that provide logits (like our Mistral integration)
pub fn create_token_data_from_logits(
    text: String,
    token_id: u32,
    logits: Vec<f32>,
    position: usize,
) -> TokenData {
    // Calculate probability from logits
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_logits: Vec<f32> = logits.iter()
        .map(|&logit| (logit - max_logit).exp())
        .collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probability = exp_logits[token_id as usize] / sum_exp;
    
    TokenData {
        text,
        token_id: Some(token_id),
        probability: Some(probability as f64),
        logits: Some(logits),
        position,
        timestamp_ms: current_timestamp(),
        alternatives: None,
    }
}

/// 📊 For models that only provide probabilities
pub fn create_token_data_from_probability(
    text: String,
    token_id: Option<u32>,
    probability: f64,
    position: usize,
) -> TokenData {
    TokenData {
        text,
        token_id,
        probability: Some(probability),
        logits: None,
        position,
        timestamp_ms: current_timestamp(),
        alternatives: None,
    }
}

/// 📝 For models that only provide text (fallback)
pub fn create_token_data_from_text(
    text: String,
    position: usize,
) -> TokenData {
    TokenData {
        text,
        token_id: None,
        probability: None,
        logits: None,
        position,
        timestamp_ms: current_timestamp(),
        alternatives: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_live_auditor_basic_flow() {
        let config = LiveAuditorConfig::default();
        let auditor = LiveResponseAuditor::new(config);
        
        let model_info = ModelInfo {
            name: "Test Model".to_string(),
            version: Some("1.0".to_string()),
            framework: "Test".to_string(),
            capabilities: ModelCapabilities {
                provides_logits: true,
                provides_probabilities: true,
                provides_alternatives: false,
                provides_attention: false,
                supports_streaming: true,
            },
        };
        
        // Start session
        let session_id = auditor.start_session(
            "Test prompt".to_string(),
            model_info,
        ).await.unwrap();
        
        // Process some tokens
        for i in 0..5 {
            let token_data = create_token_data_from_probability(
                format!("token_{}", i),
                Some(i as u32),
                0.8,
                i,
            );
            
            let metrics = auditor.process_token(session_id, token_data).await.unwrap();
            assert!(metrics.tokens_processed == i as u32 + 1);
        }
        
        // Complete session
        let session = auditor.complete_session(session_id).await.unwrap();
        assert_eq!(session.tokens.len(), 5);
        assert!(matches!(session.status, SessionStatus::Completed));
    }
    
    #[tokio::test]
    async fn test_uncertainty_calculation() {
        let config = LiveAuditorConfig::default();
        let auditor = LiveResponseAuditor::new(config);
        
        // Test entropy calculation from logits
        let logits = vec![2.0, 1.0, 0.5, 0.1, 0.1]; // High confidence distribution
        let entropy = auditor.calculate_entropy_from_logits(&logits);
        assert!(entropy > 0.0 && entropy < 3.0); // Should be relatively low entropy
        
        // Test uniform distribution (high entropy)
        let uniform_logits = vec![1.0; 100];
        let uniform_entropy = auditor.calculate_entropy_from_logits(&uniform_logits);
        assert!(uniform_entropy > entropy); // Should be higher entropy
    }
} 