// 📊 Live UQ/Hbar Audit Dashboard for OSS Models
// Real-time monitoring and alerting for uncertainty quantification

use crate::oss_logit_adapter::LiveLogitAnalysis;
use common::RiskLevel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;

/// 📈 Real-time dashboard state
#[derive(Debug, Clone)]
pub struct LiveAuditDashboard {
    /// Current session metrics
    session_state: Arc<RwLock<SessionState>>,
    /// Alert system
    alert_system: Arc<RwLock<AlertSystem>>,
    /// Historical data buffer
    history_buffer: Arc<RwLock<HistoryBuffer>>,
    /// Configuration
    config: DashboardConfig,
}

/// 📊 Current session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    /// Session start time
    pub session_start: u64,
    /// Total tokens processed
    pub total_tokens: u64,
    /// Current processing rate (tokens/sec)
    pub current_tps: f64,
    /// Average uncertainty over session
    pub session_avg_hbar: f64,
    /// Current risk level
    pub current_risk: RiskLevel,
    /// Last N uncertainty values
    pub recent_hbar_values: VecDeque<f64>,
    /// Active model information
    pub model_info: ModelInfo,
    /// Live metrics
    pub live_metrics: LiveMetrics,
}

/// 🧠 Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: String,
    pub framework: String,
    pub vocab_size: u32,
    pub context_length: u32,
}

/// ⚡ Live processing metrics  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveMetrics {
    /// Current uncertainty (ℏₛ)
    pub current_hbar: f64,
    /// Current entropy
    pub current_entropy: f64,
    /// Current confidence
    pub current_confidence: f64,
    /// Current perplexity
    pub current_perplexity: f64,
    /// Tokens processed in last second
    pub recent_token_count: u32,
    /// Processing latency (ms per token)
    pub avg_latency_ms: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
}

/// 🚨 Alert management system
#[derive(Debug, Clone)]
pub struct AlertSystem {
    /// Active alerts
    active_alerts: HashMap<String, Alert>,
    /// Alert history
    alert_history: VecDeque<Alert>,
    /// Alert thresholds
    thresholds: AlertThresholds,
    /// Notification channels
    notification_tx: Option<mpsc::UnboundedSender<AlertNotification>>,
}

/// ⚠️ Individual alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: u64,
    pub metrics: AlertMetrics,
    pub auto_resolve: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighUncertainty,
    UncertaintySpike,
    LowConfidence,
    ProcessingDelay,
    MemoryPressure,
    ModelDrift,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// 📏 Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// ℏₛ threshold for high uncertainty
    pub hbar_critical: f64,
    /// ℏₛ threshold for warning
    pub hbar_warning: f64,
    /// Confidence threshold for alerts
    pub confidence_threshold: f64,
    /// Entropy threshold
    pub entropy_threshold: f64,
    /// Processing latency threshold (ms)
    pub latency_threshold_ms: f64,
    /// Memory usage threshold (MB)
    pub memory_threshold_mb: f64,
}

/// 📊 Alert notification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertMetrics {
    pub hbar_value: f64,
    pub entropy: f64,
    pub confidence: f64,
    pub token_position: Option<usize>,
    pub token_text: Option<String>,
}

/// 📨 Alert notification
#[derive(Debug, Clone)]
pub struct AlertNotification {
    pub alert: Alert,
    pub dashboard_snapshot: DashboardSnapshot,
}

/// 📸 Dashboard snapshot for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub timestamp: u64,
    pub session_duration_mins: f64,
    pub total_tokens: u64,
    pub avg_uncertainty: f64,
    pub current_risk: RiskLevel,
    pub recent_trend: Vec<f64>,
}

/// 📈 Historical data buffer
#[derive(Debug, Clone)]
pub struct HistoryBuffer {
    /// Uncertainty history (time, ℏₛ value)
    uncertainty_history: VecDeque<(u64, f64)>,
    /// Token-level history
    token_history: VecDeque<TokenHistoryEntry>,
    /// Performance metrics history
    performance_history: VecDeque<PerformanceSnapshot>,
    /// Buffer size limits
    max_uncertainty_points: usize,
    max_token_entries: usize,
    max_performance_entries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenHistoryEntry {
    pub timestamp: u64,
    pub position: usize,
    pub token: String,
    pub uncertainty: f64,
    pub confidence: f64,
    pub local_hbar: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: u64,
    pub tokens_per_second: f64,
    pub avg_latency_ms: f64,
    pub memory_usage_mb: f64,
    pub active_alerts: u32,
}

/// ⚙️ Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Update frequency (ms)
    pub update_interval_ms: u64,
    /// History buffer sizes
    pub history_buffer_size: usize,
    /// Alert settings
    pub alert_thresholds: AlertThresholds,
    /// Enable detailed token tracking
    pub track_individual_tokens: bool,
    /// Auto-export interval (minutes)
    pub auto_export_interval_mins: Option<u64>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 100, // 10 Hz updates
            history_buffer_size: 10000,
            alert_thresholds: AlertThresholds::default(),
            track_individual_tokens: true,
            auto_export_interval_mins: Some(60), // Export every hour
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            hbar_critical: 0.4,  // Critical uncertainty
            hbar_warning: 0.8,   // Warning uncertainty  
            confidence_threshold: 0.3,  // Low confidence
            entropy_threshold: 3.0,     // High entropy
            latency_threshold_ms: 100.0, // Processing delays
            memory_threshold_mb: 2048.0, // Memory pressure
        }
    }
}

impl LiveAuditDashboard {
    /// 🚀 Initialize new live audit dashboard
    pub async fn new(config: DashboardConfig, model_info: ModelInfo) -> Self {
        let session_state = Arc::new(RwLock::new(SessionState {
            session_start: current_timestamp(),
            total_tokens: 0,
            current_tps: 0.0,
            session_avg_hbar: 0.0,
            current_risk: RiskLevel::Safe,
            recent_hbar_values: VecDeque::with_capacity(100),
            model_info,
            live_metrics: LiveMetrics::default(),
        }));

        let alert_system = Arc::new(RwLock::new(AlertSystem {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::with_capacity(1000),
            thresholds: config.alert_thresholds.clone(),
            notification_tx: None,
        }));

        let history_buffer = Arc::new(RwLock::new(HistoryBuffer {
            uncertainty_history: VecDeque::with_capacity(config.history_buffer_size),
            token_history: VecDeque::with_capacity(config.history_buffer_size),
            performance_history: VecDeque::with_capacity(1000),
            max_uncertainty_points: config.history_buffer_size,
            max_token_entries: config.history_buffer_size,
            max_performance_entries: 1000,
        }));

        Self {
            session_state,
            alert_system,
            history_buffer,
            config,
        }
    }

    /// 📊 Process new analysis result
    pub async fn process_analysis(&self, analysis: &LiveLogitAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp = current_timestamp();
        
        // Update session state
        {
            let mut state = self.session_state.write().await;
            state.total_tokens += analysis.token_uncertainties.len() as u64;
            
            // Update live metrics
            state.live_metrics = LiveMetrics {
                current_hbar: analysis.base_result.calibrated_hbar,
                current_entropy: analysis.logit_metrics.average_entropy,
                current_confidence: analysis.logit_metrics.confidence_score,
                current_perplexity: analysis.logit_metrics.perplexity,
                recent_token_count: analysis.token_uncertainties.len() as u32,
                avg_latency_ms: analysis.streaming_metrics.token_latencies_ms.iter().sum::<f64>() 
                    / analysis.streaming_metrics.token_latencies_ms.len() as f64,
                memory_usage_mb: analysis.streaming_metrics.memory_usage_mb,
            };
            
            // Update uncertainty history
            state.recent_hbar_values.push_back(analysis.base_result.calibrated_hbar);
            if state.recent_hbar_values.len() > 100 {
                state.recent_hbar_values.pop_front();
            }
            
            // Update session average
            let total_uncertainty: f64 = state.recent_hbar_values.iter().sum();
            state.session_avg_hbar = total_uncertainty / state.recent_hbar_values.len() as f64;
            
            // Update current risk
            state.current_risk = analysis.base_result.risk_level.clone();
            
            // Calculate TPS
            let session_duration = (timestamp - state.session_start) as f64 / 1000.0; // seconds
            if session_duration > 0.0 {
                state.current_tps = state.total_tokens as f64 / session_duration;
            }
        }

        // Update history buffer
        {
            let mut history = self.history_buffer.write().await;
            
            // Add uncertainty point
            history.uncertainty_history.push_back((timestamp, analysis.base_result.calibrated_hbar));
            if history.uncertainty_history.len() > history.max_uncertainty_points {
                history.uncertainty_history.pop_front();
            }
            
            // Add token-level data if enabled
            if self.config.track_individual_tokens {
                for token_uncertainty in &analysis.token_uncertainties {
                    let entry = TokenHistoryEntry {
                        timestamp,
                        position: token_uncertainty.position,
                        token: token_uncertainty.token_string.clone(),
                        uncertainty: token_uncertainty.token_entropy,
                        confidence: token_uncertainty.token_probability,
                        local_hbar: token_uncertainty.local_hbar,
                    };
                    
                    history.token_history.push_back(entry);
                    if history.token_history.len() > history.max_token_entries {
                        history.token_history.pop_front();
                    }
                }
            }
            
            // Add performance snapshot
            let perf_snapshot = PerformanceSnapshot {
                timestamp,
                tokens_per_second: analysis.streaming_metrics.throughput_tps,
                avg_latency_ms: analysis.streaming_metrics.token_latencies_ms.iter().sum::<f64>() 
                    / analysis.streaming_metrics.token_latencies_ms.len() as f64,
                memory_usage_mb: analysis.streaming_metrics.memory_usage_mb,
                active_alerts: self.alert_system.read().await.active_alerts.len() as u32,
            };
            
            history.performance_history.push_back(perf_snapshot);
            if history.performance_history.len() > history.max_performance_entries {
                history.performance_history.pop_front();
            }
        }

        // Check for alerts
        self.check_and_trigger_alerts(analysis).await?;

        Ok(())
    }

    /// 🚨 Check for alert conditions and trigger notifications
    async fn check_and_trigger_alerts(&self, analysis: &LiveLogitAnalysis) -> Result<(), Box<dyn std::error::Error>> {
        let mut alerts_to_add = Vec::new();
        let timestamp = current_timestamp();
        
        // Check uncertainty thresholds
        if analysis.base_result.calibrated_hbar <= self.config.alert_thresholds.hbar_critical {
            alerts_to_add.push(Alert {
                id: format!("hbar_critical_{}", timestamp),
                alert_type: AlertType::HighUncertainty,
                severity: AlertSeverity::Critical,
                message: format!("Critical uncertainty detected: ℏₛ = {:.4}", analysis.base_result.calibrated_hbar),
                timestamp,
                metrics: AlertMetrics {
                    hbar_value: analysis.base_result.calibrated_hbar,
                    entropy: analysis.logit_metrics.average_entropy,
                    confidence: analysis.logit_metrics.confidence_score,
                    token_position: None,
                    token_text: None,
                },
                auto_resolve: true,
            });
        } else if analysis.base_result.calibrated_hbar <= self.config.alert_thresholds.hbar_warning {
            alerts_to_add.push(Alert {
                id: format!("hbar_warning_{}", timestamp),
                alert_type: AlertType::HighUncertainty,
                severity: AlertSeverity::Warning,
                message: format!("High uncertainty: ℏₛ = {:.4}", analysis.base_result.calibrated_hbar),
                timestamp,
                metrics: AlertMetrics {
                    hbar_value: analysis.base_result.calibrated_hbar,
                    entropy: analysis.logit_metrics.average_entropy,
                    confidence: analysis.logit_metrics.confidence_score,
                    token_position: None,
                    token_text: None,
                },
                auto_resolve: true,
            });
        }

        // Check confidence threshold
        if analysis.logit_metrics.confidence_score < self.config.alert_thresholds.confidence_threshold {
            alerts_to_add.push(Alert {
                id: format!("low_confidence_{}", timestamp),
                alert_type: AlertType::LowConfidence,
                severity: AlertSeverity::Warning,
                message: format!("Low confidence detected: {:.3}", analysis.logit_metrics.confidence_score),
                timestamp,
                metrics: AlertMetrics {
                    hbar_value: analysis.base_result.calibrated_hbar,
                    entropy: analysis.logit_metrics.average_entropy,
                    confidence: analysis.logit_metrics.confidence_score,
                    token_position: None,
                    token_text: None,
                },
                auto_resolve: true,
            });
        }

        // Check individual token uncertainties
        for token_uncertainty in &analysis.token_uncertainties {
            if token_uncertainty.token_entropy > self.config.alert_thresholds.entropy_threshold {
                alerts_to_add.push(Alert {
                    id: format!("token_uncertainty_{}_{}", timestamp, token_uncertainty.position),
                    alert_type: AlertType::HighUncertainty,
                    severity: AlertSeverity::Warning,
                    message: format!("High token uncertainty: '{}' (entropy: {:.3})", 
                        token_uncertainty.token_string, token_uncertainty.token_entropy),
                    timestamp,
                    metrics: AlertMetrics {
                        hbar_value: analysis.base_result.calibrated_hbar,
                        entropy: token_uncertainty.token_entropy,
                        confidence: token_uncertainty.token_probability,
                        token_position: Some(token_uncertainty.position),
                        token_text: Some(token_uncertainty.token_string.clone()),
                    },
                    auto_resolve: true,
                });
            }
        }

        // Process alerts
        if !alerts_to_add.is_empty() {
            let mut alert_system = self.alert_system.write().await;
            
            for alert in alerts_to_add {
                // Add to active alerts
                alert_system.active_alerts.insert(alert.id.clone(), alert.clone());
                
                // Add to history
                alert_system.alert_history.push_back(alert.clone());
                if alert_system.alert_history.len() > 1000 {
                    alert_system.alert_history.pop_front();
                }
                
                // Send notification if channel exists
                if let Some(tx) = &alert_system.notification_tx {
                    let snapshot = self.create_dashboard_snapshot().await;
                    let notification = AlertNotification {
                        alert,
                        dashboard_snapshot: snapshot,
                    };
                    let _ = tx.send(notification); // Ignore send errors
                }
            }
        }

        Ok(())
    }

    /// 📸 Create current dashboard snapshot
    async fn create_dashboard_snapshot(&self) -> DashboardSnapshot {
        let state = self.session_state.read().await;
        let timestamp = current_timestamp();
        
        DashboardSnapshot {
            timestamp,
            session_duration_mins: (timestamp - state.session_start) as f64 / 60000.0,
            total_tokens: state.total_tokens,
            avg_uncertainty: state.session_avg_hbar,
            current_risk: state.current_risk.clone(),
            recent_trend: state.recent_hbar_values.iter().cloned().collect(),
        }
    }

    /// 📊 Get current dashboard state
    pub async fn get_dashboard_state(&self) -> DashboardState {
        let session = self.session_state.read().await.clone();
        let alert_system = self.alert_system.read().await;
        let history = self.history_buffer.read().await;
        
        DashboardState {
            session,
            active_alerts: alert_system.active_alerts.values().cloned().collect(),
            recent_uncertainty_history: history.uncertainty_history.iter()
                .rev().take(100).cloned().collect(),
            recent_performance: history.performance_history.iter()
                .rev().take(50).cloned().collect(),
        }
    }

    /// 🔔 Set up alert notifications
    pub async fn setup_notifications(&self) -> mpsc::UnboundedReceiver<AlertNotification> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.alert_system.write().await.notification_tx = Some(tx);
        rx
    }
}

/// 📊 Complete dashboard state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    pub session: SessionState,
    pub active_alerts: Vec<Alert>,
    pub recent_uncertainty_history: Vec<(u64, f64)>,
    pub recent_performance: Vec<PerformanceSnapshot>,
}

impl Default for LiveMetrics {
    fn default() -> Self {
        Self {
            current_hbar: 0.0,
            current_entropy: 0.0,
            current_confidence: 0.0,
            current_perplexity: 0.0,
            recent_token_count: 0,
            avg_latency_ms: 0.0,
            memory_usage_mb: 0.0,
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

#[cfg(all(test, feature = "examples"))]
mod tests {
    use super::*;
    use crate::examples::llama_integration::MockLlamaModel;
    use crate::oss_logit_adapter::{OSSLogitAdapter, OSSModelFramework, AdapterConfig};
    
    #[tokio::test]
    async fn test_dashboard_initialization() {
        let config = DashboardConfig::default();
        let model_info = ModelInfo {
            name: "Llama-2-7B".to_string(),
            size: "7B".to_string(),
            framework: "llama.cpp".to_string(),
            vocab_size: 32000,
            context_length: 4096,
        };
        
        let dashboard = LiveAuditDashboard::new(config, model_info).await;
        let state = dashboard.get_dashboard_state().await;
        
        assert_eq!(state.session.total_tokens, 0);
        assert!(state.active_alerts.is_empty());
    }
    
    #[tokio::test]
    async fn test_alert_triggering() {
        let mut config = DashboardConfig::default();
        config.alert_thresholds.hbar_critical = 1.0; // Set low threshold for testing
        
        let model_info = ModelInfo {
            name: "Test".to_string(),
            size: "Test".to_string(),
            framework: "Test".to_string(),
            vocab_size: 1000,
            context_length: 512,
        };
        
        let dashboard = LiveAuditDashboard::new(config, model_info).await;
        
        // Create mock analysis with low uncertainty (should trigger alert)
        // ... test implementation would go here
    }
} 