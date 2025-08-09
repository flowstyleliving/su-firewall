// 🔌 Audit Interface
// Clean, simple interface for integrating any model with live response auditing
// Zero dependencies on specific model implementations

use crate::live_response_auditor::{
    LiveResponseAuditor, TokenData, ModelInfo, ModelCapabilities
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// 🎯 Simple audit client that any application can use
pub struct AuditClient {
    auditor: Arc<LiveResponseAuditor>,
    current_session: Arc<RwLock<Option<Uuid>>>,
}

/// 📝 Simple request to start auditing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartAuditRequest {
    pub prompt: String,
    pub model_name: String,
    pub model_version: Option<String>,
    pub framework: String,
    pub capabilities: SimpleCapabilities,
}

/// 🔧 Simplified model capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleCapabilities {
    pub has_logits: bool,
    pub has_probabilities: bool,
    pub supports_streaming: bool,
}

/// 🎯 Simple token input for auditing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleToken {
    pub text: String,
    pub token_id: Option<u32>,
    pub probability: Option<f64>,
    pub logits: Option<Vec<f32>>,
}

/// 📊 Simple audit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleAuditResult {
    pub session_id: String,
    pub current_uncertainty: f64,
    pub average_uncertainty: f64,
    pub risk_level: String,
    pub tokens_processed: u32,
    pub alerts: Vec<SimpleAlert>,
}

/// 🚨 Simple alert format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleAlert {
    pub alert_type: String,
    pub severity: String,
    pub message: String,
    pub token_position: Option<usize>,
    pub uncertainty_value: f64,
}

impl AuditClient {
    /// 🚀 Create new audit client
    pub fn new(auditor: Arc<LiveResponseAuditor>) -> Self {
        Self {
            auditor,
            current_session: Arc::new(RwLock::new(None)),
        }
    }

    /// 📝 Start a new audit session (simple interface)
    pub async fn start_audit(
        &self,
        request: StartAuditRequest,
    ) -> Result<String, String> {
        let model_info = ModelInfo {
            name: request.model_name,
            version: request.model_version,
            framework: request.framework,
            capabilities: ModelCapabilities {
                provides_logits: request.capabilities.has_logits,
                provides_probabilities: request.capabilities.has_probabilities,
                provides_alternatives: false,
                provides_attention: false,
                supports_streaming: request.capabilities.supports_streaming,
            },
        };

        match self.auditor.start_session(request.prompt, model_info).await {
            Ok(session_id) => {
                *self.current_session.write().await = Some(session_id);
                Ok(session_id.to_string())
            },
            Err(e) => Err(e.to_string()),
        }
    }

    /// 🎯 Add a token to the current session (simple interface)
    pub async fn add_token(
        &self,
        token: SimpleToken,
    ) -> Result<SimpleAuditResult, String> {
        let session_id = {
            let session = self.current_session.read().await;
            session.ok_or("No active audit session")?
        };

        let position = self.get_current_position(session_id).await?;

        let token_data = TokenData {
            text: token.text,
            token_id: token.token_id,
            probability: token.probability,
            logits: token.logits,
            position,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            alternatives: None,
        };

        match self.auditor.process_token(session_id, token_data).await {
            Ok(metrics) => {
                let session = self.auditor.get_session(session_id).await
                    .ok_or("Session not found")?;
                
                Ok(SimpleAuditResult {
                    session_id: session_id.to_string(),
                    current_uncertainty: metrics.current_uncertainty,
                    average_uncertainty: metrics.average_uncertainty,
                    risk_level: format!("{:?}", metrics.risk_level),
                    tokens_processed: metrics.tokens_processed,
                    alerts: session.alerts.iter()
                        .map(|alert| SimpleAlert {
                            alert_type: format!("{:?}", alert.alert_type),
                            severity: format!("{:?}", alert.severity),
                            message: alert.message.clone(),
                            token_position: alert.token_position,
                            uncertainty_value: alert.uncertainty_value,
                        })
                        .collect(),
                })
            },
            Err(e) => Err(e.to_string()),
        }
    }

    /// ✅ Finish the current audit session
    pub async fn finish_audit(&self) -> Result<SimpleAuditResult, String> {
        let session_id = {
            let mut session = self.current_session.write().await;
            session.take().ok_or("No active audit session")?
        };

        match self.auditor.complete_session(session_id).await {
            Ok(session) => {
                Ok(SimpleAuditResult {
                    session_id: session_id.to_string(),
                    current_uncertainty: session.live_metrics.current_uncertainty,
                    average_uncertainty: session.live_metrics.average_uncertainty,
                    risk_level: format!("{:?}", session.live_metrics.risk_level),
                    tokens_processed: session.live_metrics.tokens_processed,
                    alerts: session.alerts.iter()
                        .map(|alert| SimpleAlert {
                            alert_type: format!("{:?}", alert.alert_type),
                            severity: format!("{:?}", alert.severity),
                            message: alert.message.clone(),
                            token_position: alert.token_position,
                            uncertainty_value: alert.uncertainty_value,
                        })
                        .collect(),
                })
            },
            Err(e) => Err(e.to_string()),
        }
    }

    /// 📊 Get current session status
    pub async fn get_status(&self) -> Result<SimpleAuditResult, String> {
        let session_id = {
            let session = self.current_session.read().await;
            session.ok_or("No active audit session")?
        };

        let session = self.auditor.get_session(session_id).await
            .ok_or("Session not found")?;

        Ok(SimpleAuditResult {
            session_id: session_id.to_string(),
            current_uncertainty: session.live_metrics.current_uncertainty,
            average_uncertainty: session.live_metrics.average_uncertainty,
            risk_level: format!("{:?}", session.live_metrics.risk_level),
            tokens_processed: session.live_metrics.tokens_processed,
            alerts: session.alerts.iter()
                .map(|alert| SimpleAlert {
                    alert_type: format!("{:?}", alert.alert_type),
                    severity: format!("{:?}", alert.severity),
                    message: alert.message.clone(),
                    token_position: alert.token_position,
                    uncertainty_value: alert.uncertainty_value,
                })
                .collect(),
        })
    }

    /// 🔢 Helper to get current position in session
    async fn get_current_position(&self, session_id: Uuid) -> Result<usize, String> {
        let session = self.auditor.get_session(session_id).await
            .ok_or("Session not found")?;
        Ok(session.tokens.len())
    }
}

/// 🌐 HTTP API wrapper for the audit interface
#[cfg(feature = "http-api")]
pub mod http_api {
    use super::*;
    use warp::Filter;
    use std::convert::Infallible;

    /// 🚀 Start HTTP API server for audit interface
    pub async fn start_audit_api_server(
        audit_client: Arc<AuditClient>,
        port: u16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = warp::any().map(move || audit_client.clone());

        // POST /audit/start
        let start_audit = warp::path!("audit" / "start")
            .and(warp::post())
            .and(warp::body::json())
            .and(client.clone())
            .and_then(handle_start_audit);

        // POST /audit/token  
        let add_token = warp::path!("audit" / "token")
            .and(warp::post())
            .and(warp::body::json())
            .and(client.clone())
            .and_then(handle_add_token);

        // POST /audit/finish
        let finish_audit = warp::path!("audit" / "finish")
            .and(warp::post())
            .and(client.clone())
            .and_then(handle_finish_audit);

        // GET /audit/status
        let get_status = warp::path!("audit" / "status")
            .and(warp::get())
            .and(client.clone())
            .and_then(handle_get_status);

        let routes = start_audit
            .or(add_token)
            .or(finish_audit)
            .or(get_status)
            .with(warp::cors().allow_any_origin());

        warp::serve(routes)
            .run(([127, 0, 0, 1], port))
            .await;

        Ok(())
    }

    async fn handle_start_audit(
        request: StartAuditRequest,
        client: Arc<AuditClient>,
    ) -> Result<impl warp::Reply, Infallible> {
        match client.start_audit(request).await {
            Ok(session_id) => Ok(warp::reply::json(&serde_json::json!({
                "success": true,
                "session_id": session_id
            }))),
            Err(error) => Ok(warp::reply::json(&serde_json::json!({
                "success": false,
                "error": error
            }))),
        }
    }

    async fn handle_add_token(
        token: SimpleToken,
        client: Arc<AuditClient>,
    ) -> Result<impl warp::Reply, Infallible> {
        match client.add_token(token).await {
            Ok(result) => Ok(warp::reply::json(&serde_json::json!({
                "success": true,
                "result": result
            }))),
            Err(error) => Ok(warp::reply::json(&serde_json::json!({
                "success": false,
                "error": error
            }))),
        }
    }

    async fn handle_finish_audit(
        client: Arc<AuditClient>,
    ) -> Result<impl warp::Reply, Infallible> {
        match client.finish_audit().await {
            Ok(result) => Ok(warp::reply::json(&serde_json::json!({
                "success": true,
                "result": result
            }))),
            Err(error) => Ok(warp::reply::json(&serde_json::json!({
                "success": false,
                "error": error
            }))),
        }
    }

    async fn handle_get_status(
        client: Arc<AuditClient>,
    ) -> Result<impl warp::Reply, Infallible> {
        match client.get_status().await {
            Ok(result) => Ok(warp::reply::json(&serde_json::json!({
                "success": true,
                "result": result
            }))),
            Err(error) => Ok(warp::reply::json(&serde_json::json!({
                "success": false,
                "error": error
            }))),
        }
    }
}

/// 📡 WebSocket API for real-time streaming
#[cfg(feature = "websocket")]
pub mod websocket_api {
    use super::*;
    use tokio_tungstenite::{WebSocketStream, tungstenite::Message};
    use futures_util::{SinkExt, StreamExt};

    /// 🌊 WebSocket handler for real-time audit streaming
    pub async fn handle_websocket_audit(
        mut ws: WebSocketStream<tokio::net::TcpStream>,
        audit_client: Arc<AuditClient>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Subscribe to real-time metrics and alerts
        let mut alert_receiver = audit_client.auditor.subscribe_alerts();
        let mut metrics_receiver = audit_client.auditor.subscribe_metrics();

        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                msg = ws.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Ok(token) = serde_json::from_str::<SimpleToken>(&text) {
                                if let Ok(result) = audit_client.add_token(token).await {
                                    let response = serde_json::to_string(&serde_json::json!({
                                        "type": "audit_result",
                                        "data": result
                                    }))?;
                                    ws.send(Message::Text(response)).await?;
                                }
                            }
                        },
                        Some(Ok(Message::Close(_))) => break,
                        Some(Err(e)) => return Err(e.into()),
                        None => break,
                    }
                },
                
                // Forward real-time alerts
                alert = alert_receiver.recv() => {
                    if let Ok(alert) = alert {
                        let simple_alert = SimpleAlert {
                            alert_type: format!("{:?}", alert.alert_type),
                            severity: format!("{:?}", alert.severity),
                            message: alert.message,
                            token_position: alert.token_position,
                            uncertainty_value: alert.uncertainty_value,
                        };
                        
                        let response = serde_json::to_string(&serde_json::json!({
                            "type": "alert",
                            "data": simple_alert
                        }))?;
                        ws.send(Message::Text(response)).await?;
                    }
                },
                
                // Forward real-time metrics
                metrics = metrics_receiver.recv() => {
                    if let Ok(metrics) = metrics {
                        let response = serde_json::to_string(&serde_json::json!({
                            "type": "metrics",
                            "data": {
                                "current_uncertainty": metrics.current_uncertainty,
                                "average_uncertainty": metrics.average_uncertainty,
                                "risk_level": format!("{:?}", metrics.risk_level),
                                "tokens_processed": metrics.tokens_processed,
                                "tokens_per_second": metrics.tokens_per_second
                            }
                        }))?;
                        ws.send(Message::Text(response)).await?;
                    }
                }
            }
        }

        Ok(())
    }
}

/// 🐍 Python bindings for easy integration
#[cfg(feature = "python")]
pub mod python_bindings {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass]
    pub struct PyAuditClient {
        client: Arc<AuditClient>,
        runtime: tokio::runtime::Runtime,
    }

    #[pymethods]
    impl PyAuditClient {
        #[new]
        pub fn new() -> PyResult<Self> {
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            // This would need the auditor to be created properly
            // For now, this is a placeholder
            todo!("Implement proper Python bindings")
        }

        pub fn start_audit(
            &self,
            prompt: String,
            model_name: String,
            framework: String,
        ) -> PyResult<String> {
            let request = StartAuditRequest {
                prompt,
                model_name,
                model_version: None,
                framework,
                capabilities: SimpleCapabilities {
                    has_logits: false,
                    has_probabilities: true,
                    supports_streaming: true,
                },
            };

            self.runtime
                .block_on(self.client.start_audit(request))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        }

        pub fn add_token(
            &self,
            text: String,
            probability: Option<f64>,
        ) -> PyResult<String> {
            let token = SimpleToken {
                text,
                token_id: None,
                probability,
                logits: None,
            };

            let result = self.runtime
                .block_on(self.client.add_token(token))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            serde_json::to_string(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        pub fn finish_audit(&self) -> PyResult<String> {
            let result = self.runtime
                .block_on(self.client.finish_audit())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            serde_json::to_string(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    #[pymodule]
    fn audit_interface(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyAuditClient>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_response_auditor::LiveAuditorConfig;

    #[tokio::test]
    async fn test_audit_interface() {
        let config = LiveAuditorConfig::default();
        let auditor = Arc::new(LiveResponseAuditor::new(config));
        let client = AuditClient::new(auditor);

        // Start audit
        let request = StartAuditRequest {
            prompt: "Test prompt".to_string(),
            model_name: "Test Model".to_string(),
            model_version: Some("1.0".to_string()),
            framework: "Test".to_string(),
            capabilities: SimpleCapabilities {
                has_logits: false,
                has_probabilities: true,
                supports_streaming: true,
            },
        };

        let session_id = client.start_audit(request).await.unwrap();
        assert!(!session_id.is_empty());

        // Add tokens
        for i in 0..3 {
            let token = SimpleToken {
                text: format!("token_{}", i),
                token_id: Some(i),
                probability: Some(0.8),
                logits: None,
            };

            let result = client.add_token(token).await.unwrap();
            assert_eq!(result.tokens_processed, i + 1);
        }

        // Finish audit
        let final_result = client.finish_audit().await.unwrap();
        assert_eq!(final_result.tokens_processed, 3);
    }
} 