// 🔍 Consolidated Audit System
// Combines audit_interface, live_audit_dashboard, live_response_auditor, and monitoring

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{info, warn, debug};
use crate::metrics;
use common::{SemanticUncertaintyResult, RequestId, RiskLevel};

/// 📊 Consolidated audit result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub request_id: RequestId,
    pub risk_level: RiskLevel,
    pub hbar_s: f64,
    pub p_fail: f64,
    pub processing_time_ms: f64,
    pub audit_metadata: AuditMetadata,
    pub monitoring_data: MonitoringData,
}

/// 📋 Audit metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditMetadata {
    pub audit_type: String,
    pub confidence: f64,
    pub flags: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// 📈 Monitoring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringData {
    pub response_time_ms: f64,
    pub memory_usage: Option<u64>,
    pub cpu_usage: Option<f32>,
    pub throughput: f64,
}

/// 🔍 Main audit system
pub struct AuditSystem {
    config: AuditConfig,
    active_audits: HashMap<RequestId, AuditContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enable_live_monitoring: bool,
    pub max_concurrent_audits: usize,
    pub audit_timeout_ms: u64,
    pub risk_thresholds: RiskThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskThresholds {
    pub critical: f64,
    pub high: f64,
    pub warning: f64,
}

#[derive(Debug)]
struct AuditContext {
    start_time: Instant,
    request_id: RequestId,
    current_risk: RiskLevel,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enable_live_monitoring: true,
            max_concurrent_audits: 100,
            audit_timeout_ms: 30000,
            risk_thresholds: RiskThresholds {
                critical: 0.8,
                high: 0.5,
                warning: 0.2,
            },
        }
    }
}

impl AuditSystem {
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            active_audits: HashMap::new(),
        }
    }

    /// Start audit for a request
    pub fn start_audit(&mut self, request_id: RequestId) -> Result<(), AuditError> {
        if self.active_audits.len() >= self.config.max_concurrent_audits {
            return Err(AuditError::TooManyActiveAudits);
        }

        let context = AuditContext {
            start_time: Instant::now(),
            request_id: request_id.clone(),
            current_risk: RiskLevel::Safe,
        };

        self.active_audits.insert(request_id, context);
        
        Ok(())
    }

    /// Complete audit with results
    pub fn complete_audit(
        &mut self,
        request_id: &RequestId,
        uncertainty_result: &SemanticUncertaintyResult,
    ) -> Result<AuditResult, AuditError> {
        let context = self.active_audits.remove(request_id)
            .ok_or(AuditError::AuditNotFound)?;

        let processing_time_ms = context.start_time.elapsed().as_secs_f64() * 1000.0;
        
        let risk_level = self.assess_risk_level(uncertainty_result.raw_hbar);
        
        let audit_result = AuditResult {
            request_id: request_id.clone(),
            risk_level: risk_level.clone(),
            hbar_s: uncertainty_result.raw_hbar,
            p_fail: uncertainty_result.delta_sigma, // Using as p_fail proxy
            processing_time_ms,
            audit_metadata: AuditMetadata {
                audit_type: "semantic_uncertainty".to_string(),
                confidence: 1.0 - uncertainty_result.raw_hbar.min(1.0),
                flags: self.generate_flags(&uncertainty_result, &risk_level),
                timestamp: chrono::Utc::now(),
            },
            monitoring_data: MonitoringData {
                response_time_ms: processing_time_ms,
                memory_usage: None, // Could integrate system metrics
                cpu_usage: None,
                throughput: 1000.0 / processing_time_ms.max(1.0),
            },
        };

        info!("Audit completed for request {}: risk={:?}, ℏₛ={:.3}", 
              request_id, risk_level, uncertainty_result.raw_hbar);

        Ok(audit_result)
    }

    /// Assess risk level based on ℏₛ value
    fn assess_risk_level(&self, hbar_s: f64) -> RiskLevel {
        if hbar_s >= self.config.risk_thresholds.critical {
            RiskLevel::Critical
        } else if hbar_s >= self.config.risk_thresholds.high {
            RiskLevel::HighRisk
        } else if hbar_s >= self.config.risk_thresholds.warning {
            RiskLevel::Warning
        } else {
            RiskLevel::Safe
        }
    }

    /// Generate audit flags based on results
    fn generate_flags(&self, result: &SemanticUncertaintyResult, risk: &RiskLevel) -> Vec<String> {
        let mut flags = Vec::new();

        match risk {
            RiskLevel::Critical => flags.push("CRITICAL_UNCERTAINTY".to_string()),
            RiskLevel::HighRisk => flags.push("HIGH_UNCERTAINTY".to_string()),
            RiskLevel::Warning => flags.push("MODERATE_UNCERTAINTY".to_string()),
            RiskLevel::Safe => {},
        }

        if result.raw_hbar > 2.0 {
            flags.push("EXTREME_HBAR".to_string());
        }

        if result.delta_mu > result.delta_sigma * 2.0 {
            flags.push("HIGH_PRECISION_DOMINANCE".to_string());
        }

        flags
    }

    /// Get dashboard data
    pub fn get_dashboard_data(&self) -> DashboardData {
        DashboardData {
            active_audits: self.active_audits.len(),
            total_processed: 0, // Would integrate with actual metrics
            average_processing_time: 0.0,
            risk_distribution: HashMap::new(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DashboardData {
    pub active_audits: usize,
    pub total_processed: u64,
    pub average_processing_time: f64,
    pub risk_distribution: HashMap<String, u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("Too many active audits")]
    TooManyActiveAudits,
    #[error("Audit not found")]
    AuditNotFound,
    #[error("Audit timeout")]
    Timeout,
}