// 📊 Cloudflare Analytics + Simple Alerts Monitoring
// Optimized for ℏₛ = 1.26 stable operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};

/// 📈 Core metrics for semantic uncertainty monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMetrics {
    pub request_count: u64,
    pub average_hbar: f32,
    pub collapse_rate: f32,
    pub response_time_p95: f64,
    pub error_rate: f32,
    pub timestamp: DateTime<Utc>,
}

/// 🚨 Alert levels based on semantic uncertainty thresholds
#[derive(Debug, Clone, PartialEq)]
pub enum AlertLevel {
    Critical,    // 🔴 ℏₛ < 0.8 or error_rate > 5%
    Warning,     // 🟡 ℏₛ < 1.0 or error_rate > 1%
    Info,        // 🔵 Normal operations
}

/// 📊 Simple monitoring system leveraging Cloudflare infrastructure
pub struct CloudflareMonitor {
    metrics_buffer: HashMap<String, SemanticMetrics>,
    alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub critical_hbar: f32,
    pub warning_hbar: f32,
    pub critical_error_rate: f32,
    pub warning_error_rate: f32,
    pub max_response_time_ms: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            critical_hbar: 0.8,
            warning_hbar: 1.0,
            critical_error_rate: 0.05,  // 5%
            warning_error_rate: 0.01,   // 1%
            max_response_time_ms: 100.0,
        }
    }
}

impl CloudflareMonitor {
    pub fn new() -> Self {
        Self {
            metrics_buffer: HashMap::new(),
            alert_thresholds: AlertThresholds::default(),
        }
    }

    /// 📝 Record semantic analysis metrics
    pub fn record_analysis(&mut self, _request_id: &str, hbar_s: f32, response_time_ms: f64, success: bool) {
        let key = format!("minute_{}", Utc::now().format("%Y%m%d_%H%M"));
        
        // First, update the metrics
        let should_alert = {
            let metrics = self.metrics_buffer.entry(key).or_insert_with(|| SemanticMetrics {
                request_count: 0,
                average_hbar: 0.0,
                collapse_rate: 0.0,
                response_time_p95: 0.0,
                error_rate: 0.0,
                timestamp: Utc::now(),
            });

            // Update metrics
            metrics.request_count += 1;
            metrics.average_hbar = (metrics.average_hbar * (metrics.request_count - 1) as f32 + hbar_s) / metrics.request_count as f32;
            
            if hbar_s < 1.0 {
                metrics.collapse_rate = ((metrics.collapse_rate * (metrics.request_count - 1) as f32) + 1.0) / metrics.request_count as f32;
            }

            if !success {
                metrics.error_rate = ((metrics.error_rate * (metrics.request_count - 1) as f32) + 1.0) / metrics.request_count as f32;
            }

            // Simple P95 approximation (good enough for monitoring)
            if response_time_ms > metrics.response_time_p95 {
                metrics.response_time_p95 = response_time_ms;
            }

            // Clone metrics for alert checking
            metrics.clone()
        };

        // 🚨 Check for immediate alerts
        self.check_alerts(&should_alert);

        info!("📊 METRICS_RECORDED | Requests: {} | ℏₛ: {:.3} | Errors: {:.1}%", 
              should_alert.request_count, should_alert.average_hbar, should_alert.error_rate * 100.0);
    }

    /// 🚨 Real-time alert checking
    fn check_alerts(&self, metrics: &SemanticMetrics) {
        let alert_level = self.determine_alert_level(metrics);
        
        match alert_level {
            AlertLevel::Critical => {
                error!("🔴 CRITICAL_ALERT | ℏₛ: {:.3} | Error Rate: {:.1}% | Response Time: {:.1}ms", 
                       metrics.average_hbar, metrics.error_rate * 100.0, metrics.response_time_p95);
                
                // In production, would trigger PagerDuty/Slack
                self.send_critical_alert(metrics);
            },
            AlertLevel::Warning => {
                warn!("🟡 WARNING_ALERT | ℏₛ: {:.3} | Error Rate: {:.1}% | Degraded Performance", 
                      metrics.average_hbar, metrics.error_rate * 100.0);
                
                self.send_warning_alert(metrics);
            },
            AlertLevel::Info => {
                // Normal operations, no alert needed
            }
        }
    }

    /// 🎯 Determine alert level using semantic uncertainty thresholds
    fn determine_alert_level(&self, metrics: &SemanticMetrics) -> AlertLevel {
        // Critical conditions
        if metrics.average_hbar < self.alert_thresholds.critical_hbar ||
           metrics.error_rate > self.alert_thresholds.critical_error_rate ||
           metrics.response_time_p95 > self.alert_thresholds.max_response_time_ms * 2.0 {
            return AlertLevel::Critical;
        }

        // Warning conditions
        if metrics.average_hbar < self.alert_thresholds.warning_hbar ||
           metrics.error_rate > self.alert_thresholds.warning_error_rate ||
           metrics.response_time_p95 > self.alert_thresholds.max_response_time_ms {
            return AlertLevel::Warning;
        }

        AlertLevel::Info
    }

    /// 🚨 Send critical alert (stub for actual implementation)
    fn send_critical_alert(&self, metrics: &SemanticMetrics) {
        // TODO: Integrate with actual alerting system
        // Could use Cloudflare Workers to call Slack webhook
        // Or send email via SendGrid API
        
        let alert_message = format!(
            "🔴 SEMANTIC UNCERTAINTY CRITICAL ALERT\n\
             Time: {}\n\
             Average ℏₛ: {:.3}\n\
             Error Rate: {:.1}%\n\
             Response Time P95: {:.1}ms\n\
             Requests: {}",
            metrics.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            metrics.average_hbar,
            metrics.error_rate * 100.0,
            metrics.response_time_p95,
            metrics.request_count
        );

        // Log for now (replace with actual alerting)
        error!("ALERT_PAYLOAD: {}", alert_message);
    }

    /// 🟡 Send warning alert (stub for actual implementation)
    fn send_warning_alert(&self, metrics: &SemanticMetrics) {
        let alert_message = format!(
            "🟡 Semantic Uncertainty Warning: ℏₛ={:.3}, ErrorRate={:.1}%, ResponseTime={:.1}ms",
            metrics.average_hbar, metrics.error_rate * 100.0, metrics.response_time_p95
        );

        warn!("WARNING_ALERT: {}", alert_message);
    }

    /// 📈 Get current metrics summary
    pub fn get_current_metrics(&self) -> Option<&SemanticMetrics> {
        let current_key = format!("minute_{}", Utc::now().format("%Y%m%d_%H%M"));
        self.metrics_buffer.get(&current_key)
    }

    /// 🧹 Cleanup old metrics (call periodically)
    pub fn cleanup_old_metrics(&mut self, retain_minutes: usize) {
        let cutoff = Utc::now() - chrono::Duration::minutes(retain_minutes as i64);
        
        self.metrics_buffer.retain(|_, metrics| {
            metrics.timestamp > cutoff
        });
    }

    /// 📊 Generate health check response
    pub fn health_check(&self) -> HealthStatus {
        if let Some(metrics) = self.get_current_metrics() {
            let alert_level = self.determine_alert_level(metrics);
            
            HealthStatus {
                status: match alert_level {
                    AlertLevel::Critical => "unhealthy".to_string(),
                    AlertLevel::Warning => "degraded".to_string(),
                    AlertLevel::Info => "healthy".to_string(),
                },
                average_hbar: metrics.average_hbar,
                error_rate: metrics.error_rate,
                response_time_p95: metrics.response_time_p95,
                request_count: metrics.request_count,
                timestamp: Utc::now(),
            }
        } else {
            HealthStatus {
                status: "unknown".to_string(),
                average_hbar: 0.0,
                error_rate: 0.0,
                response_time_p95: 0.0,
                request_count: 0,
                timestamp: Utc::now(),
            }
        }
    }
}

/// 🏥 Health check response format
#[derive(Debug, Serialize)]
pub struct HealthStatus {
    pub status: String,
    pub average_hbar: f32,
    pub error_rate: f32,
    pub response_time_p95: f64,
    pub request_count: u64,
    pub timestamp: DateTime<Utc>,
}

/// 📊 Integration with Cloudflare Analytics
pub struct CloudflareAnalytics {
    zone_id: String,
    api_token: String,
}

impl CloudflareAnalytics {
    pub fn new(zone_id: String, api_token: String) -> Self {
        Self { zone_id, api_token }
    }

    /// 📈 Send custom metrics to Cloudflare Analytics
    pub async fn send_metrics(&self, metrics: &SemanticMetrics) -> Result<(), Box<dyn std::error::Error>> {
        // This would integrate with Cloudflare's Analytics API
        // For now, just log structured data that Cloudflare Workers can capture
        
        info!(
            target: "cloudflare_analytics",
            request_count = metrics.request_count,
            average_hbar = metrics.average_hbar,
            collapse_rate = metrics.collapse_rate,
            error_rate = metrics.error_rate,
            response_time_p95 = metrics.response_time_p95,
            "📊 CLOUDFLARE_METRICS"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_basic_functionality() {
        let mut monitor = CloudflareMonitor::new();
        
        // Record some metrics
        monitor.record_analysis("req1", 1.5, 50.0, true);
        monitor.record_analysis("req2", 0.9, 75.0, true);
        monitor.record_analysis("req3", 1.2, 60.0, false);

        let metrics = monitor.get_current_metrics().unwrap();
        assert!(metrics.request_count == 3);
        assert!(metrics.average_hbar > 0.0);
        assert!(metrics.error_rate > 0.0);
    }

    #[test]
    fn test_alert_levels() {
        let monitor = CloudflareMonitor::new();
        
        // Critical alert case
        let critical_metrics = SemanticMetrics {
            request_count: 100,
            average_hbar: 0.7, // Below critical threshold
            collapse_rate: 0.1,
            response_time_p95: 250.0, // Above threshold
            error_rate: 0.08, // Above critical threshold
            timestamp: Utc::now(),
        };
        
        assert_eq!(monitor.determine_alert_level(&critical_metrics), AlertLevel::Critical);

        // Normal case
        let normal_metrics = SemanticMetrics {
            request_count: 100,
            average_hbar: 1.3, // Good
            collapse_rate: 0.01,
            response_time_p95: 45.0, // Good
            error_rate: 0.005, // Good
            timestamp: Utc::now(),
        };
        
        assert_eq!(monitor.determine_alert_level(&normal_metrics), AlertLevel::Info);
    }

    #[test]
    fn test_health_check() {
        let mut monitor = CloudflareMonitor::new();
        monitor.record_analysis("req1", 1.5, 50.0, true);
        
        let health = monitor.health_check();
        assert_eq!(health.status, "healthy");
        assert!(health.average_hbar > 0.0);
    }
}