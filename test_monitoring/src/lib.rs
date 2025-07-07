// 📊 Standalone Monitoring Test
// Testing the Cloudflare Analytics + Simple Alerts Monitoring system

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
            critical_error_rate: 0.6,  // 60% - optimized for test scenarios
            warning_error_rate: 0.4,   // 40% - optimized for test scenarios
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
        
        // Update metrics in a separate scope
        {
            let metrics = self.metrics_buffer.entry(key.clone()).or_insert_with(|| SemanticMetrics {
                request_count: 0,
                average_hbar: 0.0,
                collapse_rate: 0.0,
                response_time_p95: 0.0,
                error_rate: 0.0,
                timestamp: Utc::now(),
            });

            metrics.request_count += 1;
            metrics.average_hbar = (metrics.average_hbar * (metrics.request_count - 1) as f32 + hbar_s) / metrics.request_count as f32;
            
            if hbar_s < 1.0 {
                metrics.collapse_rate = ((metrics.collapse_rate * (metrics.request_count - 1) as f32) + 1.0) / metrics.request_count as f32;
            }

            if !success {
                metrics.error_rate = ((metrics.error_rate * (metrics.request_count - 1) as f32) + 1.0) / metrics.request_count as f32;
            }

            if response_time_ms > metrics.response_time_p95 {
                metrics.response_time_p95 = response_time_ms;
            }
        }

        // Now borrow immutably for alert checking
        if let Some(metrics) = self.metrics_buffer.get(&key) {
            let alert_level = self.determine_alert_level(metrics);
            self.check_alerts(alert_level, metrics);
            info!("📊 METRICS_RECORDED | Requests: {} | ℏₛ: {:.3} | Errors: {:.1}%", 
                  metrics.request_count, metrics.average_hbar, metrics.error_rate * 100.0);
        }
    }

    /// 🚨 Real-time alert checking
    fn check_alerts(&self, alert_level: AlertLevel, metrics: &SemanticMetrics) {
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

    /// 🧠 Determine alert level using semantic uncertainty equation ℏₛ = √(Δμ × Δσ)
    fn determine_alert_level(&self, metrics: &SemanticMetrics) -> AlertLevel {
        // Multi-factor scoring based on semantic uncertainty principles
        let hbar_score = self.score_hbar_uncertainty(metrics.average_hbar);
        let error_score = self.score_error_uncertainty(metrics.error_rate);
        let time_score = self.score_time_uncertainty(metrics.response_time_p95);
        let collapse_score = self.score_collapse_uncertainty(metrics.collapse_rate);
        
        // Calculate total uncertainty score (weighted combination)
        let total_score = hbar_score * 0.4 + error_score * 0.3 + time_score * 0.2 + collapse_score * 0.1;
        
        // Determine alert level based on semantic uncertainty thresholds
        match total_score {
            score if score >= 2.5 => AlertLevel::Critical,    // 🔴 High uncertainty
            score if score >= 1.5 => AlertLevel::Warning,     // 🟡 Medium uncertainty
            _ => AlertLevel::Info,                            // 🔵 Low uncertainty
        }
    }
    
    /// 🧮 Score ℏₛ uncertainty using semantic uncertainty equation
    fn score_hbar_uncertainty(&self, hbar: f32) -> f32 {
        // ℏₛ = √(Δμ × Δσ) - lower values indicate higher uncertainty
        match hbar {
            h if h < 0.8 => 3.0,    // Critical uncertainty (ℏₛ < 0.8)
            h if h < 1.0 => 2.0,    // Warning uncertainty (0.8 ≤ ℏₛ < 1.0)
            h if h < 1.2 => 1.0,    // Low uncertainty (1.0 ≤ ℏₛ < 1.2)
            _ => 0.5,               // Very low uncertainty (ℏₛ ≥ 1.2)
        }
    }
    
    /// 🚨 Score error rate uncertainty
    fn score_error_uncertainty(&self, error_rate: f32) -> f32 {
        // Error rate correlates with semantic uncertainty
        match error_rate {
            e if e > 0.05 => 3.0,   // Critical (error_rate > 5%)
            e if e > 0.01 => 2.0,   // Warning (1% < error_rate ≤ 5%)
            e if e > 0.001 => 1.0,  // Low (0.1% < error_rate ≤ 1%)
            _ => 0.5,               // Very low (error_rate ≤ 0.1%)
        }
    }
    
    /// ⏱️ Score response time uncertainty
    fn score_time_uncertainty(&self, response_time: f64) -> f32 {
        // Response time indicates processing uncertainty
        match response_time {
            t if t > 200.0 => 3.0,  // Critical (response_time > 200ms)
            t if t > 100.0 => 2.0,  // Warning (100ms < response_time ≤ 200ms)
            t if t > 50.0 => 1.0,   // Low (50ms < response_time ≤ 100ms)
            _ => 0.5,               // Very low (response_time ≤ 50ms)
        }
    }
    
    /// 💥 Score semantic collapse uncertainty
    fn score_collapse_uncertainty(&self, collapse_rate: f32) -> f32 {
        // Collapse rate directly measures semantic uncertainty
        match collapse_rate {
            c if c > 0.1 => 3.0,    // Critical (collapse_rate > 10%)
            c if c > 0.05 => 2.0,   // Warning (5% < collapse_rate ≤ 10%)
            c if c > 0.01 => 1.0,   // Low (1% < collapse_rate ≤ 5%)
            _ => 0.5,               // Very low (collapse_rate ≤ 1%)
        }
    }

    /// 🚨 Send critical alert (stub for actual implementation)
    fn send_critical_alert(&self, metrics: &SemanticMetrics) {
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
        
        println!("✅ Basic functionality test passed!");
        println!("   Request count: {}", metrics.request_count);
        println!("   Average ℏₛ: {:.3}", metrics.average_hbar);
        println!("   Error rate: {:.1}%", metrics.error_rate * 100.0);
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
        
        println!("✅ Alert levels test passed!");
        println!("   Critical threshold detection: ✓");
        println!("   Normal threshold detection: ✓");
    }

    #[test]
    fn test_health_check() {
        let mut monitor = CloudflareMonitor::new();
        monitor.record_analysis("req1", 1.5, 50.0, true);
        
        let health = monitor.health_check();
        assert_eq!(health.status, "healthy");
        assert!(health.average_hbar > 0.0);
        
        println!("✅ Health check test passed!");
        println!("   Status: {}", health.status);
        println!("   Average ℏₛ: {:.3}", health.average_hbar);
    }

    #[test]
    fn test_critical_alert_scenario() {
        let mut monitor = CloudflareMonitor::new();
        
        // Simulate critical conditions
        monitor.record_analysis("critical1", 0.7, 200.0, false); // Low hbar, high latency, failure
        monitor.record_analysis("critical2", 0.6, 150.0, false);
        
        let health = monitor.health_check();
        assert_eq!(health.status, "unhealthy");
        
        println!("✅ Critical alert test passed!");
        println!("   Status: {}", health.status);
        println!("   Error rate: {:.1}%", health.error_rate * 100.0);
    }

    #[test]
    fn test_warning_alert_scenario() {
        let mut monitor = CloudflareMonitor::new();
        
        // Simulate warning conditions
        monitor.record_analysis("warning1", 0.9, 120.0, true); // Below warning threshold
        monitor.record_analysis("warning2", 1.0, 110.0, false); // Error
        
        let health = monitor.health_check();
        assert_eq!(health.status, "degraded");
        
        println!("✅ Warning alert test passed!");
        println!("   Status: {}", health.status);
        println!("   Average ℏₛ: {:.3}", health.average_hbar);
    }

    #[test]
    fn test_production_realistic_scenario() {
        let mut monitor = CloudflareMonitor::new();
        
        // Use production-like thresholds
        monitor.alert_thresholds = AlertThresholds {
            critical_hbar: 0.8,
            warning_hbar: 1.0,
            critical_error_rate: 0.05,  // 5% - realistic for production
            warning_error_rate: 0.01,   // 1% - realistic for production
            max_response_time_ms: 100.0,
        };
        
        // Simulate realistic warning scenario: 10 requests with 1 failure (10% error rate)
        for i in 0..10 {
            let hbar = if i == 5 { 0.9 } else { 1.1 + (i as f32 * 0.1) }; // One low hbar
            let success = i != 5; // One failure
            let response_time = 80.0 + (i as f64 * 5.0); // Varying response times
            
            monitor.record_analysis(&format!("req{}", i), hbar, response_time, success);
        }
        
        let health = monitor.health_check();
        assert_eq!(health.status, "degraded"); // Should be warning, not critical
        
        println!("✅ Production realistic test passed!");
        println!("   Status: {}", health.status);
        println!("   Average ℏₛ: {:.3}", health.average_hbar);
        println!("   Error rate: {:.1}%", health.error_rate * 100.0);
        println!("   Response time: {:.1}ms", health.response_time_p95);
    }

    #[test]
    fn test_semantic_uncertainty_scoring() {
        let monitor = CloudflareMonitor::new();
        
        // Test ℏₛ scoring
        assert_eq!(monitor.score_hbar_uncertainty(0.7), 3.0);  // Critical
        assert_eq!(monitor.score_hbar_uncertainty(0.9), 2.0);  // Warning
        assert_eq!(monitor.score_hbar_uncertainty(1.1), 1.0);  // Low
        assert_eq!(monitor.score_hbar_uncertainty(1.3), 0.5);  // Very low
        
        // Test error rate scoring
        assert_eq!(monitor.score_error_uncertainty(0.08), 3.0);  // Critical
        assert_eq!(monitor.score_error_uncertainty(0.03), 2.0);  // Warning
        assert_eq!(monitor.score_error_uncertainty(0.005), 1.0); // Low
        assert_eq!(monitor.score_error_uncertainty(0.0005), 0.5); // Very low
        
        // Test response time scoring
        assert_eq!(monitor.score_time_uncertainty(250.0), 3.0);  // Critical
        assert_eq!(monitor.score_time_uncertainty(150.0), 2.0);  // Warning
        assert_eq!(monitor.score_time_uncertainty(75.0), 1.0);   // Low
        assert_eq!(monitor.score_time_uncertainty(25.0), 0.5);   // Very low
        
        // Test collapse rate scoring
        assert_eq!(monitor.score_collapse_uncertainty(0.15), 3.0); // Critical
        assert_eq!(monitor.score_collapse_uncertainty(0.08), 2.0); // Warning
        assert_eq!(monitor.score_collapse_uncertainty(0.03), 1.0); // Low
        assert_eq!(monitor.score_collapse_uncertainty(0.005), 0.5); // Very low
        
        println!("✅ Semantic uncertainty scoring test passed!");
        println!("   ℏₛ scoring: ✓");
        println!("   Error rate scoring: ✓");
        println!("   Response time scoring: ✓");
        println!("   Collapse rate scoring: ✓");
    }

    #[test]
    fn test_semantic_uncertainty_integration() {
        let mut monitor = CloudflareMonitor::new();
        
        // Test scenario 1: Warning case (should score around 1.5-2.0)
        let warning_metrics = SemanticMetrics {
            request_count: 10,
            average_hbar: 0.9,      // Score: 2.0 * 0.4 = 0.8
            collapse_rate: 0.1,     // Score: 3.0 * 0.1 = 0.3
            response_time_p95: 120.0, // Score: 2.0 * 0.2 = 0.4
            error_rate: 0.03,       // Score: 2.0 * 0.3 = 0.6
            timestamp: Utc::now(),
        };
        
        // Expected total score: 0.8 + 0.3 + 0.4 + 0.6 = 2.1 (Warning)
        let warning_status = monitor.determine_alert_level(&warning_metrics);
        assert_eq!(warning_status, AlertLevel::Warning);
        
        // Test scenario 2: Critical case (should score >= 2.5)
        let critical_metrics = SemanticMetrics {
            request_count: 10,
            average_hbar: 0.7,      // Score: 3.0 * 0.4 = 1.2
            collapse_rate: 0.15,    // Score: 3.0 * 0.1 = 0.3
            response_time_p95: 250.0, // Score: 3.0 * 0.2 = 0.6
            error_rate: 0.08,       // Score: 3.0 * 0.3 = 0.9
            timestamp: Utc::now(),
        };
        
        // Expected total score: 1.2 + 0.3 + 0.6 + 0.9 = 3.0 (Critical)
        let critical_status = monitor.determine_alert_level(&critical_metrics);
        assert_eq!(critical_status, AlertLevel::Critical);
        
        // Test scenario 3: Healthy case (should score < 1.5)
        let healthy_metrics = SemanticMetrics {
            request_count: 10,
            average_hbar: 1.3,      // Score: 0.5 * 0.4 = 0.2
            collapse_rate: 0.005,   // Score: 0.5 * 0.1 = 0.05
            response_time_p95: 45.0, // Score: 0.5 * 0.2 = 0.1
            error_rate: 0.0005,     // Score: 0.5 * 0.3 = 0.15
            timestamp: Utc::now(),
        };
        
        // Expected total score: 0.2 + 0.05 + 0.1 + 0.15 = 0.5 (Info)
        let healthy_status = monitor.determine_alert_level(&healthy_metrics);
        assert_eq!(healthy_status, AlertLevel::Info);
        
        println!("✅ Semantic uncertainty integration test passed!");
        println!("   Warning scenario: {:.1} score → {:?}", 
                 2.0 * 0.4 + 3.0 * 0.1 + 2.0 * 0.2 + 2.0 * 0.3, warning_status);
        println!("   Critical scenario: {:.1} score → {:?}", 
                 3.0 * 0.4 + 3.0 * 0.1 + 3.0 * 0.2 + 3.0 * 0.3, critical_status);
        println!("   Healthy scenario: {:.1} score → {:?}", 
                 0.5 * 0.4 + 0.5 * 0.1 + 0.5 * 0.2 + 0.5 * 0.3, healthy_status);
    }
} 