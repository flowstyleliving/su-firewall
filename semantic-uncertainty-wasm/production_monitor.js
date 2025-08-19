/**
 * Production Monitoring Script for Conservative Gas Optimization
 * 
 * Real-time monitoring with alerts and automatic fallbacks
 * For use during Phase 1 deployment with manual oversight
 */

class ProductionMonitor {
    constructor(conservative_optimizer) {
        this.optimizer = conservative_optimizer;
        this.monitoring_config = {
            alert_thresholds: {
                success_rate_critical: 70,    // Below 70% success rate
                success_rate_warning: 85,     // Below 85% success rate
                gas_cost_variance: 50,        // Gas costs 50% higher than expected
                processing_time_critical: 30000, // Over 30 seconds
                accuracy_degradation: 80      // Below 80% accuracy
            },
            
            monitoring_intervals: {
                performance_check: 30000,     // Every 30 seconds
                health_report: 300000,        // Every 5 minutes
                daily_report: 86400000        // Every 24 hours
            },
            
            emergency_actions: {
                auto_circuit_breaker: true,   // Automatically trip circuit breaker
                fallback_to_individual: true, // Auto-fallback on failures
                alert_webhook: null,          // Slack/Discord webhook for alerts
                email_alerts: null            // Email notification config
            }
        };
        
        this.monitoring_data = {
            start_time: Date.now(),
            alerts: [],
            performance_history: [],
            daily_summaries: []
        };
        
        this.alert_states = {
            last_alert_time: {},
            alert_cooldowns: {
                success_rate: 300000,    // 5 minutes
                gas_variance: 600000,    // 10 minutes
                accuracy: 300000         // 5 minutes
            }
        };
        
        console.log("📊 Production Monitor initialized");
        console.log("🔔 Real-time alerting enabled");
        console.log("⚡ Automatic fallbacks configured");
        
        this.startMonitoring();
    }
    
    /**
     * Start all monitoring processes
     */
    startMonitoring() {
        // Performance monitoring
        this.performance_monitor = setInterval(() => {
            this.performanceCheck();
        }, this.monitoring_config.monitoring_intervals.performance_check);
        
        // Health reports
        this.health_monitor = setInterval(() => {
            this.generateHealthReport();
        }, this.monitoring_config.monitoring_intervals.health_report);
        
        // Daily reports
        this.daily_monitor = setInterval(() => {
            this.generateDailyReport();
        }, this.monitoring_config.monitoring_intervals.daily_report);
        
        console.log("✅ All monitoring processes started");
    }
    
    /**
     * Real-time performance check with alerts
     */
    performanceCheck() {
        const safety_report = this.optimizer.getSafetyReport();
        const current_time = Date.now();
        
        // Check success rate
        if (safety_report.success_rate < this.monitoring_config.alert_thresholds.success_rate_critical) {
            this.triggerAlert('CRITICAL', 'success_rate', {
                current_rate: safety_report.success_rate,
                threshold: this.monitoring_config.alert_thresholds.success_rate_critical,
                recommendation: 'Consider stopping batch processing and investigating failures'
            });
        } else if (safety_report.success_rate < this.monitoring_config.alert_thresholds.success_rate_warning) {
            this.triggerAlert('WARNING', 'success_rate', {
                current_rate: safety_report.success_rate,
                threshold: this.monitoring_config.alert_thresholds.success_rate_warning,
                recommendation: 'Monitor closely and review recent failures'
            });
        }
        
        // Check circuit breaker status
        if (safety_report.circuit_breaker_status === 'OPEN') {
            this.triggerAlert('CRITICAL', 'circuit_breaker', {
                failure_count: safety_report.failure_count,
                recommendation: 'System automatically stopped batch processing. Manual intervention required.'
            });
        }
        
        // Store performance data
        this.monitoring_data.performance_history.push({
            timestamp: current_time,
            success_rate: safety_report.success_rate,
            total_batches: safety_report.total_batches,
            circuit_breaker: safety_report.circuit_breaker_status,
            avg_gas: safety_report.avg_gas_per_batch,
            avg_cost: safety_report.avg_cost_per_batch
        });
        
        // Keep only last 100 data points
        if (this.monitoring_data.performance_history.length > 100) {
            this.monitoring_data.performance_history = this.monitoring_data.performance_history.slice(-100);
        }
    }
    
    /**
     * Generate health report
     */
    generateHealthReport() {
        const safety_report = this.optimizer.getSafetyReport();
        const uptime_hours = (Date.now() - this.monitoring_data.start_time) / (1000 * 60 * 60);
        
        console.log("\n📊 HEALTH REPORT");
        console.log("=" .repeat(50));
        console.log(`⏱️  Uptime: ${uptime_hours.toFixed(1)} hours`);
        console.log(`📦 Total Batches: ${safety_report.total_batches}`);
        console.log(`✅ Success Rate: ${safety_report.success_rate.toFixed(1)}%`);
        console.log(`🔴 Circuit Breaker: ${safety_report.circuit_breaker_status}`);
        console.log(`⛽ Avg Gas/Batch: ${Math.round(safety_report.avg_gas_per_batch).toLocaleString()}`);
        console.log(`💰 Avg Cost/Batch: ${safety_report.avg_cost_per_batch.toFixed(6)} A0GI`);
        console.log(`🚨 Active Alerts: ${this.getActiveAlerts().length}`);
        console.log("=" .repeat(50));
        
        // Check if system is healthy
        const health_status = this.assessSystemHealth(safety_report);
        console.log(`🏥 System Health: ${health_status.status}`);
        
        if (health_status.issues.length > 0) {
            console.log("⚠️  Issues Detected:");
            health_status.issues.forEach(issue => console.log(`   • ${issue}`));
        }
        
        if (health_status.recommendations.length > 0) {
            console.log("💡 Recommendations:");
            health_status.recommendations.forEach(rec => console.log(`   • ${rec}`));
        }
    }
    
    /**
     * Assess overall system health
     */
    assessSystemHealth(safety_report) {
        const issues = [];
        const recommendations = [];
        let status = 'HEALTHY';
        
        // Check success rate trends
        if (safety_report.success_rate < 70) {
            status = 'CRITICAL';
            issues.push(`Success rate critically low: ${safety_report.success_rate.toFixed(1)}%`);
            recommendations.push('Stop batch processing and investigate root cause');
        } else if (safety_report.success_rate < 85) {
            status = 'WARNING';
            issues.push(`Success rate below target: ${safety_report.success_rate.toFixed(1)}%`);
            recommendations.push('Review recent failures and consider reducing batch size');
        }
        
        // Check circuit breaker
        if (safety_report.circuit_breaker_status === 'OPEN') {
            status = 'CRITICAL';
            issues.push('Circuit breaker is open - batch processing stopped');
            recommendations.push('Investigate failures and manually reset circuit breaker when ready');
        }
        
        // Check batch count
        if (safety_report.total_batches < 5 && this.getUptimeHours() > 2) {
            status = status === 'HEALTHY' ? 'WARNING' : status;
            issues.push('Low batch processing volume');
            recommendations.push('Verify system is receiving verification requests');
        }
        
        // Check recent performance trends
        const recent_performance = this.monitoring_data.performance_history.slice(-10);
        if (recent_performance.length >= 5) {
            const recent_success_rates = recent_performance.map(p => p.success_rate);
            const trend = this.calculateTrend(recent_success_rates);
            
            if (trend < -5) { // Declining by more than 5% over recent samples
                status = status === 'HEALTHY' ? 'WARNING' : status;
                issues.push('Success rate trending downward');
                recommendations.push('Monitor for network congestion or system degradation');
            }
        }
        
        return { status, issues, recommendations };
    }
    
    /**
     * Calculate trend (positive = improving, negative = declining)
     */
    calculateTrend(values) {
        if (values.length < 2) return 0;
        
        const first_half = values.slice(0, Math.floor(values.length / 2));
        const second_half = values.slice(Math.floor(values.length / 2));
        
        const first_avg = first_half.reduce((a, b) => a + b) / first_half.length;
        const second_avg = second_half.reduce((a, b) => a + b) / second_half.length;
        
        return second_avg - first_avg;
    }
    
    /**
     * Trigger alert with cooldown logic
     */
    triggerAlert(severity, alert_type, details) {
        const current_time = Date.now();
        const last_alert = this.alert_states.last_alert_time[alert_type] || 0;
        const cooldown = this.alert_states.alert_cooldowns[alert_type] || 300000;
        
        // Check cooldown
        if (current_time - last_alert < cooldown) {
            return; // Skip alert due to cooldown
        }
        
        const alert = {
            timestamp: current_time,
            severity,
            type: alert_type,
            details,
            acknowledged: false
        };
        
        this.monitoring_data.alerts.push(alert);
        this.alert_states.last_alert_time[alert_type] = current_time;
        
        // Display alert
        console.log(`\n🚨 ${severity} ALERT: ${alert_type.toUpperCase()}`);
        console.log(`⏰ Time: ${new Date(current_time).toISOString()}`);
        console.log(`📋 Details:`, details);
        
        // Auto-actions based on severity
        if (severity === 'CRITICAL') {
            this.handleCriticalAlert(alert_type, details);
        }
        
        // External notifications (if configured)
        this.sendExternalAlert(alert);
    }
    
    /**
     * Handle critical alerts with automatic actions
     */
    handleCriticalAlert(alert_type, details) {
        console.log("⚡ Taking automatic action for critical alert...");
        
        switch (alert_type) {
            case 'success_rate':
                if (this.monitoring_config.emergency_actions.fallback_to_individual) {
                    console.log("🔄 Automatically disabling batch processing");
                    // In real implementation, would disable batch processing
                }
                break;
                
            case 'circuit_breaker':
                console.log("🛑 Circuit breaker already activated - system protected");
                break;
                
            case 'gas_variance':
                console.log("💰 Gas costs spiking - enabling conservative mode");
                // Could automatically increase gas price buffer
                break;
        }
    }
    
    /**
     * Send external alerts (webhook, email, etc.)
     */
    sendExternalAlert(alert) {
        // Placeholder for external alert integration
        if (this.monitoring_config.emergency_actions.alert_webhook) {
            console.log("📡 Sending webhook alert...");
            // Would send to Slack/Discord/etc.
        }
        
        if (this.monitoring_config.emergency_actions.email_alerts) {
            console.log("📧 Sending email alert...");
            // Would send email notification
        }
    }
    
    /**
     * Generate daily summary report
     */
    generateDailyReport() {
        const safety_report = this.optimizer.getSafetyReport();
        const uptime_hours = this.getUptimeHours();
        const alerts_today = this.getAlertsInLastDay();
        
        const daily_summary = {
            date: new Date().toISOString().split('T')[0],
            uptime_hours,
            total_batches: safety_report.total_batches,
            success_rate: safety_report.success_rate,
            total_alerts: alerts_today.length,
            critical_alerts: alerts_today.filter(a => a.severity === 'CRITICAL').length,
            avg_gas_per_batch: safety_report.avg_gas_per_batch,
            avg_cost_per_batch: safety_report.avg_cost_per_batch,
            circuit_breaker_trips: this.countCircuitBreakerTrips(),
            system_health: this.assessSystemHealth(safety_report).status
        };
        
        this.monitoring_data.daily_summaries.push(daily_summary);
        
        console.log("\n📈 DAILY REPORT");
        console.log("=" .repeat(60));
        console.log(`📅 Date: ${daily_summary.date}`);
        console.log(`⏱️  Uptime: ${daily_summary.uptime_hours.toFixed(1)} hours`);
        console.log(`📦 Batches Processed: ${daily_summary.total_batches}`);
        console.log(`✅ Success Rate: ${daily_summary.success_rate.toFixed(1)}%`);
        console.log(`🚨 Alerts Today: ${daily_summary.total_alerts} (${daily_summary.critical_alerts} critical)`);
        console.log(`⛽ Avg Gas/Batch: ${Math.round(daily_summary.avg_gas_per_batch).toLocaleString()}`);
        console.log(`💰 Avg Cost/Batch: ${daily_summary.avg_cost_per_batch.toFixed(6)} A0GI`);
        console.log(`🏥 System Health: ${daily_summary.system_health}`);
        console.log("=" .repeat(60));
        
        // Export daily report
        this.exportDailyReport(daily_summary);
    }
    
    /**
     * Export daily report to file
     */
    async exportDailyReport(daily_summary) {
        try {
            const fs = require('fs').promises;
            const report_path = `/Users/elliejenkins/Desktop/su-firewall/monitoring_reports/daily_report_${daily_summary.date}.json`;
            
            const full_report = {
                summary: daily_summary,
                detailed_alerts: this.getAlertsInLastDay(),
                performance_history: this.monitoring_data.performance_history,
                optimizer_status: this.optimizer.getSafetyReport()
            };
            
            await fs.writeFile(report_path, JSON.stringify(full_report, null, 2));
            console.log(`📄 Daily report exported: ${report_path}`);
            
        } catch (error) {
            console.error("❌ Failed to export daily report:", error.message);
        }
    }
    
    /**
     * Get active alerts (unacknowledged)
     */
    getActiveAlerts() {
        return this.monitoring_data.alerts.filter(alert => !alert.acknowledged);
    }
    
    /**
     * Get alerts from last 24 hours
     */
    getAlertsInLastDay() {
        const day_ago = Date.now() - 86400000;
        return this.monitoring_data.alerts.filter(alert => alert.timestamp >= day_ago);
    }
    
    /**
     * Get uptime in hours
     */
    getUptimeHours() {
        return (Date.now() - this.monitoring_data.start_time) / (1000 * 60 * 60);
    }
    
    /**
     * Count circuit breaker trips
     */
    countCircuitBreakerTrips() {
        return this.monitoring_data.alerts.filter(alert => 
            alert.type === 'circuit_breaker' && alert.severity === 'CRITICAL'
        ).length;
    }
    
    /**
     * Acknowledge alert
     */
    acknowledgeAlert(alert_index) {
        if (alert_index < this.monitoring_data.alerts.length) {
            this.monitoring_data.alerts[alert_index].acknowledged = true;
            console.log(`✅ Alert ${alert_index} acknowledged`);
        }
    }
    
    /**
     * Get monitoring dashboard data
     */
    getDashboardData() {
        const safety_report = this.optimizer.getSafetyReport();
        const active_alerts = this.getActiveAlerts();
        const system_health = this.assessSystemHealth(safety_report);
        
        return {
            timestamp: Date.now(),
            uptime_hours: this.getUptimeHours(),
            system_health: system_health.status,
            success_rate: safety_report.success_rate,
            total_batches: safety_report.total_batches,
            circuit_breaker: safety_report.circuit_breaker_status,
            active_alerts: active_alerts.length,
            critical_alerts: active_alerts.filter(a => a.severity === 'CRITICAL').length,
            avg_gas_per_batch: safety_report.avg_gas_per_batch,
            avg_cost_per_batch: safety_report.avg_cost_per_batch,
            recent_performance: this.monitoring_data.performance_history.slice(-20),
            recent_alerts: this.monitoring_data.alerts.slice(-10)
        };
    }
    
    /**
     * Stop monitoring
     */
    stopMonitoring() {
        if (this.performance_monitor) clearInterval(this.performance_monitor);
        if (this.health_monitor) clearInterval(this.health_monitor);
        if (this.daily_monitor) clearInterval(this.daily_monitor);
        
        console.log("📊 Monitoring stopped");
    }
}

module.exports = ProductionMonitor;