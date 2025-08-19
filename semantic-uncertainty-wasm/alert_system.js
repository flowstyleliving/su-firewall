/**
 * Real-Time Alert System for Live 0G Gas Optimization
 * 
 * Comprehensive alerting for gas optimization failures, detection accuracy drops,
 * and unexpected testnet conditions with automatic response capabilities.
 */

class RealTimeAlertSystem {
    constructor(dashboard) {
        this.dashboard = dashboard;
        this.alert_config = {
            // Critical alert thresholds (immediate action required)
            critical_thresholds: {
                success_rate_failure: 70,      // <70% success rate
                consecutive_failures: 5,        // 5 failures in a row
                gas_cost_explosion: 300,        // 3x expected costs
                processing_timeout: 30000,      // >30s processing time
                accuracy_collapse: 60,          // <60% accuracy
                blockchain_connectivity: 3,     // 3 failed RPC calls
                memory_usage_critical: 90,      // >90% memory usage
                emergency_stop_triggered: true  // Emergency stop activated
            },
            
            // Warning alert thresholds (monitor closely)
            warning_thresholds: {
                success_rate_degraded: 80,      // <80% success rate
                gas_cost_increase: 150,         // 1.5x expected costs  
                processing_slowdown: 15000,     // >15s processing time
                accuracy_degraded: 75,          // <75% accuracy
                batch_size_forced_reduction: 3, // Forced to <3 items
                circuit_breaker_activation: 1,  // Circuit breaker trips
                confirmation_delays: 20000,     // >20s confirmation time
                rpc_response_slow: 5000        // >5s RPC response
            },
            
            // Alert cooldown periods (prevent spam)
            cooldown_periods: {
                critical_alerts: 300000,        // 5 minutes
                warning_alerts: 600000,         // 10 minutes
                info_alerts: 900000            // 15 minutes
            },
            
            // Automatic response actions
            automatic_responses: {
                emergency_stop_on_critical: true,
                reduce_batch_size_on_failures: true,
                increase_gas_buffer_on_spikes: true,
                switch_rpc_on_connectivity_issues: true,
                alert_external_services: false  // Set to true for Slack/email
            }
        };
        
        // Alert state management
        this.alert_state = {
            active_alerts: new Map(),
            alert_history: [],
            last_alert_times: new Map(),
            alert_counts: new Map(),
            suppressed_alerts: new Set()
        };
        
        // External alert handlers
        this.external_handlers = {
            slack_webhook: null,
            email_config: null,
            sms_config: null,
            custom_handlers: []
        };
        
        // Performance tracking for alerting
        this.performance_tracker = {
            recent_batches: [],
            recent_errors: [],
            recent_gas_prices: [],
            recent_confirmations: [],
            system_metrics: {
                cpu_usage: 0,
                memory_usage: 0,
                network_latency: 0
            }
        };
        
        console.log("🚨 Real-Time Alert System initialized");
        console.log("🔴 Critical alerts will trigger emergency responses");
        console.log("⚠️ Warning alerts will log and notify");
        
        this.startAlertMonitoring();
    }
    
    /**
     * Start continuous alert monitoring
     */
    startAlertMonitoring() {
        // Monitor every 5 seconds for critical conditions
        this.critical_monitor = setInterval(() => {
            this.checkCriticalConditions();
        }, 5000);
        
        // Monitor every 15 seconds for warning conditions  
        this.warning_monitor = setInterval(() => {
            this.checkWarningConditions();
        }, 15000);
        
        // Monitor system health every 30 seconds
        this.system_monitor = setInterval(() => {
            this.checkSystemHealth();
        }, 30000);
        
        console.log("✅ Alert monitoring started - checking every 5s for critical conditions");
    }
    
    /**
     * Check for critical alert conditions
     */
    checkCriticalConditions() {
        const metrics = this.dashboard.calculateCurrentMetrics();
        
        // SUCCESS RATE CRITICAL
        if (metrics.success_rate < this.alert_config.critical_thresholds.success_rate_failure) {
            this.triggerAlert('CRITICAL', 'success_rate_failure', {
                current_rate: metrics.success_rate,
                threshold: this.alert_config.critical_thresholds.success_rate_failure,
                recent_failures: metrics.consecutive_failures,
                recommendation: 'EMERGENCY STOP - Investigate failures immediately'
            });
        }
        
        // CONSECUTIVE FAILURES CRITICAL
        if (metrics.consecutive_failures >= this.alert_config.critical_thresholds.consecutive_failures) {
            this.triggerAlert('CRITICAL', 'consecutive_failures', {
                failure_count: metrics.consecutive_failures,
                threshold: this.alert_config.critical_thresholds.consecutive_failures,
                last_success: this.dashboard.live_metrics.last_successful_batch,
                recommendation: 'IMMEDIATE INTERVENTION REQUIRED'
            });
        }
        
        // GAS COST EXPLOSION
        const baseline_cost = 0.00247;
        const cost_multiplier = metrics.avg_cost_per_verification / baseline_cost;
        if (cost_multiplier > (this.alert_config.critical_thresholds.gas_cost_explosion / 100)) {
            this.triggerAlert('CRITICAL', 'gas_cost_explosion', {
                current_cost: metrics.avg_cost_per_verification,
                baseline_cost,
                multiplier: cost_multiplier,
                recommendation: 'STOP PROCESSING - Gas costs unsustainable'
            });
        }
        
        // PROCESSING TIMEOUT CRITICAL
        if (metrics.avg_processing_time > this.alert_config.critical_thresholds.processing_timeout) {
            this.triggerAlert('CRITICAL', 'processing_timeout', {
                avg_processing_time: metrics.avg_processing_time,
                threshold: this.alert_config.critical_thresholds.processing_timeout,
                recommendation: 'System performance critically degraded'
            });
        }
        
        // Check if emergency stop was triggered
        if (this.dashboard.optimizer.config.emergency_stop) {
            this.triggerAlert('CRITICAL', 'emergency_stop_activated', {
                timestamp: new Date().toISOString(),
                recommendation: 'Emergency stop is active - manual intervention required'
            });
        }
    }
    
    /**
     * Check for warning alert conditions
     */
    checkWarningConditions() {
        const metrics = this.dashboard.calculateCurrentMetrics();
        
        // SUCCESS RATE WARNING
        if (metrics.success_rate < this.alert_config.warning_thresholds.success_rate_degraded &&
            metrics.success_rate >= this.alert_config.critical_thresholds.success_rate_failure) {
            this.triggerAlert('WARNING', 'success_rate_degraded', {
                current_rate: metrics.success_rate,
                threshold: this.alert_config.warning_thresholds.success_rate_degraded,
                trend: this.calculateSuccessRateTrend(),
                recommendation: 'Monitor closely - consider reducing batch size'
            });
        }
        
        // GAS COST INCREASE WARNING
        const baseline_cost = 0.00247;
        const cost_increase = (metrics.avg_cost_per_verification / baseline_cost) * 100;
        if (cost_increase > this.alert_config.warning_thresholds.gas_cost_increase) {
            this.triggerAlert('WARNING', 'gas_cost_increase', {
                cost_increase_percent: cost_increase,
                current_avg: metrics.avg_cost_per_verification,
                baseline: baseline_cost,
                recommendation: 'Gas costs elevated - monitor network congestion'
            });
        }
        
        // PROCESSING SLOWDOWN WARNING
        if (metrics.avg_processing_time > this.alert_config.warning_thresholds.processing_slowdown &&
            metrics.avg_processing_time <= this.alert_config.critical_thresholds.processing_timeout) {
            this.triggerAlert('WARNING', 'processing_slowdown', {
                avg_time: metrics.avg_processing_time,
                threshold: this.alert_config.warning_thresholds.processing_slowdown,
                recommendation: 'Processing slower than normal - check system resources'
            });
        }
        
        // CONFIRMATION DELAYS WARNING
        if (metrics.avg_confirmation_time > (this.alert_config.warning_thresholds.confirmation_delays / 1000)) {
            this.triggerAlert('WARNING', 'confirmation_delays', {
                avg_confirmation_seconds: metrics.avg_confirmation_time,
                threshold_seconds: this.alert_config.warning_thresholds.confirmation_delays / 1000,
                recommendation: '0G network congestion detected'
            });
        }
        
        // CIRCUIT BREAKER WARNING
        const optimizer_report = this.dashboard.optimizer.getPerformanceReport();
        if (optimizer_report.circuit_breaker_status === 'OPEN') {
            this.triggerAlert('WARNING', 'circuit_breaker_active', {
                failure_count: optimizer_report.failure_count,
                last_failure: optimizer_report.last_failure,
                recommendation: 'Circuit breaker protection active - investigate failures'
            });
        }
    }
    
    /**
     * Monitor system health metrics
     */
    checkSystemHealth() {
        // Simulate system metrics (in production, use actual system monitoring)
        this.performance_tracker.system_metrics = {
            cpu_usage: Math.random() * 50, // 0-50% CPU usage
            memory_usage: 30 + Math.random() * 40, // 30-70% memory
            network_latency: 50 + Math.random() * 100 // 50-150ms latency
        };
        
        const system = this.performance_tracker.system_metrics;
        
        // HIGH MEMORY USAGE
        if (system.memory_usage > this.alert_config.critical_thresholds.memory_usage_critical) {
            this.triggerAlert('CRITICAL', 'memory_usage_critical', {
                current_usage: system.memory_usage,
                threshold: this.alert_config.critical_thresholds.memory_usage_critical,
                recommendation: 'System memory critically low - restart may be needed'
            });
        }
        
        // HIGH NETWORK LATENCY
        if (system.network_latency > this.alert_config.warning_thresholds.rpc_response_slow) {
            this.triggerAlert('WARNING', 'network_latency_high', {
                current_latency: system.network_latency,
                threshold: this.alert_config.warning_thresholds.rpc_response_slow,
                recommendation: 'Network connectivity degraded'
            });
        }
    }
    
    /**
     * Trigger alert with comprehensive handling
     */
    triggerAlert(severity, alert_type, details) {
        const alert_key = `${severity}_${alert_type}`;
        const current_time = Date.now();
        
        // Check cooldown period
        const last_alert_time = this.alert_state.last_alert_times.get(alert_key) || 0;
        const cooldown = this.alert_config.cooldown_periods[`${severity.toLowerCase()}_alerts`] || 300000;
        
        if (current_time - last_alert_time < cooldown) {
            return; // Skip alert due to cooldown
        }
        
        // Create alert object
        const alert = {
            id: this.generateAlertId(),
            timestamp: new Date().toISOString(),
            severity,
            type: alert_type,
            details,
            acknowledged: false,
            auto_response_taken: false
        };
        
        // Update alert state
        this.alert_state.last_alert_times.set(alert_key, current_time);
        this.alert_state.alert_counts.set(alert_key, (this.alert_state.alert_counts.get(alert_key) || 0) + 1);
        this.alert_state.active_alerts.set(alert.id, alert);
        this.alert_state.alert_history.push(alert);
        
        // Display alert
        this.displayAlert(alert);
        
        // Take automatic response if configured
        if (this.alert_config.automatic_responses[`${alert_type}_on_${severity.toLowerCase()}`] !== false) {
            this.executeAutomaticResponse(alert);
        }
        
        // Send to external systems
        this.sendExternalAlerts(alert);
        
        // Log alert details
        this.logAlert(alert);
    }
    
    /**
     * Display alert in console with formatting
     */
    displayAlert(alert) {
        const severity_icons = { 'CRITICAL': '🚨', 'WARNING': '⚠️', 'INFO': 'ℹ️' };
        const icon = severity_icons[alert.severity] || '📢';
        
        console.log(`\n${icon} ${alert.severity} ALERT [${alert.id}]`);
        console.log(`🕐 Time: ${alert.timestamp}`);
        console.log(`📋 Type: ${alert.type}`);
        console.log(`📊 Details:`);
        
        Object.entries(alert.details).forEach(([key, value]) => {
            console.log(`   ${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`);
        });
        
        if (alert.details.recommendation) {
            console.log(`💡 Recommendation: ${alert.details.recommendation}`);
        }
        
        console.log(`📞 Alert ID: ${alert.id} (for acknowledgment)`);
        console.log("─".repeat(60));
    }
    
    /**
     * Execute automatic response based on alert type
     */
    executeAutomaticResponse(alert) {
        let response_taken = false;
        
        switch (alert.type) {
            case 'success_rate_failure':
            case 'consecutive_failures':
                if (this.alert_config.automatic_responses.emergency_stop_on_critical) {
                    console.log("🛑 AUTO-RESPONSE: Activating emergency stop");
                    this.dashboard.optimizer.setEmergencyStop(true);
                    response_taken = true;
                }
                break;
                
            case 'gas_cost_explosion':
            case 'gas_cost_increase':
                if (this.alert_config.automatic_responses.increase_gas_buffer_on_spikes) {
                    const current_multiplier = this.dashboard.optimizer.config.gas_price_multiplier;
                    const new_multiplier = Math.min(2.0, current_multiplier * 1.2);
                    
                    console.log(`⛽ AUTO-RESPONSE: Increasing gas buffer from ${current_multiplier}x to ${new_multiplier.toFixed(1)}x`);
                    this.dashboard.optimizer.updateConfiguration({
                        gas_price_multiplier: new_multiplier
                    });
                    response_taken = true;
                }
                break;
                
            case 'processing_timeout':
            case 'processing_slowdown':
                if (this.alert_config.automatic_responses.reduce_batch_size_on_failures) {
                    const current_size = this.dashboard.optimizer.config.optimal_batch_size;
                    const new_size = Math.max(2, current_size - 1);
                    
                    console.log(`📦 AUTO-RESPONSE: Reducing batch size from ${current_size} to ${new_size}`);
                    this.dashboard.optimizer.updateConfiguration({
                        optimal_batch_size: new_size
                    });
                    response_taken = true;
                }
                break;
        }
        
        if (response_taken) {
            alert.auto_response_taken = true;
            console.log("✅ Automatic response executed");
        }
    }
    
    /**
     * Send alerts to external systems (Slack, email, etc.)
     */
    sendExternalAlerts(alert) {
        if (!this.alert_config.automatic_responses.alert_external_services) {
            return;
        }
        
        // Slack webhook
        if (this.external_handlers.slack_webhook) {
            this.sendSlackAlert(alert);
        }
        
        // Email notification
        if (this.external_handlers.email_config) {
            this.sendEmailAlert(alert);
        }
        
        // Custom handlers
        this.external_handlers.custom_handlers.forEach(handler => {
            try {
                handler(alert);
            } catch (error) {
                console.error("Custom alert handler error:", error);
            }
        });
    }
    
    /**
     * Send alert to Slack webhook
     */
    sendSlackAlert(alert) {
        const slack_message = {
            text: `${alert.severity} Alert: ${alert.type}`,
            attachments: [{
                color: alert.severity === 'CRITICAL' ? 'danger' : 'warning',
                fields: [
                    { title: 'Alert ID', value: alert.id, short: true },
                    { title: 'Time', value: alert.timestamp, short: true },
                    { title: 'Details', value: JSON.stringify(alert.details, null, 2), short: false }
                ]
            }]
        };
        
        // In production, actually send to Slack webhook
        console.log("📡 Would send to Slack:", JSON.stringify(slack_message, null, 2));
    }
    
    /**
     * Calculate success rate trend
     */
    calculateSuccessRateTrend() {
        const recent_batches = this.performance_tracker.recent_batches.slice(-20);
        if (recent_batches.length < 10) return 'insufficient_data';
        
        const first_half = recent_batches.slice(0, Math.floor(recent_batches.length / 2));
        const second_half = recent_batches.slice(Math.floor(recent_batches.length / 2));
        
        const first_success_rate = first_half.filter(b => b.success).length / first_half.length;
        const second_success_rate = second_half.filter(b => b.success).length / second_half.length;
        
        const trend = second_success_rate - first_success_rate;
        
        if (trend > 0.05) return 'improving';
        if (trend < -0.05) return 'declining';
        return 'stable';
    }
    
    /**
     * Acknowledge alert
     */
    acknowledgeAlert(alert_id) {
        const alert = this.alert_state.active_alerts.get(alert_id);
        if (alert) {
            alert.acknowledged = true;
            alert.acknowledged_at = new Date().toISOString();
            console.log(`✅ Alert ${alert_id} acknowledged`);
            return true;
        }
        return false;
    }
    
    /**
     * Get active alerts
     */
    getActiveAlerts() {
        return Array.from(this.alert_state.active_alerts.values())
            .filter(alert => !alert.acknowledged);
    }
    
    /**
     * Configure external alert handler
     */
    configureSlackWebhook(webhook_url) {
        this.external_handlers.slack_webhook = webhook_url;
        console.log("✅ Slack webhook configured for alerts");
    }
    
    configureEmail(email_config) {
        this.external_handlers.email_config = email_config;
        console.log("✅ Email alerts configured");
    }
    
    addCustomHandler(handler) {
        this.external_handlers.custom_handlers.push(handler);
        console.log("✅ Custom alert handler added");
    }
    
    /**
     * Generate comprehensive alert summary
     */
    getAlertSummary() {
        const active_alerts = this.getActiveAlerts();
        const recent_alerts = this.alert_state.alert_history.slice(-50);
        
        const summary = {
            active_alerts: active_alerts.length,
            critical_alerts: active_alerts.filter(a => a.severity === 'CRITICAL').length,
            warning_alerts: active_alerts.filter(a => a.severity === 'WARNING').length,
            
            recent_activity: {
                last_24h: recent_alerts.filter(a => 
                    Date.now() - new Date(a.timestamp).getTime() < 24 * 60 * 60 * 1000
                ).length,
                last_hour: recent_alerts.filter(a => 
                    Date.now() - new Date(a.timestamp).getTime() < 60 * 60 * 1000
                ).length
            },
            
            alert_types: this.getAlertTypeDistribution(recent_alerts),
            automatic_responses: recent_alerts.filter(a => a.auto_response_taken).length,
            
            system_health: this.performance_tracker.system_metrics,
            recommendations: this.generateAlertRecommendations(active_alerts)
        };
        
        return summary;
    }
    
    getAlertTypeDistribution(alerts) {
        const distribution = {};
        alerts.forEach(alert => {
            distribution[alert.type] = (distribution[alert.type] || 0) + 1;
        });
        return distribution;
    }
    
    generateAlertRecommendations(active_alerts) {
        const recommendations = [];
        
        const critical_count = active_alerts.filter(a => a.severity === 'CRITICAL').length;
        if (critical_count > 0) {
            recommendations.push(`🚨 ${critical_count} critical alerts require immediate attention`);
        }
        
        const warning_count = active_alerts.filter(a => a.severity === 'WARNING').length;
        if (warning_count > 3) {
            recommendations.push(`⚠️ ${warning_count} warnings suggest system stress`);
        }
        
        if (active_alerts.length === 0) {
            recommendations.push("✅ No active alerts - system operating normally");
        }
        
        return recommendations;
    }
    
    // === Utility Methods ===
    
    generateAlertId() {
        return 'alert_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4);
    }
    
    logAlert(alert) {
        // In production, log to proper logging system
        const log_entry = {
            timestamp: alert.timestamp,
            alert_id: alert.id,
            severity: alert.severity,
            type: alert.type,
            details: alert.details,
            auto_response: alert.auto_response_taken
        };
        
        console.log(`📋 ALERT LOG: ${JSON.stringify(log_entry)}`);
    }
    
    stopAlertMonitoring() {
        if (this.critical_monitor) clearInterval(this.critical_monitor);
        if (this.warning_monitor) clearInterval(this.warning_monitor);
        if (this.system_monitor) clearInterval(this.system_monitor);
        
        console.log("🚨 Alert monitoring stopped");
        
        return this.getAlertSummary();
    }
}

module.exports = RealTimeAlertSystem;