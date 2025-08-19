/**
 * Live Production Monitoring Dashboard
 * 
 * Real-time monitoring for 0G testnet gas optimization with comprehensive
 * transaction tracking, performance metrics, and alert management.
 */

class ProductionMonitoringDashboard {
    constructor(optimizer) {
        this.optimizer = optimizer;
        this.dashboard_config = {
            // Real A0GI transaction tracking
            transaction_monitoring: {
                track_every_tx: true,
                verify_confirmations: true,
                monitor_gas_prices: true,
                track_block_times: true
            },
            
            // Performance thresholds (proven from simulation)
            performance_targets: {
                success_rate_target: 85,        // Must maintain 85%+
                gas_savings_target: 20,         // Must achieve 20%+
                max_processing_time_ms: 10000,  // 10 second limit
                max_confirmation_time_s: 30,    // 30 second blockchain limit
                accuracy_threshold: 80           // 80% minimum accuracy
            },
            
            // Alert thresholds (aggressive for live deployment)
            alert_thresholds: {
                success_rate_critical: 75,      // Emergency if <75%
                success_rate_warning: 80,       // Warning if <80%
                gas_cost_spike: 100,            // Alert if costs double
                processing_timeout: 15000,      // Alert if >15s processing
                accuracy_drop: 75,              // Alert if accuracy <75%
                failed_confirmations: 3         // Alert after 3 failed TXs
            },
            
            // Data collection intervals
            monitoring_frequency: {
                realtime_check: 10000,          // Every 10 seconds
                performance_summary: 60000,     // Every minute
                detailed_report: 300000,        // Every 5 minutes
                daily_summary: 3600000          // Every hour
            }
        };
        
        // Live metrics tracking
        this.live_metrics = {
            // Transaction tracking
            total_transactions: 0,
            successful_transactions: 0,
            failed_transactions: 0,
            pending_transactions: 0,
            
            // Gas optimization tracking
            total_gas_used: 0,
            total_gas_saved: 0,
            gas_savings_history: [],
            gas_price_history: [],
            
            // Performance tracking
            processing_times: [],
            confirmation_times: [],
            accuracy_scores: [],
            
            // A0GI cost tracking
            total_a0gi_spent: 0,
            total_a0gi_saved: 0,
            cost_per_verification: [],
            
            // Error tracking
            error_log: [],
            alert_history: [],
            
            // Uptime tracking
            start_time: Date.now(),
            last_successful_batch: null,
            consecutive_failures: 0
        };
        
        // Dashboard state
        this.dashboard_active = false;
        this.alert_handlers = [];
        this.monitoring_intervals = {};
        
        console.log("📊 Production Monitoring Dashboard initialized");
        console.log("🎯 Ready to track live 0G testnet performance");
    }
    
    /**
     * Start comprehensive live monitoring
     */
    startLiveMonitoring() {
        console.log("\n🚀 Starting Live Production Monitoring");
        console.log("=" .repeat(70));
        console.log("🔴 LIVE 0G TESTNET - REAL A0GI TRANSACTIONS");
        console.log("📊 Tracking every transaction with comprehensive metrics");
        console.log("=" .repeat(70));
        
        this.dashboard_active = true;
        
        // Start monitoring intervals
        this.monitoring_intervals.realtime = setInterval(() => {
            this.performRealtimeCheck();
        }, this.dashboard_config.monitoring_frequency.realtime_check);
        
        this.monitoring_intervals.performance = setInterval(() => {
            this.generatePerformanceSummary();
        }, this.dashboard_config.monitoring_frequency.performance_summary);
        
        this.monitoring_intervals.detailed = setInterval(() => {
            this.generateDetailedReport();
        }, this.dashboard_config.monitoring_frequency.detailed_report);
        
        this.monitoring_intervals.daily = setInterval(() => {
            this.generateDailySummary();
        }, this.dashboard_config.monitoring_frequency.daily_summary);
        
        // Display initial dashboard
        this.displayDashboard();
        
        console.log("✅ Live monitoring active - tracking all metrics");
    }
    
    /**
     * Track individual batch transaction with full details
     */
    trackBatchTransaction(batch_result, transaction_details) {
        const timestamp = Date.now();
        
        // Update transaction counts
        this.live_metrics.total_transactions++;
        
        if (batch_result.success && transaction_details.confirmed) {
            this.live_metrics.successful_transactions++;
            this.live_metrics.last_successful_batch = timestamp;
            this.live_metrics.consecutive_failures = 0;
            
            // Track gas metrics
            if (transaction_details.gas_used && transaction_details.gas_saved) {
                this.live_metrics.total_gas_used += transaction_details.gas_used;
                this.live_metrics.total_gas_saved += transaction_details.gas_saved;
                
                const savings_percent = (transaction_details.gas_saved / 
                    (transaction_details.gas_used + transaction_details.gas_saved)) * 100;
                
                this.live_metrics.gas_savings_history.push({
                    timestamp,
                    savings_percent,
                    gas_used: transaction_details.gas_used,
                    gas_saved: transaction_details.gas_saved
                });
            }
            
            // Track A0GI costs
            if (transaction_details.cost_a0gi) {
                this.live_metrics.total_a0gi_spent += transaction_details.cost_a0gi;
                
                const baseline_cost = batch_result.batch_size * 0.00247; // Individual processing cost
                const savings = baseline_cost - transaction_details.cost_a0gi;
                this.live_metrics.total_a0gi_saved += savings;
                
                this.live_metrics.cost_per_verification.push({
                    timestamp,
                    cost_per_item: transaction_details.cost_a0gi / batch_result.batch_size,
                    baseline_cost: baseline_cost / batch_result.batch_size,
                    savings_per_item: savings / batch_result.batch_size
                });
            }
            
            // Track performance metrics
            if (batch_result.processing_time_ms) {
                this.live_metrics.processing_times.push({
                    timestamp,
                    time_ms: batch_result.processing_time_ms,
                    batch_size: batch_result.batch_size
                });
            }
            
            if (transaction_details.confirmation_time_ms) {
                this.live_metrics.confirmation_times.push({
                    timestamp,
                    confirmation_ms: transaction_details.confirmation_time_ms
                });
            }
            
        } else {
            this.live_metrics.failed_transactions++;
            this.live_metrics.consecutive_failures++;
            
            // Log error details
            this.live_metrics.error_log.push({
                timestamp,
                error_type: batch_result.error || 'Unknown failure',
                batch_size: batch_result.batch_size,
                transaction_hash: transaction_details.tx_hash || null,
                consecutive_failure_number: this.live_metrics.consecutive_failures
            });
        }
        
        // Track gas prices for market analysis
        if (transaction_details.gas_price_gwei) {
            this.live_metrics.gas_price_history.push({
                timestamp,
                price_gwei: transaction_details.gas_price_gwei
            });
        }
        
        // Alert checking would happen here in full implementation
        
        // Keep metrics manageable (last 1000 entries)
        this.trimMetricsHistory();
        
        // Log transaction
        this.logTransactionDetails(batch_result, transaction_details);
    }
    
    /**
     * Real-time system health check
     */
    performRealtimeCheck() {
        if (!this.dashboard_active) return;
        
        const current_metrics = this.calculateCurrentMetrics();
        
        // Check critical thresholds
        if (current_metrics.success_rate < this.dashboard_config.alert_thresholds.success_rate_critical) {
            this.triggerAlert('CRITICAL', 'success_rate_critical', {
                current_rate: current_metrics.success_rate,
                threshold: this.dashboard_config.alert_thresholds.success_rate_critical,
                consecutive_failures: this.live_metrics.consecutive_failures
            });
        }
        
        // Check gas cost spikes
        if (current_metrics.gas_cost_increase > this.dashboard_config.alert_thresholds.gas_cost_spike) {
            this.triggerAlert('WARNING', 'gas_cost_spike', {
                cost_increase_percent: current_metrics.gas_cost_increase,
                current_avg_cost: current_metrics.avg_cost_per_verification
            });
        }
        
        // Check processing timeouts
        if (current_metrics.avg_processing_time > this.dashboard_config.alert_thresholds.processing_timeout) {
            this.triggerAlert('WARNING', 'processing_timeout', {
                avg_time_ms: current_metrics.avg_processing_time,
                threshold: this.dashboard_config.alert_thresholds.processing_timeout
            });
        }
        
        // Update dashboard display
        this.updateDashboardDisplay(current_metrics);
    }
    
    /**
     * Generate comprehensive performance summary
     */
    generatePerformanceSummary() {
        const summary = this.calculateCurrentMetrics();
        const timestamp = new Date().toISOString();
        
        console.log(`\n📊 [${timestamp.split('T')[1].split('.')[0]}] Performance Summary:`);
        console.log(`   🎯 Success Rate: ${summary.success_rate.toFixed(1)}% (target: ${this.dashboard_config.performance_targets.success_rate_target}%)`);
        console.log(`   ⛽ Gas Savings: ${summary.avg_gas_savings.toFixed(1)}% (target: ${this.dashboard_config.performance_targets.gas_savings_target}%)`);
        console.log(`   💰 A0GI Saved: ${this.live_metrics.total_a0gi_saved.toFixed(6)} A0GI`);
        console.log(`   📦 Batches: ${this.live_metrics.successful_transactions}/${this.live_metrics.total_transactions}`);
        console.log(`   ⏱️ Avg Processing: ${summary.avg_processing_time.toFixed(0)}ms`);
        console.log(`   🔗 Avg Confirmation: ${summary.avg_confirmation_time.toFixed(1)}s`);
        
        // Alert on performance degradation
        if (summary.success_rate < this.dashboard_config.performance_targets.success_rate_target) {
            console.log(`   ⚠️ SUCCESS RATE BELOW TARGET`);
        }
        if (summary.avg_gas_savings < this.dashboard_config.performance_targets.gas_savings_target) {
            console.log(`   ⚠️ GAS SAVINGS BELOW TARGET`);
        }
    }
    
    /**
     * Calculate current performance metrics
     */
    calculateCurrentMetrics() {
        const total_tx = this.live_metrics.total_transactions;
        const successful_tx = this.live_metrics.successful_transactions;
        
        // Success rate
        const success_rate = total_tx > 0 ? (successful_tx / total_tx) * 100 : 0;
        
        // Gas savings
        const recent_savings = this.live_metrics.gas_savings_history.slice(-20); // Last 20 batches
        const avg_gas_savings = recent_savings.length > 0 ? 
            recent_savings.reduce((sum, s) => sum + s.savings_percent, 0) / recent_savings.length : 0;
        
        // Processing time
        const recent_processing = this.live_metrics.processing_times.slice(-20);
        const avg_processing_time = recent_processing.length > 0 ?
            recent_processing.reduce((sum, p) => sum + p.time_ms, 0) / recent_processing.length : 0;
        
        // Confirmation time
        const recent_confirmations = this.live_metrics.confirmation_times.slice(-20);
        const avg_confirmation_time = recent_confirmations.length > 0 ?
            recent_confirmations.reduce((sum, c) => sum + c.confirmation_ms, 0) / recent_confirmations.length / 1000 : 0;
        
        // Cost analysis
        const recent_costs = this.live_metrics.cost_per_verification.slice(-20);
        const avg_cost_per_verification = recent_costs.length > 0 ?
            recent_costs.reduce((sum, c) => sum + c.cost_per_item, 0) / recent_costs.length : 0;
        
        // Gas price trends
        const recent_gas_prices = this.live_metrics.gas_price_history.slice(-10);
        const avg_gas_price = recent_gas_prices.length > 0 ?
            recent_gas_prices.reduce((sum, g) => sum + g.price_gwei, 0) / recent_gas_prices.length : 0;
        
        // Cost increase detection
        const baseline_cost = 0.00247; // Individual processing cost
        const cost_increase_percent = avg_cost_per_verification > 0 ? 
            ((avg_cost_per_verification - baseline_cost) / baseline_cost) * 100 : 0;
        
        return {
            success_rate,
            avg_gas_savings,
            avg_processing_time,
            avg_confirmation_time,
            avg_cost_per_verification,
            avg_gas_price,
            gas_cost_increase: Math.abs(cost_increase_percent),
            total_transactions: total_tx,
            successful_transactions: successful_tx,
            consecutive_failures: this.live_metrics.consecutive_failures,
            uptime_hours: (Date.now() - this.live_metrics.start_time) / (1000 * 60 * 60)
        };
    }
    
    /**
     * Trigger alert with appropriate handling
     */
    triggerAlert(severity, alert_type, details) {
        const alert = {
            timestamp: new Date().toISOString(),
            severity,
            type: alert_type,
            details,
            acknowledged: false
        };
        
        this.live_metrics.alert_history.push(alert);
        
        // Display alert
        console.log(`\n🚨 ${severity} ALERT: ${alert_type.toUpperCase()}`);
        console.log(`⏰ Time: ${alert.timestamp}`);
        console.log(`📋 Details:`, JSON.stringify(details, null, 2));
        
        // Take automatic actions based on severity and type
        if (severity === 'CRITICAL') {
            this.handleCriticalAlert(alert_type, details);
        }
        
        // Call registered alert handlers
        this.alert_handlers.forEach(handler => {
            try {
                handler(alert);
            } catch (error) {
                console.error("Alert handler error:", error);
            }
        });
    }
    
    /**
     * Handle critical alerts with automatic responses
     */
    handleCriticalAlert(alert_type, details) {
        console.log("⚡ TAKING AUTOMATIC ACTION FOR CRITICAL ALERT");
        
        switch (alert_type) {
            case 'success_rate_critical':
                if (details.consecutive_failures >= 5) {
                    console.log("🛑 EMERGENCY STOP - Too many consecutive failures");
                    this.optimizer.setEmergencyStop(true);
                }
                break;
                
            case 'gas_cost_spike':
                console.log("💰 Gas costs spiking - increasing safety buffer");
                this.optimizer.updateConfiguration({
                    gas_price_multiplier: Math.min(2.0, this.optimizer.config.gas_price_multiplier * 1.2)
                });
                break;
        }
    }
    
    /**
     * Add custom alert handler
     */
    addAlertHandler(handler) {
        this.alert_handlers.push(handler);
    }
    
    /**
     * Generate detailed operational report
     */
    generateDetailedReport() {
        const metrics = this.calculateCurrentMetrics();
        const report_time = new Date().toISOString();
        
        const detailed_report = {
            report_timestamp: report_time,
            system_status: 'OPERATIONAL',
            uptime_hours: metrics.uptime_hours,
            
            // Performance metrics
            performance: {
                success_rate: metrics.success_rate,
                target_success_rate: this.dashboard_config.performance_targets.success_rate_target,
                avg_gas_savings: metrics.avg_gas_savings,
                target_gas_savings: this.dashboard_config.performance_targets.gas_savings_target,
                avg_processing_time_ms: metrics.avg_processing_time,
                avg_confirmation_time_s: metrics.avg_confirmation_time
            },
            
            // Financial metrics
            financials: {
                total_a0gi_spent: this.live_metrics.total_a0gi_spent,
                total_a0gi_saved: this.live_metrics.total_a0gi_saved,
                savings_rate_percent: this.live_metrics.total_a0gi_spent > 0 ? 
                    (this.live_metrics.total_a0gi_saved / this.live_metrics.total_a0gi_spent) * 100 : 0,
                avg_cost_per_verification: metrics.avg_cost_per_verification,
                baseline_cost_per_verification: 0.00247
            },
            
            // Transaction statistics
            transactions: {
                total: metrics.total_transactions,
                successful: metrics.successful_transactions,
                failed: metrics.total_transactions - metrics.successful_transactions,
                consecutive_failures: metrics.consecutive_failures,
                last_successful: this.live_metrics.last_successful_batch
            },
            
            // Recent alerts
            recent_alerts: this.live_metrics.alert_history.slice(-10),
            
            // Operational recommendations
            recommendations: this.generateOperationalRecommendations(metrics)
        };
        
        // Save detailed report
        this.saveReport('detailed', detailed_report);
        
        return detailed_report;
    }
    
    generateOperationalRecommendations(metrics) {
        const recommendations = [];
        
        // Performance recommendations
        if (metrics.success_rate < 85) {
            recommendations.push({
                priority: 'HIGH',
                category: 'performance',
                message: 'Success rate below target - consider reducing batch size',
                action: 'reduce_batch_size'
            });
        }
        
        if (metrics.avg_gas_savings < 20) {
            recommendations.push({
                priority: 'MEDIUM',
                category: 'optimization', 
                message: 'Gas savings below target - review batch efficiency',
                action: 'optimize_batching'
            });
        }
        
        if (metrics.consecutive_failures >= 3) {
            recommendations.push({
                priority: 'HIGH',
                category: 'reliability',
                message: 'Multiple consecutive failures - investigate root cause',
                action: 'investigate_failures'
            });
        }
        
        // Scaling recommendations
        if (metrics.success_rate >= 90 && metrics.avg_gas_savings >= 25) {
            recommendations.push({
                priority: 'LOW',
                category: 'scaling',
                message: 'System performing well - ready for Phase 2 scaling',
                action: 'consider_phase2'
            });
        }
        
        return recommendations;
    }
    
    /**
     * Emergency procedures and rollback
     */
    initiateEmergencyRollback(reason) {
        console.log("\n🚨 INITIATING EMERGENCY ROLLBACK");
        console.log(`📋 Reason: ${reason}`);
        
        // Step 1: Stop batch processing
        this.optimizer.setEmergencyStop(true);
        console.log("✅ Batch processing stopped");
        
        // Step 2: Reset circuit breaker
        this.optimizer.resetCircuitBreaker();
        console.log("✅ Circuit breaker reset");
        
        // Step 3: Save emergency report
        const emergency_report = {
            timestamp: new Date().toISOString(),
            reason,
            metrics_at_rollback: this.calculateCurrentMetrics(),
            recent_errors: this.live_metrics.error_log.slice(-10),
            recent_alerts: this.live_metrics.alert_history.slice(-5)
        };
        
        this.saveReport('emergency', emergency_report);
        console.log("✅ Emergency report saved");
        
        // Step 4: Provide recovery instructions
        console.log("\n📋 RECOVERY INSTRUCTIONS:");
        console.log("1. Review emergency report in monitoring_reports/");
        console.log("2. Address root cause of failures");
        console.log("3. Test individual processing mode");
        console.log("4. Re-enable batch processing with optimizer.setEmergencyStop(false)");
        console.log("5. Monitor closely for 1 hour before scaling");
        
        return emergency_report;
    }
    
    // === Utility Methods ===
    
    displayDashboard() {
        const metrics = this.calculateCurrentMetrics();
        
        console.log("\n" + "=" .repeat(70));
        console.log("📊 LIVE 0G PRODUCTION DASHBOARD");
        console.log("=" .repeat(70));
        console.log(`🕐 Uptime: ${metrics.uptime_hours.toFixed(1)} hours`);
        console.log(`🎯 Success Rate: ${metrics.success_rate.toFixed(1)}% (Target: 85%+)`);
        console.log(`⛽ Gas Savings: ${metrics.avg_gas_savings.toFixed(1)}% (Target: 20%+)`);
        console.log(`💰 A0GI Saved: ${this.live_metrics.total_a0gi_saved.toFixed(6)}`);
        console.log(`📦 Transactions: ${metrics.successful_transactions}/${metrics.total_transactions}`);
        console.log(`🚨 Active Alerts: ${this.live_metrics.alert_history.filter(a => !a.acknowledged).length}`);
        console.log("=" .repeat(70));
    }
    
    updateDashboardDisplay(metrics) {
        // Update display every 10th check to avoid spam
        if (this.live_metrics.total_transactions % 10 === 0) {
            this.displayDashboard();
        }
    }
    
    trimMetricsHistory() {
        const max_entries = 1000;
        
        if (this.live_metrics.gas_savings_history.length > max_entries) {
            this.live_metrics.gas_savings_history = this.live_metrics.gas_savings_history.slice(-max_entries/2);
        }
        if (this.live_metrics.processing_times.length > max_entries) {
            this.live_metrics.processing_times = this.live_metrics.processing_times.slice(-max_entries/2);
        }
        if (this.live_metrics.cost_per_verification.length > max_entries) {
            this.live_metrics.cost_per_verification = this.live_metrics.cost_per_verification.slice(-max_entries/2);
        }
    }
    
    logTransactionDetails(batch_result, transaction_details) {
        const log_entry = {
            timestamp: new Date().toISOString(),
            batch_id: batch_result.batch_id || 'unknown',
            success: batch_result.success,
            batch_size: batch_result.batch_size,
            processing_time_ms: batch_result.processing_time_ms,
            gas_used: transaction_details.gas_used,
            gas_saved: transaction_details.gas_saved,
            cost_a0gi: transaction_details.cost_a0gi,
            tx_hash: transaction_details.tx_hash,
            confirmation_time_ms: transaction_details.confirmation_time_ms,
            error: batch_result.error || null
        };
        
        // Append to log file (in production, use proper logging)
        console.log(`📋 TX LOG: ${JSON.stringify(log_entry)}`);
    }
    
    saveReport(report_type, report_data) {
        const timestamp = new Date().toISOString().split('T')[0];
        const filename = `${report_type}_report_${timestamp}.json`;
        
        try {
            const fs = require('fs');
            const path = '/Users/elliejenkins/Desktop/su-firewall/monitoring_reports/' + filename;
            fs.writeFileSync(path, JSON.stringify(report_data, null, 2));
            console.log(`💾 ${report_type} report saved: ${filename}`);
        } catch (error) {
            console.error(`❌ Failed to save ${report_type} report:`, error.message);
        }
    }
    
    stopMonitoring() {
        this.dashboard_active = false;
        
        Object.values(this.monitoring_intervals).forEach(interval => {
            clearInterval(interval);
        });
        
        console.log("📊 Live monitoring stopped");
        
        // Generate final report
        const final_report = this.generateDetailedReport();
        console.log("✅ Final monitoring report generated");
        
        return final_report;
    }
}

module.exports = ProductionMonitoringDashboard;