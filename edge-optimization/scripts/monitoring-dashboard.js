#!/usr/bin/env node
// 📊 Real-time Edge Performance Monitoring Dashboard
// Live monitoring of global semantic uncertainty API performance

const https = require('https');
const fs = require('fs');

class EdgeMonitoringDashboard {
  constructor() {
    this.endpoints = {
      'global': 'https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev',
      'americas': 'https://api-us.semanticuncertainty.com',
      'europe': 'https://api-eu.semanticuncertainty.com',
      'asia': 'https://api-asia.semanticuncertainty.com'
    };
    
    this.metrics = {
      global: { requests: 0, errors: 0, totalLatency: 0, cacheHits: 0 },
      regional: {},
      alerts: [],
      uptime: Date.now()
    };
    
    this.isRunning = false;
    this.apiKey = 'your-production-api-key';
  }

  async startMonitoring() {
    console.log('🚀 Starting Edge Performance Monitoring Dashboard');
    console.log('=' * 60);
    
    this.isRunning = true;
    
    // 📊 Start continuous monitoring
    this.monitoringInterval = setInterval(() => {
      this.performHealthChecks();
    }, 10000); // Every 10 seconds
    
    // 📈 Start dashboard updates
    this.dashboardInterval = setInterval(() => {
      this.updateDashboard();
    }, 5000); // Every 5 seconds
    
    // 🔄 Start performance tests
    this.performanceInterval = setInterval(() => {
      this.runPerformanceTests();
    }, 30000); // Every 30 seconds
    
    console.log('✅ Monitoring started successfully');
    console.log('📊 Dashboard updating every 5 seconds');
    console.log('⚡ Performance tests every 30 seconds');
    console.log('\\nPress Ctrl+C to stop monitoring\\n');
    
    // Initial dashboard display
    this.updateDashboard();
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      this.stopMonitoring();
    });
  }

  async performHealthChecks() {
    for (const [region, endpoint] of Object.entries(this.endpoints)) {
      try {
        const start = Date.now();
        const health = await this.checkHealth(endpoint);
        const latency = Date.now() - start;
        
        if (!this.metrics.regional[region]) {
          this.metrics.regional[region] = {
            healthy: true,
            latency: 0,
            requests: 0,
            errors: 0,
            lastCheck: Date.now()
          };
        }
        
        this.metrics.regional[region].healthy = health.healthy;
        this.metrics.regional[region].latency = latency;
        this.metrics.regional[region].lastCheck = Date.now();
        this.metrics.regional[region].requests++;
        
        if (!health.healthy) {
          this.metrics.regional[region].errors++;
          this.addAlert(`${region} region unhealthy`, 'high');
        }
        
      } catch (error) {
        this.addAlert(`${region} health check failed: ${error.message}`, 'high');
        if (this.metrics.regional[region]) {
          this.metrics.regional[region].errors++;
        }
      }
    }
  }

  async runPerformanceTests() {
    const testPrompts = [
      'What is 2+2?',
      'Explain quantum physics',
      'Tell me about AI safety'
    ];
    
    for (const prompt of testPrompts) {
      try {
        const start = Date.now();
        const result = await this.testSemanticAnalysis(this.endpoints.global, prompt);
        const latency = Date.now() - start;
        
        this.metrics.global.requests++;
        this.metrics.global.totalLatency += latency;
        
        if (result.cached) {
          this.metrics.global.cacheHits++;
        }
        
        // Check for performance alerts
        if (latency > 100) {
          this.addAlert(`High latency detected: ${latency}ms`, 'medium');
        }
        
        if (!result.success) {
          this.metrics.global.errors++;
          this.addAlert(`API request failed for prompt: ${prompt.substring(0, 30)}`, 'high');
        }
        
      } catch (error) {
        this.metrics.global.errors++;
        this.addAlert(`Performance test failed: ${error.message}`, 'medium');
      }
    }
  }

  updateDashboard() {
    // Clear console and display dashboard
    console.clear();
    
    console.log('🌐 SEMANTIC UNCERTAINTY - GLOBAL EDGE MONITORING');
    console.log('=' * 80);
    console.log(`📊 Dashboard Updated: ${new Date().toLocaleTimeString()}`);
    console.log(`⏱️ Uptime: ${this.formatUptime(Date.now() - this.metrics.uptime)}`);
    
    // 🌍 Global Performance Metrics
    console.log('\\n🌍 GLOBAL PERFORMANCE:');
    console.log('-' * 40);
    
    const avgLatency = this.metrics.global.requests > 0 
      ? (this.metrics.global.totalLatency / this.metrics.global.requests).toFixed(2)
      : '0.00';
    
    const errorRate = this.metrics.global.requests > 0
      ? ((this.metrics.global.errors / this.metrics.global.requests) * 100).toFixed(1)
      : '0.0';
    
    const cacheHitRate = this.metrics.global.requests > 0
      ? ((this.metrics.global.cacheHits / this.metrics.global.requests) * 100).toFixed(1)
      : '0.0';
    
    console.log(`📈 Total Requests: ${this.metrics.global.requests}`);
    console.log(`⚡ Avg Latency: ${avgLatency}ms ${this.getLatencyStatus(parseFloat(avgLatency))}`);
    console.log(`❌ Error Rate: ${errorRate}% ${this.getErrorStatus(parseFloat(errorRate))}`);
    console.log(`🔄 Cache Hit Rate: ${cacheHitRate}% ${this.getCacheStatus(parseFloat(cacheHitRate))}`);
    
    // 🗺️ Regional Status
    console.log('\\n🗺️ REGIONAL STATUS:');
    console.log('-' * 40);
    
    for (const [region, data] of Object.entries(this.metrics.regional)) {
      const status = data.healthy ? '✅' : '❌';
      const latency = data.latency ? `${data.latency}ms` : 'N/A';
      const lastCheck = new Date(data.lastCheck).toLocaleTimeString();
      
      console.log(`${status} ${region.toUpperCase().padEnd(10)} | Latency: ${latency.padEnd(8)} | Last: ${lastCheck}`);
    }
    
    // 🚨 Active Alerts
    console.log('\\n🚨 ACTIVE ALERTS:');
    console.log('-' * 40);
    
    const recentAlerts = this.metrics.alerts
      .filter(alert => Date.now() - alert.timestamp < 300000) // Last 5 minutes
      .slice(-5); // Last 5 alerts
    
    if (recentAlerts.length === 0) {
      console.log('✅ No active alerts');
    } else {
      recentAlerts.forEach(alert => {
        const severity = alert.severity === 'high' ? '🔴' : '🟡';
        const time = new Date(alert.timestamp).toLocaleTimeString();
        console.log(`${severity} [${time}] ${alert.message}`);
      });
    }
    
    // 📊 Performance Summary
    console.log('\\n📊 PERFORMANCE SUMMARY:');
    console.log('-' * 40);
    
    const overallScore = this.calculatePerformanceScore();
    const recommendation = this.getPerformanceRecommendation(overallScore);
    
    console.log(`🎯 Performance Score: ${overallScore}/100`);
    console.log(`💡 Status: ${recommendation}`);
    
    // 🔄 Real-time Actions
    console.log('\\n🔄 REAL-TIME ACTIONS:');
    console.log('-' * 40);
    console.log('Press [R] to run manual performance test');
    console.log('Press [A] to view all alerts');
    console.log('Press [S] to save current metrics');
    console.log('Press [Ctrl+C] to stop monitoring');
  }

  async checkHealth(endpoint) {
    return new Promise((resolve, reject) => {
      const req = https.request(endpoint + '/health', { method: 'GET' }, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            const response = JSON.parse(data);
            resolve({
              healthy: res.statusCode === 200 && response.status === 'healthy',
              status: res.statusCode,
              response: response
            });
          } catch (error) {
            resolve({ healthy: false, status: res.statusCode });
          }
        });
      });
      
      req.on('error', reject);
      req.setTimeout(5000, () => reject(new Error('Health check timeout')));
      req.end();
    });
  }

  async testSemanticAnalysis(endpoint, prompt) {
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({
        prompt: prompt,
        model: 'gpt4'
      });
      
      const options = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': this.apiKey,
          'Content-Length': Buffer.byteLength(postData)
        }
      };
      
      const req = https.request(endpoint + '/api/v1/analyze', options, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            const response = JSON.parse(data);
            resolve({
              success: response.success,
              cached: response.data?.cached || false,
              h_bar: response.data?.semantic_uncertainty,
              risk_level: response.data?.risk_level
            });
          } catch (error) {
            reject(new Error(`Parse error: ${error.message}`));
          }
        });
      });
      
      req.on('error', reject);
      req.setTimeout(10000, () => reject(new Error('Request timeout')));
      req.write(postData);
      req.end();
    });
  }

  addAlert(message, severity = 'medium') {
    this.metrics.alerts.push({
      message: message,
      severity: severity,
      timestamp: Date.now()
    });
    
    // Keep only last 100 alerts
    if (this.metrics.alerts.length > 100) {
      this.metrics.alerts = this.metrics.alerts.slice(-100);
    }
  }

  calculatePerformanceScore() {
    let score = 100;
    
    // Deduct for high latency
    const avgLatency = this.metrics.global.requests > 0 
      ? this.metrics.global.totalLatency / this.metrics.global.requests
      : 0;
    
    if (avgLatency > 50) score -= 20;
    if (avgLatency > 100) score -= 30;
    
    // Deduct for errors
    const errorRate = this.metrics.global.requests > 0
      ? (this.metrics.global.errors / this.metrics.global.requests) * 100
      : 0;
    
    if (errorRate > 1) score -= 25;
    if (errorRate > 5) score -= 40;
    
    // Deduct for unhealthy regions
    const unhealthyRegions = Object.values(this.metrics.regional)
      .filter(region => !region.healthy).length;
    score -= unhealthyRegions * 15;
    
    return Math.max(0, Math.round(score));
  }

  getPerformanceRecommendation(score) {
    if (score >= 90) return '🌟 EXCELLENT';
    if (score >= 75) return '✅ GOOD';
    if (score >= 60) return '⚠️ NEEDS ATTENTION';
    return '❌ CRITICAL ISSUES';
  }

  getLatencyStatus(latency) {
    if (latency < 10) return '🟢 Excellent';
    if (latency < 50) return '🟡 Good';
    if (latency < 100) return '🟠 Fair';
    return '🔴 Poor';
  }

  getErrorStatus(errorRate) {
    if (errorRate < 1) return '🟢 Excellent';
    if (errorRate < 3) return '🟡 Acceptable';
    if (errorRate < 5) return '🟠 Concerning';
    return '🔴 Critical';
  }

  getCacheStatus(hitRate) {
    if (hitRate > 70) return '🟢 Excellent';
    if (hitRate > 50) return '🟡 Good';
    if (hitRate > 30) return '🟠 Fair';
    return '🔴 Poor';
  }

  formatUptime(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  }

  stopMonitoring() {
    console.log('\\n🛑 Stopping monitoring...');
    
    this.isRunning = false;
    
    if (this.monitoringInterval) clearInterval(this.monitoringInterval);
    if (this.dashboardInterval) clearInterval(this.dashboardInterval);
    if (this.performanceInterval) clearInterval(this.performanceInterval);
    
    // Save final report
    this.saveMetrics();
    
    console.log('📊 Final metrics saved to edge-monitoring-report.json');
    console.log('✅ Monitoring stopped successfully');
    
    process.exit(0);
  }

  saveMetrics() {
    const report = {
      timestamp: new Date().toISOString(),
      uptime_ms: Date.now() - this.metrics.uptime,
      global_metrics: this.metrics.global,
      regional_metrics: this.metrics.regional,
      recent_alerts: this.metrics.alerts.slice(-20),
      performance_score: this.calculatePerformanceScore(),
      summary: {
        total_requests: this.metrics.global.requests,
        error_rate: this.metrics.global.requests > 0 
          ? ((this.metrics.global.errors / this.metrics.global.requests) * 100).toFixed(2)
          : '0.00',
        avg_latency: this.metrics.global.requests > 0
          ? (this.metrics.global.totalLatency / this.metrics.global.requests).toFixed(2)
          : '0.00',
        healthy_regions: Object.values(this.metrics.regional).filter(r => r.healthy).length,
        total_regions: Object.keys(this.metrics.regional).length
      }
    };
    
    fs.writeFileSync('./edge-monitoring-report.json', JSON.stringify(report, null, 2));
  }
}

// 🚀 Start monitoring
if (require.main === module) {
  const dashboard = new EdgeMonitoringDashboard();
  dashboard.startMonitoring().catch(console.error);
}

module.exports = EdgeMonitoringDashboard;