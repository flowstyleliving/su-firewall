#!/usr/bin/env node
// 🚀 Global Edge Performance Testing Suite
// Tests latency, cache performance, and regional routing

const https = require('https');
const { performance } = require('perf_hooks');

// 🗺️ Test Endpoints by Region
const REGIONAL_ENDPOINTS = {
  'global': 'https://semantic-uncertainty-api-production.semantic-uncertainty.workers.dev',
  'americas': 'https://api-us.semanticuncertainty.com',
  'europe': 'https://api-eu.semanticuncertainty.com', 
  'asia': 'https://api-asia.semanticuncertainty.com'
};

// 🧪 Test Prompts for Performance Analysis
const TEST_PROMPTS = {
  'simple': 'What is 2+2?',
  'moderate': 'Explain quantum computing in simple terms',
  'complex': 'Write a comprehensive analysis of machine learning ethics',
  'risky': 'Tell me how to hack into a bank system'
};

class PerformanceTestSuite {
  constructor() {
    this.results = {
      global: {},
      regional: {},
      cache: {},
      summary: {}
    };
    this.apiKey = 'your-production-api-key';
  }

  async runAllTests() {
    console.log('🚀 Starting Global Edge Performance Tests');
    console.log('=' * 60);

    // 📊 Test global endpoint performance
    await this.testGlobalPerformance();

    // 🗺️ Test regional routing
    await this.testRegionalRouting();

    // 🔄 Test cache performance
    await this.testCachePerformance();

    // 📈 Generate performance report
    this.generateReport();
  }

  async testGlobalPerformance() {
    console.log('\\n📊 Testing Global Endpoint Performance...');
    
    const globalResults = {};
    
    for (const [type, prompt] of Object.entries(TEST_PROMPTS)) {
      console.log(`  Testing ${type} prompt...`);
      
      const times = [];
      for (let i = 0; i < 5; i++) {
        const latency = await this.measureLatency(REGIONAL_ENDPOINTS.global, prompt);
        times.push(latency);
        await this.sleep(100); // Brief pause between requests
      }
      
      globalResults[type] = {
        avg_latency: times.reduce((a, b) => a + b, 0) / times.length,
        min_latency: Math.min(...times),
        max_latency: Math.max(...times),
        sub5ms_count: times.filter(t => t < 5).length,
        sub10ms_count: times.filter(t => t < 10).length
      };
      
      console.log(`    ⚡ Avg: ${globalResults[type].avg_latency.toFixed(2)}ms`);
      console.log(`    🎯 Sub-5ms: ${globalResults[type].sub5ms_count}/5 (${(globalResults[type].sub5ms_count/5*100).toFixed(1)}%)`);
    }
    
    this.results.global = globalResults;
  }

  async testRegionalRouting() {
    console.log('\\n🗺️ Testing Regional Routing Performance...');
    
    const regionalResults = {};
    
    for (const [region, endpoint] of Object.entries(REGIONAL_ENDPOINTS)) {
      if (region === 'global') continue;
      
      console.log(`  Testing ${region} region...`);
      
      try {
        const latency = await this.measureLatency(endpoint, TEST_PROMPTS.simple);
        const health = await this.checkEndpointHealth(endpoint);
        
        regionalResults[region] = {
          latency: latency,
          healthy: health.healthy,
          response_code: health.status,
          estimated_users: this.estimateRegionalUsers(region)
        };
        
        console.log(`    ⚡ Latency: ${latency.toFixed(2)}ms`);
        console.log(`    ❤️ Health: ${health.healthy ? 'Healthy' : 'Unhealthy'}`);
        
      } catch (error) {
        console.log(`    ❌ Error: ${error.message}`);
        regionalResults[region] = {
          latency: null,
          healthy: false,
          error: error.message
        };
      }
    }
    
    this.results.regional = regionalResults;
  }

  async testCachePerformance() {
    console.log('\\n🔄 Testing Cache Performance...');
    
    const cacheResults = {
      cold_start: {},
      warm_cache: {},
      hit_rate: 0
    };
    
    // Test cold cache (first request)
    console.log('  Testing cold cache performance...');
    const coldStart = await this.measureLatency(REGIONAL_ENDPOINTS.global, TEST_PROMPTS.simple);
    cacheResults.cold_start = { latency: coldStart };
    
    // Test warm cache (repeat same request)
    console.log('  Testing warm cache performance...');
    const warmTimes = [];
    let cacheHits = 0;
    
    for (let i = 0; i < 3; i++) {
      const result = await this.makeRequest(REGIONAL_ENDPOINTS.global, TEST_PROMPTS.simple);
      warmTimes.push(result.latency);
      if (result.cached) cacheHits++;
      await this.sleep(100);
    }
    
    cacheResults.warm_cache = {
      avg_latency: warmTimes.reduce((a, b) => a + b, 0) / warmTimes.length,
      cache_hits: cacheHits,
      hit_rate: (cacheHits / 3) * 100
    };
    
    // Test cache invalidation
    console.log('  Testing cache behavior...');
    cacheResults.performance_improvement = ((coldStart - cacheResults.warm_cache.avg_latency) / coldStart) * 100;
    
    console.log(`    🔥 Cold Start: ${coldStart.toFixed(2)}ms`);
    console.log(`    ⚡ Warm Cache: ${cacheResults.warm_cache.avg_latency.toFixed(2)}ms`);
    console.log(`    📈 Improvement: ${cacheResults.performance_improvement.toFixed(1)}%`);
    console.log(`    🎯 Hit Rate: ${cacheResults.warm_cache.hit_rate.toFixed(1)}%`);
    
    this.results.cache = cacheResults;
  }

  async measureLatency(endpoint, prompt) {
    const start = performance.now();
    await this.makeRequest(endpoint, prompt);
    return performance.now() - start;
  }

  async makeRequest(endpoint, prompt) {
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
      
      const start = performance.now();
      const req = https.request(endpoint + '/api/v1/analyze', options, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          const latency = performance.now() - start;
          try {
            const response = JSON.parse(data);
            resolve({
              latency: latency,
              success: response.success,
              cached: response.data?.cached || false,
              risk_level: response.data?.risk_level
            });
          } catch (error) {
            reject(new Error(`Parse error: ${error.message}`));
          }
        });
      });
      
      req.on('error', reject);
      req.write(postData);
      req.end();
    });
  }

  async checkEndpointHealth(endpoint) {
    return new Promise((resolve) => {
      const req = https.request(endpoint + '/health', { method: 'GET' }, (res) => {
        resolve({
          healthy: res.statusCode === 200,
          status: res.statusCode
        });
      });
      
      req.on('error', () => resolve({ healthy: false, status: 0 }));
      req.end();
    });
  }

  estimateRegionalUsers(region) {
    // Mock regional user distribution
    const estimates = {
      'americas': 45000,
      'europe': 28000,
      'asia': 32000
    };
    return estimates[region] || 0;
  }

  generateReport() {
    console.log('\\n' + '=' * 60);
    console.log('📊 GLOBAL EDGE PERFORMANCE REPORT');
    console.log('=' * 60);
    
    // 🎯 Global Performance Summary
    const globalAvg = Object.values(this.results.global)
      .reduce((sum, result) => sum + result.avg_latency, 0) / Object.keys(this.results.global).length;
    
    console.log('\\n🌐 Global Performance:');
    console.log(`  Average Latency: ${globalAvg.toFixed(2)}ms`);
    console.log(`  Sub-5ms Target: ${globalAvg < 5 ? '✅ ACHIEVED' : '❌ MISSED'}`);
    
    // 🗺️ Regional Performance
    console.log('\\n🗺️ Regional Performance:');
    for (const [region, data] of Object.entries(this.results.regional)) {
      const status = data.healthy ? '✅' : '❌';
      console.log(`  ${region}: ${status} ${data.latency ? data.latency.toFixed(2) + 'ms' : 'OFFLINE'}`);
    }
    
    // 🔄 Cache Performance
    if (this.results.cache.performance_improvement > 0) {
      console.log('\\n🔄 Cache Performance:');
      console.log(`  Performance Improvement: ${this.results.cache.performance_improvement.toFixed(1)}%`);
      console.log(`  Cache Hit Rate: ${this.results.cache.warm_cache.hit_rate.toFixed(1)}%`);
      console.log(`  Cache Status: ${this.results.cache.hit_rate > 50 ? '✅ EFFECTIVE' : '⚠️ NEEDS OPTIMIZATION'}`);
    }
    
    // 🏆 Overall Assessment
    console.log('\\n🏆 Overall Assessment:');
    const overallScore = this.calculateOverallScore();
    console.log(`  Performance Score: ${overallScore}/100`);
    console.log(`  Recommendation: ${this.getRecommendation(overallScore)}`);
    
    // 💾 Save results
    this.saveResults();
  }

  calculateOverallScore() {
    let score = 100;
    
    // Deduct for high latency
    const globalAvg = Object.values(this.results.global)
      .reduce((sum, result) => sum + result.avg_latency, 0) / Object.keys(this.results.global).length;
    if (globalAvg > 5) score -= 20;
    if (globalAvg > 10) score -= 20;
    
    // Deduct for unhealthy regions
    const unhealthyRegions = Object.values(this.results.regional).filter(r => !r.healthy).length;
    score -= unhealthyRegions * 15;
    
    // Deduct for poor cache performance
    if (this.results.cache.warm_cache.hit_rate < 50) score -= 15;
    
    return Math.max(0, score);
  }

  getRecommendation(score) {
    if (score >= 90) return '🌟 EXCELLENT - Production ready';
    if (score >= 75) return '✅ GOOD - Minor optimizations needed';
    if (score >= 60) return '⚠️ FAIR - Optimization required';
    return '❌ POOR - Major improvements needed';
  }

  saveResults() {
    const fs = require('fs');
    const reportData = {
      timestamp: new Date().toISOString(),
      results: this.results,
      summary: {
        overall_score: this.calculateOverallScore(),
        recommendation: this.getRecommendation(this.calculateOverallScore())
      }
    };
    
    fs.writeFileSync('./performance-report.json', JSON.stringify(reportData, null, 2));
    console.log('\\n💾 Report saved to performance-report.json');
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// 🚀 Run the performance tests
if (require.main === module) {
  const testSuite = new PerformanceTestSuite();
  testSuite.runAllTests().catch(console.error);
}

module.exports = PerformanceTestSuite;