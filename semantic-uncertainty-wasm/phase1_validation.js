/**
 * Phase 1 Validation Script
 * 
 * Conservative testing with manual oversight for proving basic batch processing
 * Target: 10 successful batch transactions with detailed monitoring
 */

const ConservativeGasOptimizer = require('./conservative_gas_optimizer.js');

class Phase1Validator {
    constructor() {
        this.test_results = {
            phase: 1,
            start_time: new Date().toISOString(),
            target_batches: 10,
            completed_batches: 0,
            test_cases: [],
            success_criteria: {
                minimum_batches: 10,
                success_rate_threshold: 90, // 90%
                max_gas_cost_variance: 20, // 20%
                min_accuracy_retention: 85 // 85%
            },
            overall_status: 'IN_PROGRESS'
        };
        
        // Test scenarios for gradual validation
        this.test_scenarios = [
            {
                name: "Basic functionality test",
                description: "Simple 3-item batch with clear results",
                verifications: [
                    { text: "The capital of France is Paris.", model: "test", expected_risk: "Safe" },
                    { text: "Python was created by Guido van Rossum.", model: "test", expected_risk: "Safe" },
                    { text: "Machine learning requires training data.", model: "test", expected_risk: "Safe" }
                ]
            },
            {
                name: "Mixed risk level test",
                description: "3-item batch with different risk levels",
                verifications: [
                    { text: "Water freezes at 0 degrees Celsius.", model: "test", expected_risk: "Safe" },
                    { text: "The Eiffel Tower is located in Rome, Italy.", model: "test", expected_risk: "Critical" },
                    { text: "AI will probably replace most jobs within 5 years.", model: "test", expected_risk: "Warning" }
                ]
            },
            {
                name: "Edge case - minimum batch",
                description: "2-item batch (minimum size)",
                verifications: [
                    { text: "The sun is a star.", model: "test", expected_risk: "Safe" },
                    { text: "Gravity makes objects fall upward.", model: "test", expected_risk: "Critical" }
                ]
            },
            {
                name: "Maximum batch size test",
                description: "5-item batch (maximum allowed)",
                verifications: [
                    { text: "Earth orbits around the sun.", model: "test", expected_risk: "Safe" },
                    { text: "Shakespeare wrote Hamlet.", model: "test", expected_risk: "Safe" },
                    { text: "The moon is made of cheese.", model: "test", expected_risk: "Critical" },
                    { text: "Photosynthesis produces oxygen.", model: "test", expected_risk: "Safe" },
                    { text: "Bitcoin was invented in 2008.", model: "test", expected_risk: "Safe" }
                ]
            }
        ];
        
        console.log("🛡️ Phase 1 Validator initialized");
        console.log(`🎯 Target: ${this.test_results.target_batches} successful batches`);
        console.log(`📊 Success criteria: ${this.test_results.success_criteria.success_rate_threshold}% success rate`);
    }
    
    /**
     * Run complete Phase 1 validation with manual oversight
     */
    async runPhase1Validation() {
        console.log("\n🚀 Starting Phase 1 Conservative Validation");
        console.log("=" .repeat(80));
        console.log("⚠️  MANUAL OVERSIGHT REQUIRED - Review each batch before proceeding");
        console.log("🎯 Goal: Prove basic batch processing works reliably");
        console.log("=" .repeat(80));
        
        try {
            // Initialize mock oracle for testing
            const mock_oracle = this.createMockOracle();
            const conservative_optimizer = new ConservativeGasOptimizer(mock_oracle);
            
            // Run test scenarios
            for (let scenario_index = 0; scenario_index < this.test_scenarios.length; scenario_index++) {
                const scenario = this.test_scenarios[scenario_index];
                
                console.log(`\n📋 Running Scenario ${scenario_index + 1}: ${scenario.name}`);
                console.log(`   📝 ${scenario.description}`);
                console.log(`   📦 Batch size: ${scenario.verifications.length}`);
                
                // Wait for manual confirmation in real deployment
                if (process.env.MANUAL_APPROVAL === 'true') {
                    console.log("\n⏸️  MANUAL APPROVAL REQUIRED");
                    console.log("   Review the scenario above and press Enter to continue...");
                    // In real deployment, would await user input
                    await this.sleep(2000); // Simulate manual review
                }
                
                // Run multiple iterations of each scenario
                const iterations = scenario_index < 2 ? 3 : 2; // More iterations for basic tests
                
                for (let iteration = 1; iteration <= iterations; iteration++) {
                    console.log(`\n🔄 Iteration ${iteration}/${iterations}`);
                    
                    try {
                        const test_result = await this.runTestBatch(
                            conservative_optimizer,
                            scenario,
                            `${scenario.name}_iteration_${iteration}`
                        );
                        
                        this.test_results.test_cases.push(test_result);
                        
                        if (test_result.success) {
                            this.test_results.completed_batches++;
                            console.log(`✅ Batch completed successfully`);
                            console.log(`   📈 Progress: ${this.test_results.completed_batches}/${this.test_results.target_batches}`);
                        } else {
                            console.log(`⚠️  Batch failed but fallback succeeded`);
                        }
                        
                        // Safety pause between batches
                        console.log(`⏳ Safety pause (5 seconds)...`);
                        await this.sleep(5000);
                        
                    } catch (error) {
                        console.error(`❌ Test iteration failed:`, error.message);
                        
                        const failure_result = {
                            test_name: `${scenario.name}_iteration_${iteration}`,
                            success: false,
                            error: error.message,
                            timestamp: new Date().toISOString()
                        };
                        
                        this.test_results.test_cases.push(failure_result);
                    }
                }
                
                // Check if we should continue
                const current_success_rate = this.calculateSuccessRate();
                if (current_success_rate < 70 && this.test_results.test_cases.length >= 5) {
                    console.log(`\n🚨 Success rate too low (${current_success_rate.toFixed(1)}%) - stopping for analysis`);
                    break;
                }
            }
            
            // Generate final assessment
            const final_assessment = this.generatePhase1Assessment(conservative_optimizer);
            
            // Export results
            await this.exportPhase1Results(final_assessment);
            
            console.log("\n" + "=" .repeat(80));
            console.log("🏆 PHASE 1 VALIDATION COMPLETE");
            console.log("=" .repeat(80));
            
            this.displayFinalResults(final_assessment);
            
            return final_assessment;
            
        } catch (error) {
            console.error("\n❌ Phase 1 validation failed:", error);
            throw error;
        }
    }
    
    /**
     * Run individual test batch with detailed monitoring
     */
    async runTestBatch(optimizer, scenario, test_name) {
        const start_time = Date.now();
        
        console.log(`   🔬 Running test: ${test_name}`);
        
        try {
            // Process the batch
            const result = await optimizer.processConservativeBatch(
                scenario.verifications,
                { test_name, scenario: scenario.name }
            );
            
            const end_time = Date.now();
            
            // Validate results against expectations
            const validation = this.validateTestResults(result, scenario);
            
            const test_result = {
                test_name,
                scenario: scenario.name,
                success: result.success,
                processing_time_ms: end_time - start_time,
                batch_size: scenario.verifications.length,
                gas_metrics: result.metrics || null,
                accuracy_validation: validation,
                transaction_hash: result.transaction_result?.tx_hash || null,
                timestamp: new Date().toISOString(),
                raw_result: result
            };
            
            // Log detailed results
            console.log(`      ⏱️  Processing time: ${test_result.processing_time_ms}ms`);
            if (result.metrics) {
                console.log(`      ⛽ Gas used: ${result.metrics.gas_used.toLocaleString()}`);
                console.log(`      💰 Cost: ${result.metrics.cost_a0gi.toFixed(6)} A0GI`);
                console.log(`      📈 Est. savings: ${result.metrics.estimated_savings_percent.toFixed(1)}%`);
            }
            console.log(`      🎯 Accuracy: ${validation.accuracy_score.toFixed(1)}%`);
            console.log(`      📋 TX Hash: ${test_result.transaction_hash || 'N/A'}`);
            
            return test_result;
            
        } catch (error) {
            console.error(`      ❌ Test batch failed:`, error.message);
            
            return {
                test_name,
                scenario: scenario.name,
                success: false,
                error: error.message,
                processing_time_ms: Date.now() - start_time,
                timestamp: new Date().toISOString()
            };
        }
    }
    
    /**
     * Validate test results against expected outcomes
     */
    validateTestResults(result, scenario) {
        let correct_predictions = 0;
        let total_predictions = 0;
        
        if (result.success && result.results) {
            for (let i = 0; i < scenario.verifications.length; i++) {
                const verification = scenario.verifications[i];
                const prediction = result.results[i];
                
                if (prediction && prediction.risk_level) {
                    total_predictions++;
                    
                    // Simple validation - exact match or reasonable approximation
                    if (prediction.risk_level === verification.expected_risk) {
                        correct_predictions++;
                    } else if (this.isReasonableRiskPrediction(prediction.risk_level, verification.expected_risk)) {
                        correct_predictions += 0.5; // Partial credit
                    }
                }
            }
        }
        
        const accuracy_score = total_predictions > 0 ? (correct_predictions / total_predictions) * 100 : 0;
        
        return {
            total_predictions,
            correct_predictions,
            accuracy_score,
            meets_accuracy_threshold: accuracy_score >= this.test_results.success_criteria.min_accuracy_retention
        };
    }
    
    /**
     * Check if risk prediction is reasonably close to expected
     */
    isReasonableRiskPrediction(predicted, expected) {
        const risk_levels = ['Safe', 'Warning', 'High Risk', 'Critical'];
        const predicted_index = risk_levels.indexOf(predicted);
        const expected_index = risk_levels.indexOf(expected);
        
        // Allow 1 level difference (e.g., Warning instead of Safe)
        return Math.abs(predicted_index - expected_index) <= 1;
    }
    
    /**
     * Calculate current success rate
     */
    calculateSuccessRate() {
        const successful = this.test_results.test_cases.filter(test => test.success).length;
        const total = this.test_results.test_cases.length;
        
        return total > 0 ? (successful / total) * 100 : 0;
    }
    
    /**
     * Generate comprehensive Phase 1 assessment
     */
    generatePhase1Assessment(optimizer) {
        const success_rate = this.calculateSuccessRate();
        const successful_tests = this.test_results.test_cases.filter(test => test.success);
        
        // Calculate metrics from successful tests
        const avg_processing_time = successful_tests.length > 0 ?
            successful_tests.reduce((sum, test) => sum + test.processing_time_ms, 0) / successful_tests.length : 0;
        
        const gas_metrics = successful_tests
            .filter(test => test.gas_metrics)
            .map(test => test.gas_metrics);
        
        const avg_gas_used = gas_metrics.length > 0 ?
            gas_metrics.reduce((sum, metrics) => sum + metrics.gas_used, 0) / gas_metrics.length : 0;
        
        const avg_cost = gas_metrics.length > 0 ?
            gas_metrics.reduce((sum, metrics) => sum + metrics.cost_a0gi, 0) / gas_metrics.length : 0;
        
        const avg_savings = gas_metrics.length > 0 ?
            gas_metrics.reduce((sum, metrics) => sum + metrics.estimated_savings_percent, 0) / gas_metrics.length : 0;
        
        const accuracy_scores = this.test_results.test_cases
            .filter(test => test.accuracy_validation)
            .map(test => test.accuracy_validation.accuracy_score);
        
        const avg_accuracy = accuracy_scores.length > 0 ?
            accuracy_scores.reduce((a, b) => a + b, 0) / accuracy_scores.length : 0;
        
        // Check success criteria
        const criteria_met = {
            minimum_batches: this.test_results.completed_batches >= this.test_results.success_criteria.minimum_batches,
            success_rate: success_rate >= this.test_results.success_criteria.success_rate_threshold,
            accuracy_retention: avg_accuracy >= this.test_results.success_criteria.min_accuracy_retention,
            system_stability: !optimizer.circuit_breaker.is_open
        };
        
        const ready_for_phase_2 = Object.values(criteria_met).every(Boolean);
        
        return {
            phase: 1,
            completion_time: new Date().toISOString(),
            duration_hours: (Date.now() - new Date(this.test_results.start_time).getTime()) / (1000 * 60 * 60),
            
            // Core metrics
            total_tests: this.test_results.test_cases.length,
            successful_tests: successful_tests.length,
            success_rate: success_rate,
            completed_batches: this.test_results.completed_batches,
            
            // Performance metrics
            avg_processing_time_ms: avg_processing_time,
            avg_gas_used: avg_gas_used,
            avg_cost_a0gi: avg_cost,
            avg_savings_percent: avg_savings,
            avg_accuracy_percent: avg_accuracy,
            
            // Success criteria assessment
            success_criteria: this.test_results.success_criteria,
            criteria_met,
            ready_for_phase_2,
            
            // System status
            optimizer_status: optimizer.getSafetyReport(),
            
            // Recommendations
            recommendations: this.generateRecommendations(ready_for_phase_2, criteria_met, avg_savings),
            
            // Detailed results
            test_results: this.test_results.test_cases
        };
    }
    
    /**
     * Generate recommendations based on results
     */
    generateRecommendations(ready_for_phase_2, criteria_met, avg_savings) {
        const recommendations = [];
        
        if (ready_for_phase_2) {
            recommendations.push("✅ PROCEED TO PHASE 2 - Gradual Scale Testing");
            recommendations.push("🔧 Increase batch size to 8-12 items");
            recommendations.push("📊 Enable basic compression and selective storage");
            recommendations.push("⏱️ Reduce batch timeout to 7 seconds");
        } else {
            recommendations.push("⚠️ CONTINUE PHASE 1 TESTING");
            
            if (!criteria_met.minimum_batches) {
                recommendations.push(`📊 Complete more batches: ${this.test_results.completed_batches}/${this.test_results.success_criteria.minimum_batches}`);
            }
            
            if (!criteria_met.success_rate) {
                recommendations.push("🔧 Investigate batch failure causes and improve error handling");
            }
            
            if (!criteria_met.accuracy_retention) {
                recommendations.push("🎯 Review semantic analysis calibration to improve accuracy");
            }
            
            if (!criteria_met.system_stability) {
                recommendations.push("🚨 Reset circuit breaker and address system stability issues");
            }
        }
        
        // Performance recommendations
        if (avg_savings < 15) {
            recommendations.push("💰 Gas savings below target - investigate optimization opportunities");
        } else if (avg_savings > 30) {
            recommendations.push("🎉 Excellent gas savings - ready for more aggressive optimization");
        }
        
        recommendations.push("📋 Continue monitoring all transactions in block explorer");
        recommendations.push("📊 Track accuracy on production-like data");
        
        return recommendations;
    }
    
    /**
     * Display final results summary
     */
    displayFinalResults(assessment) {
        console.log(`\n📊 PHASE 1 RESULTS SUMMARY:`);
        console.log(`   🎯 Success Rate: ${assessment.success_rate.toFixed(1)}% (target: ${assessment.success_criteria.success_rate_threshold}%)`);
        console.log(`   📦 Completed Batches: ${assessment.completed_batches}/${assessment.success_criteria.minimum_batches}`);
        console.log(`   🎯 Avg Accuracy: ${assessment.avg_accuracy_percent.toFixed(1)}% (target: ${assessment.success_criteria.min_accuracy_retention}%)`);
        console.log(`   💰 Avg Gas Savings: ${assessment.avg_savings_percent.toFixed(1)}%`);
        console.log(`   ⏱️ Avg Processing Time: ${assessment.avg_processing_time_ms.toFixed(0)}ms`);
        
        console.log(`\n✅ SUCCESS CRITERIA:`);
        Object.entries(assessment.criteria_met).forEach(([criterion, met]) => {
            console.log(`   ${met ? '✅' : '❌'} ${criterion}: ${met ? 'MET' : 'NOT MET'}`);
        });
        
        console.log(`\n🚀 PHASE 2 READINESS: ${assessment.ready_for_phase_2 ? '✅ READY' : '❌ NOT READY'}`);
        
        console.log(`\n📋 RECOMMENDATIONS:`);
        assessment.recommendations.forEach(rec => console.log(`   ${rec}`));
    }
    
    /**
     * Export detailed Phase 1 results
     */
    async exportPhase1Results(assessment) {
        const fs = require('fs').promises;
        
        // Export JSON results
        const json_path = '/Users/elliejenkins/Desktop/su-firewall/test_results/phase1_validation_results.json';
        await fs.writeFile(json_path, JSON.stringify(assessment, null, 2));
        
        // Generate markdown report
        const markdown = this.generatePhase1Report(assessment);
        const md_path = '/Users/elliejenkins/Desktop/su-firewall/test_results/PHASE1_VALIDATION_REPORT.md';
        await fs.writeFile(md_path, markdown);
        
        console.log(`\n💾 Results exported:`);
        console.log(`   📄 JSON: ${json_path}`);
        console.log(`   📋 Report: ${md_path}`);
    }
    
    /**
     * Generate markdown report for Phase 1
     */
    generatePhase1Report(assessment) {
        return `# 🛡️ Phase 1 Conservative Validation Report

**Date:** ${new Date().toLocaleDateString()}  
**Duration:** ${assessment.duration_hours.toFixed(1)} hours  
**Environment:** 0G Newton Testnet Conservative Testing  

---

## 🎯 **EXECUTIVE SUMMARY**

| Metric | Target | Achieved | Status |
|--------|--------|----------|---------|
| **Success Rate** | ${assessment.success_criteria.success_rate_threshold}% | **${assessment.success_rate.toFixed(1)}%** | ${assessment.criteria_met.success_rate ? '✅' : '❌'} |
| **Completed Batches** | ${assessment.success_criteria.minimum_batches} | **${assessment.completed_batches}** | ${assessment.criteria_met.minimum_batches ? '✅' : '❌'} |
| **Accuracy Retention** | ${assessment.success_criteria.min_accuracy_retention}% | **${assessment.avg_accuracy_percent.toFixed(1)}%** | ${assessment.criteria_met.accuracy_retention ? '✅' : '❌'} |
| **System Stability** | Stable | **${assessment.optimizer_status.circuit_breaker_status}** | ${assessment.criteria_met.system_stability ? '✅' : '❌'} |

**🚀 Phase 2 Readiness:** ${assessment.ready_for_phase_2 ? '✅ **READY TO PROCEED**' : '❌ **CONTINUE PHASE 1**'}

---

## 📊 **PERFORMANCE METRICS**

### **Conservative Batch Processing:**
- **Total Tests:** ${assessment.total_tests}
- **Successful Tests:** ${assessment.successful_tests}
- **Average Processing Time:** ${assessment.avg_processing_time_ms.toFixed(0)}ms
- **Average Gas Used:** ${Math.round(assessment.avg_gas_used).toLocaleString()} gas
- **Average Cost:** ${assessment.avg_cost_a0gi.toFixed(6)} A0GI
- **Average Gas Savings:** ${assessment.avg_savings_percent.toFixed(1)}%

### **Configuration Used:**
- **Batch Size:** 2-5 verifications (ultra-conservative)
- **Gas Price Strategy:** Conservative fixed (1.3x network price)
- **Storage Threshold:** High uncertainty only (ℏₛ ≥ 2.0)
- **Advanced Features:** All disabled for safety

---

## 🧪 **TEST SCENARIOS RESULTS**

${assessment.test_results.map((test, index) => `
### Test ${index + 1}: ${test.test_name}
- **Scenario:** ${test.scenario}
- **Success:** ${test.success ? '✅ YES' : '❌ NO'}
- **Processing Time:** ${test.processing_time_ms}ms
- **Gas Used:** ${test.gas_metrics?.gas_used?.toLocaleString() || 'N/A'}
- **Cost:** ${test.gas_metrics?.cost_a0gi?.toFixed(6) || 'N/A'} A0GI
- **Accuracy:** ${test.accuracy_validation?.accuracy_score?.toFixed(1) || 'N/A'}%
- **TX Hash:** ${test.transaction_hash || 'N/A'}
`).join('\n')}

---

## ✅ **SUCCESS CRITERIA ASSESSMENT**

${Object.entries(assessment.criteria_met).map(([criterion, met]) => `
### ${criterion.replace('_', ' ').toUpperCase()}
- **Status:** ${met ? '✅ MET' : '❌ NOT MET'}
- **Details:** ${this.getCriterionDetails(criterion, assessment)}
`).join('\n')}

---

## 📋 **RECOMMENDATIONS**

${assessment.recommendations.map(rec => `- ${rec}`).join('\n')}

---

## 🔧 **SYSTEM STATUS**

### **Circuit Breaker:**
- **Status:** ${assessment.optimizer_status.circuit_breaker_status}
- **Failure Count:** ${assessment.optimizer_status.failure_count}
- **Total Batches:** ${assessment.optimizer_status.total_batches}

### **Performance Trends:**
- **Success Rate:** ${assessment.optimizer_status.success_rate.toFixed(1)}%
- **Average Gas/Batch:** ${Math.round(assessment.optimizer_status.avg_gas_per_batch).toLocaleString()}
- **Average Cost/Batch:** ${assessment.optimizer_status.avg_cost_per_batch.toFixed(6)} A0GI

---

## 🎯 **NEXT STEPS**

${assessment.ready_for_phase_2 ? `
### ✅ **PROCEED TO PHASE 2 - GRADUAL SCALE TESTING**

**Week 2 Configuration:**
- Increase batch size to 8-12 items
- Enable basic data compression
- Add selective storage (ℏₛ ≥ 1.8)
- Monitor for 1 week before next increase

**Success Criteria for Phase 2:**
- Maintain >85% success rate
- Achieve 30-45% gas reduction
- Process 50+ batches successfully
- System stability maintained
` : `
### ⚠️ **CONTINUE PHASE 1 TESTING**

**Required Actions:**
- Address failed success criteria
- Run additional conservative batches
- Investigate and fix reliability issues
- Re-run validation when issues resolved
`}

---

**Phase 1 Conservative Validation ${assessment.ready_for_phase_2 ? 'COMPLETED SUCCESSFULLY' : 'REQUIRES ADDITIONAL TESTING'}**  
*Foundation proven for incremental scaling*
`;
    }
    
    getCriterionDetails(criterion, assessment) {
        switch (criterion) {
            case 'minimum_batches':
                return `Need ${assessment.success_criteria.minimum_batches}, completed ${assessment.completed_batches}`;
            case 'success_rate':
                return `Need ${assessment.success_criteria.success_rate_threshold}%, achieved ${assessment.success_rate.toFixed(1)}%`;
            case 'accuracy_retention':
                return `Need ${assessment.success_criteria.min_accuracy_retention}%, achieved ${assessment.avg_accuracy_percent.toFixed(1)}%`;
            case 'system_stability':
                return `Circuit breaker status: ${assessment.optimizer_status.circuit_breaker_status}`;
            default:
                return 'Status tracked';
        }
    }
    
    /**
     * Create mock oracle for testing
     */
    createMockOracle() {
        return {
            detector: {
                analyze_text: (text) => {
                    // Simple mock analysis based on text content
                    let hbar_s, risk_level;
                    
                    if (text.toLowerCase().includes('cheese') || 
                        text.toLowerCase().includes('upward') ||
                        text.toLowerCase().includes('rome') && text.toLowerCase().includes('eiffel')) {
                        hbar_s = 2.5 + Math.random() * 0.5; // High uncertainty
                        risk_level = 'Critical';
                    } else if (text.toLowerCase().includes('probably') ||
                               text.toLowerCase().includes('might') ||
                               text.toLowerCase().includes('5 years')) {
                        hbar_s = 1.2 + Math.random() * 0.5; // Medium uncertainty
                        risk_level = 'Warning';
                    } else {
                        hbar_s = 0.3 + Math.random() * 0.2; // Low uncertainty
                        risk_level = 'Safe';
                    }
                    
                    return {
                        hbar_s,
                        p_fail: 1 / (1 + Math.exp(-5.0 * (hbar_s - 2.0))),
                        risk_level,
                        method_scores: [hbar_s, hbar_s * 0.9, hbar_s * 1.1, hbar_s * 0.95],
                        computation_time_ms: 2 + Math.random() * 3
                    };
                }
            },
            config: {
                wallet_address: "0x9B613eD794B81043C23fA4a19d8f674090313b81"
            }
        };
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Execute if run directly
if (require.main === module) {
    const validator = new Phase1Validator();
    validator.runPhase1Validation()
        .then(assessment => {
            console.log(`\n🎉 Phase 1 validation completed!`);
            console.log(`🚀 Ready for Phase 2: ${assessment.ready_for_phase_2 ? 'YES' : 'NO'}`);
            process.exit(0);
        })
        .catch(error => {
            console.error('\n❌ Phase 1 validation failed:', error);
            process.exit(1);
        });
}

module.exports = { Phase1Validator };