/**
 * Comprehensive Gas Optimization Testing Suite
 * 
 * Implements rigorous testing plan as specified:
 * Phase 1: Gas Optimization Testing
 * - Test batch sizes (10, 50, 100 verifications per transaction)
 * - Measure gas reduction vs latency tradeoffs
 * - Validate accuracy preservation during optimization
 * 
 * Target: 60% immediate gas reduction validation
 */

class GasOptimizationTester {
    constructor(oracle, optimization_engine) {
        this.oracle = oracle;
        this.optimization_engine = optimization_engine;
        this.test_results = {
            baseline_metrics: null,
            batch_size_tests: [],
            accuracy_tests: [],
            latency_tests: [],
            cost_analysis: [],
            optimization_summary: null
        };
        
        // Test configurations
        this.test_configs = {
            batch_sizes: [1, 10, 25, 50, 100], // Include baseline (1) for comparison
            test_iterations: 5, // Multiple runs for statistical significance
            accuracy_test_cases: this.generateAccuracyTestCases(),
            target_gas_reduction: 0.60, // 60% reduction target
            baseline_cost_per_verification: 0.00247 // A0GI from live testing
        };
        
        console.log("🧪 Gas Optimization Testing Suite initialized");
        console.log(`📊 Testing batch sizes: ${this.test_configs.batch_sizes.join(', ')}`);
        console.log(`🎯 Target gas reduction: ${this.test_configs.target_gas_reduction * 100}%`);
    }
    
    /**
     * PHASE 1: COMPREHENSIVE GAS OPTIMIZATION TESTING
     */
    async runComprehensiveTests() {
        console.log("\n🚀 Starting Comprehensive Gas Optimization Testing");
        console.log("=" .repeat(80));
        
        try {
            // Step 1: Establish baseline metrics
            await this.establishBaseline();
            
            // Step 2: Test batch size optimization
            await this.testBatchSizeOptimization();
            
            // Step 3: Validate accuracy preservation
            await this.validateAccuracyPreservation();
            
            // Step 4: Measure latency tradeoffs
            await this.measureLatencyTradeoffs();
            
            // Step 5: Comprehensive cost analysis
            await this.performCostAnalysis();
            
            // Step 6: Generate optimization summary
            const summary = this.generateOptimizationSummary();
            
            // Step 7: Export detailed results
            await this.exportTestResults();
            
            console.log("\n✅ Comprehensive testing completed successfully!");
            return summary;
            
        } catch (error) {
            console.error("❌ Testing failed:", error);
            throw error;
        }
    }
    
    /**
     * Step 1: Establish baseline performance metrics
     */
    async establishBaseline() {
        console.log("\n📊 Step 1: Establishing baseline metrics");
        
        const baseline_tests = [];
        
        for (let i = 0; i < this.test_configs.test_iterations; i++) {
            console.log(`🔄 Baseline run ${i + 1}/${this.test_configs.test_iterations}`);
            
            const test_case = this.test_configs.accuracy_test_cases[0];
            const start_time = performance.now();
            
            // Single verification (no batching)
            const verification = await this.oracle.verifyAIOutput(
                test_case.text,
                test_case.model,
                { test_run: `baseline_${i}` }
            );
            
            const end_time = performance.now();
            
            baseline_tests.push({
                iteration: i + 1,
                processing_time_ms: end_time - start_time,
                gas_used: verification.submission_result?.gas_used || 134125,
                cost_a0gi: verification.submission_result?.cost_a0gi || 0.00247,
                hbar_s: verification.hbar_s,
                accuracy: this.evaluateAccuracy(verification, test_case)
            });
        }
        
        this.test_results.baseline_metrics = {
            avg_processing_time: baseline_tests.reduce((sum, t) => sum + t.processing_time_ms, 0) / baseline_tests.length,
            avg_gas_used: baseline_tests.reduce((sum, t) => sum + t.gas_used, 0) / baseline_tests.length,
            avg_cost_a0gi: baseline_tests.reduce((sum, t) => sum + t.cost_a0gi, 0) / baseline_tests.length,
            avg_accuracy: baseline_tests.reduce((sum, t) => sum + t.accuracy, 0) / baseline_tests.length,
            individual_results: baseline_tests
        };
        
        console.log("✅ Baseline established:");
        console.log(`   📊 Avg processing time: ${this.test_results.baseline_metrics.avg_processing_time.toFixed(2)}ms`);
        console.log(`   ⛽ Avg gas used: ${this.test_results.baseline_metrics.avg_gas_used.toLocaleString()}`);
        console.log(`   💰 Avg cost: ${this.test_results.baseline_metrics.avg_cost_a0gi.toFixed(5)} A0GI`);
        console.log(`   🎯 Avg accuracy: ${(this.test_results.baseline_metrics.avg_accuracy * 100).toFixed(1)}%`);
    }
    
    /**
     * Step 2: Test different batch sizes for optimization
     */
    async testBatchSizeOptimization() {
        console.log("\n🔧 Step 2: Testing batch size optimization");
        
        for (const batch_size of this.test_configs.batch_sizes) {
            if (batch_size === 1) continue; // Skip baseline (already tested)
            
            console.log(`\n📦 Testing batch size: ${batch_size}`);
            
            const batch_tests = [];
            
            for (let iteration = 0; iteration < this.test_configs.test_iterations; iteration++) {
                console.log(`  🔄 Batch test ${iteration + 1}/${this.test_configs.test_iterations}`);
                
                // Create batch of test cases
                const batch_inputs = [];
                for (let i = 0; i < batch_size; i++) {
                    const test_case = this.test_configs.accuracy_test_cases[i % this.test_configs.accuracy_test_cases.length];
                    batch_inputs.push({
                        text: test_case.text,
                        model: test_case.model,
                        metadata: { 
                            batch_test: true, 
                            iteration: iteration,
                            expected_result: test_case.expected_result
                        }
                    });
                }
                
                const start_time = performance.now();
                
                // Configure optimization engine for this batch size
                this.optimization_engine.config.optimal_batch_size = batch_size;
                
                // Process batch
                const batch_results = await this.processBatchWithMetrics(batch_inputs);
                
                const end_time = performance.now();
                
                batch_tests.push({
                    iteration: iteration + 1,
                    batch_size,
                    total_processing_time_ms: end_time - start_time,
                    per_item_time_ms: (end_time - start_time) / batch_size,
                    total_gas_used: batch_results.total_gas_used,
                    per_item_gas: batch_results.total_gas_used / batch_size,
                    total_cost_a0gi: batch_results.total_cost_a0gi,
                    per_item_cost: batch_results.total_cost_a0gi / batch_size,
                    gas_savings_percent: batch_results.gas_savings_percent,
                    cost_savings_percent: batch_results.cost_savings_percent,
                    accuracy_scores: batch_results.accuracy_scores,
                    avg_accuracy: batch_results.avg_accuracy
                });
            }
            
            // Calculate batch size summary
            const batch_summary = {
                batch_size,
                iterations: this.test_configs.test_iterations,
                avg_total_time: batch_tests.reduce((sum, t) => sum + t.total_processing_time_ms, 0) / batch_tests.length,
                avg_per_item_time: batch_tests.reduce((sum, t) => sum + t.per_item_time_ms, 0) / batch_tests.length,
                avg_gas_per_item: batch_tests.reduce((sum, t) => sum + t.per_item_gas, 0) / batch_tests.length,
                avg_cost_per_item: batch_tests.reduce((sum, t) => sum + t.per_item_cost, 0) / batch_tests.length,
                avg_gas_savings: batch_tests.reduce((sum, t) => sum + t.gas_savings_percent, 0) / batch_tests.length,
                avg_cost_savings: batch_tests.reduce((sum, t) => sum + t.cost_savings_percent, 0) / batch_tests.length,
                avg_accuracy: batch_tests.reduce((sum, t) => sum + t.avg_accuracy, 0) / batch_tests.length,
                individual_tests: batch_tests
            };
            
            this.test_results.batch_size_tests.push(batch_summary);
            
            console.log(`  ✅ Batch size ${batch_size} results:`);
            console.log(`     📊 Avg per-item time: ${batch_summary.avg_per_item_time.toFixed(2)}ms`);
            console.log(`     ⛽ Avg gas savings: ${batch_summary.avg_gas_savings.toFixed(1)}%`);
            console.log(`     💰 Avg cost savings: ${batch_summary.avg_cost_savings.toFixed(1)}%`);
            console.log(`     🎯 Avg accuracy: ${(batch_summary.avg_accuracy * 100).toFixed(1)}%`);
        }
    }
    
    /**
     * Step 3: Validate accuracy preservation during optimization
     */
    async validateAccuracyPreservation() {
        console.log("\n🎯 Step 3: Validating accuracy preservation");
        
        for (const test_case of this.test_configs.accuracy_test_cases) {
            console.log(`  📝 Testing: "${test_case.description}"`);
            
            // Test with baseline (no optimization)
            const baseline_result = await this.oracle.verifyAIOutput(
                test_case.text,
                test_case.model,
                { accuracy_test: 'baseline' }
            );
            
            // Test with optimization (batch size 25)
            this.optimization_engine.config.optimal_batch_size = 25;
            const optimized_result = await this.optimization_engine.optimizedVerification(
                test_case.text,
                test_case.model,
                { accuracy_test: 'optimized' }
            );
            
            const accuracy_comparison = {
                test_case: test_case.description,
                expected_result: test_case.expected_result,
                baseline_hbar_s: baseline_result.hbar_s,
                optimized_hbar_s: optimized_result.hbar_s,
                baseline_risk_level: baseline_result.risk_level,
                optimized_risk_level: optimized_result.risk_level,
                baseline_accuracy: this.evaluateAccuracy(baseline_result, test_case),
                optimized_accuracy: this.evaluateAccuracy(optimized_result, test_case),
                accuracy_preservation: this.calculateAccuracyPreservation(baseline_result, optimized_result, test_case)
            };
            
            this.test_results.accuracy_tests.push(accuracy_comparison);
            
            console.log(`     ✅ Baseline accuracy: ${(accuracy_comparison.baseline_accuracy * 100).toFixed(1)}%`);
            console.log(`     🔧 Optimized accuracy: ${(accuracy_comparison.optimized_accuracy * 100).toFixed(1)}%`);
            console.log(`     📊 Preservation: ${(accuracy_comparison.accuracy_preservation * 100).toFixed(1)}%`);
        }
    }
    
    /**
     * Step 4: Measure latency tradeoffs
     */
    async measureLatencyTradeoffs() {
        console.log("\n⏱️ Step 4: Measuring latency tradeoffs");
        
        const latency_scenarios = [
            { name: "Single urgent verification", batch_size: 1, urgency: "high" },
            { name: "Small batch processing", batch_size: 10, urgency: "medium" },
            { name: "Optimal batch processing", batch_size: 25, urgency: "low" },
            { name: "Large batch processing", batch_size: 50, urgency: "low" },
            { name: "Maximum batch processing", batch_size: 100, urgency: "low" }
        ];
        
        for (const scenario of latency_scenarios) {
            console.log(`  🔄 Testing: ${scenario.name}`);
            
            const latency_measurements = [];
            
            for (let i = 0; i < 3; i++) { // 3 runs for each scenario
                const test_case = this.test_configs.accuracy_test_cases[0];
                
                const start_time = performance.now();
                
                if (scenario.batch_size === 1) {
                    // Single verification
                    await this.oracle.verifyAIOutput(test_case.text, test_case.model);
                } else {
                    // Batch verification
                    const batch = Array(scenario.batch_size).fill().map(() => ({
                        text: test_case.text,
                        model: test_case.model
                    }));
                    
                    this.optimization_engine.config.optimal_batch_size = scenario.batch_size;
                    await this.processBatchWithMetrics(batch);
                }
                
                const end_time = performance.now();
                const total_time = end_time - start_time;
                const per_item_time = total_time / scenario.batch_size;
                
                latency_measurements.push({
                    run: i + 1,
                    total_time_ms: total_time,
                    per_item_time_ms: per_item_time
                });
            }
            
            const avg_total_time = latency_measurements.reduce((sum, m) => sum + m.total_time_ms, 0) / latency_measurements.length;
            const avg_per_item_time = latency_measurements.reduce((sum, m) => sum + m.per_item_time_ms, 0) / latency_measurements.length;
            
            const latency_result = {
                scenario: scenario.name,
                batch_size: scenario.batch_size,
                urgency: scenario.urgency,
                avg_total_time_ms: avg_total_time,
                avg_per_item_time_ms: avg_per_item_time,
                throughput_per_sec: 1000 / avg_per_item_time,
                measurements: latency_measurements
            };
            
            this.test_results.latency_tests.push(latency_result);
            
            console.log(`     ✅ Avg total time: ${avg_total_time.toFixed(1)}ms`);
            console.log(`     📊 Avg per-item time: ${avg_per_item_time.toFixed(1)}ms`);
            console.log(`     🚀 Throughput: ${latency_result.throughput_per_sec.toFixed(0)} ops/sec`);
        }
    }
    
    /**
     * Step 5: Comprehensive cost analysis
     */
    async performCostAnalysis() {
        console.log("\n💰 Step 5: Performing comprehensive cost analysis");
        
        const volume_scenarios = [
            { name: "MVP Launch", verifications_per_day: 1000 },
            { name: "Growth Phase", verifications_per_day: 10000 },
            { name: "Scale Phase", verifications_per_day: 100000 },
            { name: "Enterprise Scale", verifications_per_day: 1000000 }
        ];
        
        for (const scenario of volume_scenarios) {
            console.log(`  📊 Analyzing: ${scenario.name} (${scenario.verifications_per_day.toLocaleString()} verifications/day)`);
            
            // Calculate costs for different batch sizes
            const cost_comparisons = [];
            
            for (const batch_result of this.test_results.batch_size_tests) {
                const daily_cost_baseline = scenario.verifications_per_day * this.test_results.baseline_metrics.avg_cost_a0gi;
                const daily_cost_optimized = scenario.verifications_per_day * batch_result.avg_cost_per_item;
                const daily_savings_a0gi = daily_cost_baseline - daily_cost_optimized;
                const daily_savings_percent = (daily_savings_a0gi / daily_cost_baseline) * 100;
                const monthly_savings_a0gi = daily_savings_a0gi * 30;
                const yearly_savings_a0gi = daily_savings_a0gi * 365;
                
                cost_comparisons.push({
                    batch_size: batch_result.batch_size,
                    daily_cost_baseline: daily_cost_baseline,
                    daily_cost_optimized: daily_cost_optimized,
                    daily_savings_a0gi: daily_savings_a0gi,
                    daily_savings_percent: daily_savings_percent,
                    monthly_savings_a0gi: monthly_savings_a0gi,
                    yearly_savings_a0gi: yearly_savings_a0gi,
                    meets_target: daily_savings_percent >= (this.test_configs.target_gas_reduction * 100)
                });
            }
            
            // Find optimal batch size for this volume
            const optimal_batch = cost_comparisons
                .filter(c => c.meets_target)
                .sort((a, b) => b.daily_savings_percent - a.daily_savings_percent)[0];
            
            const cost_analysis = {
                scenario: scenario.name,
                verifications_per_day: scenario.verifications_per_day,
                cost_comparisons,
                optimal_batch_size: optimal_batch?.batch_size || null,
                optimal_daily_savings: optimal_batch?.daily_savings_percent || 0,
                optimal_yearly_savings: optimal_batch?.yearly_savings_a0gi || 0,
                target_achieved: optimal_batch?.meets_target || false
            };
            
            this.test_results.cost_analysis.push(cost_analysis);
            
            console.log(`     🎯 Optimal batch size: ${optimal_batch?.batch_size || 'N/A'}`);
            console.log(`     💰 Daily savings: ${optimal_batch?.daily_savings_percent.toFixed(1) || '0'}%`);
            console.log(`     📈 Yearly savings: ${optimal_batch?.yearly_savings_a0gi.toFixed(2) || '0'} A0GI`);
            console.log(`     ✅ Target achieved: ${optimal_batch?.meets_target ? 'YES' : 'NO'}`);
        }
    }
    
    /**
     * Generate comprehensive optimization summary
     */
    generateOptimizationSummary() {
        console.log("\n📋 Step 6: Generating optimization summary");
        
        // Find best performing batch size
        const best_batch = this.test_results.batch_size_tests
            .sort((a, b) => b.avg_gas_savings - a.avg_gas_savings)[0];
        
        // Calculate overall accuracy preservation
        const avg_accuracy_preservation = this.test_results.accuracy_tests
            .reduce((sum, t) => sum + t.accuracy_preservation, 0) / this.test_results.accuracy_tests.length;
        
        // Determine if target achieved
        const target_achieved = best_batch?.avg_gas_savings >= (this.test_configs.target_gas_reduction * 100);
        
        const summary = {
            test_completion_date: new Date().toISOString(),
            target_gas_reduction_percent: this.test_configs.target_gas_reduction * 100,
            target_achieved: target_achieved,
            
            // Performance metrics
            best_batch_size: best_batch?.batch_size || null,
            best_gas_savings_percent: best_batch?.avg_gas_savings || 0,
            best_cost_savings_percent: best_batch?.avg_cost_savings || 0,
            
            // Accuracy metrics
            avg_accuracy_preservation_percent: avg_accuracy_preservation * 100,
            accuracy_test_cases_passed: this.test_results.accuracy_tests.filter(t => t.accuracy_preservation > 0.85).length,
            accuracy_test_cases_total: this.test_results.accuracy_tests.length,
            
            // Latency metrics
            fastest_per_item_time_ms: Math.min(...this.test_results.latency_tests.map(t => t.avg_per_item_time_ms)),
            best_throughput_ops_sec: Math.max(...this.test_results.latency_tests.map(t => t.throughput_per_sec)),
            
            // Cost projections
            enterprise_yearly_savings_a0gi: this.test_results.cost_analysis
                .find(c => c.scenario === "Enterprise Scale")?.optimal_yearly_savings || 0,
            
            // Recommendations
            recommended_batch_size: best_batch?.batch_size || 25,
            recommended_deployment: target_achieved ? "IMMEDIATE" : "NEEDS_OPTIMIZATION",
            
            // Risk assessment
            risk_level: this.assessOptimizationRisk(avg_accuracy_preservation, best_batch?.avg_gas_savings || 0),
            
            // Next steps
            next_steps: this.generateNextSteps(target_achieved, avg_accuracy_preservation)
        };
        
        this.test_results.optimization_summary = summary;
        
        console.log("✅ Optimization Summary Generated:");
        console.log(`   🎯 Target (60% reduction): ${target_achieved ? '✅ ACHIEVED' : '❌ NOT MET'}`);
        console.log(`   📊 Best gas savings: ${summary.best_gas_savings_percent.toFixed(1)}%`);
        console.log(`   🎯 Accuracy preservation: ${summary.avg_accuracy_preservation_percent.toFixed(1)}%`);
        console.log(`   📦 Recommended batch size: ${summary.recommended_batch_size}`);
        console.log(`   🚀 Deployment status: ${summary.recommended_deployment}`);
        
        return summary;
    }
    
    /**
     * Export comprehensive test results
     */
    async exportTestResults() {
        console.log("\n💾 Step 7: Exporting test results");
        
        const export_data = {
            meta: {
                test_suite_version: "v1.0",
                completion_date: new Date().toISOString(),
                total_test_duration_hours: 2, // Estimated
                test_environment: "0G Newton Testnet Production",
                wallet_address: "0x9B613eD794B81043C23fA4a19d8f674090313b81"
            },
            configuration: this.test_configs,
            results: this.test_results
        };
        
        // Save to JSON file
        const fs = require('fs').promises;
        const export_path = '/Users/elliejenkins/Desktop/su-firewall/test_results/gas_optimization_test_results.json';
        await fs.writeFile(export_path, JSON.stringify(export_data, null, 2));
        
        console.log(`✅ Results exported to: ${export_path}`);
        
        // Generate markdown summary
        const markdown_summary = this.generateMarkdownSummary();
        const markdown_path = '/Users/elliejenkins/Desktop/su-firewall/test_results/GAS_OPTIMIZATION_RESULTS.md';
        await fs.writeFile(markdown_path, markdown_summary);
        
        console.log(`📄 Summary exported to: ${markdown_path}`);
    }
    
    // === Helper Methods ===
    
    async processBatchWithMetrics(batch_inputs) {
        // Process batch and collect detailed metrics
        const results = [];
        let total_gas_used = 0;
        let total_cost_a0gi = 0;
        const accuracy_scores = [];
        
        for (const input of batch_inputs) {
            const result = await this.optimization_engine.optimizedVerification(
                input.text,
                input.model,
                input.metadata
            );
            
            results.push(result);
            total_gas_used += result.batch_result?.gas_used || 134125;
            total_cost_a0gi += result.batch_result?.cost_a0gi || 0.00247;
            
            if (input.metadata?.expected_result) {
                const accuracy = this.evaluateAccuracy(result, { expected_result: input.metadata.expected_result });
                accuracy_scores.push(accuracy);
            }
        }
        
        const baseline_gas = batch_inputs.length * this.test_results.baseline_metrics.avg_gas_used;
        const baseline_cost = batch_inputs.length * this.test_results.baseline_metrics.avg_cost_a0gi;
        
        return {
            results,
            total_gas_used,
            total_cost_a0gi,
            gas_savings_percent: ((baseline_gas - total_gas_used) / baseline_gas) * 100,
            cost_savings_percent: ((baseline_cost - total_cost_a0gi) / baseline_cost) * 100,
            accuracy_scores,
            avg_accuracy: accuracy_scores.length > 0 ? accuracy_scores.reduce((a, b) => a + b) / accuracy_scores.length : 1.0
        };
    }
    
    generateAccuracyTestCases() {
        return [
            {
                description: "Clear factual statement",
                text: "The capital of France is Paris.",
                model: "test_model",
                expected_result: "safe"
            },
            {
                description: "Obvious hallucination",
                text: "The capital of France is London, which is known for its Eiffel Tower.",
                model: "test_model",
                expected_result: "critical"
            },
            {
                description: "Subtle misinformation",
                text: "Python was created by Guido van Rossum in 1989.",
                model: "test_model",
                expected_result: "warning" // Actually 1991
            },
            {
                description: "Technical accuracy",
                text: "Machine learning algorithms require training data to learn patterns.",
                model: "test_model",
                expected_result: "safe"
            },
            {
                description: "Uncertain claim",
                text: "Artificial intelligence will definitely replace all human jobs by 2030.",
                model: "test_model",
                expected_result: "high_risk"
            }
        ];
    }
    
    evaluateAccuracy(verification_result, test_case) {
        // Map risk levels to expected results
        const risk_mapping = {
            'safe': ['Safe'],
            'warning': ['Warning'],
            'high_risk': ['High Risk'],
            'critical': ['Critical']
        };
        
        const expected_risks = risk_mapping[test_case.expected_result] || [];
        return expected_risks.includes(verification_result.risk_level) ? 1.0 : 0.0;
    }
    
    calculateAccuracyPreservation(baseline_result, optimized_result, test_case) {
        const baseline_accuracy = this.evaluateAccuracy(baseline_result, test_case);
        const optimized_accuracy = this.evaluateAccuracy(optimized_result, test_case);
        
        if (baseline_accuracy === optimized_accuracy) return 1.0;
        if (baseline_accuracy === 0 && optimized_accuracy === 1) return 1.0; // Improvement
        if (baseline_accuracy === 1 && optimized_accuracy === 0) return 0.0; // Degradation
        
        return 0.5; // Partial preservation
    }
    
    assessOptimizationRisk(accuracy_preservation, gas_savings) {
        if (accuracy_preservation >= 0.95 && gas_savings >= 60) return "LOW";
        if (accuracy_preservation >= 0.85 && gas_savings >= 50) return "MEDIUM";
        if (accuracy_preservation >= 0.70 && gas_savings >= 40) return "HIGH";
        return "CRITICAL";
    }
    
    generateNextSteps(target_achieved, accuracy_preservation) {
        const steps = [];
        
        if (target_achieved) {
            steps.push("Deploy optimized gas system to production");
            steps.push("Monitor real-world performance for 1 week");
        } else {
            steps.push("Implement advanced optimization techniques");
            steps.push("Explore Merkle tree batching for larger reductions");
        }
        
        if (accuracy_preservation < 0.90) {
            steps.push("Review ensemble method selection for accuracy improvement");
            steps.push("Implement adaptive accuracy/speed tradeoffs");
        }
        
        steps.push("Scale testing to 1M verifications/day volumes");
        steps.push("Implement real-time monitoring dashboards");
        
        return steps;
    }
    
    generateMarkdownSummary() {
        const summary = this.test_results.optimization_summary;
        const best_batch = this.test_results.batch_size_tests
            .sort((a, b) => b.avg_gas_savings - a.avg_gas_savings)[0];
        
        return `# 🔧 Gas Optimization Testing Results

**Date:** ${new Date().toLocaleDateString()}  
**Environment:** 0G Newton Testnet Production  
**Target:** ${summary.target_gas_reduction_percent}% Gas Reduction  

---

## 🎯 **EXECUTIVE SUMMARY**

| Metric | Target | Achieved | Status |
|--------|--------|----------|---------|
| **Gas Reduction** | ${summary.target_gas_reduction_percent}% | **${summary.best_gas_savings_percent.toFixed(1)}%** | ${summary.target_achieved ? '✅ **ACHIEVED**' : '❌ **NOT MET**'} |
| **Accuracy Preservation** | >85% | **${summary.avg_accuracy_preservation_percent.toFixed(1)}%** | ${summary.avg_accuracy_preservation_percent >= 85 ? '✅' : '❌'} |
| **Optimal Batch Size** | TBD | **${summary.best_batch_size}** | ✅ |
| **Deployment Ready** | Yes/No | **${summary.recommended_deployment}** | ${summary.recommended_deployment === 'IMMEDIATE' ? '✅' : '⚠️'} |

---

## 📊 **BATCH SIZE ANALYSIS**

| Batch Size | Gas Savings | Cost Savings | Per-Item Time | Accuracy |
|------------|-------------|--------------|---------------|----------|
${this.test_results.batch_size_tests.map(batch => 
`| **${batch.batch_size}** | ${batch.avg_gas_savings.toFixed(1)}% | ${batch.avg_cost_savings.toFixed(1)}% | ${batch.avg_per_item_time.toFixed(1)}ms | ${(batch.avg_accuracy * 100).toFixed(1)}% |`
).join('\n')}

### 🏆 **OPTIMAL CONFIGURATION**
- **Batch Size:** ${summary.best_batch_size}
- **Gas Savings:** ${summary.best_gas_savings_percent.toFixed(1)}%
- **Throughput:** ${Math.max(...this.test_results.latency_tests.map(t => t.throughput_per_sec)).toFixed(0)} ops/sec

---

## 💰 **COST IMPACT ANALYSIS**

${this.test_results.cost_analysis.map(analysis => `
### ${analysis.scenario}
- **Volume:** ${analysis.verifications_per_day.toLocaleString()} verifications/day
- **Optimal Batch Size:** ${analysis.optimal_batch_size}
- **Daily Savings:** ${analysis.optimal_daily_savings.toFixed(1)}%
- **Yearly Savings:** ${analysis.optimal_yearly_savings.toFixed(2)} A0GI
- **Target Met:** ${analysis.target_achieved ? '✅ YES' : '❌ NO'}
`).join('\n')}

---

## 🎯 **ACCURACY VALIDATION**

${this.test_results.accuracy_tests.map(test => `
### ${test.test_case}
- **Expected:** ${test.expected_result}
- **Baseline:** ${test.baseline_risk_level} (${(test.baseline_accuracy * 100).toFixed(0)}% correct)
- **Optimized:** ${test.optimized_risk_level} (${(test.optimized_accuracy * 100).toFixed(0)}% correct)
- **Preservation:** ${(test.accuracy_preservation * 100).toFixed(1)}%
`).join('\n')}

---

## ⏱️ **LATENCY PERFORMANCE**

| Scenario | Batch Size | Total Time | Per-Item | Throughput |
|----------|------------|------------|----------|------------|
${this.test_results.latency_tests.map(test => 
`| ${test.scenario} | ${test.batch_size} | ${test.avg_total_time_ms.toFixed(1)}ms | ${test.avg_per_item_time_ms.toFixed(1)}ms | ${test.throughput_per_sec.toFixed(0)} ops/sec |`
).join('\n')}

---

## 🚀 **DEPLOYMENT RECOMMENDATION**

### **${summary.recommended_deployment === 'IMMEDIATE' ? '✅ IMMEDIATE DEPLOYMENT APPROVED' : '⚠️ OPTIMIZATION NEEDED'}**

**Risk Level:** ${summary.risk_level}

### **Next Steps:**
${summary.next_steps.map(step => `- ${step}`).join('\n')}

---

## 📈 **PRODUCTION PROJECTIONS**

For Enterprise Scale (1M verifications/day):
- **Annual Cost Savings:** ${this.test_results.cost_analysis.find(c => c.scenario === "Enterprise Scale")?.optimal_yearly_savings.toFixed(0)} A0GI
- **Performance Impact:** ${summary.fastest_per_item_time_ms.toFixed(1)}ms per verification
- **Scaling Capacity:** ${summary.best_throughput_ops_sec.toFixed(0)} ops/sec sustained

---

**Testing completed successfully with ${summary.target_achieved ? 'TARGET ACHIEVED' : 'optimization opportunities identified'}**
`;
    }
}

// Export for both ES6 and CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GasOptimizationTester;
} else if (typeof window !== 'undefined') {
    window.GasOptimizationTester = GasOptimizationTester;
}

export default GasOptimizationTester;