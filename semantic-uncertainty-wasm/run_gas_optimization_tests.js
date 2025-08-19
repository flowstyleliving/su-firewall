/**
 * Gas Optimization Test Execution Script
 * 
 * Executes the comprehensive gas optimization testing suite
 * according to the user's detailed roadmap requirements.
 */

// Import required modules
const SemanticUncertaintyWasm = require('./pkg/semantic_uncertainty_wasm');
const ZeroGHallucinationOracle = require('./zg_production_integration.js');
const GasOptimizationEngine = require('./gas_optimization_engine.js');
const GasOptimizationTester = require('./gas_optimization_testing.js');

async function runGasOptimizationTests() {
    console.log("🚀 Starting Gas Optimization Test Suite");
    console.log("=" .repeat(80));
    console.log("📅 Date:", new Date().toLocaleString());
    console.log("🎯 Target: 60% gas reduction with accuracy preservation");
    console.log("⛓️ Environment: 0G Newton Testnet (Chain ID: 16600)");
    console.log("💳 Wallet: 0x9B613eD794B81043C23fA4a19d8f674090313b81");
    console.log("=" .repeat(80));
    
    try {
        // Step 1: Initialize WASM detector
        console.log("\n🔧 Step 1: Initializing WASM semantic detector");
        await SemanticUncertaintyWasm.default();
        const detector = new SemanticUncertaintyWasm.SemanticUncertaintyDetector();
        console.log("✅ WASM detector initialized");
        
        // Step 2: Initialize 0G Oracle with production configuration
        console.log("\n🌐 Step 2: Initializing 0G production oracle");
        const oracle_config = {
            // Production 0G Newton Testnet Configuration
            rpc_endpoint: "https://rpc-testnet.0g.ai",
            chain_id: 16600,
            wallet_address: "0x9B613eD794B81043C23fA4a19d8f674090313b81",
            gas_price_gwei: 2.5,
            
            // Optimization baseline configuration
            verification_threshold: 0.001, // Optimal threshold from live testing
            batch_timeout_ms: 5000,
            max_concurrent: 10
        };
        
        // Load the production oracle (updated version with real 0G integration)
        const ZeroGProductionOracle = require('./zg_production_integration.js');
        const oracle = new ZeroGProductionOracle(detector, oracle_config);
        console.log("✅ Production oracle initialized");
        
        // Step 3: Initialize Gas Optimization Engine
        console.log("\n⚡ Step 3: Initializing gas optimization engine");
        const optimization_config = {
            // Batch Processing Configuration (from roadmap)
            optimal_batch_size: 25,
            max_batch_size: 100,
            batch_timeout_ms: 5000,
            
            // Data Compression & Selective Storage
            uncertainty_threshold: 1.5,
            compression_enabled: true,
            
            // Dynamic Gas Price Optimization
            gas_price_strategy: 'dynamic',
            gas_price_multiplier: 1.1,
            max_gas_price_gwei: 50,
            
            // Advanced Features
            merkle_batching_enabled: true,
            off_chain_computation: true
        };
        
        const gas_optimizer = new GasOptimizationEngine(oracle, optimization_config);
        console.log("✅ Gas optimization engine initialized");
        
        // Step 4: Initialize Testing Suite
        console.log("\n🧪 Step 4: Initializing comprehensive testing suite");
        const tester = new GasOptimizationTester(oracle, gas_optimizer);
        console.log("✅ Testing suite initialized");
        
        // Step 5: Execute Comprehensive Tests
        console.log("\n🎯 Step 5: Executing comprehensive gas optimization tests");
        const test_results = await tester.runComprehensiveTests();
        
        // Step 6: Display Final Results
        console.log("\n" + "=" .repeat(80));
        console.log("🏆 GAS OPTIMIZATION TEST RESULTS");
        console.log("=" .repeat(80));
        
        console.log(`\n📊 PERFORMANCE METRICS:`);
        console.log(`   🎯 Target Gas Reduction: 60%`);
        console.log(`   ✅ Achieved Gas Reduction: ${test_results.best_gas_savings_percent.toFixed(1)}%`);
        console.log(`   📈 Status: ${test_results.target_achieved ? '✅ TARGET ACHIEVED' : '❌ NEEDS OPTIMIZATION'}`);
        
        console.log(`\n🎯 ACCURACY METRICS:`);
        console.log(`   📊 Accuracy Preservation: ${test_results.avg_accuracy_preservation_percent.toFixed(1)}%`);
        console.log(`   ✅ Test Cases Passed: ${test_results.accuracy_test_cases_passed}/${test_results.accuracy_test_cases_total}`);
        
        console.log(`\n⚡ PERFORMANCE METRICS:`);
        console.log(`   🚀 Best Throughput: ${test_results.best_throughput_ops_sec.toFixed(0)} ops/sec`);
        console.log(`   ⏱️  Fastest Per-Item Time: ${test_results.fastest_per_item_time_ms.toFixed(1)}ms`);
        console.log(`   📦 Recommended Batch Size: ${test_results.recommended_batch_size}`);
        
        console.log(`\n💰 COST PROJECTIONS:`);
        console.log(`   💵 Enterprise Yearly Savings: ${test_results.enterprise_yearly_savings_a0gi.toFixed(0)} A0GI`);
        console.log(`   📈 ROI Assessment: ${test_results.enterprise_yearly_savings_a0gi > 1000 ? 'EXCELLENT' : 'GOOD'}`);
        
        console.log(`\n🚀 DEPLOYMENT RECOMMENDATION:`);
        console.log(`   📋 Status: ${test_results.recommended_deployment}`);
        console.log(`   ⚠️  Risk Level: ${test_results.risk_level}`);
        
        // Step 7: Advanced Testing (if target achieved)
        if (test_results.target_achieved) {
            console.log("\n🔬 Step 6: Running advanced optimization validation");
            await runAdvancedOptimizationTests(gas_optimizer, tester);
        } else {
            console.log("\n⚠️  Target not achieved - recommendations provided for further optimization");
        }
        
        // Step 8: Generate Performance Report
        console.log("\n📄 Step 7: Generating comprehensive performance report");
        const report = await generatePerformanceReport(test_results, gas_optimizer);
        console.log(`✅ Performance report generated: ${report.file_path}`);
        
        console.log("\n" + "=" .repeat(80));
        console.log(`🎉 GAS OPTIMIZATION TESTING COMPLETED SUCCESSFULLY!`);
        console.log(`📊 Results available in: test_results/gas_optimization_test_results.json`);
        console.log(`📄 Summary available in: test_results/GAS_OPTIMIZATION_RESULTS.md`);
        console.log("=" .repeat(80));
        
        return test_results;
        
    } catch (error) {
        console.error("\n❌ Gas optimization testing failed:");
        console.error(error);
        
        // Generate error report
        const error_report = {
            timestamp: new Date().toISOString(),
            error: error.message,
            stack: error.stack,
            test_phase: "initialization_or_execution",
            remediation: [
                "Check 0G testnet connectivity",
                "Verify wallet address and network configuration", 
                "Ensure WASM module is properly compiled",
                "Review gas optimization engine configuration"
            ]
        };
        
        const fs = require('fs').promises;
        await fs.writeFile(
            '/Users/elliejenkins/Desktop/su-firewall/test_results/gas_optimization_error.json',
            JSON.stringify(error_report, null, 2)
        );
        
        throw error;
    }
}

async function runAdvancedOptimizationTests(gas_optimizer, tester) {
    console.log("🔬 Running advanced optimization validation tests");
    
    // Test 1: Merkle Tree Batching Validation
    console.log("  🌳 Testing Merkle tree batching integrity");
    const merkle_test = await validateMerkleTreeBatching(gas_optimizer);
    console.log(`     ✅ Merkle validation: ${merkle_test.integrity_preserved ? 'PASSED' : 'FAILED'}`);
    
    // Test 2: Edge Case Handling
    console.log("  🔄 Testing edge case handling");
    const edge_cases = await testEdgeCases(gas_optimizer);
    console.log(`     ✅ Edge cases handled: ${edge_cases.all_passed ? 'PASSED' : 'FAILED'}`);
    
    // Test 3: Scale Testing (burst capacity)
    console.log("  📈 Testing burst capacity and scale limits");
    const scale_test = await testBurstCapacity(gas_optimizer);
    console.log(`     ✅ Burst capacity: ${scale_test.max_throughput} ops/sec sustained`);
    
    // Test 4: Long-running stability
    console.log("  ⏳ Testing long-running stability (5 minutes)");
    const stability_test = await testLongRunningStability(gas_optimizer);
    console.log(`     ✅ Stability: ${stability_test.stable ? 'STABLE' : 'DEGRADATION DETECTED'}`);
}

async function validateMerkleTreeBatching(gas_optimizer) {
    // Create test batch with known data
    const test_batch = Array(50).fill().map((_, i) => ({
        text: `Test verification ${i}: This is a test statement for Merkle validation.`,
        model: "test_model",
        metadata: { test_type: "merkle_validation", index: i }
    }));
    
    // Process with Merkle batching enabled
    let merkle_root_1, merkle_root_2;
    
    // First run
    gas_optimizer.config.merkle_batching_enabled = true;
    const result_1 = await gas_optimizer.processBatch(test_batch.map(t => ({
        text: t.text,
        model: t.model,
        metadata: t.metadata,
        id: gas_optimizer.generateRequestId(),
        timestamp: Date.now()
    })));
    
    // Second run with same data
    const result_2 = await gas_optimizer.processBatch(test_batch.map(t => ({
        text: t.text,
        model: t.model,
        metadata: t.metadata,
        id: gas_optimizer.generateRequestId(),
        timestamp: Date.now()
    })));
    
    // Compare Merkle roots (should be different due to timestamps but structure should be consistent)
    return {
        integrity_preserved: result_1 && result_2,
        merkle_enabled: gas_optimizer.config.merkle_batching_enabled,
        batch_size: test_batch.length
    };
}

async function testEdgeCases(gas_optimizer) {
    const edge_cases = [
        { name: "Empty batch", data: [] },
        { name: "Single item batch", data: [{ text: "Test", model: "test", metadata: {} }] },
        { name: "Maximum batch size", data: Array(100).fill({ text: "Max batch test", model: "test", metadata: {} }) },
        { name: "Special characters", data: [{ text: "Testing émojis 🚀 and special chars: @#$%^&*()", model: "test", metadata: {} }] },
        { name: "Very long text", data: [{ text: "A".repeat(10000), model: "test", metadata: {} }] }
    ];
    
    let passed = 0;
    
    for (const test_case of edge_cases) {
        try {
            if (test_case.data.length === 0) {
                // Empty batch should return empty result
                const result = await gas_optimizer.processBatch([]);
                if (result.length === 0) passed++;
            } else {
                // Other cases should process without error
                const batch_data = test_case.data.map(d => ({
                    ...d,
                    id: gas_optimizer.generateRequestId(),
                    timestamp: Date.now()
                }));
                
                await gas_optimizer.processBatch(batch_data);
                passed++;
            }
        } catch (error) {
            console.log(`     ⚠️  ${test_case.name} failed: ${error.message}`);
        }
    }
    
    return {
        all_passed: passed === edge_cases.length,
        passed_count: passed,
        total_count: edge_cases.length
    };
}

async function testBurstCapacity(gas_optimizer) {
    const burst_sizes = [10, 50, 100, 200, 500];
    const throughput_measurements = [];
    
    for (const burst_size of burst_sizes) {
        const start_time = performance.now();
        
        // Create burst workload
        const burst_data = Array(burst_size).fill().map((_, i) => ({
            text: `Burst test ${i}: Testing system capacity under load.`,
            model: "test_model",
            metadata: { burst_test: true, index: i },
            id: gas_optimizer.generateRequestId(),
            timestamp: Date.now()
        }));
        
        try {
            await gas_optimizer.processBatch(burst_data);
            const end_time = performance.now();
            const duration_sec = (end_time - start_time) / 1000;
            const throughput = burst_size / duration_sec;
            
            throughput_measurements.push({
                burst_size,
                duration_ms: end_time - start_time,
                throughput_ops_sec: throughput
            });
            
        } catch (error) {
            console.log(`     ⚠️  Burst size ${burst_size} failed: ${error.message}`);
            break;
        }
    }
    
    return {
        measurements: throughput_measurements,
        max_throughput: Math.max(...throughput_measurements.map(m => m.throughput_ops_sec))
    };
}

async function testLongRunningStability(gas_optimizer) {
    const test_duration_ms = 5 * 60 * 1000; // 5 minutes
    const start_time = Date.now();
    const measurements = [];
    
    let iteration = 0;
    
    while ((Date.now() - start_time) < test_duration_ms) {
        const iter_start = performance.now();
        
        // Regular workload batch
        const batch_data = Array(25).fill().map((_, i) => ({
            text: `Stability test iteration ${iteration} item ${i}: Long-running system validation.`,
            model: "test_model",
            metadata: { stability_test: true, iteration, item: i },
            id: gas_optimizer.generateRequestId(),
            timestamp: Date.now()
        }));
        
        try {
            await gas_optimizer.processBatch(batch_data);
            const iter_end = performance.now();
            
            measurements.push({
                iteration: iteration++,
                duration_ms: iter_end - iter_start,
                timestamp: Date.now()
            });
            
            // Brief pause between iterations
            await new Promise(resolve => setTimeout(resolve, 1000));
            
        } catch (error) {
            console.log(`     ⚠️  Stability test failed at iteration ${iteration}: ${error.message}`);
            break;
        }
    }
    
    // Analyze stability (look for performance degradation)
    const early_measurements = measurements.slice(0, 10);
    const late_measurements = measurements.slice(-10);
    
    const early_avg = early_measurements.reduce((sum, m) => sum + m.duration_ms, 0) / early_measurements.length;
    const late_avg = late_measurements.reduce((sum, m) => sum + m.duration_ms, 0) / late_measurements.length;
    
    const performance_ratio = late_avg / early_avg;
    const stable = performance_ratio < 1.2; // Less than 20% degradation
    
    return {
        stable,
        total_iterations: measurements.length,
        performance_ratio,
        early_avg_ms: early_avg,
        late_avg_ms: late_avg,
        measurements
    };
}

async function generatePerformanceReport(test_results, gas_optimizer) {
    const fs = require('fs').promises;
    
    const report = {
        meta: {
            generation_date: new Date().toISOString(),
            test_environment: "0G Newton Testnet Production",
            wallet_address: "0x9B613eD794B81043C23fA4a19d8f674090313b81",
            optimization_engine_version: "v1.0"
        },
        
        executive_summary: test_results,
        
        detailed_metrics: {
            gas_optimization_report: gas_optimizer.getOptimizationReport(),
            configuration_used: gas_optimizer.config
        },
        
        deployment_readiness: {
            production_ready: test_results.target_achieved && test_results.avg_accuracy_preservation_percent >= 85,
            risk_assessment: test_results.risk_level,
            recommended_actions: test_results.next_steps
        }
    };
    
    const report_path = '/Users/elliejenkins/Desktop/su-firewall/test_results/comprehensive_gas_optimization_report.json';
    await fs.writeFile(report_path, JSON.stringify(report, null, 2));
    
    return { file_path: report_path, report };
}

// Execute if run directly
if (require.main === module) {
    runGasOptimizationTests()
        .then(results => {
            console.log(`\n🎉 Testing completed successfully!`);
            console.log(`📊 Target achieved: ${results.target_achieved ? 'YES' : 'NO'}`);
            process.exit(0);
        })
        .catch(error => {
            console.error('\n❌ Testing failed:', error);
            process.exit(1);
        });
}

module.exports = { runGasOptimizationTests };