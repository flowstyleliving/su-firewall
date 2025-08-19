/**
 * 0G Testnet Connection Test
 * 
 * This script tests the actual connection to 0G testnet
 * and validates RPC endpoints, chain info, and basic functionality.
 */

// Test configuration
const TEST_CONFIG = {
    // 0G Newton Testnet
    newton_rpc: "https://evmrpc-testnet.0g.ai",
    newton_chain_id: 16600,
    
    // 0G Galileo Testnet  
    galileo_rpc: "https://evmrpc-testnet.0g.ai",
    galileo_chain_id: 16601,
    
    // Alternative RPC endpoints
    alternative_rpcs: [
        "https://0g-json-rpc-public.originstake.com",
        "https://og-testnet-jsonrpc.blockhub.id"
    ],
    
    // Network info
    native_token: "A0GI",
    faucet_url: "https://faucet.0g.ai",
    explorer_url: "https://chainscan-newton.0g.ai"
};

class ZeroGConnectionTester {
    constructor() {
        this.results = {
            rpc_tests: [],
            network_info: {},
            performance_metrics: {},
            errors: []
        };
    }
    
    async runAllTests() {
        console.log("🧪 Starting 0G Testnet Connection Tests");
        console.log("=" * 50);
        
        try {
            // Test 1: RPC Endpoint Connectivity
            await this.testRPCConnectivity();
            
            // Test 2: Network Information
            await this.testNetworkInfo();
            
            // Test 3: Block Information
            await this.testBlockInfo();
            
            // Test 4: Gas Price Information
            await this.testGasInfo();
            
            // Test 5: Performance Metrics
            await this.testPerformance();
            
            // Test 6: Error Handling
            await this.testErrorHandling();
            
            this.generateTestReport();
            
        } catch (error) {
            console.error("❌ Test suite failed:", error);
            this.results.errors.push(`Test suite failure: ${error.message}`);
        }
    }
    
    async testRPCConnectivity() {
        console.log("\n🔗 Testing RPC Connectivity...");
        
        // Test primary Newton RPC
        await this.testSingleRPC("Newton Primary", TEST_CONFIG.newton_rpc);
        
        // Test alternative RPCs
        for (let i = 0; i < TEST_CONFIG.alternative_rpcs.length; i++) {
            await this.testSingleRPC(`Alternative ${i + 1}`, TEST_CONFIG.alternative_rpcs[i]);
        }
    }
    
    async testSingleRPC(name, rpcUrl) {
        const startTime = performance.now();
        
        try {
            console.log(`  Testing ${name}: ${rpcUrl}`);
            
            const response = await this.rpcCall(rpcUrl, "eth_blockNumber");
            const blockNumber = parseInt(response.result, 16);
            const responseTime = performance.now() - startTime;
            
            const result = {
                name,
                url: rpcUrl,
                status: "✅ SUCCESS",
                block_number: blockNumber,
                response_time_ms: responseTime.toFixed(2),
                error: null
            };
            
            this.results.rpc_tests.push(result);
            console.log(`    ✅ Success: Block ${blockNumber} (${responseTime.toFixed(2)}ms)`);
            
        } catch (error) {
            const result = {
                name,
                url: rpcUrl,
                status: "❌ FAILED",
                block_number: null,
                response_time_ms: null,
                error: error.message
            };
            
            this.results.rpc_tests.push(result);
            this.results.errors.push(`${name} RPC failed: ${error.message}`);
            console.log(`    ❌ Failed: ${error.message}`);
        }
    }
    
    async testNetworkInfo() {
        console.log("\n🌐 Testing Network Information...");
        
        const workingRPC = this.getWorkingRPC();
        if (!workingRPC) {
            console.log("  ⚠️ No working RPC found, skipping network tests");
            return;
        }
        
        try {
            // Test chain ID
            const chainIdResponse = await this.rpcCall(workingRPC, "eth_chainId");
            const chainId = parseInt(chainIdResponse.result, 16);
            
            // Test network version
            const netVersionResponse = await this.rpcCall(workingRPC, "net_version");
            const netVersion = netVersionResponse.result;
            
            this.results.network_info = {
                chain_id: chainId,
                net_version: netVersion,
                expected_newton_chain_id: TEST_CONFIG.newton_chain_id,
                expected_galileo_chain_id: TEST_CONFIG.galileo_chain_id,
                chain_match: chainId === TEST_CONFIG.newton_chain_id || chainId === TEST_CONFIG.galileo_chain_id
            };
            
            console.log(`  Chain ID: ${chainId}`);
            console.log(`  Net Version: ${netVersion}`);
            console.log(`  Expected: ${TEST_CONFIG.newton_chain_id} (Newton) or ${TEST_CONFIG.galileo_chain_id} (Galileo)`);
            console.log(`  Match: ${this.results.network_info.chain_match ? '✅' : '❌'}`);
            
        } catch (error) {
            console.log(`  ❌ Network info failed: ${error.message}`);
            this.results.errors.push(`Network info test failed: ${error.message}`);
        }
    }
    
    async testBlockInfo() {
        console.log("\n📦 Testing Block Information...");
        
        const workingRPC = this.getWorkingRPC();
        if (!workingRPC) return;
        
        try {
            // Get latest block
            const blockResponse = await this.rpcCall(workingRPC, "eth_getBlockByNumber", ["latest", false]);
            const block = blockResponse.result;
            
            if (block) {
                const blockInfo = {
                    number: parseInt(block.number, 16),
                    hash: block.hash,
                    timestamp: parseInt(block.timestamp, 16),
                    transaction_count: block.transactions.length,
                    gas_limit: parseInt(block.gasLimit, 16),
                    gas_used: parseInt(block.gasUsed, 16)
                };
                
                this.results.network_info.latest_block = blockInfo;
                
                console.log(`  Block Number: ${blockInfo.number}`);
                console.log(`  Block Hash: ${blockInfo.hash}`);
                console.log(`  Timestamp: ${new Date(blockInfo.timestamp * 1000).toISOString()}`);
                console.log(`  Transactions: ${blockInfo.transaction_count}`);
                console.log(`  Gas Used: ${blockInfo.gas_used.toLocaleString()} / ${blockInfo.gas_limit.toLocaleString()}`);
                
            } else {
                throw new Error("No block data returned");
            }
            
        } catch (error) {
            console.log(`  ❌ Block info failed: ${error.message}`);
            this.results.errors.push(`Block info test failed: ${error.message}`);
        }
    }
    
    async testGasInfo() {
        console.log("\n⛽ Testing Gas Information...");
        
        const workingRPC = this.getWorkingRPC();
        if (!workingRPC) return;
        
        try {
            // Get gas price
            const gasPriceResponse = await this.rpcCall(workingRPC, "eth_gasPrice");
            const gasPrice = parseInt(gasPriceResponse.result, 16);
            const gasPriceGwei = gasPrice / 1e9;
            
            this.results.network_info.gas_price = {
                wei: gasPrice,
                gwei: gasPriceGwei
            };
            
            console.log(`  Gas Price: ${gasPrice} wei (${gasPriceGwei.toFixed(2)} Gwei)`);
            
            // Estimate gas for a simple transaction
            const estimateResponse = await this.rpcCall(workingRPC, "eth_estimateGas", [{
                from: "0x0000000000000000000000000000000000000000",
                to: "0x0000000000000000000000000000000000000001",
                value: "0x0"
            }]);
            
            const gasEstimate = parseInt(estimateResponse.result, 16);
            this.results.network_info.gas_estimate = gasEstimate;
            
            console.log(`  Gas Estimate (simple tx): ${gasEstimate.toLocaleString()}`);
            
        } catch (error) {
            console.log(`  ❌ Gas info failed: ${error.message}`);
            this.results.errors.push(`Gas info test failed: ${error.message}`);
        }
    }
    
    async testPerformance() {
        console.log("\n⚡ Testing Performance Metrics...");
        
        const workingRPC = this.getWorkingRPC();
        if (!workingRPC) return;
        
        const iterations = 10;
        const times = [];
        
        console.log(`  Running ${iterations} requests to measure performance...`);
        
        try {
            for (let i = 0; i < iterations; i++) {
                const start = performance.now();
                await this.rpcCall(workingRPC, "eth_blockNumber");
                const elapsed = performance.now() - start;
                times.push(elapsed);
                
                // Small delay between requests
                await this.sleep(100);
            }
            
            const avgTime = times.reduce((a, b) => a + b) / times.length;
            const minTime = Math.min(...times);
            const maxTime = Math.max(...times);
            
            this.results.performance_metrics = {
                iterations,
                avg_response_time_ms: avgTime.toFixed(2),
                min_response_time_ms: minTime.toFixed(2),
                max_response_time_ms: maxTime.toFixed(2),
                requests_per_second: (1000 / avgTime).toFixed(2)
            };
            
            console.log(`  Average Response: ${avgTime.toFixed(2)}ms`);
            console.log(`  Min/Max: ${minTime.toFixed(2)}ms / ${maxTime.toFixed(2)}ms`);
            console.log(`  Est. Throughput: ${(1000 / avgTime).toFixed(2)} req/sec`);
            
        } catch (error) {
            console.log(`  ❌ Performance test failed: ${error.message}`);
            this.results.errors.push(`Performance test failed: ${error.message}`);
        }
    }
    
    async testErrorHandling() {
        console.log("\n🚨 Testing Error Handling...");
        
        const workingRPC = this.getWorkingRPC();
        if (!workingRPC) return;
        
        // Test invalid method
        try {
            await this.rpcCall(workingRPC, "invalid_method");
            console.log("  ⚠️ Expected error for invalid method, but got success");
        } catch (error) {
            console.log(`  ✅ Correct error for invalid method: ${error.message}`);
        }
        
        // Test invalid parameters
        try {
            await this.rpcCall(workingRPC, "eth_getBlockByNumber", ["invalid_block", false]);
            console.log("  ⚠️ Expected error for invalid block, but got success");
        } catch (error) {
            console.log(`  ✅ Correct error for invalid block: ${error.message}`);
        }
    }
    
    async rpcCall(rpcUrl, method, params = []) {
        const request = {
            jsonrpc: "2.0",
            method: method,
            params: params,
            id: Date.now()
        };
        
        const response = await fetch(rpcUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(`RPC Error: ${result.error.message || result.error}`);
        }
        
        return result;
    }
    
    getWorkingRPC() {
        const workingTest = this.results.rpc_tests.find(test => test.status.includes("SUCCESS"));
        return workingTest ? workingTest.url : null;
    }
    
    generateTestReport() {
        console.log("\n" + "=" * 60);
        console.log("📊 0G TESTNET CONNECTION TEST REPORT");
        console.log("=" * 60);
        
        // RPC Results
        console.log("\n🔗 RPC Endpoint Results:");
        this.results.rpc_tests.forEach(test => {
            console.log(`  ${test.status} ${test.name}`);
            console.log(`    URL: ${test.url}`);
            if (test.block_number) {
                console.log(`    Block: ${test.block_number} | Response: ${test.response_time_ms}ms`);
            }
            if (test.error) {
                console.log(`    Error: ${test.error}`);
            }
        });
        
        // Network Info
        if (Object.keys(this.results.network_info).length > 0) {
            console.log("\n🌐 Network Information:");
            const info = this.results.network_info;
            
            if (info.chain_id) {
                console.log(`  Chain ID: ${info.chain_id} ${info.chain_match ? '✅' : '❌'}`);
            }
            
            if (info.latest_block) {
                console.log(`  Latest Block: ${info.latest_block.number}`);
                console.log(`  Block Time: ${new Date(info.latest_block.timestamp * 1000).toLocaleString()}`);
            }
            
            if (info.gas_price) {
                console.log(`  Gas Price: ${info.gas_price.gwei.toFixed(2)} Gwei`);
            }
        }
        
        // Performance
        if (Object.keys(this.results.performance_metrics).length > 0) {
            console.log("\n⚡ Performance Metrics:");
            const perf = this.results.performance_metrics;
            console.log(`  Average Response: ${perf.avg_response_time_ms}ms`);
            console.log(`  Throughput: ~${perf.requests_per_second} req/sec`);
        }
        
        // Errors
        if (this.results.errors.length > 0) {
            console.log("\n❌ Errors Encountered:");
            this.results.errors.forEach(error => {
                console.log(`  • ${error}`);
            });
        }
        
        // Summary
        const successfulRPCs = this.results.rpc_tests.filter(test => test.status.includes("SUCCESS")).length;
        const totalRPCs = this.results.rpc_tests.length;
        
        console.log("\n📈 Test Summary:");
        console.log(`  RPC Success Rate: ${successfulRPCs}/${totalRPCs} (${(successfulRPCs/totalRPCs*100).toFixed(1)}%)`);
        console.log(`  Network Connectivity: ${successfulRPCs > 0 ? '✅ OPERATIONAL' : '❌ FAILED'}`);
        console.log(`  Ready for Production: ${successfulRPCs > 0 && this.results.errors.length === 0 ? '✅ YES' : '❌ NO'}`);
        
        console.log("\n🌐 Additional Resources:");
        console.log(`  Faucet: ${TEST_CONFIG.faucet_url}`);
        console.log(`  Explorer: ${TEST_CONFIG.explorer_url}`);
        console.log(`  Documentation: https://docs.0g.ai`);
        
        console.log("\n" + "=" * 60);
        
        return this.results;
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Export for use in both Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ZeroGConnectionTester;
} else if (typeof window !== 'undefined') {
    window.ZeroGConnectionTester = ZeroGConnectionTester;
}

// Auto-run if in Node.js environment
if (typeof require !== 'undefined' && require.main === module) {
    const tester = new ZeroGConnectionTester();
    tester.runAllTests().then(() => {
        process.exit(0);
    }).catch((error) => {
        console.error("Test suite crashed:", error);
        process.exit(1);
    });
}

export default ZeroGConnectionTester;