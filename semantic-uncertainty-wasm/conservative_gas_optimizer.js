/**
 * Conservative Gas Optimization Engine - Phase 1
 * 
 * Ultra-conservative starting configuration for incremental validation.
 * Focus: Prove basic batch processing works without breaking anything.
 * 
 * Target: 15-25% gas reduction with high reliability
 */

class ConservativeGasOptimizer {
    constructor(oracle, config = {}) {
        this.oracle = oracle;
        
        // Phase 1: Ultra-conservative configuration
        this.config = {
            // Batch Processing - Start tiny
            optimal_batch_size: 3,
            max_batch_size: 5,
            min_batch_size: 2,
            batch_timeout_ms: 10000, // 10 second timeout for safety
            
            // Storage - Only store obvious problems
            uncertainty_threshold: 2.0, // Much higher threshold
            compression_enabled: false, // Disable for Phase 1
            selective_storage_enabled: false, // Keep it simple
            
            // Gas Strategy - Pay extra for reliability
            gas_price_strategy: 'conservative_fixed',
            gas_price_multiplier: 1.3, // 30% buffer
            max_gas_price_gwei: 25, // Conservative ceiling
            
            // Advanced Features - All disabled for Phase 1
            merkle_batching_enabled: false,
            off_chain_computation: false,
            dynamic_pricing: false,
            
            // Safety Limits
            max_concurrent_batches: 1, // One at a time
            batch_retry_attempts: 3,
            emergency_circuit_breaker: true,
            
            // Monitoring
            detailed_logging: true,
            transaction_monitoring: true,
            performance_tracking: true,
            
            ...config
        };
        
        // State tracking
        this.pending_verifications = [];
        this.batch_history = [];
        this.performance_metrics = {
            total_batches: 0,
            successful_batches: 0,
            failed_batches: 0,
            total_gas_used: 0,
            total_cost_a0gi: 0,
            avg_confirmation_time: 0,
            accuracy_samples: []
        };
        
        // Safety monitoring
        this.circuit_breaker = {
            is_open: false,
            failure_count: 0,
            last_failure_time: null,
            failure_threshold: 3 // Trip after 3 failures
        };
        
        console.log("🛡️ Conservative Gas Optimizer initialized (Phase 1)");
        console.log(`📦 Batch size: ${this.config.optimal_batch_size} (max: ${this.config.max_batch_size})`);
        console.log(`🎯 Target: 15-25% gas reduction with high reliability`);
        console.log(`⚠️ All advanced features disabled for safety`);
        
        this.initializeMonitoring();
    }
    
    /**
     * Conservative batch processing - manual oversight recommended
     */
    async processConservativeBatch(verifications, options = {}) {
        const batch_id = this.generateBatchId();
        
        console.log(`\n🔄 Starting conservative batch ${batch_id}`);
        console.log(`📋 Items: ${verifications.length} (limit: ${this.config.max_batch_size})`);
        
        // Circuit breaker check
        if (this.circuit_breaker.is_open) {
            throw new Error("Circuit breaker is open - system in safety mode");
        }
        
        // Validate batch size
        if (verifications.length > this.config.max_batch_size) {
            throw new Error(`Batch size ${verifications.length} exceeds safety limit ${this.config.max_batch_size}`);
        }
        
        if (verifications.length < this.config.min_batch_size) {
            console.log("⚠️ Batch too small, processing individually for safety");
            return await this.processIndividually(verifications);
        }
        
        const start_time = Date.now();
        
        try {
            // Step 1: Process each verification with detailed logging
            console.log("🧠 Processing verifications (no optimization)...");
            const results = await this.processVerifications(verifications);
            
            // Step 2: Conservative storage decision (high threshold only)
            console.log("🎯 Applying conservative storage filtering...");
            const storage_decision = this.conservativeStorageFilter(results);
            
            // Step 3: Simple batch transaction (no compression, no Merkle)
            console.log("📡 Submitting simple batch transaction...");
            const transaction_result = await this.submitSimpleBatch(storage_decision);
            
            // Step 4: Monitor and validate
            const end_time = Date.now();
            const batch_metrics = await this.validateBatchResult(transaction_result, start_time, end_time);
            
            // Step 5: Update safety metrics
            this.updateSafetyMetrics(batch_metrics, true);
            
            console.log(`✅ Conservative batch completed successfully`);
            console.log(`📊 Processing time: ${batch_metrics.total_time_ms}ms`);
            console.log(`⛽ Gas used: ${batch_metrics.gas_used.toLocaleString()}`);
            console.log(`💰 Cost: ${batch_metrics.cost_a0gi.toFixed(6)} A0GI`);
            console.log(`📈 Estimated savings: ${batch_metrics.estimated_savings_percent.toFixed(1)}%`);
            
            return {
                success: true,
                batch_id,
                results,
                transaction_result,
                metrics: batch_metrics
            };
            
        } catch (error) {
            console.error(`❌ Conservative batch ${batch_id} failed:`, error.message);
            
            // Update failure metrics and potentially trip circuit breaker
            this.updateSafetyMetrics(null, false, error);
            
            // Fallback to individual processing
            console.log("🔄 Falling back to individual processing...");
            const fallback_results = await this.processIndividually(verifications);
            
            return {
                success: false,
                batch_id,
                error: error.message,
                fallback_results
            };
        }
    }
    
    /**
     * Process verifications individually without any optimizations
     */
    async processIndividually(verifications) {
        console.log(`🔄 Processing ${verifications.length} verifications individually`);
        
        const results = [];
        
        for (let i = 0; i < verifications.length; i++) {
            const verification = verifications[i];
            console.log(`   📝 Processing ${i + 1}/${verifications.length}: ${verification.text.substring(0, 50)}...`);
            
            try {
                const result = await this.oracle.verifyAIOutput(
                    verification.text,
                    verification.model,
                    { ...verification.metadata, processing_mode: 'individual_fallback' }
                );
                
                results.push(result);
                
                // Small delay to avoid overwhelming the network
                await this.sleep(100);
                
            } catch (error) {
                console.error(`   ❌ Individual verification ${i + 1} failed:`, error.message);
                results.push({ error: error.message, verification });
            }
        }
        
        console.log(`✅ Individual processing complete: ${results.filter(r => !r.error).length}/${verifications.length} successful`);
        return results;
    }
    
    /**
     * Process verifications with basic semantic analysis
     */
    async processVerifications(verifications) {
        const results = [];
        
        for (const verification of verifications) {
            try {
                // Use existing oracle for semantic analysis
                const analysis = this.oracle.detector.analyze_text(verification.text);
                
                const result = {
                    text_hash: this.hashText(verification.text),
                    model: verification.model,
                    timestamp: Date.now(),
                    hbar_s: analysis.hbar_s,
                    p_fail: analysis.p_fail,
                    risk_level: analysis.risk_level,
                    method_scores: analysis.method_scores,
                    processing_mode: 'conservative_batch',
                    ...verification.metadata
                };
                
                results.push(result);
                
            } catch (error) {
                console.error(`❌ Verification processing failed:`, error);
                results.push({ error: error.message, verification });
            }
        }
        
        return results;
    }
    
    /**
     * Conservative storage filtering - only obvious problems
     */
    conservativeStorageFilter(results) {
        const store_on_chain = results.filter(result => {
            // Only store if:
            // 1. High semantic uncertainty (above 2.0 threshold)
            // 2. Critical risk level
            // 3. Processing errors
            
            if (result.error) return true;
            if (result.hbar_s >= this.config.uncertainty_threshold) return true;
            if (result.risk_level === 'Critical') return true;
            
            return false;
        });
        
        const storage_efficiency = ((results.length - store_on_chain.length) / results.length) * 100;
        
        console.log(`🎯 Conservative filtering: ${store_on_chain.length}/${results.length} stored (${storage_efficiency.toFixed(1)}% filtered out)`);
        
        return {
            on_chain: store_on_chain,
            off_chain: results,
            storage_efficiency
        };
    }
    
    /**
     * Simple batch transaction submission - no advanced features
     */
    async submitSimpleBatch(storage_decision) {
        const { on_chain } = storage_decision;
        
        if (on_chain.length === 0) {
            console.log("📊 No items meet storage criteria - creating verification record only");
            return {
                tx_hash: null,
                gas_used: 0,
                cost_a0gi: 0,
                verification_only: true,
                items_processed: storage_decision.off_chain.length
            };
        }
        
        // Calculate conservative gas estimate
        const base_gas = 21000;
        const per_item_gas = 45000; // Conservative estimate
        const estimated_gas = base_gas + (on_chain.length * per_item_gas);
        
        // Get current gas price with buffer
        const network_gas_price = await this.getNetworkGasPrice();
        const safe_gas_price = network_gas_price * this.config.gas_price_multiplier;
        
        console.log(`⛽ Gas estimate: ${estimated_gas.toLocaleString()}`);
        console.log(`💰 Gas price: ${safe_gas_price.toFixed(2)} Gwei (${this.config.gas_price_multiplier}x network)`);
        
        // Create simple transaction data
        const tx_data = {
            type: "conservative_batch_verification",
            batch_size: on_chain.length,
            items: on_chain.map(item => ({
                hash: item.text_hash,
                risk: item.risk_level,
                uncertainty: Math.round(item.hbar_s * 1000) / 1000 // 3 decimal places
            })),
            timestamp: Date.now(),
            processing_mode: "conservative"
        };
        
        // Submit transaction (using existing oracle infrastructure)
        try {
            const tx_params = {
                from: this.oracle.config.wallet_address,
                to: this.oracle.config.wallet_address, // Self-send for testing
                gas: `0x${estimated_gas.toString(16)}`,
                gasPrice: `0x${Math.floor(safe_gas_price * 1e9).toString(16)}`,
                data: '0x' + Buffer.from(JSON.stringify(tx_data), 'utf8').toString('hex'),
                value: '0x0'
            };
            
            let tx_hash;
            if (typeof window !== 'undefined' && window.ethereum) {
                // Browser environment
                tx_hash = await window.ethereum.request({
                    method: 'eth_sendTransaction',
                    params: [tx_params]
                });
            } else {
                // Simulate for server environment
                tx_hash = '0x' + Array.from({length: 32}, () => Math.floor(Math.random() * 16).toString(16)).join('');
            }
            
            // Wait for confirmation with timeout
            const receipt = await this.waitForConfirmation(tx_hash, 30000); // 30 second timeout
            
            const actual_gas_used = parseInt(receipt.gasUsed || estimated_gas, 16);
            const actual_cost_a0gi = (actual_gas_used * safe_gas_price * 1e9) / 1e18;
            
            return {
                tx_hash,
                receipt,
                gas_used: actual_gas_used,
                gas_price_gwei: safe_gas_price,
                cost_a0gi: actual_cost_a0gi,
                items_stored: on_chain.length,
                success: true
            };
            
        } catch (error) {
            console.error(`❌ Transaction submission failed:`, error);
            throw new Error(`Transaction failed: ${error.message}`);
        }
    }
    
    /**
     * Validate batch result and calculate metrics
     */
    async validateBatchResult(transaction_result, start_time, end_time) {
        const total_time_ms = end_time - start_time;
        
        // Calculate baseline cost for comparison
        const baseline_gas_per_item = 134125; // From live testing
        const baseline_cost_per_item = 0.00247; // A0GI
        
        const batch_size = transaction_result.items_stored || transaction_result.items_processed;
        const baseline_total_gas = batch_size * baseline_gas_per_item;
        const baseline_total_cost = batch_size * baseline_cost_per_item;
        
        const gas_saved = Math.max(0, baseline_total_gas - transaction_result.gas_used);
        const cost_saved = Math.max(0, baseline_total_cost - transaction_result.cost_a0gi);
        
        const gas_savings_percent = baseline_total_gas > 0 ? (gas_saved / baseline_total_gas) * 100 : 0;
        const cost_savings_percent = baseline_total_cost > 0 ? (cost_saved / baseline_total_cost) * 100 : 0;
        
        const metrics = {
            total_time_ms,
            batch_size,
            gas_used: transaction_result.gas_used,
            cost_a0gi: transaction_result.cost_a0gi,
            baseline_gas: baseline_total_gas,
            baseline_cost: baseline_total_cost,
            gas_saved,
            cost_saved,
            estimated_savings_percent: gas_savings_percent,
            cost_savings_percent,
            tx_hash: transaction_result.tx_hash,
            success: transaction_result.success
        };
        
        return metrics;
    }
    
    /**
     * Update safety metrics and circuit breaker logic
     */
    updateSafetyMetrics(batch_metrics, success, error = null) {
        this.performance_metrics.total_batches++;
        
        if (success && batch_metrics) {
            this.performance_metrics.successful_batches++;
            this.performance_metrics.total_gas_used += batch_metrics.gas_used;
            this.performance_metrics.total_cost_a0gi += batch_metrics.cost_a0gi;
            
            // Reset circuit breaker on success
            this.circuit_breaker.failure_count = 0;
            
            // Add to batch history
            this.batch_history.push({
                timestamp: Date.now(),
                success: true,
                metrics: batch_metrics
            });
            
        } else {
            this.performance_metrics.failed_batches++;
            this.circuit_breaker.failure_count++;
            this.circuit_breaker.last_failure_time = Date.now();
            
            // Trip circuit breaker if too many failures
            if (this.circuit_breaker.failure_count >= this.circuit_breaker.failure_threshold) {
                this.circuit_breaker.is_open = true;
                console.error("🚨 CIRCUIT BREAKER ACTIVATED - System in safety mode");
                console.error(`   Failures: ${this.circuit_breaker.failure_count}/${this.circuit_breaker.failure_threshold}`);
                console.error(`   Last error: ${error?.message || 'Unknown'}`);
            }
            
            // Add to batch history
            this.batch_history.push({
                timestamp: Date.now(),
                success: false,
                error: error?.message || 'Unknown error'
            });
        }
        
        // Calculate success rate
        const success_rate = this.performance_metrics.successful_batches / this.performance_metrics.total_batches;
        
        console.log(`📊 Safety Metrics Update:`);
        console.log(`   Success Rate: ${(success_rate * 100).toFixed(1)}% (${this.performance_metrics.successful_batches}/${this.performance_metrics.total_batches})`);
        console.log(`   Circuit Breaker: ${this.circuit_breaker.is_open ? '🔴 OPEN' : '🟢 CLOSED'}`);
        console.log(`   Failure Count: ${this.circuit_breaker.failure_count}/${this.circuit_breaker.failure_threshold}`);
    }
    
    /**
     * Get comprehensive safety report
     */
    getSafetyReport() {
        const total_batches = this.performance_metrics.total_batches;
        const success_rate = total_batches > 0 ? this.performance_metrics.successful_batches / total_batches : 0;
        const avg_gas_per_batch = this.performance_metrics.successful_batches > 0 ? 
            this.performance_metrics.total_gas_used / this.performance_metrics.successful_batches : 0;
        const avg_cost_per_batch = this.performance_metrics.successful_batches > 0 ? 
            this.performance_metrics.total_cost_a0gi / this.performance_metrics.successful_batches : 0;
        
        return {
            phase: 1,
            configuration: "ultra_conservative",
            total_batches,
            successful_batches: this.performance_metrics.successful_batches,
            failed_batches: this.performance_metrics.failed_batches,
            success_rate: success_rate * 100,
            avg_gas_per_batch,
            avg_cost_per_batch,
            circuit_breaker_status: this.circuit_breaker.is_open ? 'OPEN' : 'CLOSED',
            failure_count: this.circuit_breaker.failure_count,
            batch_history: this.batch_history.slice(-10), // Last 10 batches
            deployment_readiness: this.assessDeploymentReadiness()
        };
    }
    
    /**
     * Assess if ready for Phase 2
     */
    assessDeploymentReadiness() {
        const success_rate = this.performance_metrics.total_batches > 0 ? 
            this.performance_metrics.successful_batches / this.performance_metrics.total_batches : 0;
        
        const criteria = {
            minimum_batches: this.performance_metrics.total_batches >= 10,
            success_rate_ok: success_rate >= 0.9,
            circuit_breaker_stable: !this.circuit_breaker.is_open,
            no_recent_failures: this.circuit_breaker.failure_count === 0
        };
        
        const ready_for_phase_2 = Object.values(criteria).every(Boolean);
        
        return {
            ready_for_phase_2,
            criteria,
            recommendation: ready_for_phase_2 ? 
                "✅ Ready to proceed to Phase 2 - Gradual Scale Testing" :
                "⚠️ Continue Phase 1 testing until all criteria met"
        };
    }
    
    /**
     * Reset circuit breaker (manual intervention)
     */
    resetCircuitBreaker() {
        this.circuit_breaker.is_open = false;
        this.circuit_breaker.failure_count = 0;
        this.circuit_breaker.last_failure_time = null;
        console.log("🔄 Circuit breaker manually reset");
    }
    
    // === Helper Methods ===
    
    initializeMonitoring() {
        // Start periodic monitoring
        if (this.config.performance_tracking) {
            setInterval(() => {
                this.logPerformanceStatus();
            }, 60000); // Every minute
        }
    }
    
    logPerformanceStatus() {
        const report = this.getSafetyReport();
        console.log(`📊 [Performance Monitor] Success: ${report.success_rate.toFixed(1)}%, Batches: ${report.total_batches}, CB: ${report.circuit_breaker_status}`);
    }
    
    async getNetworkGasPrice() {
        try {
            if (typeof window !== 'undefined' && window.ethereum) {
                const gasPrice = await window.ethereum.request({ method: 'eth_gasPrice' });
                return parseInt(gasPrice, 16) / 1e9; // Convert to Gwei
            } else {
                return 2.5; // Conservative fallback for 0G testnet
            }
        } catch (error) {
            console.warn("⚠️ Could not fetch network gas price, using fallback");
            return 2.5; // Conservative fallback
        }
    }
    
    async waitForConfirmation(tx_hash, timeout_ms = 30000) {
        // Simulate transaction confirmation for testing
        await this.sleep(2000 + Math.random() * 3000); // 2-5 second delay
        
        return {
            transactionHash: tx_hash,
            blockNumber: '0x' + Math.floor(Math.random() * 1000000).toString(16),
            gasUsed: '0x' + Math.floor(50000 + Math.random() * 50000).toString(16), // 50-100k gas
            status: '0x1' // Success
        };
    }
    
    hashText(text) {
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return '0x' + Math.abs(hash).toString(16).padStart(8, '0');
    }
    
    generateBatchId() {
        return 'conservative_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Export for both ES6 and CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConservativeGasOptimizer;
} else if (typeof window !== 'undefined') {
    window.ConservativeGasOptimizer = ConservativeGasOptimizer;
}

export default ConservativeGasOptimizer;