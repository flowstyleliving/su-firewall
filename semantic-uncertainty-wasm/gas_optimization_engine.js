/**
 * Gas Optimization Engine for 0G Semantic Uncertainty Firewall
 * 
 * Implements comprehensive gas optimization strategies:
 * - Batch Processing (60% reduction target)
 * - Data Compression (minimize on-chain storage)
 * - Selective Storage (high-uncertainty only)
 * - Dynamic Gas Price Adjustment
 * - Off-chain Computation with Proofs
 * - Merkle Tree Batching
 */

class GasOptimizationEngine {
    constructor(oracle, config = {}) {
        this.oracle = oracle;
        this.config = {
            // Batch Processing Configuration
            optimal_batch_size: config.optimal_batch_size || 25,
            max_batch_size: config.max_batch_size || 100,
            batch_timeout_ms: config.batch_timeout_ms || 5000,
            
            // Storage Optimization
            uncertainty_threshold: config.uncertainty_threshold || 1.5, // Only store high uncertainty
            compression_enabled: config.compression_enabled || true,
            
            // Gas Price Optimization
            gas_price_strategy: config.gas_price_strategy || 'dynamic', // 'fixed', 'dynamic', 'aggressive'
            gas_price_multiplier: config.gas_price_multiplier || 1.1,
            max_gas_price_gwei: config.max_gas_price_gwei || 50,
            
            // Advanced Features
            merkle_batching_enabled: config.merkle_batching_enabled || true,
            off_chain_computation: config.off_chain_computation || true,
            
            ...config
        };
        
        // Batch Management
        this.pending_verifications = [];
        this.batch_queue = [];
        this.batch_timer = null;
        
        // Gas Tracking
        this.gas_usage_history = [];
        this.current_gas_price = null;
        
        // Performance Metrics
        this.optimization_metrics = {
            total_verifications: 0,
            batched_verifications: 0,
            gas_saved: 0,
            cost_saved_a0gi: 0,
            avg_batch_size: 0,
            compression_ratio: 0
        };
        
        console.log("🔧 Gas Optimization Engine initialized");
        console.log(`📦 Optimal batch size: ${this.config.optimal_batch_size}`);
        console.log(`💰 Gas price strategy: ${this.config.gas_price_strategy}`);
        
        this.initializeOptimizations();
    }
    
    async initializeOptimizations() {
        try {
            // Initialize gas price monitoring
            await this.updateCurrentGasPrice();
            
            // Start batch processing timer
            this.startBatchProcessing();
            
            console.log("✅ Gas optimization systems online");
            
        } catch (error) {
            console.error("❌ Failed to initialize gas optimizations:", error);
        }
    }
    
    /**
     * IMMEDIATE OPTIMIZATION 1: BATCH PROCESSING
     * Group multiple verifications into single transactions for 60% gas reduction
     */
    async optimizedVerification(text, model, metadata = {}) {
        const verification_request = {
            text,
            model,
            metadata,
            timestamp: Date.now(),
            id: this.generateRequestId()
        };
        
        // Add to batch queue
        this.pending_verifications.push(verification_request);
        
        console.log(`📦 Added to batch queue (${this.pending_verifications.length}/${this.config.optimal_batch_size})`);
        
        // If batch is full, process immediately
        if (this.pending_verifications.length >= this.config.optimal_batch_size) {
            return await this.processBatch();
        }
        
        // Otherwise, set timer for batch processing
        this.scheduleBatchProcessing();
        
        // Return promise that resolves when batch is processed
        return new Promise((resolve, reject) => {
            verification_request.resolve = resolve;
            verification_request.reject = reject;
        });
    }
    
    async processBatch() {
        if (this.pending_verifications.length === 0) {
            return [];
        }
        
        const batch = [...this.pending_verifications];
        this.pending_verifications = [];
        
        console.log(`🚀 Processing batch of ${batch.length} verifications`);
        
        try {
            const start_time = performance.now();
            
            // STEP 1: Off-chain computation for all verifications
            const off_chain_results = await this.computeOffChain(batch);
            
            // STEP 2: Apply selective storage (only high uncertainty)
            const filtered_results = this.applySelectiveStorage(off_chain_results);
            
            // STEP 3: Compress data for on-chain storage
            const compressed_data = this.compressVerificationData(filtered_results);
            
            // STEP 4: Create Merkle tree for batch verification
            const merkle_proof = this.createMerkleProof(compressed_data);
            
            // STEP 5: Submit optimized batch transaction
            const batch_result = await this.submitOptimizedBatch(compressed_data, merkle_proof);
            
            const processing_time = performance.now() - start_time;
            
            // Update metrics
            this.updateOptimizationMetrics(batch, batch_result, processing_time);
            
            // Resolve all pending promises
            this.resolveBatchPromises(batch, off_chain_results, batch_result);
            
            console.log(`✅ Batch processed: ${batch.length} verifications in ${processing_time.toFixed(2)}ms`);
            console.log(`💰 Gas saved: ${batch_result.gas_saved} (${batch_result.gas_savings_percent.toFixed(1)}%)`);
            
            return off_chain_results;
            
        } catch (error) {
            console.error(`❌ Batch processing failed:`, error);
            this.rejectBatchPromises(batch, error);
            throw error;
        }
    }
    
    /**
     * IMMEDIATE OPTIMIZATION 2: OFF-CHAIN COMPUTATION
     * Perform semantic analysis off-chain, only store proofs on-chain
     */
    async computeOffChain(batch) {
        console.log(`🧠 Computing ${batch.length} verifications off-chain`);
        
        const results = [];
        
        for (const request of batch) {
            try {
                // Perform full semantic analysis off-chain
                const analysis = this.oracle.detector.analyze_text(request.text);
                
                // Create verification record
                const verification = {
                    request_id: request.id,
                    text_hash: this.hashText(request.text),
                    model: request.model,
                    timestamp: request.timestamp,
                    
                    // Semantic analysis results
                    hbar_s: analysis.hbar_s,
                    p_fail: analysis.p_fail,
                    risk_level: analysis.risk_level,
                    method_scores: analysis.method_scores,
                    
                    // Optimization metadata
                    computation_location: 'off_chain',
                    batch_id: this.generateBatchId(),
                    
                    ...request.metadata
                };
                
                results.push(verification);
                
            } catch (error) {
                console.error(`❌ Off-chain computation failed for request ${request.id}:`, error);
                
                // Create error result
                results.push({
                    request_id: request.id,
                    error: error.message,
                    text_hash: this.hashText(request.text),
                    model: request.model,
                    timestamp: request.timestamp
                });
            }
        }
        
        console.log(`✅ Off-chain computation complete: ${results.length} results`);
        return results;
    }
    
    /**
     * IMMEDIATE OPTIMIZATION 3: SELECTIVE STORAGE
     * Only store high-uncertainty results on-chain to minimize storage costs
     */
    applySelectiveStorage(results) {
        const filtered_results = results.filter(result => {
            // Store on-chain if:
            // 1. High semantic uncertainty (above threshold)
            // 2. Critical or High Risk classification
            // 3. Processing errors (for debugging)
            
            if (result.error) return true; // Always store errors
            if (result.hbar_s >= this.config.uncertainty_threshold) return true;
            if (result.risk_level === 'Critical' || result.risk_level === 'High Risk') return true;
            
            return false; // Low uncertainty results stay off-chain
        });
        
        const storage_efficiency = ((results.length - filtered_results.length) / results.length * 100);
        
        console.log(`🎯 Selective storage applied: ${filtered_results.length}/${results.length} stored (${storage_efficiency.toFixed(1)}% reduction)`);
        
        return {
            on_chain: filtered_results,
            off_chain: results,
            storage_efficiency
        };
    }
    
    /**
     * IMMEDIATE OPTIMIZATION 4: DATA COMPRESSION
     * Compress verification data to minimize on-chain storage costs
     */
    compressVerificationData(storage_result) {
        const { on_chain, off_chain } = storage_result;
        
        console.log(`🗜️ Compressing ${on_chain.length} on-chain verifications`);
        
        const compressed = {
            // Batch metadata
            batch_id: this.generateBatchId(),
            timestamp: Date.now(),
            total_verifications: off_chain.length,
            stored_verifications: on_chain.length,
            
            // Compressed verification data
            verifications: on_chain.map(v => this.compressVerification(v)),
            
            // Summary statistics for off-chain verifications
            off_chain_summary: this.createOffChainSummary(off_chain),
            
            // Merkle root for integrity verification
            merkle_root: null // Will be set by createMerkleProof
        };
        
        const original_size = JSON.stringify(on_chain).length;
        const compressed_size = JSON.stringify(compressed.verifications).length;
        const compression_ratio = (original_size - compressed_size) / original_size;
        
        console.log(`✅ Compression complete: ${(compression_ratio * 100).toFixed(1)}% size reduction`);
        
        this.optimization_metrics.compression_ratio = compression_ratio;
        
        return compressed;
    }
    
    compressVerification(verification) {
        // Use abbreviated field names and quantized values to reduce storage
        return {
            id: verification.request_id,
            h: verification.text_hash,
            m: verification.model,
            t: verification.timestamp,
            u: Math.round(verification.hbar_s * 10000), // 4 decimal precision
            p: Math.round(verification.p_fail * 10000),
            r: this.encodeRiskLevel(verification.risk_level),
            s: verification.method_scores.map(s => Math.round(s * 1000)), // 3 decimal precision
            e: verification.error ? verification.error.substring(0, 100) : null
        };
    }
    
    createOffChainSummary(off_chain_results) {
        const safe_results = off_chain_results.filter(r => r.risk_level === 'Safe');
        
        return {
            total: off_chain_results.length,
            safe: safe_results.length,
            avg_hbar_s: safe_results.reduce((sum, r) => sum + r.hbar_s, 0) / safe_results.length || 0,
            avg_p_fail: safe_results.reduce((sum, r) => sum + r.p_fail, 0) / safe_results.length || 0
        };
    }
    
    /**
     * ADVANCED OPTIMIZATION 1: MERKLE TREE BATCHING
     * Create cryptographic proof for batch integrity verification
     */
    createMerkleProof(compressed_data) {
        if (!this.config.merkle_batching_enabled) {
            return { merkle_root: null, proof: null };
        }
        
        console.log(`🌳 Creating Merkle proof for batch integrity`);
        
        // Create leaf hashes for each verification
        const leaves = compressed_data.verifications.map(v => this.hashObject(v));
        
        // Build Merkle tree
        const merkle_tree = this.buildMerkleTree(leaves);
        const merkle_root = merkle_tree[merkle_tree.length - 1][0]; // Root is at top level
        
        // Update compressed data with Merkle root
        compressed_data.merkle_root = merkle_root;
        
        console.log(`✅ Merkle root: ${merkle_root}`);
        
        return {
            merkle_root,
            proof: merkle_tree,
            leaf_count: leaves.length
        };
    }
    
    buildMerkleTree(leaves) {
        if (leaves.length === 0) return [];
        
        let current_level = leaves;
        const tree = [current_level];
        
        while (current_level.length > 1) {
            const next_level = [];
            
            for (let i = 0; i < current_level.length; i += 2) {
                const left = current_level[i];
                const right = current_level[i + 1] || left; // Handle odd number of nodes
                const parent = this.hashPair(left, right);
                next_level.push(parent);
            }
            
            current_level = next_level;
            tree.push(current_level);
        }
        
        return tree;
    }
    
    /**
     * IMMEDIATE OPTIMIZATION 5: DYNAMIC GAS PRICE OPTIMIZATION
     */
    async updateCurrentGasPrice() {
        try {
            const network_gas_price = await this.oracle.rpcCall('eth_gasPrice');
            const network_price_gwei = parseInt(network_gas_price, 16) / 1e9;
            
            let optimized_price_gwei;
            
            switch (this.config.gas_price_strategy) {
                case 'aggressive':
                    // 10% below network price for slower but cheaper transactions
                    optimized_price_gwei = Math.max(1, network_price_gwei * 0.9);
                    break;
                    
                case 'dynamic':
                    // Adjust based on network congestion and urgency
                    optimized_price_gwei = this.calculateDynamicGasPrice(network_price_gwei);
                    break;
                    
                case 'fixed':
                default:
                    optimized_price_gwei = network_price_gwei * this.config.gas_price_multiplier;
                    break;
            }
            
            // Cap at maximum gas price
            this.current_gas_price = Math.min(optimized_price_gwei, this.config.max_gas_price_gwei);
            
            console.log(`⛽ Gas price updated: ${this.current_gas_price.toFixed(2)} Gwei (network: ${network_price_gwei.toFixed(2)} Gwei)`);
            
        } catch (error) {
            console.error("❌ Failed to update gas price:", error);
            this.current_gas_price = 20; // Fallback to 20 Gwei
        }
    }
    
    calculateDynamicGasPrice(network_price_gwei) {
        // Analyze recent gas usage patterns and network congestion
        const recent_blocks = this.gas_usage_history.slice(-10);
        
        if (recent_blocks.length < 5) {
            // Not enough data, use conservative approach
            return network_price_gwei * 1.05;
        }
        
        const avg_confirmation_time = recent_blocks.reduce((sum, block) => sum + block.confirmation_time, 0) / recent_blocks.length;
        const congestion_factor = Math.max(0.8, Math.min(1.3, avg_confirmation_time / 2000)); // Target 2s confirmation
        
        return network_price_gwei * congestion_factor;
    }
    
    /**
     * OPTIMIZED BATCH TRANSACTION SUBMISSION
     */
    async submitOptimizedBatch(compressed_data, merkle_proof) {
        console.log(`📡 Submitting optimized batch transaction`);
        
        const start_time = performance.now();
        
        try {
            // Calculate gas estimates
            const base_gas = 21000; // Base transaction cost
            const storage_gas = this.estimateStorageGas(compressed_data);
            const computation_gas = 30000; // Contract computation overhead
            
            const total_estimated_gas = base_gas + storage_gas + computation_gas;
            
            // Optimize gas price
            await this.updateCurrentGasPrice();
            
            // Create optimized transaction
            const tx_params = {
                from: this.oracle.config.wallet_address,
                to: this.oracle.config.oracle_contract || this.oracle.config.wallet_address,
                gas: `0x${total_estimated_gas.toString(16)}`,
                gasPrice: `0x${Math.floor(this.current_gas_price * 1e9).toString(16)}`,
                data: this.encodeOptimizedData(compressed_data, merkle_proof),
                value: '0x0'
            };
            
            console.log(`⛽ Estimated gas: ${total_estimated_gas.toLocaleString()}`);
            console.log(`💰 Gas price: ${this.current_gas_price.toFixed(2)} Gwei`);
            
            // Submit transaction
            let tx_hash;
            if (window.ethereum) {
                tx_hash = await window.ethereum.request({
                    method: 'eth_sendTransaction',
                    params: [tx_params],
                });
            } else {
                // Use RPC for server-side submission
                const response = await this.oracle.rpcCall('eth_sendRawTransaction', [tx_params.data]);
                tx_hash = response.result;
            }
            
            // Wait for confirmation
            const receipt = await this.oracle.waitForTransactionConfirmation(tx_hash);
            
            const confirmation_time = performance.now() - start_time;
            
            // Calculate gas savings vs individual transactions
            const individual_cost = compressed_data.total_verifications * 134125; // Previous per-verification cost
            const batch_cost = parseInt(receipt.gasUsed, 16);
            const gas_saved = individual_cost - batch_cost;
            const gas_savings_percent = (gas_saved / individual_cost) * 100;
            
            // Track gas usage for future optimization
            this.gas_usage_history.push({
                timestamp: Date.now(),
                batch_size: compressed_data.total_verifications,
                gas_used: batch_cost,
                gas_price_gwei: this.current_gas_price,
                confirmation_time
            });
            
            // Keep only recent history
            if (this.gas_usage_history.length > 100) {
                this.gas_usage_history = this.gas_usage_history.slice(-50);
            }
            
            return {
                tx_hash,
                receipt,
                batch_size: compressed_data.total_verifications,
                gas_used: batch_cost,
                gas_saved,
                gas_savings_percent,
                confirmation_time,
                cost_a0gi: (batch_cost * this.current_gas_price * 1e9) / 1e18
            };
            
        } catch (error) {
            console.error(`❌ Batch transaction failed:`, error);
            throw error;
        }
    }
    
    /**
     * BATCH PROCESSING SCHEDULING
     */
    startBatchProcessing() {
        // Process batches every 5 seconds or when full
        this.batch_timer = setInterval(() => {
            if (this.pending_verifications.length > 0) {
                this.processBatch().catch(error => {
                    console.error("❌ Scheduled batch processing failed:", error);
                });
            }
        }, this.config.batch_timeout_ms);
    }
    
    scheduleBatchProcessing() {
        // Don't schedule if already scheduled or if batch is already full
        if (this.batch_timer || this.pending_verifications.length >= this.config.optimal_batch_size) {
            return;
        }
        
        // Schedule batch processing after timeout
        setTimeout(() => {
            if (this.pending_verifications.length > 0) {
                this.processBatch().catch(error => {
                    console.error("❌ Scheduled batch processing failed:", error);
                });
            }
        }, this.config.batch_timeout_ms);
    }
    
    /**
     * OPTIMIZATION METRICS AND REPORTING
     */
    updateOptimizationMetrics(batch, batch_result, processing_time) {
        this.optimization_metrics.total_verifications += batch.length;
        this.optimization_metrics.batched_verifications += batch.length;
        this.optimization_metrics.gas_saved += batch_result.gas_saved;
        this.optimization_metrics.cost_saved_a0gi += batch_result.gas_saved * this.current_gas_price * 1e9 / 1e18;
        
        // Calculate running average batch size
        const total_batches = Math.ceil(this.optimization_metrics.batched_verifications / this.config.optimal_batch_size);
        this.optimization_metrics.avg_batch_size = this.optimization_metrics.batched_verifications / total_batches;
    }
    
    getOptimizationReport() {
        const total_savings_percent = this.optimization_metrics.total_verifications > 0 ? 
            (this.optimization_metrics.gas_saved / (this.optimization_metrics.total_verifications * 134125)) * 100 : 0;
        
        return {
            ...this.optimization_metrics,
            total_savings_percent,
            current_gas_price_gwei: this.current_gas_price,
            batch_queue_length: this.pending_verifications.length,
            optimization_efficiency: this.calculateOptimizationEfficiency()
        };
    }
    
    calculateOptimizationEfficiency() {
        // Combine multiple efficiency metrics into single score
        const gas_efficiency = Math.min(100, this.optimization_metrics.compression_ratio * 100);
        const batch_efficiency = Math.min(100, (this.optimization_metrics.avg_batch_size / this.config.optimal_batch_size) * 100);
        const storage_efficiency = this.optimization_metrics.storage_efficiency || 0;
        
        return (gas_efficiency + batch_efficiency + storage_efficiency) / 3;
    }
    
    /**
     * UTILITY METHODS
     */
    generateRequestId() {
        return 'req_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
    }
    
    generateBatchId() {
        return 'batch_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
    }
    
    hashText(text) {
        // Simple hash for demo - use crypto.subtle.digest in production
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return '0x' + Math.abs(hash).toString(16).padStart(8, '0');
    }
    
    hashObject(obj) {
        const str = JSON.stringify(obj);
        return this.hashText(str);
    }
    
    hashPair(left, right) {
        return this.hashText(left + right);
    }
    
    encodeRiskLevel(risk_level) {
        const levels = { 'Safe': 0, 'Warning': 1, 'High Risk': 2, 'Critical': 3 };
        return levels[risk_level] || 0;
    }
    
    estimateStorageGas(compressed_data) {
        // Estimate gas cost for storing compressed data
        const data_size = JSON.stringify(compressed_data).length;
        return Math.ceil(data_size / 32) * 20000; // Approximate storage cost
    }
    
    encodeOptimizedData(compressed_data, merkle_proof) {
        // Encode data for blockchain storage
        const data = {
            compressed_data,
            merkle_proof: merkle_proof.merkle_root,
            optimization_version: 'v1.0'
        };
        
        return '0x' + Buffer.from(JSON.stringify(data), 'utf8').toString('hex');
    }
    
    resolveBatchPromises(batch, results, batch_result) {
        batch.forEach((request, index) => {
            if (request.resolve) {
                request.resolve({
                    ...results[index],
                    batch_result,
                    optimization_applied: true
                });
            }
        });
    }
    
    rejectBatchPromises(batch, error) {
        batch.forEach(request => {
            if (request.reject) {
                request.reject(error);
            }
        });
    }
    
    cleanup() {
        if (this.batch_timer) {
            clearInterval(this.batch_timer);
            this.batch_timer = null;
        }
    }
}

// Export for both ES6 and CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GasOptimizationEngine;
} else if (typeof window !== 'undefined') {
    window.GasOptimizationEngine = GasOptimizationEngine;
}

export default GasOptimizationEngine;