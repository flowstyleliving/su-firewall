/**
 * 🧮 PURE MATHEMATICAL SEMANTIC UNCERTAINTY SYSTEM
 * 
 * No text analysis - purely mathematical semantic uncertainty calculation
 * based on numerical feature vectors and statistical distributions
 */

class PureMathSemanticUncertainty {
    constructor() {
        this.name = "Pure Mathematical Semantic Uncertainty Engine";
        this.version = "1.0.0";
        this.golden_scale = 3.4;
        console.log(`🧮 ${this.name} v${this.version} initialized`);
    }
    
    /**
     * Calculate semantic uncertainty from pure numerical input
     * @param {Array} feature_vector - Numerical features [f1, f2, ..., fn]
     * @param {Object} distribution_params - Statistical distribution parameters
     * @returns {Object} Semantic uncertainty analysis
     */
    analyzeFromFeatures(feature_vector, distribution_params = null) {
        console.log(`🔢 Analyzing feature vector: [${feature_vector.slice(0, 5).map(f => f.toFixed(3)).join(', ')}${feature_vector.length > 5 ? '...' : ''}]`);
        
        // Extract mathematical properties
        const math_properties = this.extractMathematicalProperties(feature_vector);
        
        // Calculate core uncertainty components
        const delta_mu = this.calculatePrecisionDivergence(feature_vector, math_properties);
        const delta_sigma = this.calculateFlexibilityDivergence(feature_vector, math_properties);
        
        // Apply quantum-inspired uncertainty principle
        const raw_hbar_s = Math.sqrt(delta_mu * delta_sigma);
        const calibrated_hbar_s = raw_hbar_s * this.golden_scale;
        
        // Classification based on mathematical thresholds
        const classification = this.classifyUncertainty(calibrated_hbar_s);
        
        return {
            feature_vector,
            mathematical_properties: math_properties,
            delta_mu: delta_mu,
            delta_sigma: delta_sigma,
            raw_uncertainty: raw_hbar_s,
            calibrated_uncertainty: calibrated_hbar_s,
            classification: classification,
            computation_time: performance.now() % 1 // Simulated
        };
    }
    
    /**
     * Extract mathematical properties from feature vector
     */
    extractMathematicalProperties(features) {
        const n = features.length;
        const mean = features.reduce((sum, f) => sum + f, 0) / n;
        const variance = features.reduce((sum, f) => sum + Math.pow(f - mean, 2), 0) / n;
        const std_dev = Math.sqrt(variance);
        
        // Higher-order moments
        const skewness = this.calculateSkewness(features, mean, std_dev);
        const kurtosis = this.calculateKurtosis(features, mean, std_dev);
        
        // Information theoretic measures
        const entropy = this.calculateEntropy(features);
        const complexity = this.calculateKolmogorovComplexity(features);
        
        // Stability measures
        const autocorrelation = this.calculateAutocorrelation(features);
        const spectral_density = this.calculateSpectralDensity(features);
        
        return {
            dimension: n,
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
            entropy,
            complexity,
            autocorrelation,
            spectral_density,
            l2_norm: Math.sqrt(features.reduce((sum, f) => sum + f * f, 0)),
            max_eigenvalue: this.estimateMaxEigenvalue(features)
        };
    }
    
    /**
     * Calculate precision component (Δμ) using Jensen-Shannon-like divergence
     */
    calculatePrecisionDivergence(features, props) {
        // Measure how "concentrated" vs "dispersed" the feature distribution is
        const concentration_measure = props.std_dev / (Math.abs(props.mean) + 1e-8);
        const entropy_normalized = props.entropy / Math.log(features.length);
        
        // High concentration + low entropy = high precision (low Δμ)
        const precision_factor = concentration_measure * entropy_normalized;
        
        // Add non-linearity for better separation
        const delta_mu = Math.exp(-precision_factor) + (props.complexity / 100);
        
        return Math.max(0.01, Math.min(delta_mu, 10.0));
    }
    
    /**
     * Calculate flexibility component (Δσ) using KL-divergence-like measure  
     */
    calculateFlexibilityDivergence(features, props) {
        // Measure how "flexible" vs "rigid" the feature distribution is
        const flexibility_from_moments = Math.abs(props.skewness) + Math.abs(props.kurtosis - 3);
        const autocorr_flexibility = 1.0 - Math.abs(props.autocorrelation);
        const spectral_flexibility = props.spectral_density;
        
        // Combine flexibility measures
        const delta_sigma = (flexibility_from_moments + autocorr_flexibility + spectral_flexibility) / 3.0;
        
        return Math.max(0.01, Math.min(delta_sigma, 10.0));
    }
    
    /**
     * Mathematical helper functions
     */
    calculateSkewness(features, mean, std_dev) {
        if (std_dev === 0) return 0;
        const n = features.length;
        const sum_cubed = features.reduce((sum, f) => sum + Math.pow((f - mean) / std_dev, 3), 0);
        return sum_cubed / n;
    }
    
    calculateKurtosis(features, mean, std_dev) {
        if (std_dev === 0) return 3; // Normal kurtosis
        const n = features.length;
        const sum_fourth = features.reduce((sum, f) => sum + Math.pow((f - mean) / std_dev, 4), 0);
        return sum_fourth / n;
    }
    
    calculateEntropy(features) {
        // Discretize features into bins for entropy calculation
        const bins = 10;
        const min_val = Math.min(...features);
        const max_val = Math.max(...features);
        const bin_width = (max_val - min_val) / bins;
        
        if (bin_width === 0) return 0;
        
        const histogram = new Array(bins).fill(0);
        features.forEach(f => {
            const bin = Math.floor((f - min_val) / bin_width);
            const clamped_bin = Math.max(0, Math.min(bins - 1, bin));
            histogram[clamped_bin]++;
        });
        
        const n = features.length;
        let entropy = 0;
        histogram.forEach(count => {
            if (count > 0) {
                const p = count / n;
                entropy -= p * Math.log2(p);
            }
        });
        
        return entropy;
    }
    
    calculateKolmogorovComplexity(features) {
        // Approximate K-complexity using compression-like measure
        const n = features.length;
        const unique_values = new Set(features.map(f => f.toFixed(3))).size;
        const complexity_ratio = unique_values / n;
        
        // Add pattern detection
        let pattern_score = 0;
        for (let i = 1; i < n; i++) {
            const diff = Math.abs(features[i] - features[i-1]);
            if (diff < 0.01) pattern_score += 1; // Repetition detected
        }
        
        const pattern_complexity = 1.0 - (pattern_score / (n - 1));
        return (complexity_ratio + pattern_complexity) * 50; // Scale to reasonable range
    }
    
    calculateAutocorrelation(features) {
        if (features.length < 2) return 0;
        
        const n = features.length;
        const mean = features.reduce((sum, f) => sum + f, 0) / n;
        
        let numerator = 0;
        let denominator = 0;
        
        for (let i = 0; i < n - 1; i++) {
            numerator += (features[i] - mean) * (features[i + 1] - mean);
            denominator += Math.pow(features[i] - mean, 2);
        }
        
        return denominator === 0 ? 0 : numerator / denominator;
    }
    
    calculateSpectralDensity(features) {
        // Simple spectral density approximation using variance of differences
        if (features.length < 2) return 0;
        
        const diffs = [];
        for (let i = 1; i < features.length; i++) {
            diffs.push(features[i] - features[i-1]);
        }
        
        const diff_mean = diffs.reduce((sum, d) => sum + d, 0) / diffs.length;
        const diff_variance = diffs.reduce((sum, d) => sum + Math.pow(d - diff_mean, 2), 0) / diffs.length;
        
        return Math.sqrt(diff_variance);
    }
    
    estimateMaxEigenvalue(features) {
        // Power method approximation for dominant eigenvalue
        // Create simple covariance-like matrix and estimate largest eigenvalue
        const n = Math.min(features.length, 10); // Limit for performance
        let v = features.slice(0, n);
        
        // Normalize
        const norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
        if (norm > 0) {
            v = v.map(val => val / norm);
        }
        
        // Simple power iteration (few steps)
        for (let iter = 0; iter < 3; iter++) {
            const new_v = v.map(val => val * features.slice(0, n).reduce((sum, f) => sum + f * val, 0));
            const new_norm = Math.sqrt(new_v.reduce((sum, val) => sum + val * val, 0));
            if (new_norm > 0) {
                v = new_v.map(val => val / new_norm);
            }
        }
        
        return Math.abs(v.reduce((sum, val, i) => sum + val * features[i], 0));
    }
    
    /**
     * Classify uncertainty level based on mathematical thresholds
     */
    classifyUncertainty(hbar_s) {
        if (hbar_s < 2.0) {
            return {
                level: 'CRITICAL',
                class: 'critical',
                action: '🚫 REJECT - High mathematical uncertainty',
                confidence: 1.0 - (hbar_s / 2.0),
                description: 'Low semantic uncertainty indicates mathematical inconsistency'
            };
        } else if (hbar_s < 4.0) {
            return {
                level: 'WARNING', 
                class: 'warning',
                action: '⚠️ REVIEW - Moderate mathematical uncertainty',
                confidence: (hbar_s - 2.0) / 2.0,
                description: 'Moderate uncertainty suggests ambiguous mathematical properties'
            };
        } else {
            return {
                level: 'SAFE',
                class: 'safe', 
                action: '✅ ACCEPT - Low mathematical uncertainty',
                confidence: Math.min(1.0, (hbar_s - 4.0) / 4.0),
                description: 'High semantic uncertainty indicates mathematical consistency'
            };
        }
    }
    
    /**
     * Generate test vectors for different types of mathematical content
     */
    generateTestVectors() {
        return {
            // Chaotic/random features (should be CRITICAL - hallucination-like)
            chaotic: Array.from({length: 20}, () => Math.random() * 10 - 5),
            
            // Smooth/predictable features (should be SAFE - legitimate-like)
            smooth: Array.from({length: 20}, (_, i) => Math.sin(i * 0.1) + Math.cos(i * 0.05)),
            
            // Mixed features (should be WARNING - suspicious-like)  
            mixed: Array.from({length: 20}, (_, i) => i % 3 === 0 ? Math.random() * 5 : Math.sin(i * 0.2))
        };
    }
    
    /**
     * Run complete mathematical analysis demo
     */
    runMathDemo() {
        console.log("\n" + "🧮".repeat(60));
        console.log("🔢 PURE MATHEMATICAL SEMANTIC UNCERTAINTY DEMO");
        console.log("🧮".repeat(60));
        console.log("🎯 No text analysis - purely mathematical feature vectors");
        console.log("⚡ Quantum-inspired uncertainty: ℏₛ = √(Δμ × Δσ) × 3.4");
        console.log("🧮".repeat(60));
        
        const test_vectors = this.generateTestVectors();
        
        Object.entries(test_vectors).forEach(([type, features]) => {
            console.log(`\n🔍 ANALYZING ${type.toUpperCase()} MATHEMATICAL FEATURES:`);
            
            const result = this.analyzeFromFeatures(features);
            
            console.log(`📊 MATHEMATICAL ANALYSIS:`);
            console.log(`   🧮 Raw ℏₛ: ${result.raw_uncertainty.toFixed(3)}`);
            console.log(`   ⚡ Calibrated ℏₛ: ${result.calibrated_uncertainty.toFixed(3)}`);
            console.log(`   🎯 Classification: ${result.classification.level}`);
            console.log(`   📋 Action: ${result.classification.action}`);
            console.log(`   💡 Confidence: ${(result.classification.confidence * 100).toFixed(1)}%`);
            console.log(`   📈 Mathematical Properties:`);
            console.log(`      • Entropy: ${result.mathematical_properties.entropy.toFixed(3)}`);
            console.log(`      • Complexity: ${result.mathematical_properties.complexity.toFixed(3)}`);
            console.log(`      • Autocorrelation: ${result.mathematical_properties.autocorrelation.toFixed(3)}`);
            console.log(`      • Spectral Density: ${result.mathematical_properties.spectral_density.toFixed(3)}`);
        });
        
        console.log("\n💡 PURE MATH INSIGHTS:");
        console.log("   🧮 No text processing required - works on any numerical data");
        console.log("   ⚡ Universal semantic uncertainty for any mathematical representation");
        console.log("   📊 Physics-inspired uncertainty principle applied to feature spaces");
        console.log("   🔢 Could work on embeddings, sensor data, financial time series, etc.");
        
        console.log("\n🎉 PURE MATHEMATICAL SEMANTIC UNCERTAINTY COMPLETE!");
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PureMathSemanticUncertainty };
}

// Run demo if called directly
if (typeof require !== 'undefined' && require.main === module) {
    const system = new PureMathSemanticUncertainty();
    system.runMathDemo();
}