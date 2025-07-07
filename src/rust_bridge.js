// 🦀⚡ Rust-JavaScript Bridge for Semantic Uncertainty Engine
// Connects JavaScript Worker to Rust WASM when available, falls back to optimized JS

export class RustSemanticBridge {
  constructor() {
    this.wasmModule = null;
    this.isWasmLoaded = false;
    this.initPromise = this.initialize();
  }

  async initialize() {
    try {
      // Try to load WASM module
      if (typeof WebAssembly !== 'undefined') {
        // In production, this would load the actual WASM file
        // const wasmModule = await import('./core-engine/pkg/semantic_uncertainty_engine.js');
        // this.wasmModule = wasmModule;
        // this.isWasmLoaded = true;
        console.log('WASM not available, using optimized JavaScript fallback');
      }
    } catch (error) {
      console.warn('Failed to load WASM module, using JavaScript fallback:', error);
    }
  }

  async analyze(prompt, output, config = {}) {
    await this.initPromise;

    if (this.isWasmLoaded && this.wasmModule) {
      return this.analyzeWasm(prompt, output, config);
    } else {
      return this.analyzeJavaScript(prompt, output, config);
    }
  }

  // WASM-based analysis (when available)
  async analyzeWasm(prompt, output, config) {
    try {
      const analyzer = new this.wasmModule.WasmSemanticAnalyzer();
      const result = await analyzer.analyze(prompt, output);
      return JSON.parse(result);
    } catch (error) {
      console.error('WASM analysis failed, falling back to JavaScript:', error);
      return this.analyzeJavaScript(prompt, output, config);
    }
  }

  // Optimized JavaScript fallback (Rust algorithm ported)
  analyzeJavaScript(prompt, output, config = {}) {
    const startTime = Date.now();
    
    // Configuration with Rust-like defaults
    const rustConfig = {
      embeddingDims: config.embeddingDims || 128,
      maxSequenceLength: config.maxSequenceLength || 256,
      entropyMinThreshold: config.entropyMinThreshold || 0.1,
      jsMinThreshold: config.jsMinThreshold || 0.01,
      collapseThreshold: config.collapseThreshold || 1.0,
      fastMode: config.fastMode !== false,
      useSIMD: config.useSIMD !== false,
      ...config
    };

    try {
      // Generate embeddings using Rust-ported algorithm
      const promptEmbedding = this.embedTextFast(prompt, rustConfig);
      const outputEmbedding = this.embedTextFast(output, rustConfig);

      // Compute semantic uncertainty metrics (Rust algorithm)
      const deltaMu = this.computeDeltaMu(outputEmbedding, rustConfig);
      const deltaSigma = this.computeDeltaSigma(promptEmbedding, outputEmbedding, rustConfig);
      
      // Calculate ℏₛ = √(Δμ × Δσ) with stability check
      const hBarS = (deltaMu > 0 && deltaSigma > 0) ? 
        Math.sqrt(deltaMu * deltaSigma) : 0.0;

      const collapseRisk = hBarS < rustConfig.collapseThreshold;
      const processingTimeMs = Date.now() - startTime;

      return {
        request_id: this.generateRequestId(),
        hbar_s: parseFloat(hBarS.toFixed(4)),
        delta_mu: parseFloat(deltaMu.toFixed(4)),
        delta_sigma: parseFloat(deltaSigma.toFixed(4)),
        collapse_risk: collapseRisk,
        processing_time_ms: processingTimeMs,
        embedding_dims: rustConfig.embeddingDims,
        timestamp: new Date().toISOString(),
        engine: 'rust-js-bridge'
      };

    } catch (error) {
      console.error('JavaScript analysis failed:', error);
      throw new Error(`Semantic analysis failed: ${error.message}`);
    }
  }

  // Port of Rust embed_text_fast_inplace function
  embedTextFast(text, config) {
    if (!text || text.length === 0) {
      return new Array(config.embeddingDims).fill(0.0);
    }

    const embedding = new Array(config.embeddingDims).fill(0.0);
    
    // Deterministic hash-based embedding (ported from Rust)
    for (let i = 0; i < Math.min(text.length, config.maxSequenceLength); i++) {
      const char = text.charCodeAt(i);
      
      // Simple hash function (equivalent to Rust DefaultHasher approach)
      let hash = this.hashCombine(char, i);
      const idx = Math.abs(hash) % embedding.length;
      
      // Distribution function (ported from Rust)
      const value = Math.tanh(Math.sin(hash * 0.1) + Math.cos(i * 0.05));
      embedding[idx] += value;
    }

    // Normalize vector (ported from Rust normalize_simd)
    this.normalizeVector(embedding);
    
    return embedding;
  }

  // Port of Rust hash function
  hashCombine(a, b) {
    // Simple hash combination (simplified from Rust DefaultHasher)
    let hash = a;
    hash = ((hash << 5) - hash) + b;
    hash = hash & hash; // Convert to 32-bit integer
    return hash;
  }

  // Port of Rust normalize_simd function
  normalizeVector(vec) {
    const sumSquares = vec.reduce((sum, x) => sum + x * x, 0);
    
    if (sumSquares > 1e-10) {
      const invNorm = 1.0 / Math.sqrt(sumSquares);
      for (let i = 0; i < vec.length; i++) {
        vec[i] *= invNorm;
      }
    } else {
      // Handle zero vector (ported from Rust)
      const uniformVal = 1.0 / Math.sqrt(vec.length);
      vec.fill(uniformVal);
    }
  }

  // Port of Rust compute_delta_mu_simd function
  computeDeltaMu(embedding, config) {
    const entropy = this.entropyApprox(embedding);
    
    // Precision is inverse of entropy with stability (ported from Rust)
    return 1.0 / Math.max(Math.sqrt(entropy), config.entropyMinThreshold);
  }

  // Port of Rust entropy_approx_simd function
  entropyApprox(embedding) {
    let entropy = 0.0;
    
    for (const value of embedding) {
      const p = Math.abs(value);
      if (p > 1e-10) {
        entropy += -p * Math.log(p);
      }
    }
    
    return entropy;
  }

  // Port of Rust compute_delta_sigma_simd function
  computeDeltaSigma(promptEmb, outputEmb, config) {
    // Convert to probability distributions (ported from Rust)
    const p = this.toProbabilityFast(promptEmb);
    const q = this.toProbabilityFast(outputEmb);
    
    // Compute Jensen-Shannon divergence (ported from Rust)
    const jsDivergence = this.jsDivergence(p, q);
    
    return Math.max(Math.sqrt(jsDivergence), config.jsMinThreshold);
  }

  // Port of Rust to_probability_fast function
  toProbabilityFast(vec) {
    const sum = vec.reduce((sum, x) => sum + Math.abs(x), 0);
    
    if (sum > 1e-10) {
      return vec.map(x => Math.abs(x) / sum);
    } else {
      return new Array(vec.length).fill(1.0 / vec.length);
    }
  }

  // Port of Rust js_divergence_simd function
  jsDivergence(p, q) {
    let jsSum = 0.0;
    
    for (let i = 0; i < p.length; i++) {
      const pi = p[i];
      const qi = q[i];
      
      if (pi > 1e-10 && qi > 1e-10) {
        const m = (pi + qi) * 0.5;
        if (m > 1e-10) {
          jsSum += 0.5 * (pi * Math.log(pi / m) + qi * Math.log(qi / m));
        }
      }
    }
    
    return jsSum;
  }

  // Generate unique request ID (ported from Rust RequestId)
  generateRequestId() {
    return 'xxxx-xxxx-4xxx-yxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  // Performance metrics
  getEngineInfo() {
    return {
      engine_type: this.isWasmLoaded ? 'rust-wasm' : 'rust-js-bridge',
      wasm_available: this.isWasmLoaded,
      performance_mode: 'ultra-fast',
      algorithms: 'rust-ported'
    };
  }
}

// Export singleton instance
export const rustBridge = new RustSemanticBridge();