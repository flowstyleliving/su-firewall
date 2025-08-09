// 🚀 Semantic Uncertainty Runtime - Cloudflare Worker
// Core Equation: ℏₛ = √(Δμ × Δσ) - Fisher Information Implementation
// Precision via Cramér-Rao Bound, Flexibility via Fisher-Rao Metric

// Worker status
let workerStatus = 'active';

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type'
};

// Fisher Information Implementation
class FisherInformationAnalyzer {
  constructor() {
    this.epsilon = 1e-8; // Small value for numerical stability
  }

  // Calculate Fisher Information for a probability distribution
  calculateFisherInformation(probabilities) {
    const n = probabilities.length;
    let fisherInfo = 0;
    
    for (let i = 0; i < n; i++) {
      const p = Math.max(probabilities[i], this.epsilon);
      // Fisher information for multinomial distribution: I(θ) = n/p
      fisherInfo += 1 / p;
    }
    
    return fisherInfo;
  }

  // Calculate Cramér-Rao bound for precision (Δμ)
  calculatePrecision(promptFreq, outputFreq) {
    // Combine distributions for parameter estimation
    const combinedFreq = { ...promptFreq };
    for (const [word, freq] of Object.entries(outputFreq)) {
      combinedFreq[word] = (combinedFreq[word] || 0) + freq;
    }
    
    // Normalize to probabilities
    const total = Object.values(combinedFreq).reduce((sum, freq) => sum + freq, 0);
    const probabilities = Object.values(combinedFreq).map(freq => freq / total);
    
    // Calculate Fisher information
    const fisherInfo = this.calculateFisherInformation(probabilities);
    
    // Precision via Cramér-Rao bound: Δμ = 1/√(n * I(θ))
    const n = Object.keys(combinedFreq).length;
    const precision = 1 / Math.sqrt(n * fisherInfo);
    
    return Math.max(precision, this.epsilon);
  }

  // Calculate Fisher-Rao metric for flexibility (Δσ)
  calculateFlexibility(promptFreq, outputFreq) {
    // Create Fisher-Rao metric tensor
    const allWords = new Set([...Object.keys(promptFreq), ...Object.keys(outputFreq)]);
    const wordArray = Array.from(allWords);
    const n = wordArray.length;
    
    if (n === 0) return this.epsilon;
    
    // Calculate Fisher-Rao metric tensor g_ij
    let metricSum = 0;
    let totalPairs = 0;
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const wordI = wordArray[i];
        const wordJ = wordArray[j];
        
        // Get probabilities
        const pI = (promptFreq[wordI] || 0) + (outputFreq[wordI] || 0);
        const pJ = (promptFreq[wordJ] || 0) + (outputFreq[wordJ] || 0);
        
        if (pI > 0 && pJ > 0) {
          // Fisher-Rao metric: g_ij = E[∂log p/∂θ_i * ∂log p/∂θ_j]
          // For multinomial: g_ij = δ_ij / p_i where δ_ij is Kronecker delta
          const metric = (i === j) ? 1 / pI : 0;
          metricSum += metric;
          totalPairs++;
        }
      }
    }
    
    // Flexibility as inverse of average metric curvature
    const avgMetric = totalPairs > 0 ? metricSum / totalPairs : 1;
    const flexibility = 1 / (1 + avgMetric); // Normalized to [0,1]
    
    return Math.max(flexibility, this.epsilon);
  }

  // Calculate semantic uncertainty using Fisher information framework
  calculateSemanticUncertainty(prompt, output) {
    // Clean and normalize text
    const cleanPrompt = prompt.toLowerCase().replace(/[^\w\s]/g, ' ').trim();
    const cleanOutput = output.toLowerCase().replace(/[^\w\s]/g, ' ').trim();
    
    const promptWords = cleanPrompt.split(/\s+/).filter(word => word.length > 0);
    const outputWords = cleanOutput.split(/\s+/).filter(word => word.length > 0);
    
    if (promptWords.length === 0 || outputWords.length === 0) {
      return {
        precision: this.epsilon,
        flexibility: this.epsilon,
        semantic_uncertainty: this.epsilon
      };
    }
    
    // Calculate word frequency distributions
    const promptFreq = {};
    const outputFreq = {};
    
    promptWords.forEach(word => {
      promptFreq[word] = (promptFreq[word] || 0) + 1;
    });
    
    outputWords.forEach(word => {
      outputFreq[word] = (outputFreq[word] || 0) + 1;
    });
    
    // Calculate precision via Cramér-Rao bound
    const precision = this.calculatePrecision(promptFreq, outputFreq);
    
    // Calculate flexibility via Fisher-Rao metric
    const flexibility = this.calculateFlexibility(promptFreq, outputFreq);
    
    // Calculate semantic uncertainty: ℏₛ = √(Δμ × Δσ)
    const semanticUncertainty = Math.sqrt(precision * flexibility);
    
    return {
      precision,
      flexibility,
      semantic_uncertainty: semanticUncertainty
    };
  }
}

// Calculate semantic uncertainty using Fisher information framework
async function calculateSemanticUncertainty(prompt, output) {
  console.log('🚀 Using Fisher Information Framework');
  
  try {
    const analyzer = new FisherInformationAnalyzer();
    const result = analyzer.calculateSemanticUncertainty(prompt, output);
    
    // Determine risk level based on uncertainty
    let riskLevel = 'Safe';
    if (result.semantic_uncertainty < 0.3) {
      riskLevel = 'Critical';
    } else if (result.semantic_uncertainty < 0.5) {
      riskLevel = 'Warning';
    } else if (result.semantic_uncertainty < 0.7) {
      riskLevel = 'HighRisk';
    }
    
    return {
      method: 'fisher_information',
      core_equation: 'ℏₛ = √(Δμ × Δσ)',
      precision: result.precision,
      flexibility: result.flexibility,
      semantic_uncertainty: result.semantic_uncertainty,
      worker_debug: 'FISHER_INFORMATION_ACTIVE',
      raw_hbar: result.semantic_uncertainty,
      risk_level: riskLevel,
      processing_time_ms: 0,
      request_id: 'fisher-' + Date.now(),
      timestamp: new Date().toISOString()
    };
    
  } catch (error) {
    console.error('❌ Fisher information calculation failed:', error);
    throw error;
  }
}

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);
    
    // Health check endpoint
    if (url.pathname === '/health' || url.pathname === '/api/v1/health') {
      return new Response(JSON.stringify({
        status: 'healthy',
        runtime: 'semantic-uncertainty-runtime',
        version: '1.0.0',
        core_equation: 'ℏₛ = √(Δμ × Δσ)',
        worker_status: workerStatus,
        timestamp: new Date().toISOString()
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Analysis endpoint
    if (url.pathname === '/api/v1/analyze' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { prompt, output } = body;
        
        if (!prompt || !output) {
          return new Response(JSON.stringify({
            error: 'Missing required fields: prompt and output'
          }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }
        

        
        // Calculate semantic uncertainty
        const result = await calculateSemanticUncertainty(prompt, output);
        
        return new Response(JSON.stringify(result), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
        
      } catch (error) {
        console.error('Analysis endpoint error:', error);
        return new Response(JSON.stringify({
          error: 'Analysis failed',
          details: error.message
        }), {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
    }



    // Default response
    return new Response(JSON.stringify({
      error: 'Endpoint not found',
      available_endpoints: ['/health', '/api/v1/health', '/api/v1/analyze']
    }), { 
      status: 404,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    });
  }
};